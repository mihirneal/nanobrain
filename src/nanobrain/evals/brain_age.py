"""Brain age regression: frozen encoder, pooled tokens (patch mean or cls), ridge head."""

import argparse
import datetime
import gzip
import json
import logging
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from huggingface_hub import snapshot_download
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import nanobrain.utils.misc as misc
from nanobrain.evals.models.base import ModelTransform, ModelWrapper
from nanobrain.evals.models.registry import create_model, list_models

logger = logging.getLogger()

ALPHAS = np.logspace(-3, 6, 19)
HF_REPO = "medarc/nanobrain-evals"
HF_DIR = "age"


def download_data() -> Path:
    """The repo's age/ dir, downloaded into $HF_HOME/hub unless already cached."""
    snapshot = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=f"{HF_DIR}/*")
    return Path(snapshot) / HF_DIR


def load_subjects(data_root: Path) -> list[dict]:
    """Load subjects from the validation layout."""
    rows = []
    for path in sorted(data_root.glob("preprocessed/*/*/t1w.nii.gz")):
        rel_dir = path.parent.relative_to(data_root / "preprocessed")
        labels = json.loads((data_root / "labels" / rel_dir / "labels.json").read_text())
        age = float(labels["age"])
        rows.append({"subject": rel_dir.parts[0], "t1w": path, "age": age})
    assert rows, f"no images found in {data_root}/preprocessed"
    return rows


class SubjectDataset(Dataset):
    def __init__(self, rows: list[dict], transform: ModelTransform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img = nib.Nifti1Image.from_bytes(gzip.decompress(self.rows[index]["t1w"].read_bytes()))
        return self.transform(img)


@torch.inference_mode()
def embed(
    model: ModelWrapper, dataset: SubjectDataset, device: torch.device, workers: int, pool: str
) -> np.ndarray:
    """(N, D) features, the mean of the patch tokens (avg) or the cls token (cls)."""
    loader = DataLoader(dataset, batch_size=1, num_workers=workers)
    features = []
    start = time.perf_counter()
    for index, batch in enumerate(loader):
        embeds = model(misc.send_data(batch, device))
        if pool == "cls":
            assert embeds.cls_embeds is not None, "the model has no cls token"
            features.append(embeds.cls_embeds.flatten(1))
        else:
            features.append(embeds.patch_embeds.mean(dim=1))
        if (index + 1) % 100 == 0:
            logger.info(f"embedded {index + 1}/{len(dataset)} ({time.perf_counter() - start:.0f}s)")
    return torch.cat(features).cpu().numpy()


def cross_validate(
    features: np.ndarray, ages: np.ndarray, seed: int = 0, n_folds: int = 20
) -> np.ndarray:
    """Out-of-fold age for every subject, each predicted by a head fit on the other folds."""
    oof = np.zeros(len(ages), dtype=float)
    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(folds.split(features)):
        head = Pipeline([("scaler", StandardScaler()), ("ridge", RidgeCV(alphas=ALPHAS))])
        head.fit(features[train], ages[train])
        oof[test] = head.predict(features[test])
        logger.info(
            f"fold {fold + 1}/{n_folds} n={len(test)} alpha={head[-1].alpha_:.3g} "
            f"r2={r2_score(ages[test], oof[test]):.3f} "
            f"mae={mean_absolute_error(ages[test], oof[test]):.2f}"
        )
    return oof


def metrics(y: np.ndarray, oof: np.ndarray) -> dict:
    return {
        "r2": float(r2_score(y, oof)),
        "mae": float(mean_absolute_error(y, oof)),
    }


def score(
    y: np.ndarray, oof: np.ndarray, seed: int = 0, n_boot: int = 2000, alpha: float = 0.05
) -> dict:
    """Both metrics, each with a percentile CI resampling subjects with replacement."""
    rng = np.random.default_rng(seed)
    resamples = rng.integers(0, len(y), size=(n_boot, len(y)))

    summary = {}
    for name, point in metrics(y, oof).items():
        samples = [metrics(y[rows], oof[rows])[name] for rows in resamples]
        low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        summary[name] = point
        summary[f"{name}_ci_low"] = float(low)
        summary[f"{name}_ci_high"] = float(high)
    return summary


def main(args: argparse.Namespace):
    if args.output_dir is None:
        run_name = args.ckpt_path.stem
        if args.random_init:
            run_name += "_random"
        if args.pool != "avg":
            run_name += f"_{args.pool}"
        args.output_dir = args.ckpt_path.parent / "evals" / "brain_age" / run_name
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    misc.setup_logging(logger, run_dir / "log.txt")
    misc.random_seed(args.seed)
    config = {key: str(value) for key, value in vars(args).items()}
    logger.info(f"brain age eval, {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"sha: {misc.git_sha()}")
    logger.info(f"config: {json.dumps(config)}")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    data_root = download_data()
    logger.info(f"data root: {data_root}")
    rows = load_subjects(data_root)
    ages = np.array([row["age"] for row in rows])
    logger.info(
        f"dataset: {len(rows)} subjects, age {ages.min():.0f}-{ages.max():.0f} "
        f"mean {ages.mean():.1f}"
    )

    transform, model = create_model(
        args.model, ckpt_path=args.ckpt_path, random_init=args.random_init
    )
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = SubjectDataset(rows, transform)
    features = embed(model, dataset, device, args.workers, args.pool)

    oof = cross_validate(features, ages, seed=args.seed)
    summary = score(ages, oof, seed=args.seed)

    preds = [
        {"subject": row["subject"], "age": float(age), "pred": float(pred)}
        for row, age, pred in zip(rows, ages, oof)
    ]
    (run_dir / "preds.json").write_text("".join(json.dumps(pred) + "\n" for pred in preds))

    record = {
        "model": args.model,
        "ckpt_path": str(args.ckpt_path),
        "random_init": args.random_init,
        "pool": args.pool,
        **summary,
    }
    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")
    logger.info("result: " + "  ".join(f"{k}={v:.4f}" for k, v in summary.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_models(), required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default <ckpt dir>/evals/brain_age/<ckpt>[_random][_cls]",
    )
    parser.add_argument(
        "--random-init", action="store_true", help="random init weights, as a baseline"
    )
    parser.add_argument(
        "--pool", choices=["avg", "cls"], default="avg", help="mean of patch tokens, or CLS token"
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
