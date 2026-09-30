"""Stroke lesion segmentation on ISLES-2022 + SOOP: frozen encoder, patch tokens, linear probes.
Protocol (fixed): 5-fold CV over subjects, stratified by dataset and lesion size. Voxels no token
covers are 0. Each modality gets the one probability cut that maximizes its mean out-of-fold
Dice. Dice, lesion-wise
F1 and absolute volume difference are reported per modality and averaged over modalities (the
mean of the per-modality means), with bootstrap CIs over subjects.
"""

import argparse
import datetime
import json
import logging
import math
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from nibabel.processing import resample_from_to, smooth_image
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import nanobrain.utils.misc as misc
from nanobrain.evals.models.base import ModelTransform, ModelWrapper
from nanobrain.evals.models.registry import create_model, list_models
from nanobrain.model import patchify3d, unpatchify3d

logger = logging.getLogger()

HF_REPO = "medarc/nanobrain-evals"
HF_DIR = "stroke"
MODALITIES = ("dwi", "adc", "flair")

# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------


def download_data() -> Path:
    snapshot = snapshot_download(HF_REPO, repo_type="dataset", allow_patterns=f"{HF_DIR}/*")
    return Path(snapshot) / HF_DIR


def load_subjects(data_root: Path) -> list[dict]:
    """Load subjects from the validation layout."""
    rows = []
    for seg_path in sorted(data_root.glob("labels/*/*/seg.nii.gz")):
        rel_dir = seg_path.parent.relative_to(data_root / "labels")
        image_dir = data_root / "preprocessed" / rel_dir
        labels = json.loads((seg_path.parent / "labels.json").read_text())
        images = {mod: image_dir / f"{mod}.nii.gz" for mod in MODALITIES}
        rows.append(
            {
                "subject": rel_dir.parts[0],
                "seg": seg_path,
                "images": {mod: path for mod, path in images.items() if path.exists()},
                "lesion_ml": float(labels["lesion_ml"]),
                "dataset": labels["dataset"],
            }
        )
    assert rows, f"no labels found in {data_root}/labels"
    return rows


def load(path: Path) -> nib.Nifti1Image:
    """The nii.gz as released, with a trailing singleton axis dropped (SOOP's DWI is 4D)."""
    return nib.funcs.squeeze_image(nib.load(path))


def to_spacing(img: nib.Nifti1Image, affine: np.ndarray) -> nib.Nifti1Image:
    """Smooth the axes finer than the target grid's voxels, like the model transform does for
    images, so resampling averages over each target voxel instead of sampling a point."""
    spacing = float(np.linalg.norm(affine[:3, :3], axis=0).min())
    zooms = img.header.get_zooms()[:3]
    if all(z >= spacing for z in zooms):
        return img
    return smooth_image(img, fwhm=[math.sqrt(max(spacing**2 - z**2, 0)) for z in zooms])


class SubjectDataset(Dataset):
    """One transformed sample per modality the subject has, each with the lesion mask resampled
    onto its grid as a lesion fraction per voxel."""

    def __init__(self, rows: list[dict], transform: ModelTransform, modalities: list[str]):
        self.rows = rows
        self.transform = transform
        self.modalities = modalities

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        seg_img = nib.load(row["seg"])
        seg_img = nib.Nifti1Image(np.asarray(seg_img.dataobj, dtype=np.float32), seg_img.affine)
        sample = {}
        for mod in self.modalities:
            if mod in row["images"]:
                sample[mod] = self.transform(load(row["images"][mod]))
                grid = (tuple(sample[mod]["image"].shape), sample[mod]["affine"].numpy())
                seg = resample_from_to(to_spacing(seg_img, grid[1]), grid, order=1)
                sample[mod]["seg"] = torch.from_numpy(seg.get_fdata(dtype=np.float32))
        return sample


# ----------------------------------------------------------------------------------------------
# features
# ----------------------------------------------------------------------------------------------


@torch.inference_mode()
def embed(
    model: ModelWrapper,
    dataset: SubjectDataset,
    device: torch.device,
    workers: int,
) -> tuple[list[dict], dict]:
    """Per subject and modality: token ids on the patch grid, token features, the lesion
    fraction of every voxel of every token's patch, and the grid affine. Plus the patch
    geometry, the same for every sample."""
    loader = DataLoader(dataset, batch_size=1, num_workers=workers)
    subjects, geometry = [], None
    lost = {mod: 0.0 for mod in dataset.modalities}
    total = {mod: 0.0 for mod in dataset.modalities}
    start = time.perf_counter()
    for index, batch in enumerate(loader):
        subject = {}
        for mod in [mod for mod in dataset.modalities if mod in batch]:
            sample = misc.send_data(batch[mod], device)
            grid = model(sample).grid_embeds[0]  # [D gx gy gz]
            if geometry is None:
                image_shape = tuple(sample["image"].shape[1:])
                patch = image_shape[0] // grid.shape[1]
                assert image_shape == tuple(g * patch for g in grid.shape[1:]), (
                    f"patch grid {tuple(grid.shape[1:])} doesn't tile the image {image_shape}"
                )
                geometry = {"image_shape": image_shape, "patch": patch}
                logger.info(f"geometry: {geometry}")

            # tokens are where the model embedded a patch
            ids = (grid.abs().amax(0) > 0).flatten().nonzero()[:, 0]
            targets = patchify3d(sample["seg"], geometry["patch"])[0, ids]
            lost[mod] += float(sample["seg"].sum() - targets.sum())
            total[mod] += float(sample["seg"].sum())

            subject[mod] = {
                "ids": ids.cpu(),
                "feats": grid.flatten(1).T[ids].to(torch.bfloat16).cpu(),
                "targets": targets.half().cpu(),
                "affine": batch[mod]["affine"][0].numpy(),
            }
        subjects.append(subject)
        if (index + 1) % 50 == 0:
            logger.info(f"embedded {index + 1}/{len(dataset)} ({time.perf_counter() - start:.0f}s)")
    lost_pct = " ".join(f"{m}={100 * lost[m] / max(total[m], 1e-6):.2f}%" for m in lost)
    logger.info(f"lesion volume no token covers: {lost_pct}")
    return subjects, geometry


# ----------------------------------------------------------------------------------------------
# linear probe
# ----------------------------------------------------------------------------------------------

# fixed for every model, never tuned per model
LINEAR_EPOCHS = 10
LINEAR_BATCH = 4096
LINEAR_LR = 1e-3
LINEAR_WD = 1e-4


def feature_stats(samples: list[dict], device: torch.device) -> tuple[Tensor, Tensor]:
    """Per feature mean and std over the training tokens."""
    total, total_sq, n = 0.0, 0.0, 0
    for s in samples:
        x = s["feats"].to(device).float()
        total = total + x.sum(0)
        total_sq = total_sq + (x**2).sum(0)
        n += len(x)
    mean = total / n
    std = (total_sq / n - mean**2).clamp_min(1e-12).sqrt()
    return mean, std


def cosine_lr(step: int, total: int, lr: float) -> float:
    return 0.5 * lr * (1 + math.cos(math.pi * step / total))


def fit_linear(train: list[dict], geometry: dict, device: torch.device, seed: int):
    """The probe of one modality, fit on the training subjects' samples of that modality. Each
    token predicts the lesion fraction of every voxel of its patch."""
    x = torch.cat([s["feats"] for s in train]).to(device)
    y = torch.cat([s["targets"] for s in train]).to(device)
    mean, std = feature_stats(train, device)

    torch.manual_seed(seed)
    head = nn.Linear(x.shape[1], y.shape[1]).to(device)
    nn.init.zeros_(head.weight)
    rate = y.float().mean().clamp(1e-6, 1 - 1e-6)
    nn.init.constant_(head.bias, float(torch.log(rate / (1 - rate))))
    opt = torch.optim.AdamW(head.parameters(), lr=LINEAR_LR, weight_decay=LINEAR_WD)

    steps_per_epoch = math.ceil(len(x) / LINEAR_BATCH)
    total, step = LINEAR_EPOCHS * steps_per_epoch, 0
    for _ in range(LINEAR_EPOCHS):
        for idx in torch.randperm(len(x), device=device).split(LINEAR_BATCH):
            for group in opt.param_groups:
                group["lr"] = cosine_lr(step, total, LINEAR_LR)
            logits = head((x[idx].float() - mean) / std)
            loss = F.binary_cross_entropy_with_logits(logits, y[idx].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1

    @torch.inference_mode()
    def predict(s: dict) -> Tensor:
        """The probability of every voxel of the sample's grid, 0 where there's no token."""
        probs = torch.sigmoid(head((s["feats"].to(device).float() - mean) / std))
        num_patches = math.prod(size // geometry["patch"] for size in geometry["image_shape"])
        dense = probs.new_zeros(num_patches, probs.shape[1])
        dense[s["ids"].to(device)] = probs
        return unpatchify3d(dense[None], geometry["image_shape"], geometry["patch"])[0]

    return predict


# ----------------------------------------------------------------------------------------------
# protocol
# ----------------------------------------------------------------------------------------------

N_FOLDS = 5
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 2)
SIZE_BINS = {"lt1ml": (0, 1), "1to10ml": (1, 10), "gt10ml": (10, np.inf)}


def to_native(
    volume: Tensor, affine: np.ndarray, row: dict
) -> tuple[np.ndarray, np.ndarray, float]:
    """A volume on the model's grid (with its voxel to world affine) sampled at the native label
    voxels, trilinear, 0 outside the grid. Plus the true lesion mask and the label voxel volume
    in ml."""
    seg_img = nib.load(row["seg"])
    truth = np.asarray(seg_img.dataobj) > 0
    # native voxel -> model voxel, then grid_sample's [-1, 1] coordinates, last axis first
    to_model = torch.tensor(np.linalg.solve(affine, seg_img.affine), dtype=torch.float32)
    ijk = torch.stack(torch.meshgrid(*map(torch.arange, truth.shape), indexing="ij"), -1)
    xyz = ijk.reshape(-1, 3).float() @ to_model[:3, :3].T + to_model[:3, 3]
    size = torch.tensor(volume.shape, dtype=torch.float32)
    coords = (2 * xyz / (size - 1) - 1).flip(-1).to(volume.device)
    out = F.grid_sample(
        volume[None, None].float(),
        coords.view(1, -1, 1, 1, 3),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    voxel_ml = float(np.prod(seg_img.header.get_zooms()[:3]) / 1000)
    return out.view(truth.shape).cpu().numpy(), truth, voxel_ml


# 26-connectivity, what separates one lesion from another
CONNECTIVITY = np.ones((3, 3, 3))


def lesion_f1(pred: np.ndarray, truth_labels: np.ndarray, n_true: int) -> float:
    """Lesion-wise F1, each lesion a connected component. A true lesion is found when any
    predicted voxel touches it, a predicted lesion is a false alarm when it touches no true one.
    1 when both are empty."""
    pred_labels, n_pred = ndimage.label(pred, structure=CONNECTIVITY)
    if n_true + n_pred == 0:
        return 1.0
    found = len(np.unique(truth_labels[pred & (truth_labels > 0)]))
    false_alarms = n_pred - len(np.unique(pred_labels[pred & (truth_labels > 0)]))
    return 2 * found / (2 * found + false_alarms + (n_true - found))


def threshold_curves(probs: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    """Dice, lesion-wise F1 and predicted voxels at every threshold. Dice is 1 when both are
    empty."""
    curves = {name: np.zeros(len(THRESHOLDS)) for name in ["dice", "f1", "predicted"]}
    n_true = truth.sum()
    truth_labels, n_lesions = ndimage.label(truth, structure=CONNECTIVITY)
    for i, threshold in enumerate(THRESHOLDS):
        pred = probs >= threshold
        curves["predicted"][i] = pred.sum()
        denom = curves["predicted"][i] + n_true
        curves["dice"][i] = 2 * (pred & truth).sum() / denom if denom else 1.0
        curves["f1"][i] = lesion_f1(pred, truth_labels, n_lesions)
    return curves


def cross_validate(
    rows: list[dict],
    subjects: list[dict],
    geometry: dict,
    modalities: list[str],
    device: torch.device,
    probs_dir: Path,
    seed: int = 0,
) -> dict[str, dict[str, np.ndarray]]:
    """Out-of-fold Dice, lesion-wise F1 and predicted volume curves for every subject and
    modality, NaN where the subject doesn't have the modality. Every out-of-fold probability map
    is saved to probs_dir/<modality>, on the subject's label grid."""
    curves = {
        mod: {name: np.full((len(rows), len(THRESHOLDS)), np.nan) for name in CURVES}
        for mod in modalities
    }
    strata = [f"{row['dataset']}_{size_bin(row['lesion_ml'])}" for row in rows]
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(folds.split(rows, strata)):
        start = time.perf_counter()
        best = {}
        for mod in modalities:
            has = [i for i in train if mod in subjects[i]]
            predict = fit_linear([subjects[i][mod] for i in has], geometry, device, seed + fold)
            tested = [i for i in test if mod in subjects[i]]
            for i in tested:
                sample = subjects[i][mod]
                probs, truth, voxel_ml = to_native(predict(sample), sample["affine"], rows[i])
                subject_curves = threshold_curves(probs, truth)
                curves[mod]["dice"][i] = subject_curves["dice"]
                curves[mod]["f1"][i] = subject_curves["f1"]
                curves[mod]["pred_ml"][i] = subject_curves["predicted"] * voxel_ml
                out = probs_dir / mod / f"{rows[i]['subject']}.nii.gz"
                out.parent.mkdir(parents=True, exist_ok=True)
                probs_u8 = np.round(probs * 255).astype(np.uint8)
                nib.save(nib.Nifti1Image(probs_u8, nib.load(rows[i]["seg"]).affine), out)
            best[mod] = curves[mod]["dice"][tested].mean(0).max()
        logger.info(
            f"fold {fold + 1}/{N_FOLDS} n={len(test)} best fold dice "
            + " ".join(f"{mod}={v:.3f}" for mod, v in best.items())
            + f" ({time.perf_counter() - start:.0f}s)"
        )
    return curves


CURVES = ["dice", "f1", "pred_ml"]


def size_bin(lesion_ml: float) -> str:
    return next(name for name, (low, high) in SIZE_BINS.items() if low < lesion_ml <= high)


def metrics(per_subject: dict[str, dict[str, np.ndarray]], rows: np.ndarray) -> dict:
    """Each metric per modality, over the subjects that have it, and averaged over modalities."""
    out = {}
    for mod, v in per_subject.items():
        out[f"{mod}_dice"] = float(np.nanmean(v["dice"][rows]))
        out[f"{mod}_lesion_f1"] = float(np.nanmean(v["f1"][rows]))
        out[f"{mod}_avd_ml"] = float(np.nanmean(v["abs_err_ml"][rows]))
    for name in ["dice", "lesion_f1", "avd_ml"]:
        out[name] = float(np.mean([out[f"{mod}_{name}"] for mod in per_subject]))
    return out


def score(
    curves: dict[str, dict[str, np.ndarray]],
    true_ml: np.ndarray,
    seed: int = 0,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Each modality at its own cut, the one maximizing its mean out-of-fold Dice. The metrics
    per modality ("<mod>_dice", ...) and averaged over modalities ("dice", ...), each with a
    percentile CI resampling subjects. Plus Dice per lesion size bin."""
    summary, per_subject = {}, {}
    for mod, c in curves.items():
        best = int(np.nanmean(c["dice"], axis=0).argmax())
        summary[f"{mod}_threshold"] = float(THRESHOLDS[best])
        per_subject[mod] = {
            "dice": c["dice"][:, best],
            "f1": c["f1"][:, best],
            "abs_err_ml": np.abs(c["pred_ml"][:, best] - true_ml),
        }

    rng = np.random.default_rng(seed)
    everyone = np.arange(len(true_ml))
    resamples = rng.integers(0, len(true_ml), size=(n_boot, len(true_ml)))
    samples = [metrics(per_subject, rows) for rows in resamples]
    for name, point in metrics(per_subject, everyone).items():
        low, high = np.nanpercentile(
            [s[name] for s in samples], [100 * alpha / 2, 100 * (1 - alpha / 2)]
        )
        summary[name] = point
        summary[f"{name}_ci_low"] = float(low)
        summary[f"{name}_ci_high"] = float(high)
    for name, (low, high) in SIZE_BINS.items():
        keep = np.flatnonzero((true_ml > low) & (true_ml <= high))
        for mod, v in per_subject.items():
            summary[f"{mod}_dice_{name}"] = float(np.nanmean(v["dice"][keep]))
        summary[f"dice_{name}"] = float(np.mean([summary[f"{m}_dice_{name}"] for m in per_subject]))
        summary[f"n_{name}"] = int(len(keep))
    return summary, per_subject


# ----------------------------------------------------------------------------------------------
# entrypoint
# ----------------------------------------------------------------------------------------------


def main(args: argparse.Namespace):
    if args.output_dir is None:
        run_name = f"{args.model}_{args.ckpt_path.stem}"
        if args.random_init:
            run_name += "_random"
        run_name += "_" + "-".join(args.modalities)
        args.output_dir = args.ckpt_path.parent / "evals" / "stroke" / run_name
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    misc.setup_logging(logger, run_dir / "log.txt")
    misc.random_seed(args.seed)
    config = {key: str(value) for key, value in vars(args).items()}
    logger.info(f"stroke segmentation eval, {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info(f"sha: {misc.git_sha()}")
    logger.info(f"config: {json.dumps(config)}")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    data_root = download_data()
    logger.info(f"data root: {data_root}")
    rows = load_subjects(data_root)
    true_ml = np.array([row["lesion_ml"] for row in rows])
    datasets = sorted({row["dataset"] for row in rows})
    counts = {d: sum(row["dataset"] == d for row in rows) for d in datasets}
    has = {mod: sum(mod in row["images"] for row in rows) for mod in args.modalities}
    logger.info(
        f"dataset: {len(rows)} subjects {counts}, lesion median {np.median(true_ml):.1f} ml, "
        f"subjects per modality {has}"
    )

    transform, model = create_model(
        args.model, ckpt_path=args.ckpt_path, random_init=args.random_init
    )
    device = torch.device(args.device)
    model.to(device).eval()
    dataset = SubjectDataset(rows, transform, args.modalities)
    subjects, geometry = embed(model, dataset, device, args.workers)
    model.cpu()

    curves = cross_validate(
        rows, subjects, geometry, args.modalities, device, run_dir / "probs", seed=args.seed
    )
    summary, per_subject = score(curves, true_ml, seed=args.seed)

    preds = []
    for i, row in enumerate(rows):
        pred = {"subject": row["subject"], "dataset": row["dataset"], "lesion_ml": row["lesion_ml"]}
        for mod, v in per_subject.items():
            if not np.isnan(v["dice"][i]):
                pred[f"{mod}_dice"], pred[f"{mod}_f1"] = float(v["dice"][i]), float(v["f1"][i])
        preds.append(pred)
    (run_dir / "preds.json").write_text("".join(json.dumps(pred) + "\n" for pred in preds))

    record = {
        "model": args.model,
        "ckpt_path": str(args.ckpt_path),
        "random_init": args.random_init,
        "modalities": args.modalities,
        **summary,
    }
    (run_dir / "metrics.json").write_text(json.dumps(record) + "\n")
    for group in [*args.modalities, "mean"]:
        prefix = "" if group == "mean" else f"{group}_"
        values = [
            f"{name}={summary[prefix + name]:.3f} [{summary[prefix + name + '_ci_low']:.3f}, "
            f"{summary[prefix + name + '_ci_high']:.3f}]"
            for name in ["dice", "lesion_f1", "avd_ml"]
        ]
        logger.info(f"result {group}: " + "  ".join(values))
    logger.info(
        "thresholds: " + " ".join(f"{m}={summary[f'{m}_threshold']}" for m in args.modalities)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list_models(), required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="default <ckpt dir>/evals/stroke/<model>_<ckpt>[_random]_<modalities>",
    )
    parser.add_argument(
        "--random-init", action="store_true", help="random init weights, as a baseline"
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        choices=MODALITIES,
        default=["dwi", "adc", "flair"],
        help="one probe each, scored on the subjects that have it, scores averaged",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
