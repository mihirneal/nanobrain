"""nanobrain ViTMAE3D: the nii.gz transform, the encoder wrapper, and its registered constructor."""

import logging

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from nibabel.processing import resample_from_to
from torch import Tensor

import nanobrain.utils.preprocessing as preproc
from nanobrain.data import TARGET_SPACING, fit_to_shape, resample_image
from nanobrain.evals.models.base import Embeddings
from nanobrain.evals.models.registry import register_model
from nanobrain.model import ViTMAE3D
from nanobrain.prepare import VMAX_QUANTILE, VMIN_QUANTILE

logger = logging.getLogger()


class NanobrainTransform:
    """The nanobrain pretraining data pipeline, prepare.process_image followed by
    data.process_sample, applied to a raw nii.gz. Shared by models trained on that data.
    """

    def __init__(self, grid_size: tuple[int, int, int]):
        self.grid_size = tuple(grid_size)

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]:
        # prepare.process_image: head mask, conform to the grid fov, clip to value quantiles
        mask_img = preproc.threshold_mask(img)
        fit_img = preproc.conform_image(
            img,
            mask_img,
            min_voxel_size=TARGET_SPACING,
            max_fov=tuple(float(size) for size in self.grid_size),
        )
        mask_img = resample_from_to(mask_img, fit_img, order=0)

        data = fit_img.get_fdata(dtype=np.float32)
        mask = np.asarray(mask_img.dataobj) > 0
        values = data[mask]
        vmin, vmax = np.quantile(values, [VMIN_QUANTILE, VMAX_QUANTILE])
        assert vmax > vmin, f"degenerate value range {vmin=}, {vmax=}"
        values = np.clip((values - vmin) / (vmax - vmin), 0, 1)

        # data.process_sample: z-score in the mask, resample to 1mm, fit to the grid
        values = torch.from_numpy(values).float()
        values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)
        mask = torch.from_numpy(mask)
        image = torch.zeros(mask.shape, dtype=values.dtype)
        image.masked_scatter_(mask, values)

        spacing = tuple(np.linalg.norm(fit_img.affine[:3, :3], axis=0).tolist())
        image = resample_image(image, spacing=spacing, target_spacing=TARGET_SPACING)
        image = fit_to_shape(image, target_shape=self.grid_size)
        mask = resample_image(
            mask.float(), spacing=spacing, target_spacing=TARGET_SPACING, mode="nearest"
        )
        mask = fit_to_shape(mask, target_shape=self.grid_size)
        return {"image": image * mask, "mask": mask}


class ViTMAE3DWrapper(nn.Module):
    def __init__(self, model: ViTMAE3D):
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        device_type = batch["image"].device.type
        with torch.autocast(device_type, torch.bfloat16, enabled=device_type == "cuda"):
            embeds = self.model.forward_embedding(batch["image"], batch["mask"])
        return Embeddings(*(None if x is None else x.float() for x in embeds))


@register_model
def vitmae3d(
    *, ckpt_path: str, random_init: bool = False
) -> tuple[NanobrainTransform, ViTMAE3DWrapper]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ckpt["args"]
    model = ViTMAE3D(
        grid_size=tuple(args["grid_size"]),
        patch_size=args["patch_size"],
        **args["model_kwargs"],
    )
    # random init keeps the architecture, as a baseline
    if random_init:
        logger.info(f"built {ckpt_path} architecture with random init weights")
    else:
        model.load_state_dict(ckpt["model"])
        logger.info(f"loaded {ckpt_path} (epoch {ckpt['epoch']})")
    model.requires_grad_(False)
    return NanobrainTransform(model.grid_size), ViTMAE3DWrapper(model)
