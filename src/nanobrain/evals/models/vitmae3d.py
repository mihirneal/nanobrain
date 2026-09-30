"""nanobrain ViTMAE3D: the nii.gz transform, the encoder wrapper, and its registered constructor."""

import logging
import math

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
        affine = grid_affine(fit_img, image.shape, self.grid_size)
        image = fit_to_shape(image, target_shape=self.grid_size)
        mask = resample_image(
            mask.float(), spacing=spacing, target_spacing=TARGET_SPACING, mode="nearest"
        )
        mask = fit_to_shape(mask, target_shape=self.grid_size)
        return {"image": image * mask, "mask": mask, "affine": torch.from_numpy(affine)}


def grid_affine(
    fit_img: nib.Nifti1Image,
    resampled_shape: tuple[int, int, int],
    grid_size: tuple[int, int, int],
) -> np.ndarray:
    """Voxel to world affine of the final grid, following resample_image (trilinear,
    align_corners=False) and then fit_to_shape (x/y centered, z top aligned)."""
    in_shape = np.array(fit_img.shape[:3], dtype=float)
    out_shape = np.array(resampled_shape, dtype=float)
    scale = in_shape / out_shape
    diff = np.array(grid_size) - out_shape
    before = np.array([diff[0] // 2, diff[1] // 2, diff[2]])
    # final voxel f is resampled voxel f - before, which samples fit_img voxel
    # (f - before + 0.5) * scale - 0.5
    to_fit = np.eye(4)
    to_fit[:3, :3] = np.diag(scale)
    to_fit[:3, 3] = (0.5 - before) * scale - 0.5
    return fit_img.affine @ to_fit


class ViTMAE3DWrapper(nn.Module):
    def __init__(self, model: ViTMAE3D):
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, Tensor]) -> Embeddings:
        device_type = batch["image"].device.type
        # forward_embedding, keeping the patch ids to place the tokens on the patch grid
        with torch.autocast(device_type, torch.bfloat16, enabled=device_type == "cuda"):
            patches, _, coord, patch_ids = self.model.sample_patches(batch["image"], batch["mask"])
            embeds = self.model.forward_encoder(patches, coord).float()
        num_cls = self.model.class_tokens
        cls_embeds = embeds[:, :num_cls] if num_cls else None
        patch_embeds = embeds[:, num_cls:]

        # patch ids index the patch grid in patchify3d order
        B, _, D = patch_embeds.shape
        grid_shape = [size // self.model.patch_size for size in self.model.grid_size]
        grid_embeds = patch_embeds.new_zeros(B, math.prod(grid_shape), D)
        grid_embeds.scatter_(1, patch_ids[..., None].expand(-1, -1, D), patch_embeds)
        grid_embeds = grid_embeds.transpose(1, 2).reshape(B, D, *grid_shape)
        return Embeddings(cls_embeds, patch_embeds, grid_embeds)


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
