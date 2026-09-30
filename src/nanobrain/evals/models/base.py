"""Model interface for the evals."""

from typing import Callable, NamedTuple

import nibabel as nib
import torch.nn as nn
from torch import Tensor


class Embeddings(NamedTuple):
    cls_embeds: Tensor | None
    """cls embeddings [B C D]"""

    patch_embeds: Tensor | None
    """patch embeddings [B L D]"""

    grid_embeds: Tensor | None = None
    """patch embeddings on the patch grid [B D gx gy gz], zero where the model has no token.
    The patch grid tiles the transform's image grid, for dense tasks like segmentation."""


class ModelWrapper(nn.Module):
    """
    Wrap a frozen encoder. Takes a batch of transformed samples and returns embeddings.
    """

    def forward(self, batch: dict[str, Tensor]) -> Embeddings: ...


class ModelTransform:
    """
    Model specific data transform. Takes a raw nii.gz image and returns a sample with all
    the model's preprocessing applied. Runs in the dataloader workers. For dense tasks the
    sample also has "affine", the voxel to world affine of the transformed image grid.
    """

    def __call__(self, img: nib.Nifti1Image) -> dict[str, Tensor]: ...


ModelTransformPair = tuple[ModelTransform, ModelWrapper]

ModelFn = Callable[..., ModelTransformPair]
