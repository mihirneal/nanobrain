from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

GRID_SHAPE = (192, 240, 192)
TARGET_SPACING = 1.0


class BrainNpzDataset(Dataset):
    def __init__(self, root: str | Path, filelist: str | Path):
        self.root = Path(root)
        self.paths = Path(filelist).read_text().strip().splitlines()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        with np.load(self.root / path) as npz:
            affine = npz["affine"]
            sample = {
                "path": path,
                "values": torch.from_numpy(npz["values"]),
                "mask": torch.from_numpy(npz["mask"]),
                "shape": tuple(npz["shape"].tolist()),
                "affine": torch.from_numpy(affine),
                "spacing": tuple(np.linalg.norm(affine[:3, :3], axis=0).tolist()),
            }
        return sample


def process_batch(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    # each sample is resampled to 1 mm and written straight into its slot of a preallocated
    # batch, cropped or zero padded to GRID_SHAPE 
    device = samples[0]["values"].device
    images = torch.zeros(len(samples), *GRID_SHAPE, device=device)
    masks = torch.zeros(len(samples), *GRID_SHAPE, device=device)
    for ii, sample in enumerate(samples):
        spacing = sample["spacing"]

        # normalize values
        values = sample["values"].float()
        values = values / values.max()
        values = (values - values.mean()) / values.std(correction=0).clamp_min(1e-6)

        # decode mask
        mask = brle_to_dense(sample["mask"], shape=sample["shape"])

        # make dense image
        image = torch.zeros(mask.shape, dtype=values.dtype, device=values.device)
        image.masked_scatter_(mask, values)

        # resizing. the mask resamples as uint8, nearest keeps it 0/1
        image = resample_image(image, spacing=spacing, target_spacing=TARGET_SPACING)
        mask = resample_image(
            mask.to(torch.uint8), spacing=spacing, target_spacing=TARGET_SPACING, mode="nearest"
        )

        # crop/pad into the batch and apply mask
        src, dst = fit_slices(image.shape, GRID_SHAPE)
        masks[ii][dst] = mask[src]
        torch.mul(image[src], mask[src], out=images[ii][dst])
    return {"image": images, "mask": masks}


def fit_slices(shape: tuple[int, ...], target_shape: tuple[int, ...]) -> tuple[tuple, tuple]:
    """Source and destination slices of fit_to_shape's pad (x/y centered, z top aligned)."""
    src, dst = [], []
    for axis, (size, target) in enumerate(zip(shape, target_shape)):
        diff = target - size
        before = diff // 2 if axis < 2 else diff
        n = min(size + before, target) - max(before, 0)
        src.append(slice(max(-before, 0), max(-before, 0) + n))
        dst.append(slice(max(before, 0), max(before, 0) + n))
    return tuple(src), tuple(dst)


def brle_to_dense(mask_rle: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    # mask_rle is a binary run length encoding, alternating false and true runs starting
    # with false
    X, Y, Z = shape
    run_lengths = mask_rle.long()
    run_flags = torch.arange(len(run_lengths), device=mask_rle.device) % 2 == 1
    mask = torch.repeat_interleave(run_flags, run_lengths, output_size=X * Y * Z)
    return mask.reshape(shape)


def resample_image(
    image: torch.Tensor,
    spacing: tuple[float, float, float],
    target_spacing: float = 1.0,
    mode: str = "trilinear",
) -> torch.Tensor:
    new_shape = [round(size * sp / target_spacing) for size, sp in zip(image.shape, spacing)]
    if list(image.shape) == new_shape:
        return image
    image = F.interpolate(image[None, None], size=new_shape, mode=mode)
    return image[0, 0]


def fit_to_shape(
    image: torch.Tensor, target_shape: tuple[int, int, int] = GRID_SHAPE
) -> torch.Tensor:
    # x/y centered and z top aligned, matching the crop in conform_image
    # F.pad takes pads last dim first, negative pads crop
    pad = []
    for axis in reversed(range(3)):
        diff = target_shape[axis] - image.shape[axis]
        if axis < 2:
            before = diff // 2
        else:
            before = diff
        pad += [before, diff - before]
    return F.pad(image, pad)
