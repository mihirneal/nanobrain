# Walnut v2

Release recipe for walnut v2: a ViT-L MAE (316M parameters) pretrained on FOMO-300K.

- Config: `src/nanobrain/config/walnut_v2.yaml` (merged over `default_pretrain.yaml`; no command-line overrides besides output paths).
- Model: ViT-L, 4 mm patches on a 1 mm 192x240x192 grid; 85% of brain tokens hidden in 8 mm cubes (4,096 visible, 4,096 reconstructed).
- Objective: pixel reconstruction plus level-set targets (in-mask fraction above 8 intensity thresholds, 2 mm sub-blocks, residual and left-right mirror channels), a decoder-side level head, and an MLP age head on FOMO-300K age metadata whose gradient into the encoder ramps in from 65% of training.
- Optimizer: Muon for the block matrices (2x lr), AdamW for the rest; base lr 1e-3, 5 warmup epochs, linear decay over the last 4; decoder qk norm.
- Data: `src/nanobrain/config/fomo300k_images_qc.txt`, 253418 FOMO-300K images passing QC, sha256 `fc179058bff4f842c96ef593a08e0045ebdbba184e7339f62b4ccf17356c6ab7`, read from `$BRAINMARKS_SMRI_DATA/fomo300k`. 20 epochs = 158,386 steps of 32 images.
- Hardware: 8 H100 80 GB (DDP, 4 images per GPU), about 15 h.
- Environment: Python 3.12, torch 2.14.0+cu130, flash-attn-3 3.0.0; full package list in `environment.txt` (`uv.lock` pins the same versions).

## Reproduce

```bash
git checkout <release commit>   # a clean checkout: the run refuses uncommitted or untracked files (require_clean_git)
experiments/walnut_v2/train.sh max walnut_v2
```

The run's `log.txt` and `summary.json` record the commit, `dirty: false`, the train-list sha256 and the data root.
GPU kernels are not bit-deterministic, so a rerun matches the released checkpoint's scores within seed noise, not bit for bit.
