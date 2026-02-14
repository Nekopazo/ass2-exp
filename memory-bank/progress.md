# Progress Log

## Date
2026-02-14

## Scope
Continue implementation plan Phase 0 only. Phase 1 not started.

## Completed

### Step 0.1 - Project directory structure
Checked directory structure under `ass2-exp/`.
`tree` command is unavailable in current environment, so used `find -maxdepth 2` for equivalent verification.
Verified existing directories:
- `configs/`
- `data/`
- `data/splits/`
- `src/keras/`
- `src/pytorch/`
- `logs/`
- `logs/checkpoints/`
- `results/`
- `analysis/`
- `plots/`
- `scripts/`

### Step 0.2 - Environment info
Re-collected environment info using the required virtual environment:
- `source ~/envs/my_jupyter_env/bin/activate`
Updated `environment.txt` with:
- python version
- tensorflow version
- torch version
- nvcc output
- cudnn version from torch
- nvidia-smi output

### Step 0.3 - Dependencies and lock file
Updated `requirements.txt` with pinned key dependencies from the active environment.
Regenerated lock file using:
- `pip freeze > requirements_lock.txt`

### Step 0.4 - GPU availability check
Ran both framework GPU checks in `my_jupyter_env` and recorded outputs in `environment.txt`.
Current result in this runtime context:
- TensorFlow GPU list: empty (`[]`)
- PyTorch CUDA availability: `False`

## Notes / Current blockers
GPU is not visible in this runtime context (`nvidia-smi` NVML init error, CUDA device not detected by TF/Torch), so plan-level validation expecting A40 cannot be confirmed here.

## Next action
Wait for user validation of Phase 0 results before starting Phase 1.

---

## Update
2026-02-14 (Phase 1 implementation started, Phase 2 not started)

### What was implemented
- Added executable Phase 1 data-preparation script:
  - `scripts/prepare_phase1_data.py`
- Script covers plan Steps 1.1 to 1.7 in one pipeline:
  - load/download CIFAR-10 and CIFAR-100
  - load Tiny-ImageNet from local `.npy`, local `data/tiny-imagenet-200/`, or HuggingFace (`datasets`) fallback
  - generate fixed stratified splits with `seed=42`
  - save split index files into `data/splits/`
  - compute training-only channel mean/std and write `configs/dataset_stats.yaml`
  - perform required split/statistics validations and print verification outputs

### Validation run status
- Ran with required environment:
  - `source ~/envs/my_jupyter_env/bin/activate`
- Script syntax check passed:
  - `python -m py_compile scripts/prepare_phase1_data.py`
- End-to-end execution is currently blocked by network/DNS restrictions:
  - CIFAR download failed with `URLError: [Errno -3] Temporary failure in name resolution`
  - `pip install datasets` also failed due network resolution error

### Blockers
- Runtime cannot reach external dataset/package hosts, so online download path cannot complete in current context.

### Ready offline fallback
Phase 1 can continue immediately once dataset files are provided locally:
- CIFAR auto-detected by torchvision cache under `data/` (or extracted dataset files in expected CIFAR layout)
- Tiny-ImageNet accepted as either:
  - `data/tiny_imagenet_{train,test}_{images,labels}.npy`, or
  - extracted folder `data/tiny-imagenet-200/` (official structure with `train/`, `val/`, `wnids.txt`)

---

## Update
2026-02-14 (Phase 2 implementation completed, Phase 3 not started)

### What was implemented
- Added `configs/train_config.yaml` for Step 2.1 with required sections:
  - `optimizer`, `lr`, `training`, `epochs`, `augmentation`, `input_size`
- Added `configs/experiment_matrix.yaml` for Step 2.2 with required matrices:
  - `stage1` and `stage2` (datasets/models/precisions/seeds/frameworks)

### Validation run status
- Ran validation with required environment:
  - `source ~/envs/my_jupyter_env/bin/activate`
- YAML parse checks passed for:
  - `configs/train_config.yaml`
  - `configs/experiment_matrix.yaml`
  - existing `configs/dataset_stats.yaml`
- Combination count checks:
  - Stage 1: `2 * 2 * 1 * 3 * 2 = 24`
  - Stage 2: `3 * 3 * 2 * 5 * 2 = 180`

### Next action
Stop here and wait for your test/validation. Phase 3 has not been started.
