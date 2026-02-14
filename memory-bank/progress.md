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

---

## Update
2026-02-14 (Phase 3 implementation completed, Phase 4 not started)

### What was implemented
- Added PyTorch model builders:
  - `src/pytorch/models.py`
  - Implemented `mobilenetv2`, `resnet50`, `convnext_tiny` with required CIFAR/Tiny stem adjustments and dataset-specific classifier heads.
- Added Keras model builders:
  - `src/keras/models.py`
  - Implemented `MobileNetV2`, `ResNet50`, `ConvNeXtTiny` with required CIFAR/Tiny stem adjustments and dataset-specific classifier heads.
- Added package init files for imports:
  - `src/__init__.py`
  - `src/pytorch/__init__.py`
  - `src/keras/__init__.py`
- Added Phase 3.7 parameter-comparison generator:
  - `scripts/generate_param_comparison.py`
  - Generates `results/param_comparison.csv` and validates model forward shape checks and parameter-diff thresholds.

### Validation run status
- Ran with required environment:
  - `source ~/envs/my_jupyter_env/bin/activate`
- Syntax checks passed:
  - `python -m py_compile src/pytorch/models.py src/keras/models.py scripts/generate_param_comparison.py`
- Executed:
  - `python scripts/generate_param_comparison.py`
- Output generated:
  - `results/param_comparison.csv` with 18 rows (3 models × 2 input sizes × 3 class counts)
- Validation checks passed:
  - Forward output shape checks all passed for both frameworks.
  - ResNet-50 Tiny path check passed: pre-global-pool feature map is 4x4.
  - Parameter diff thresholds passed:
    - MobileNetV2: max diff_pct = 0.0%
    - ResNet-50: max diff_pct = 0.112921% (<1%)
    - ConvNeXt-Tiny: max diff_pct = 0.012421% (<5%)

### Next action
Stop here and wait for your test/validation. Phase 4 has not been started.

---

## Update
2026-02-14 (Phase 4 implementation completed, Phase 5 not started)

### What was implemented
- Added PyTorch data pipeline:
  - `src/pytorch/data.py`
  - Implemented fixed split loading from `data/splits/`, train/val/test selection, dataset-specific augmentation, normalization using `configs/dataset_stats.yaml`, and DataLoader construction.
  - Enforced required DataLoader defaults for training/eval:
    - `num_workers=8`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2` (when workers > 0)
    - `drop_last=True` only for train; `drop_last=False` for val/test.
  - Added Tiny-ImageNet grayscale-safe RGB expansion logic.
- Added Keras data pipeline:
  - `src/keras/data.py`
  - Implemented matching split loading, preprocessing, and `tf.data` pipeline with:
    - `map(..., num_parallel_calls=tf.data.AUTOTUNE)`
    - `prefetch(tf.data.AUTOTUNE)`
    - `shuffle(...)` for train only
    - `batch(..., drop_remainder=True)` for train and `drop_remainder=False` for val/test.
  - Added Tiny-ImageNet grayscale-safe RGB expansion logic.
- Updated package exports:
  - `src/pytorch/__init__.py`
  - `src/keras/__init__.py`

### Validation run status
- Ran with required environment:
  - `source ~/envs/my_jupyter_env/bin/activate`
- Syntax checks passed:
  - `python -m py_compile src/pytorch/data.py src/keras/data.py src/pytorch/__init__.py src/keras/__init__.py`
- Executed Phase 4 verification script (sandbox-limited fallback used `num_workers=0` only for this validation run due multiprocessing semaphore permission limits).
- Validation checks passed:
  - CIFAR-10: train/val counts = 45,000 / 5,000.
  - CIFAR-100: train/val counts = 45,000 / 5,000.
  - Tiny-ImageNet: train/val counts = 90,000 / 10,000.
  - Batch shapes:
    - PyTorch train batch: `(256, 3, 32, 32)` for CIFAR, `(256, 3, 64, 64)` for Tiny-ImageNet.
    - Keras train batch: `(256, 32, 32, 3)` for CIFAR, `(256, 64, 64, 3)` for Tiny-ImageNet.
  - Normalized value ranges observed in first train batch were within expected rough bounds (about `[-2, 2.2]`).
  - Cross-framework key consistency test (same CIFAR-10 val image, no augmentation):
    - max absolute pixel diff after normalization = `0.0` (< `1e-5`).
  - Tiny-ImageNet grayscale expansion spot check:
    - Found gray-like sample and confirmed shape `(64, 64, 3)`.

### Next action
Stop here and wait for your test/validation. Phase 5 has not been started.

---

## Update
2026-02-14 (Phase 5 implementation started and completed for code delivery, Phase 6 not started)

### What was implemented
- Added shared training utilities for cross-framework consistency:
  - `src/common/training.py`
  - `src/common/__init__.py`
  - Includes dataset metadata helpers, optimizer-step calculation for gradient accumulation, and a shared per-step warmup+cosine LR schedule.
- Added PyTorch training loop:
  - `src/pytorch/train.py`
  - Supports CLI args: `--dataset --model --precision --seed --config` (plus `--epochs_override --num_workers` for controlled validation runs).
  - Implements:
    - seed setup (Python/NumPy/PyTorch/CUDA), cuDNN deterministic settings
    - SGD (momentum/nesterov/weight_decay)
    - per-step warmup + cosine LR with `T_max = total_steps - warmup_steps`
    - FP32/AMP (`torch.cuda.amp`)
    - gradient accumulation with loss scaling by `gradient_accumulation_steps`
    - epoch logging (`train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `val_macro_f1`, `epoch_time_seconds`, `learning_rate`)
    - best checkpoint selection by `val_accuracy` and save to `logs/checkpoints/*.pt`
    - best-checkpoint test evaluation and JSON save to `logs/*_test.json`
- Added Keras custom training loop:
  - `src/keras/train.py`
  - Implements same CLI and metrics protocol as PyTorch.
  - Uses `tf.GradientTape` custom loop (not `model.fit`) with:
    - deterministic setup (`enable_op_determinism`)
    - SGD hyperparameter alignment
    - shared per-step warmup + cosine LR behavior
    - FP32/AMP (`tf.keras.mixed_precision` + `LossScaleOptimizer`)
    - gradient accumulation with loss divided by accumulation steps
    - best-checkpoint selection by `val_accuracy`, checkpoint save to `logs/checkpoints/*.h5`, reload, and test evaluation output to `logs/*_test.json`

### Validation run status
- Ran with required environment:
  - `source ~/envs/my_jupyter_env/bin/activate`
- Syntax checks passed:
  - `python -m py_compile src/common/__init__.py src/common/training.py src/pytorch/train.py src/keras/train.py`
- Runtime smoke checks passed:
  - Shared LR schedule key points verified (`step 0 = 0`, warmup/cosine transition values correct)
  - One-step PyTorch training smoke run on CIFAR-10 completed
  - One-step Keras training smoke run on CIFAR-10 completed
  - Keras `.h5` checkpoint save/load roundtrip verified with model containing modified layers

### Notes
- In current runtime context, CUDA device is still not available due driver/runtime mismatch; full 3-epoch validation runs from Step 5.1/5.2 and LR/AMP benchmark validations (Steps 5.3/5.4) are prepared but not fully executed here.
- Stopped at Phase 5 implementation as requested. Phase 6 has not been started.
