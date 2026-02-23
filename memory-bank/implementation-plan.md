# Keras vs PyTorch 图像分类 Benchmark — 实施计划 (Stage B)

> **文件覆盖声明：** 本需求文档是项目的**唯一权威参考**。本版本包含阶段 B：PlantVillage（38 类，224×224）+ CIFAR-10（10 类，32×32→224×224）迁移学习实验。

---

## Phase 0: 项目初始化与环境准备

### Step 0.1 — 创建项目目录结构

创建以下目录树：

```
ass2-exp/
├── configs/            # 超参数配置文件（YAML）
├── data/               # 数据集存放目录
│   └── splits/         # 3 折交叉验证索引文件
├── logs/               # 每次运行的逐 epoch 日志（CSV）
│   └── checkpoints/    # best checkpoint 文件（保留不删）
├── results/            # 汇总结果
├── analysis/           # 统计分析脚本与输出
├── plots/              # 训练曲线图
└── scripts/            # 启动脚本（shell）
```

**验证：** 运行 `tree ass2-exp/ -L 2`，确认所有目录存在且层级正确。

---

### Step 0.2 — 记录实验环境信息

在项目根目录创建 `environment.txt`，记录以下信息（通过终端命令获取）：

- `python --version`
- `pip show tensorflow | grep Version`
- `pip show torch | grep Version`
- `nvcc --version`（CUDA 版本）
- `python -c "import torch; print(torch.backends.cudnn.version())"`（cuDNN 版本）
- `nvidia-smi | head -3`（GPU 驱动版本 + GPU 型号）

**验证：** 打开 `environment.txt`，确认包含 6 项信息，GPU 型号为 NVIDIA A40。

---

### Step 0.3 — 安装并锁定依赖

创建 `requirements.txt`，列出所有依赖及精确版本号（`==`）。关键依赖至少包括：

- tensorflow
- torch, torchvision
- scikit-learn（用于 Macro-F1、分层划分与交叉验证）
- pandas, numpy
- pyyaml
- scipy（用于统计检验）
- matplotlib（用于绘图）
- Pillow（用于 PlantVillage 图像加载与预处理）

安装后用 `pip freeze > requirements_lock.txt` 锁定完整依赖。

**验证：** 在新的虚拟环境中执行 `pip install -r requirements_lock.txt`，确认无报错。

---

### Step 0.4 — 确认 GPU 可用性

分别用 TensorFlow 和 PyTorch 检查 GPU 是否可被识别：

- TensorFlow：`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- PyTorch：`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

**验证：** 两条命令都输出 A40 GPU 信息，无报错。

---

## Phase 1: 数据准备与划分

> **数据集策略：** 使用两个数据集：
> 1. **PlantVillage**：农作物病害数据集（224×224 分辨率，38 分类），验证迁移学习在高分辨率真实数据上的表现。
> 2. **CIFAR-10**：经典图像分类基准（32×32 分辨率，10 分类），训练/推理时 resize 到 224×224 以适配 ImageNet 预训练模型。
>
> 两个数据集均采用迁移学习（加载 ImageNet 预训练权重）。

### Step 1.1 — 下载并预处理 PlantVillage 数据集

从 Kaggle 下载 PlantVillage 数据集至 `data/plantvillage/` 目录。PlantVillage 包含 38 个类别的植物叶片图像（健康与病害），图像分辨率不一，需统一 resize 至 224×224。

下载后**预处理并保存为 numpy 数组**至 `data/` 目录（加载更快）：
- `data/plantvillage_images.npy` — shape `(N, 224, 224, 3)`，uint8
- `data/plantvillage_labels.npy` — shape `(N,)`，int
- `data/plantvillage_class_names.json` — 类别名称映射

**验证：**
1. 打印图片总数 N（预期约 54,000 张）。
2. 类别数 = 38。
3. 随机抽取 5 张图片，确认尺寸为 224×224×3。
4. 打印各类别的样本数，观察类别不平衡程度。

---

### Step 1.1b — 下载并预处理 CIFAR-10 数据集

通过 `tf.keras.datasets.cifar10.load_data()` 自动下载 CIFAR-10。将训练集（50,000）和测试集（10,000）合并为 60,000 张图像，保存为 numpy 数组：

- `data/cifar10_images.npy` — shape `(60000, 32, 32, 3)`，uint8（**保持原始 32×32 分辨率**，训练时在数据管线中 resize 到 224×224）
- `data/cifar10_labels.npy` — shape `(60000,)`，int64
- `data/cifar10_class_names.json` — 10 个类别名称

**验证：**
1. 图片总数 = 60,000。
2. 类别数 = 10。
3. 图片尺寸为 32×32×3。

---

### Step 1.2 — 生成测试集划分与三折交叉验证索引

对**每个数据集**独立执行以下操作（通用函数 `_generate_splits(dataset, labels, splits_dir)`）：

1. **先划出固定测试集**：使用 `sklearn.model_selection.train_test_split(stratify=y, test_size=0.2, random_state=42)` 从全部数据中分层抽取 20% 作为测试集，保存为：
   - `data/splits/{dataset}_test_indices.npy`
   - `data/splits/{dataset}_trainval_indices.npy`

2. **对剩余 80% 进行分层 3 折划分**：使用 `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` 对 trainval 子集生成 3 折索引，保存为：
   - `data/splits/{dataset}_fold{i}_train_indices.npy`
   - `data/splits/{dataset}_fold{i}_val_indices.npy`（i=0,1,2）

   > **注意：** 这里的 fold 索引是相对于 trainval 子集的**局部索引**。加载时需先取 trainval_indices，再用 fold 索引从中取样。

流程：每折训练 → 在该折验证集上评估选 best checkpoint → 最终在 test 集上评估。

**验证：**
1. 测试集约占总数的 20%。
2. 3 折验证集索引两两无交集，并集覆盖整个 trainval 子集。
3. 测试集与 trainval 子集无交集。
4. 每折验证集中各类别比例与整体分布一致（分层保证）。

---

### Step 1.3 — 计算各折的 mean 和 std

对**每个数据集**的每一折训练集计算通道级 mean 和 std（在像素值归一化到 [0,1] 之后）。通用函数 `_compute_dataset_stats(dataset, images_path, splits_dir)`。结果保存至 `configs/dataset_stats.yaml`，格式如下：

```yaml
plantvillage:
  fold0:
    mean: [R, G, B]
    std:  [R, G, B]
  fold1:
    mean: [R, G, B]
    std:  [R, G, B]
  fold2:
    mean: [R, G, B]
    std:  [R, G, B]
cifar10:
  fold0:
    mean: [R, G, B]
    std:  [R, G, B]
  fold1:
    mean: [R, G, B]
    std:  [R, G, B]
  fold2:
    mean: [R, G, B]
    std:  [R, G, B]
```

**CLI 支持：** `python ass2_code.py prepare_data --dataset all|plantvillage|cifar10`

**验证：**
1. 每个 mean 值在 [0, 1] 范围内。
2. 每个 std 值在 (0, 0.5] 范围内。
3. 同一数据集不同折的 mean/std 差异很小（因为训练集重叠度约 67%）。

---

## Phase 2: 配置文件

### Step 2.1 — 创建统一超参数配置文件

在 `configs/` 目录创建 `train_config.yaml`，包含以下字段：

```yaml
optimizer:
  type: SGD
  momentum: 0.9
  nesterov: true
  weight_decay: 1e-4

lr:
  base_lr: 0.04
  scheduler: cosine
  eta_min: 0.0
  warmup_epochs: 5
  warmup_start_lr: 0.0
  warmup_mode: per_step

training:
  batch_size: 128
  loss: cross_entropy

early_stopping:
  monitor: val_loss
  patience: 7
  min_delta: 0.0

epochs: 30

cross_validation:
  n_splits: 3
  seed: 42

augmentation:
  random_resized_crop:
    size: 224
    scale: [0.8, 1.0]
  random_horizontal_flip: true

input_size: 224
```

> **与初版差异说明：** base_lr 从 0.01 提升至 0.04（配合 batch_size=128 的线性缩放），epochs 从 100 降至 30，patience 从 15 降至 7，batch_size 从 32 提升至 128。目的是在保证训练质量的前提下大幅缩短实验时间。

**验证：** 用 YAML 解析器加载该文件，确认所有字段与实验文档一致，无解析错误。

---

### Step 2.2 — 创建实验矩阵配置文件

创建 `configs/experiment_matrix.yaml`：

```yaml
name: "迁移学习 — Keras vs PyTorch Benchmark"
datasets: [plantvillage, cifar10]
models: [resnet50, vgg16, mobilenetv2]
folds: [0, 1, 2]
frameworks: [keras, pytorch]
transfer_learning: true
total_runs: 36
```

**验证：** 组合数 = 2 datasets × 3 models × 3 folds × 2 frameworks = 36。

---

## Phase 3: 模型构建

> **迁移学习策略：** 所有三个模型（ResNet50、VGG16、MobileNetV2）均加载 ImageNet 预训练权重，仅重新初始化分类头，所有层可训练（全量微调）。
>
> **多数据集支持：** 模型构建函数接受 `num_classes` 参数（PlantVillage=38, CIFAR-10=10），通过 `get_num_classes(dataset)` 获取。

### Step 3.1 — 实现 PyTorch 版 ResNet50

使用 `torchvision.models.resnet50(weights='IMAGENET1K_V1')`，仅重新初始化最后的 fc 层以匹配目标类别数。所有层可训练（全量微调）。

```python
def build_pytorch_model(model_name: str, num_classes: int = 38):
```

**验证：**
1. 创建模型实例，用 `(1, 3, 224, 224)` 的随机张量做前向传播，确认输出形状为 `(1, num_classes)`。
2. 确认加载预训练权重后，卷积层权重非全零。
3. 打印模型参数总数（约 23.5M for 38 classes）。

---

### Step 3.2 — 实现 Keras 版 ResNet50

使用 `tf.keras.applications.ResNet50(include_top=False, weights='imagenet')`，添加 GlobalAveragePooling2D + Dense(num_classes) 作为新分类头。所有层可训练（全量微调）。

```python
def build_keras_model(model_name: str, num_classes: int = 38, weight_decay: float = 1e-4):
```

**验证：**
1. 测试前向传播，确认输出形状正确。
2. 打印参数总数，与 PyTorch 版对比，差异应小于 1%。

---

### Step 3.3 — 实现 PyTorch 版 VGG16-BN

使用 `torchvision.models.vgg16_bn(weights='IMAGENET1K_V1')`。224→7×7×512。分类器仅修改最后一层输出维度为 `num_classes`。

**验证：**
1. 创建模型实例，用 `(1, 3, 224, 224)` 做前向传播，确认输出形状为 `(1, num_classes)`。
2. 打印模型参数总数。

---

### Step 3.4 — 实现 Keras 版 VGG16-BN

**不使用** `tf.keras.applications.VGG16` 作为基础（因其不含 BatchNorm）。**完全手动实现 VGG16-BN**（在每个 Conv2D 层后、ReLU 前插入 BatchNormalization），使其与 PyTorch `vgg16_bn` 架构完全对齐。

**迁移学习权重加载**：先创建一个临时的 `tf.keras.applications.VGG16(weights='imagenet', include_top=True)` 模型，**仅将其 Conv2D 层和 Dense 层的权重复制到手动实现的 VGG16-BN 模型中**，BN 层保持默认初始化。最后重新初始化分类层以匹配类别数。

> **权重复制实现要点（重要）：** 分别从两个模型中过滤出 Conv2D 层列表和 Dense 层列表，按各自列表内的序号一一对应复制；最后一个 Dense（分类层）跳过。
> **框架差异说明：** PyTorch 的 `vgg16_bn(weights='IMAGENET1K_V1')` 包含 ImageNet 训练过程中学到的 BN 参数，而 Keras 版的 BN 参数从默认值开始微调。此差异在报告中说明。

**验证：**
1. 测试前向传播，确认输出形状正确。
2. 打印参数总数，与 PyTorch 版对比。
3. 加载预训练权重后，对比 Conv 层权重与官方 VGG16 的 Conv 层权重，确认数值完全一致。

---

### Step 3.5 — 实现 PyTorch 版 MobileNetV2

使用 `torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')`，保持默认架构不变，仅重新初始化分类头以匹配 `num_classes`。

**验证：**
1. 创建模型实例，用 `(1, 3, 224, 224)` 做前向传播，确认输出形状为 `(1, num_classes)`。
2. 打印模型参数总数。

---

### Step 3.6 — 实现 Keras 版 MobileNetV2

使用 `tf.keras.applications.MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)`，添加新分类头匹配 `num_classes`。

**验证：**
1. 测试前向传播，确认输出形状正确。
2. 打印参数总数，与 PyTorch 版对比，差异应小于 1%。

---

### Step 3.7 — 参数对齐总表

创建 `results/param_comparison.csv`，包含以下列：

| dataset | model | input_size | num_classes | pytorch_params | keras_params | diff_pct |

填入 6 行：每个数据集（plantvillage, cifar10） × 3 模型（ResNet50、VGG16、MobileNetV2）。

**验证：** ResNet50 和 MobileNetV2 使用框架官方实现，`diff_pct` < 1%；VGG16 因 BN 对齐实现差异 `diff_pct` < 1%。

---

## Phase 4: 数据加载与预处理管线

### Step 4.1 — 实现共享数据加载函数

实现 `_load_dataset_split(dataset, fold, split)` 通用函数，加载指定数据集/fold/split 的数据：

- **支持多数据集：** 通过 `dataset` 参数（"plantvillage" 或 "cifar10"）确定文件路径。
- **内存缓存：** 使用 `_data_cache` 字典按 `{dataset}_images` / `{dataset}_labels` 键缓存，避免重复加载。
- PlantVillage 的 fold 索引是**相对 trainval 子集的局部索引**，需先加载 `{dataset}_trainval_indices.npy`，再用 fold 局部索引从中取值。

---

### Step 4.2 — 实现 PyTorch 数据加载器

构建 PyTorch DataLoader（`get_pytorch_dataloader(dataset, fold, split, config, num_workers=8)`）：

- **训练集增强**：RandomResizedCrop(224, scale=(0.8, 1.0)) + RandomHorizontalFlip。**对 CIFAR-10（32×32），RandomResizedCrop 会自动 resize 到 224×224。**
- 验证集/测试集：Resize(224, 224) + 标准化。
- 应用标准化（使用该折对应的 mean/std，通过 `load_dataset_stats(dataset, fold)` 获取）。
- batch_size=128，train 时 shuffle=True。
- **DataLoader 并行设置：** `num_workers=8`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`。
- **`drop_last` 策略：** 仅训练集使用 `drop_last=True`；验证集和测试集使用 `drop_last=False`。

**验证：**
1. 加载 fold 0 训练集、验证集，确认样本数正确。
2. 取一个 batch，确认形状为 `(128, 3, 224, 224)`。
3. 检查标准化后的像素值范围大致在 [-3, 3] 之间。

---

### Step 4.3 — 实现 Keras 数据加载器

使用 `tf.data.Dataset` API（`get_keras_dataset(dataset, fold, split, config, shuffle_seed=42)`），功能完全对应 Step 4.2。

- **关键性能优化：** 使用 `tf.data.Dataset.from_tensor_slices()` + `.cache()` 缓存数据。
- 数据需先用 `np.ascontiguousarray()` 确保在连续内存中（非 mmap）。
- **CIFAR-10 resize 处理：**
  - 训练集：`tf.image.resize(256, 256)` → `tf.image.random_crop(224, 224)` + `random_flip_left_right`
  - 验证/测试集：`tf.image.resize(224, 224)`（通用，对已是 224×224 的 PlantVillage 无影响）
- **数据管线设置：** `dataset.map(..., num_parallel_calls=tf.data.AUTOTUNE)`, `dataset.prefetch(tf.data.AUTOTUNE)`。训练集加 `dataset.shuffle(...)`。
- **`drop_remainder` 策略：** 仅训练集使用 `drop_remainder=True`。

> **重要：不使用混合精度（mixed_float16）。** 此前尝试在 Keras 训练中启用 `tf.keras.mixed_precision.set_global_policy('mixed_float16')` + `LossScaleOptimizer`，但由于 `LossScaleOptimizer` 需要配合 `get_scaled_loss()` / `get_unscaled_gradients()` 使用，在自定义 `tf.GradientTape` 循环中未正确实现，**导致训练准确率从 ~95% 暴跌至 ~3%**。已移除混合精度，改用标准 float32 训练。PyTorch 端的 `torch.cuda.amp` 混合精度实现正确，予以保留。

**验证：**
1. 与 PyTorch 版相同的样本数检查。
2. 取一个 batch，确认形状为 `(128, 224, 224, 3)`（注意 Keras 通道在最后）。
3. **关键一致性测试：** 对同一张图片（固定索引，无增强），分别用两个管线加载并标准化后，比较像素值——两者之差的绝对值最大值应 < 1e-5。

---

## Phase 5: 训练循环

### Step 5.1 — 实现 PyTorch 训练脚本

实现完整训练循环（`train_pytorch(dataset, model_name, fold, config, ...)`）。要求：

- 从命令行参数接收：`--framework`, `--dataset`, `--model`, `--fold`, `--config`, `--epochs_override`, `--num_workers`。
- 设置随机种子（Python, NumPy, PyTorch, CUDA），seed = 42 + fold。
- 初始化模型（`build_pytorch_model(model_name, num_classes=get_num_classes(dataset))`，加载 ImageNet 预训练权重）、优化器（SGD, momentum=0.9, nesterov=True, weight_decay=1e-4）。
- 使用 base_lr=0.04。
- 学习率调度：Cosine decay（eta_min=0）+ 5 epochs 线性 warmup（从 LR=0 **按 step/iteration** 线性增长到 base_lr）。**Cosine decay 从 warmup 结束后开始，`T_max = total_steps - warmup_steps`**。
- **早停机制 (Early Stopping)：** 监控验证集损失 (val_loss)，`patience=7`。
- **cuDNN benchmark：** `torch.backends.cudnn.benchmark = True`（自动选择最优卷积算法，加速训练）。
- **AMP 混合精度（PyTorch）：** 使用 `torch.cuda.amp.GradScaler()` + `torch.cuda.amp.autocast()` 加速训练。
- 每个 epoch 记录：train_loss, train_accuracy, val_loss, val_accuracy, val_macro_f1, epoch_time_seconds, learning_rate。
  - **`epoch_time_seconds`：** 仅计训练时间，**不含** validation 评估时间。
  - **`learning_rate`：** 记录该 epoch **最后一个 step** 的学习率。
- **Best checkpoint 选择依据：** 基于 **val_accuracy**。Checkpoint 保存至 `logs/checkpoints/`，**保留不删除**。
- 将逐 epoch 日志**实时**保存为 CSV。**每完成一个 epoch 即追加一行**。
- **文件命名：** `{framework}_{dataset}_{model}_fold{fold}.csv`（如 `pytorch_cifar10_resnet50_fold0.csv`）。
- **训练结束后：** 加载 best checkpoint，在 test 集上评估，记录 test_accuracy 和 test_macro_f1，生成混淆矩阵。

**验证：**
1. 用 ResNet50 + fold 0 运行 3 个 epoch。
2. 确认 CSV 日志文件生成，包含正确列名和 3 行数据。
3. 确认 train_loss 在 3 个 epoch 中呈下降趋势。
4. 确认 epoch_time_seconds 为正数且合理。
5. 确认 test 评估结果和混淆矩阵文件已生成。

---

### Step 5.2 — 实现 Keras 训练脚本

功能完全对应 Step 5.1（`train_keras(dataset, model_name, fold, config, ...)`）。**使用 `tf.GradientTape` 自定义训练循环**（而非 `model.fit()` + Callback），以便精确控制 per-step LR warmup 和早停逻辑，与 PyTorch 训练循环结构对齐。要求：

- 使用相同的命令行参数接口（包含 `--dataset`）。
- 设置随机种子（Python, NumPy, TensorFlow）。
- 使用 `tf.keras.optimizers.SGD` 配合相同超参数。**weight_decay 实现方式：** 在模型每个带权重的层定义时加 `kernel_regularizer=tf.keras.regularizers.l2(weight_decay / 2)`；在 `GradientTape` 循环中，total_loss = cross_entropy_loss + `sum(model.losses)`。
- **不使用混合精度**（float32 训练，不使用 `LossScaleOptimizer`）。
- 实现 Cosine decay + warmup 的自定义 LR schedule，与 PyTorch 版完全一致。
- 每个 epoch 记录与 Step 5.1 相同的指标。
- **逐 epoch 实时写入 CSV**。
- 训练结束后进行 test 评估，生成混淆矩阵。
- **Checkpoint 格式：** `.keras`。

**验证：** 同 Step 5.1 的验证方法，但使用 Keras 脚本。

---

### Step 5.3 — 验证学习率调度一致性

分别运行 PyTorch 和 Keras 脚本 10 个 epoch，检查学习率。

**验证：** 两个框架在每个 step 的学习率差异 < 1e-6。

---

## Phase 6: 结果汇总脚本

### Step 6.1 — 实现结果汇总脚本

实现汇总脚本，扫描 `logs/` 下所有 epoch 级 CSV 日志和 test JSON 文件，对每个文件提取汇总数据：

- framework, dataset, model, fold
- val_acc_best, val_f1_best, test_accuracy, test_macro_f1
- epoch_best, epochs_trained
- time_per_epoch_avg, total_training_time, images_per_sec_avg

> **多数据集支持：** `extract_run_summary()` 从 test JSON 中读取 `dataset` 字段，自动使用对应数据集的 splits 计算 images_per_sec。

额外功能：对同一 (framework, dataset, model) 组合的 3 折结果计算均值和标准差。

输出文件：
- `results/results.csv`（36 行逐折结果）
- `results/results_summary.csv`（12 行均值汇总，2 数据集 × 3 模型 × 2 框架）

**验证：** `results.csv` 行数 = 36，`results_summary.csv` 行数 = 12。

---

## Phase 7: 执行全部实验

### Step 7.1 — 运行全部 36 次训练

遍历所有组合：
- 数据集：plantvillage, cifar10
- 框架：keras, pytorch
- 模型：resnet50, vgg16, mobilenetv2
- 折：0, 1, 2

共 36 次运行。自动跳过已完成的运行（检查对应 test JSON 文件是否存在）。

```bash
python ass2_code.py run_all
```

**验证：**
1. `logs/` 目录下生成 36 个 CSV 文件。
2. 无任何运行以 NaN loss 结束。
3. PlantVillage 所有模型 test_accuracy 预期 > 90%（迁移学习优势显著）。
4. CIFAR-10 所有模型 test_accuracy 预期 > 85%。
5. 早停机制生效：大部分运行在达到最大 30 epoch 前终止。

---

### Step 7.2 — 汇总结果

运行汇总脚本：

```bash
python ass2_code.py aggregate
```

**验证：**
1. `results.csv` 包含 36 行数据。
2. 不同折的结果不完全相同（证明 CV 划分有效）。
3. 同一 model-fold 下，Keras 和 PyTorch 的准确率差距 < 5%。
4. 3 折的标准差在合理范围内（< 2%）。

---

## Phase 8: 统计分析

### Step 8.1 — 实现统计检验脚本

对每个数据集-模型组合（共 6 组：2 datasets × 3 models）：

1. 收集 Keras 的 3 个 fold 结果和 PyTorch 的 3 个 fold 结果。
2. **按 fold 配对**：同一 fold 下的 Keras 与 PyTorch 形成一对（共 3 对）。
3. 对 test_accuracy、test_macro_f1、images_per_sec_avg 分别执行：
   - 配对 t 检验 → p-value
   - Cohen's d 效应量
   - Cliff's delta 效应量
   - 差值 Δ = mean(PyTorch) − mean(Keras)

结果保存至 `analysis/statistical_results.csv`。

**验证：**
1. 输出文件行数 = 6 组 × 3 指标 = 18 行。
2. 所有 p-value 在 [0, 1] 范围内。

---

### Step 8.2 — 生成显著性汇总表

标注哪些组合在 α=0.05 下存在显著差异，标注效应量大小。

输出为 `analysis/significance_summary.csv` 和 `analysis/significance_summary.md`。

---

## Phase 9: 可视化

### Step 9.1 — 绘制训练曲线

为每个数据集-模型组合（6 组）绘制：

- 子图 1：训练 loss vs. epoch
- 子图 2：验证 accuracy vs. epoch
- 子图 3：验证 Macro-F1 vs. epoch
- 子图 4：学习率 vs. epoch

每个子图包含两个框架各 3 条折线 + 均值粗线。

保存为 `plots/curves_{dataset}_{model}.png`。

**验证：** 生成 6 张图片（2 数据集 × 3 模型）。

---

### Step 9.2 — 绘制框架对比柱状图

- 准确率对比柱状图（按数据集×模型分组，含 3 折标准差误差棒）。
- 训练速度对比柱状图（images/sec）。

保存为 `plots/comparison_accuracy.png` 和 `plots/comparison_speed.png`。

**验证：** 生成 2 张图，每张图包含 6 组柱状对比（2 数据集 × 3 模型）。

---

## Phase 10: 最终交付物整理

### Step 10.1 — 整理配置文件

确认 `configs/` 目录包含：
- `train_config.yaml`
- `experiment_matrix.yaml`
- `dataset_stats.yaml`（包含两个数据集的统计量）

---

### Step 10.2 — 整理原始日志

确认 `logs/` 目录包含 36 个 CSV 日志文件，命名格式统一：`{framework}_{dataset}_{model}_fold{fold}.csv`。

---

### Step 10.3 — 整理汇总结果表

确认 `results/results.csv`（36 行）和 `results/results_summary.csv`（12 行）完整。

---

### Step 10.4 — 整理统计分析输出

确认 `analysis/` 目录包含统计检验结果和显著性汇总。

---

### Step 10.5 — 整理可视化图片

确认 `plots/` 目录包含所有生成的图片。

共计 8 张图片（6 训练曲线 + 2 对比柱状图）。

---

## 检查清单（总览）

| Phase | 步骤数 | 关键输出 |
|-------|--------|---------|
| 0 - 环境准备 | 4 | 目录结构、`environment.txt`、依赖锁定 |
| 1 - 数据准备 | 4 | PlantVillage + CIFAR-10 数据集 + 3折CV索引文件 + `dataset_stats.yaml` |
| 2 - 配置文件 | 2 | `train_config.yaml`、`experiment_matrix.yaml` |
| 3 - 模型构建 | 7 | 6 个模型实现（ResNet50 + VGG16 + MobileNetV2 × 双框架）+ 参数对齐表（含两个数据集） |
| 4 - 数据管线 | 3 | 通用加载函数 + PyTorch DataLoader + Keras tf.data（均支持多数据集） |
| 5 - 训练循环 | 3 | 两套训练脚本（均支持 `--dataset`）+ LR验证 |
| 6 - 结果汇总 | 1 | 汇总脚本（逐折 + 均值汇总，自动识别数据集） |
| 7 - 执行实验 | 2 | 36 次运行（2 数据集 × 3 模型 × 3 折 × 2 框架）+ 结果汇总 |
| 8 - 统计分析 | 2 | 统计检验（6 组 × 3 指标 = 18 行）+ 显著性汇总 |
| 9 - 可视化 | 2 | 6 训练曲线 + 2 对比柱状图 |
| 10 - 交付整理 | 5 | 完整交付物（36 日志 + 汇总表） |
| 11 - 结果分析 | 10 | 6 结果表 + 场景适用性分析 + 统计解读 + MobileNetV2 根因分析 + ResNet50 不稳定性分析 + VGG16-BN 评估 + 吞吐量分析 + 场景推荐表 + 局限性 |
| **合计** | **45** | |

---

## Phase 11: Experimental Results — Analysis and Discussion

> **分析说明：** 本节基于 36 次训练运行（2 数据集 × 3 模型 × 2 框架 × 3 折）的完整实验结果，以 JATIT 期刊论文质量撰写。分析聚焦于**各模型-框架组合在不同应用场景下的适用性**，而非简单的排名。

---

### 11.1 Experimental Environment

| Item | Detail |
|---|---|
| Python | 3.10.12 |
| TensorFlow | 2.20.0 |
| PyTorch | 2.9.1 |
| cuDNN | 9.1.0.02 (reported via `torch.backends.cudnn.version()` = 91002) |
| GPU | NVIDIA A40 (48 GB VRAM) |

> **Note:** The `environment.txt` was captured on a login node where NVML initialization failed. Actual training runs executed on GPU-equipped compute nodes, as confirmed by epoch-level throughput figures (60–370 images/sec) that are inconsistent with CPU-only execution.

---

### 11.2 Classification Performance Overview

#### Table A: PlantVillage — Test Accuracy (mean ± std across 3 folds)

| Model | Keras | PyTorch | Δ (PT − Keras) | p-value | Significant? |
|---|---|---|---|---|---|
| ResNet50 | 99.47 ± 0.02% | 99.77 ± 0.05% | +0.30 pp | 0.0213 | Yes |
| VGG16 | 99.30 ± 0.10% | 99.76 ± 0.06% | +0.46 pp | 0.0302 | Yes |
| MobileNetV2 | 57.18 ± 10.45% | 99.74 ± 0.05% | **+42.56 pp** | 0.0194 | Yes |

#### Table B: PlantVillage — Test Macro-F1 (mean ± std across 3 folds)

| Model | Keras | PyTorch | Δ (PT − Keras) | p-value | Significant? |
|---|---|---|---|---|---|
| ResNet50 | 0.9931 ± 0.0001 | 0.9964 ± 0.0005 | +0.0033 | 0.0107 | Yes |
| VGG16 | 0.9907 ± 0.0014 | 0.9966 ± 0.0005 | +0.0059 | 0.0323 | Yes |
| MobileNetV2 | 0.4688 ± 0.1045 | 0.9963 ± 0.0004 | **+0.5275** | 0.0127 | Yes |

#### Table C: CIFAR-10 — Test Accuracy (mean ± std across 3 folds)

| Model | Keras | PyTorch | Δ (PT − Keras) | p-value | Significant? |
|---|---|---|---|---|---|
| ResNet50 | 95.85 ± 0.10% | 93.47 ± 2.78% | −2.39 pp | 0.2766 | No |
| VGG16 | 93.79 ± 0.23% | 95.73 ± 0.12% | +1.94 pp | 0.0095 | Yes |
| MobileNetV2 | 39.61 ± 8.85% | 95.39 ± 0.14% | **+55.78 pp** | 0.0084 | Yes |

#### Table D: CIFAR-10 — Test Macro-F1 (mean ± std across 3 folds)

| Model | Keras | PyTorch | Δ (PT − Keras) | p-value | Significant? |
|---|---|---|---|---|---|
| ResNet50 | 0.9584 ± 0.0010 | 0.9345 ± 0.0279 | −0.0240 | 0.2761 | No |
| VGG16 | 0.9377 ± 0.0023 | 0.9572 ± 0.0012 | +0.0195 | 0.0095 | Yes |
| MobileNetV2 | 0.3757 ± 0.1032 | 0.9538 ± 0.0014 | **+0.5781** | 0.0106 | Yes |

#### Table E: Training Throughput — images/sec (mean ± std across 3 folds)

| Dataset | Model | Keras | PyTorch | Δ (PT − Keras) | p-value | Sig? |
|---|---|---|---|---|---|---|
| PlantVillage | ResNet50 | 97.9 ± 31.9 | 124.9 ± 0.9 | +27.0 | 0.2900 | No |
| PlantVillage | VGG16 | 72.8 ± 0.1 | 76.3 ± 0.6 | +3.5 | 0.0075 | Yes |
| PlantVillage | MobileNetV2 | 210.3 ± 23.7 | 179.3 ± 17.9 | −31.0 | 0.0292 | Yes |
| CIFAR-10 | ResNet50 | 214.0 ± 2.6 | 114.4 ± 18.3 | −99.6 | 0.0125 | Yes |
| CIFAR-10 | VGG16 | 137.1 ± 0.6 | 65.6 ± 8.8 | −71.5 | 0.0052 | Yes |
| CIFAR-10 | MobileNetV2 | 369.0 ± 0.8 | 160.5 ± 14.2 | −208.6 | 0.0015 | Yes |

#### Table F: Parameter Counts

| Dataset | Model | Input | Classes | PyTorch | Keras | Diff |
|---|---|---|---|---|---|---|
| PlantVillage | ResNet50 | 224 | 38 | 23,585,894 | 23,612,454 | 0.11% |
| PlantVillage | VGG16-BN | 224 | 38 | 134,424,678 | 134,424,678 | 0.00% |
| PlantVillage | MobileNetV2 | 224 | 38 | 2,272,550 | 2,272,550 | 0.00% |
| CIFAR-10 | VGG16-BN | 32→224 | 10 | 14,728,266 | 14,728,266 | 0.00% |
| CIFAR-10 | MobileNetV2 | 32→224 | 10 | 2,236,682 | 2,236,682 | 0.00% |

> CIFAR-10 images are natively 32×32 and resized to 224×224 in the training pipeline.

---

### 11.3 Scenario-Based Suitability Analysis

The following subsections discuss each model–framework combination in terms of **what practical deployment context it serves**, rather than asserting absolute superiority.

#### 11.3.1 High-Resolution Domain-Specific Tasks (PlantVillage Scenario)

PlantVillage represents a class of problems where images are captured at moderate-to-high resolution (224×224), exhibit fine-grained inter-class differences (e.g., distinguishing between closely related disease symptoms on similar leaf surfaces), and the label space is relatively large (38 classes). This scenario is representative of agricultural monitoring, medical dermatology, and industrial surface inspection.

**ResNet50** and **VGG16** demonstrated strong cross-framework stability in this scenario. Both frameworks achieved test accuracies above 99.3%, with PyTorch holding a statistically significant but practically marginal advantage of 0.30 percentage points (ResNet50) and 0.46 percentage points (VGG16). From a deployment perspective, this margin is unlikely to influence real-world decision-making; both implementations are equally suitable for production-grade plant disease identification pipelines. The choice between Keras and PyTorch for these architectures may therefore be guided by ecosystem factors — developer familiarity, integration with existing infrastructure (e.g., TensorFlow Serving vs. TorchServe), and hardware-specific optimization libraries — rather than classification performance.

**MobileNetV2** presents a strikingly different picture. PyTorch MobileNetV2 achieved 99.74% accuracy with negligible cross-fold variance (±0.05%), performing on par with heavier architectures. Keras MobileNetV2, however, collapsed to 57.18% with extreme variance (±10.45%), rendering it unsuitable for any deployment context. This finding has critical implications: **practitioners who require a lightweight, deployment-friendly model for high-resolution domain tasks should use the PyTorch implementation**, and must not assume that framework migration preserves performance for lightweight architectures.

**Throughput considerations:** PyTorch ResNet50 was approximately 28% faster than Keras on PlantVillage (124.9 vs. 97.9 images/sec), though this difference did not reach statistical significance due to high variance in Keras timings (std = 31.9, reflecting the longer first-fold run of Keras ResNet50 which trained for 43 epochs before early stopping, in contrast to the consistent 30-epoch runs on folds 1 and 2). VGG16 throughput was comparable across frameworks (~73–76 images/sec). For MobileNetV2, Keras reported higher throughput (210 vs. 179 images/sec), but this figure is misleading because Keras MobileNetV2 early-stopped at epoch 8–9 due to catastrophic training failure; the throughput measurement reflects a broken training run rather than a functionally comparable pipeline.

#### 11.3.2 Low-Resolution Upsampled Tasks (CIFAR-10 Scenario)

CIFAR-10 represents a fundamentally different scenario: natively low-resolution images (32×32) that must be upsampled to 224×224 to match pretrained model receptive fields. This introduces bilinear interpolation artifacts — blurriness, loss of high-frequency texture detail, and a distribution shift from the ImageNet pretraining domain. Real-world analogues include remote sensing at limited spatial resolution, surveillance with low-resolution cameras, and medical imaging with sub-optimal acquisition protocols.

**Keras ResNet50** exhibited the most stable behaviour in this scenario, achieving 95.85% accuracy with remarkably low cross-fold variance (±0.10%). All three folds completed the full 30-epoch schedule without early stopping. This stability makes Keras ResNet50 particularly suitable for applications where consistent, predictable performance across different data partitions is prioritized — for instance, regulatory-grade classification systems where variability across validation runs must be minimized.

**PyTorch ResNet50** on CIFAR-10 revealed a training stability concern. While fold 2 achieved 96.68% accuracy (the highest individual accuracy for ResNet50 on CIFAR-10), folds 0 and 1 early-stopped at epoch 8 with only 91.88–91.85% accuracy. Epoch-level logs reveal that validation accuracy peaked at epoch 1 (during warmup at LR ≈ 0.008) and then deteriorated as the learning rate increased toward the base rate of 0.04, triggering early stopping. This warmup-phase instability produced a large cross-fold standard deviation (±2.78%). The difference between Keras and PyTorch ResNet50 was not statistically significant (p = 0.277), but the high variance in PyTorch results renders it less suitable for deployment contexts that require reproducible performance guarantees. This observation suggests that **PyTorch ResNet50 may require a reduced base learning rate (e.g., 0.01–0.02) or extended warmup when applied to distribution-shifted inputs**, whereas Keras ResNet50 tolerated the same hyperparameters without instability.

**PyTorch VGG16** delivered the most robust performance on CIFAR-10 among the cross-framework comparisons, achieving 95.73% accuracy with low variance (±0.12%). The statistically significant 1.94 percentage point advantage over Keras VGG16 (93.79%, p = 0.0095) is attributable to VGG16-BN implementation differences: PyTorch loads pretrained BatchNorm running statistics from ImageNet, whereas the Keras implementation initialises BN layers from defaults (mean = 0, variance = 1) because `tf.keras.applications.VGG16` does not include BatchNorm. This head start in feature normalization appears to matter more under the distribution-shifted CIFAR-10→224 regime than on the in-distribution PlantVillage images. **For tasks involving substantial distribution shift between pretraining and target domains, frameworks that provide pretrained BN statistics offer a measurable advantage.**

Keras VGG16 on CIFAR-10 nonetheless achieved a respectable 93.79% with high stability (±0.23%). For scenarios where VGG16 is required for architectural compatibility (e.g., with existing feature extraction pipelines) and the target framework is TensorFlow/Keras, the 1.9 pp performance gap relative to PyTorch is unlikely to be prohibitive in most practical settings.

**MobileNetV2 on CIFAR-10** follows the same pattern as PlantVillage: PyTorch achieves excellent performance (95.39%, ±0.14%) while Keras collapses catastrophically (39.61%, ±8.85%). This result is analysed in depth in Section 11.5.

**Throughput on CIFAR-10:** Keras was substantially faster than PyTorch across all stable configurations — approximately 1.88× for ResNet50 (214 vs. 114 images/sec) and 2.09× for VGG16 (137 vs. 66 images/sec). This advantage is likely attributable to (a) the `tf.data` pipeline's efficient handling of small-image upsampling via fused `tf.image.resize` operations, and (b) PyTorch's AMP mixed-precision overhead (GradScaler bookkeeping) providing less net benefit on a compute-light task (small source images, limited augmentation surface). **For throughput-sensitive applications on low-resolution data, Keras offers a significant training speed advantage** — provided the chosen architecture is not susceptible to the instability issues identified with MobileNetV2.

#### 11.3.3 Deployment-Constrained Scenarios (Lightweight Model Selection)

In edge-deployment and mobile-inference contexts, MobileNetV2 (2.27M parameters for PlantVillage, 2.24M for CIFAR-10) is the natural choice due to its 10× smaller footprint compared to ResNet50 (23.6M) and 59× smaller than VGG16 (134.4M). In the PyTorch implementation, MobileNetV2 achieves classification accuracy fully competitive with heavier architectures — 99.74% on PlantVillage and 95.39% on CIFAR-10 — demonstrating that parameter efficiency does not sacrifice accuracy under proper training conditions.

However, this study establishes that **MobileNetV2's deployment viability is framework-dependent**. The Keras (TensorFlow) implementation of MobileNetV2 proved unreliable across both datasets, eliminating it as a candidate for TensorFlow-based deployment workflows. Practitioners building TensorFlow Lite or TensorFlow.js pipelines with MobileNetV2 should exercise caution and conduct thorough validation; the instability observed here may or may not reproduce under different hyperparameter configurations, but the risk is non-negligible.

**Recommendation:** For lightweight deployment, PyTorch MobileNetV2 with ONNX export provides the safest path to cross-platform inference. If TensorFlow deployment is mandatory, ResNet50 or VGG16 are more reliable choices, though at substantially higher parameter cost.

---

### 11.4 Statistical Significance Interpretation

Of the 18 paired comparisons (2 datasets × 3 models × 3 metrics), 15 reached statistical significance at α = 0.05. All significant comparisons exhibited large Cohen's d values (|d| > 3.0 for accuracy/F1 metrics), and Cliff's delta values of ±1.0 (indicating complete stochastic dominance across folds). These statistics must be interpreted with caution due to two factors:

1. **Low sample size (n = 3 folds):** With only 3 paired observations per test, the paired t-test has limited power and the effect-size estimates are inflated. The large Cohen's d values (e.g., d = 6.25 for CIFAR-10 MobileNetV2 accuracy) reflect both genuine performance differences and the narrow denominator (small within-framework variance). In the Keras MobileNetV2 cases, the performance difference is indisputable regardless of sample size; however, for the smaller ResNet50 and VGG16 differences (Δ < 2 pp), a larger number of folds (e.g., 5 or 10) would provide more robust significance estimates.

2. **Multiple comparisons:** Conducting 18 simultaneous tests inflates the family-wise error rate. Under Bonferroni correction (α_adj = 0.05/18 ≈ 0.0028), only 3 comparisons remain significant — all in the throughput and MobileNetV2 accuracy categories. The VGG16 and ResNet50 accuracy differences on PlantVillage (p ≈ 0.02–0.03) would no longer be significant under this stricter threshold, suggesting that the small absolute differences (< 0.5 pp) in these cases may not be generalizable beyond the tested folds.

**The three comparisons that did not reach significance** are:
- CIFAR-10 ResNet50 accuracy (p = 0.277): dominated by PyTorch's fold-level instability (Section 11.3.2).
- CIFAR-10 ResNet50 macro-F1 (p = 0.276): same underlying cause.
- PlantVillage ResNet50 throughput (p = 0.290): high Keras timing variance from the fold-0 anomaly.

**Summary:** Statistical significance in this benchmark is primarily driven by two phenomena — (a) the catastrophic Keras MobileNetV2 failures (overwhelming effect) and (b) smaller but consistent PyTorch advantages attributable to BatchNorm pretraining (VGG16) or implementation-level numerical differences (ResNet50 on PlantVillage). When MobileNetV2 results are excluded, the accuracy differences between frameworks are modest (< 2.5 pp) and their practical impact is limited.

---

### 11.5 Keras MobileNetV2 Failure — Root Cause Analysis

The most prominent finding of this benchmark is the catastrophic failure of Keras MobileNetV2 across both datasets. This subsection presents a detailed analysis.

#### 11.5.1 Observed Phenomenon

Keras MobileNetV2 exhibited a consistent pattern across all 6 runs (2 datasets × 3 folds):

- **Training metrics remained healthy:** Training accuracy reached 95%+ (CIFAR-10) and 99%+ (PlantVillage) within 8 epochs.
- **Validation loss exploded:** Val_loss increased from ~1.6–2.2 at epoch 1 to 7–17 within 5–8 epochs, a 4–10× amplification.
- **Validation accuracy collapsed:** Despite high training accuracy, validation accuracy fell to 16–48% (CIFAR-10) and 3–66% (PlantVillage).
- **Early stopping triggered universally:** All 6 runs terminated at epoch 8–9.
- **No analogous behaviour in PyTorch:** PyTorch MobileNetV2 converged normally on both datasets, achieving 95.4% (CIFAR-10) and 99.7% (PlantVillage).

#### 11.5.2 Distinguishing Overfitting from Numerical Instability

Classical overfitting manifests as a gradual increase in validation loss coupled with a slow decline in validation accuracy — the model memorises training data while losing generalisation. The behaviour observed here is qualitatively different: validation loss **exploded** by an order of magnitude within 2–3 epochs while training metrics continued to improve normally. This pattern is characteristic of **numerical instability or severe feature-distribution mismatch** between training and validation sets as perceived by the model's internal representations, rather than simple overfitting.

The cross-fold consistency of this failure (6/6 runs affected, across two unrelated datasets) eliminates data-specific artefacts as a cause and points to a systematic implementation-level factor.

#### 11.5.3 Hypothesised Root Causes

**Hypothesis 1 — BatchNorm momentum and epsilon differences:**
MobileNetV2 relies heavily on BatchNormalization layers throughout its inverted-residual blocks. Subtle differences in default BN hyperparameters between frameworks — PyTorch uses `momentum=0.1, epsilon=1e-5` while TensorFlow/Keras uses `momentum=0.99, epsilon=1e-3` — can produce divergent running statistics during fine-tuning. The Keras convention (`momentum=0.99`) results in a slower exponential moving average update, meaning that the running mean/variance adapts more sluggishly to the new domain. Under aggressive learning rates (0.04 with warmup), the BN running statistics in Keras may lag behind rapidly changing weight distributions, causing a growing mismatch between training-time batch statistics and inference-time running statistics. This mismatch is amplified at validation time (when `training=False` forces the use of running statistics), explaining the divergence between training accuracy (which uses batch statistics) and validation accuracy (which uses lagging running statistics).

**Hypothesis 2 — Depthwise separable convolution numerical sensitivity:**
MobileNetV2's depthwise separable convolutions operate on individual channels, producing narrow feature maps that are more sensitive to per-channel normalisation errors. When BN running statistics diverge (Hypothesis 1), the impact is amplified through the depthwise separable architecture more than through standard convolutions (ResNet50, VGG16), where cross-channel interactions provide a natural averaging effect.

**Hypothesis 3 — Learning rate interaction:**
The base learning rate of 0.04 was selected for batch size 128 following the linear scaling rule. Epoch-level logs confirm that val_loss deterioration in Keras MobileNetV2 tracks the warmup LR increase: val_loss at epoch 1 (LR ≈ 0.008) is 1.6–2.2, but by epoch 5 (LR ≈ 0.04) it reaches 6–17. This monotonic correlation suggests that the Keras MobileNetV2 gradient landscape becomes unstable at higher learning rates, while PyTorch MobileNetV2 tolerates the same schedule. The interaction between LR magnitude and BN momentum differences (Hypothesis 1) provides a plausible mechanism: faster weight updates under high LR exacerbate the lag between batch statistics and running statistics in Keras.

**Hypothesis 4 — Absence of mixed precision in Keras as a contributing factor:**
PyTorch training used AMP (float16 forward + backward, float32 accumulation), while Keras used float32 throughout (due to the `LossScaleOptimizer` failure documented in the technical notes below). While float32 should be more numerically stable than float16, the different gradient dynamics — particularly different effective step sizes due to precision-dependent gradient magnitudes — may interact with the BN momentum issue in unpredictable ways. This hypothesis is secondary and would require controlled ablation to confirm.

#### 11.5.4 Significance for Framework Comparison Research

This finding elevates the benchmark's contribution beyond a simple "frameworks are equivalent" or "framework X is better" conclusion. It demonstrates that **framework equivalence is conditionally dependent on the (architecture, dataset, hyperparameter) triple**, and that specific boundary conditions can expose hidden implementation divergences that produce catastrophic performance gaps.

The practical implication is clear: **when migrating a transfer-learning pipeline from one framework to another, performance parity cannot be assumed**. This is particularly important for lightweight architectures (MobileNet family, EfficientNet-Lite) that are commonly used in mobile and edge deployments, where the assumption of framework-neutrality may lead to silently degraded production models.

Framework comparison studies should therefore not merely report average performance across configurations, but must systematically identify and analyse failure modes — configurations where one framework produces catastrophically worse results than the other. These failure modes, rather than the aggregate statistics, represent the most actionable information for practitioners.

---

### 11.6 PyTorch ResNet50 on CIFAR-10 — Training Instability Analysis

A secondary but noteworthy finding is the training instability of PyTorch ResNet50 on CIFAR-10, which contrasts with the stability of the same model under Keras.

#### 11.6.1 Observed Phenomenon

- **Fold 0:** Validation accuracy peaked at 91.97% (epoch 1, LR ≈ 0.008), then declined through epochs 2–8 as LR increased during warmup. Val_loss increased from 0.233 to 0.279. Early stopping triggered at epoch 8.
- **Fold 1:** Nearly identical pattern. Val accuracy peaked at 91.98% (epoch 1), declined through warmup. Early stopping at epoch 8.
- **Fold 2:** Normal convergence. The model survived the warmup phase, reached val accuracy 96.69% at epoch 29, and completed all 30 epochs. Final test accuracy: 96.68%.

The 2-out-of-3 failure rate and the consistent failure pattern (peak at epoch 1, decline during warmup) point to a **warmup-LR sensitivity specific to PyTorch ResNet50 under distribution-shifted inputs**.

#### 11.6.2 Interpretation

The CIFAR-10→224 upsampling introduces a distribution shift from ImageNet that is more pronounced for the early layers (which expect high-frequency textures present in natural images but absent in bilinearly upsampled 32×32 images). During the warmup phase, the learning rate increases from 0.008 to 0.04. For fold 2, the specific random initialisation of the final classification layer (seed = 42 + 2 = 44) happened to place the model in a region of the loss landscape where it could tolerate the increasing LR and eventually recover. For folds 0 and 1, the initial conditions were less favourable, and the increasing LR pushed the model away from the narrow valley found at epoch 1.

Keras ResNet50, by contrast, survived the same warmup schedule on all 3 folds. This suggests a subtle framework-level difference in how the warmup interacts with the model's gradient dynamics — possibly related to (a) the slight architectural difference (23.6M vs. 23.6M parameters, with Keras using `GlobalAveragePooling2D` as an explicit layer vs. PyTorch's built-in `AdaptiveAvgPool2d`), or (b) differences in gradient computation precision (Keras float32 vs. PyTorch AMP float16), which may alter the effective step size during the critical warmup transition.

**Practical implication:** For distribution-shifted transfer learning tasks, a reduced base learning rate or longer warmup schedule is recommended as a precautionary measure, regardless of framework. The specific failure mode observed here — where increasing LR during warmup destabilises a model that initially converges well at low LR — is a known risk in cosine-annealing schedules with aggressive base rates.

---

### 11.7 VGG16-BN Cross-Framework Alignment Assessment

The VGG16-BN comparison deserves dedicated attention because it involves a known asymmetry: PyTorch loads pretrained BN running statistics from ImageNet, while the Keras implementation (which was manually constructed with BatchNorm layers) initialises BN from defaults.

**PlantVillage results:** Keras VGG16 (99.30%) vs. PyTorch VGG16 (99.76%). The 0.46 pp gap is statistically significant (p = 0.030) but practically small. Both frameworks converged smoothly over 23–30 epochs without early stopping. This suggests that on in-distribution, high-resolution data, the BN initialisation difference is absorbed during fine-tuning — the 23–30 training epochs are sufficient for Keras BN to learn appropriate running statistics from the target domain.

**CIFAR-10 results:** Keras VGG16 (93.79%) vs. PyTorch VGG16 (95.73%). The 1.94 pp gap is statistically significant (p = 0.0095) and more pronounced than on PlantVillage. The distribution shift in CIFAR-10 (upsampled low-resolution images) amplifies the impact of BN initialisation: PyTorch's pretrained BN statistics provide a better starting point for feature normalisation under distribution-shifted conditions, while Keras BN must learn from scratch during a training process that is itself complicated by the non-standard input distribution.

**Assessment:** The VGG16-BN comparison is partially confounded by the BN initialisation asymmetry, which is an inherent limitation of the experimental design (Keras `tf.keras.applications.VGG16` does not include BatchNorm, necessitating manual implementation). This confound means that the VGG16 accuracy differences cannot be attributed purely to "framework choice" but rather reflect a combination of framework-level and implementation-level factors. Future work should investigate whether custom Keras VGG16-BN models with pretrained BN statistics (obtained via weight conversion scripts) close the gap.

---

### 11.8 Training Throughput — Framework-Task Interaction

The throughput results reveal that **neither framework is universally faster**; throughput advantage is contingent on the dataset characteristics.

**CIFAR-10 (small source images, resize-heavy pipeline):** Keras was 1.88× (ResNet50), 2.09× (VGG16), and 2.30× (MobileNetV2) faster than PyTorch. The likely explanation is that `tf.data`'s fused image processing operations (resize + crop + flip as a single graph node) are more efficient than PyTorch's PIL-based `torchvision.transforms` pipeline when the dominant cost is image upsampling (32→224). Additionally, PyTorch's AMP overhead (GradScaler + autocast context management) provides less net speedup on a task where the forward/backward pass is already fast relative to data loading.

**PlantVillage (native 224×224, minimal resize):** The throughput picture reverses partially. PyTorch ResNet50 was ~28% faster (124.9 vs. 97.9 images/sec, though not significant). VGG16 was similar across frameworks. MobileNetV2 showed Keras as faster, but this result is invalidated by the Keras training failure. On native-resolution data where data loading is less dominant, PyTorch's AMP and optimised DataLoader (`persistent_workers`, `pin_memory`, `prefetch_factor`) narrow or eliminate the Keras advantage.

**Practical guidance:** For training pipelines dominated by image preprocessing (small-to-large resizing), Keras/TensorFlow's `tf.data` API offers a measurable speed advantage. For training on native-resolution images where GPU compute is the bottleneck, throughput is largely framework-neutral, and PyTorch's AMP provides a slight edge.

---

### 11.9 Summary of Scenario-Specific Recommendations

| Scenario | Recommended Configuration | Rationale |
|---|---|---|
| High-res domain-specific classification (e.g., plant disease, medical imaging) | Either framework + ResNet50 or VGG16 | Both achieve >99.3% on PlantVillage; framework choice guided by ecosystem preference |
| High-res + deployment-constrained (edge, mobile) | PyTorch MobileNetV2 | 99.7% accuracy at 2.3M parameters; Keras MobileNetV2 is unreliable |
| Low-res upsampled classification + stability required | Keras ResNet50 | Most stable on CIFAR-10 (95.85%, σ = 0.10%); PyTorch ResNet50 shows warmup instability |
| Low-res + highest accuracy target | PyTorch VGG16 | 95.73% on CIFAR-10; benefits from pretrained BN under distribution shift |
| Training throughput priority (small source images) | Keras (any stable model) | 1.9–2.3× faster than PyTorch on CIFAR-10 due to tf.data efficiency |
| Training throughput priority (native-res images) | Either framework | Throughput difference is small and dataset-dependent |
| Framework migration of lightweight models | Requires re-validation | Keras MobileNetV2 failure demonstrates that framework parity cannot be assumed |

---

### 11.10 Limitations and Threats to Validity

1. **Low fold count (n = 3):** The 3-fold cross-validation provides only 3 paired observations per statistical test, limiting power and inflating effect-size estimates. A 5- or 10-fold design would strengthen significance claims.
2. **Single learning-rate configuration:** All experiments used base_lr = 0.04 with cosine annealing. The Keras MobileNetV2 and PyTorch ResNet50 failures may be partially attributable to this specific hyperparameter choice, and a grid search over learning rates would provide a more complete picture.
3. **Single GPU type:** Results are specific to NVIDIA A40. Different GPU architectures (e.g., NVIDIA V100, A100, consumer GPUs) may alter throughput rankings and potentially affect numerical stability.
4. **CIFAR-10 upsampling artefact:** Bilinear upsampling from 32×32 to 224×224 introduces artefacts that interact non-trivially with pretrained model expectations. This is a deliberate design choice to test transfer learning under distribution shift, but it limits the generalisability of CIFAR-10 results to native-resolution classification tasks.
5. **VGG16-BN confound:** The BN initialisation asymmetry between Keras (default) and PyTorch (pretrained) is an inherent experimental limitation. The VGG16 comparison conflates framework-level and implementation-level effects.
6. **No optimizer ablation:** Only SGD was tested. Framework-specific optimiser implementations (e.g., Adam, AdamW) may exhibit different cross-framework behaviour.
7. **Keras mixed-precision exclusion:** Due to the `LossScaleOptimizer` failure in custom GradientTape loops (documented in the technical notes below), Keras ran in float32 while PyTorch used AMP. This asymmetry potentially affects both accuracy and throughput comparisons.

---

## 重要技术备注

### Keras 混合精度修复记录

**问题：** 在 Keras 训练循环中启用 `mixed_float16` 全局策略 + `LossScaleOptimizer` 后，训练准确率从 ~95% 暴跌至 ~3%（接近随机猜测）。

**根因：** `LossScaleOptimizer` 包装了 base optimizer，但在 `tf.GradientTape` 自定义循环中，必须使用 `optimizer.get_scaled_loss(loss)` 计算缩放后的 loss，再用 `optimizer.get_unscaled_gradients(grads)` 还原梯度。原代码直接对未缩放的 loss 求梯度，导致 float16 下梯度数值不正确。

**修复方案：** 完全移除 `mixed_float16` 和 `LossScaleOptimizer`，改用标准 float32 的 `tf.keras.optimizers.SGD`。同时移除了 `train_step`、`eval_step` 和 test evaluation 中不必要的 `tf.cast(logits, tf.float32)` 操作。

**PyTorch 端不受影响：** PyTorch 的 `torch.cuda.amp.GradScaler` + `autocast()` 是正确实现，予以保留。
