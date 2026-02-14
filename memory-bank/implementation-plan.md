# Keras vs PyTorch 图像分类 Benchmark — 实施计划

---

## Phase 0: 项目初始化与环境准备

### Step 0.1 — 创建项目目录结构

创建以下目录树：

```
ass2-exp/
├── configs/            # 超参数配置文件（YAML）
├── data/               # 数据集存放目录
│   └── splits/         # 固定划分索引文件
├── src/
│   ├── keras/          # Keras/TensorFlow 训练代码
│   └── pytorch/        # PyTorch 训练代码
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
- scikit-learn（用于 Macro-F1）
- pandas, numpy
- pyyaml
- scipy（用于统计检验）
- matplotlib（用于绘图）

安装后用 `pip freeze > requirements_lock.txt` 锁定完整依赖。

**验证：** 在新的虚拟环境中执行 `pip install -r requirements_lock.txt`，确认无报错；运行 `python -c "import tensorflow; import torch; print('OK')"` 确认两框架可正常导入。

---

### Step 0.4 — 确认 GPU 可用性

分别用 TensorFlow 和 PyTorch 检查 GPU 是否可被识别：

- TensorFlow：`python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- PyTorch：`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

**验证：** 两条命令都输出 A40 GPU 信息，无报错。

---

## Phase 1: 数据准备与固定划分

### Step 1.1 — 下载 CIFAR-10 数据集

使用 torchvision 或 keras.datasets 下载 CIFAR-10 至 `data/` 目录。

**验证：** 打印训练集形状 `(50000, 32, 32, 3)`，测试集形状 `(10000, 32, 32, 3)`，类别数 = 10。

---

### Step 1.2 — 下载 CIFAR-100 数据集

同上，下载 CIFAR-100。

**验证：** 训练集形状 `(50000, 32, 32, 3)`，测试集形状 `(10000, 32, 32, 3)`，类别数 = 100。

---

### Step 1.3 — 下载 Tiny-ImageNet 数据集

使用 HuggingFace `datasets` 库下载 Tiny-ImageNet-200 至 `data/` 目录。Tiny-ImageNet 官方没有带标签的 test set，因此使用官方 **val 集作为 test 集**，训练仅使用官方 train 集（不再从 train 中切验证集）。

```python
from datasets import load_dataset
ds = load_dataset("zh-plus/tiny-imagenet")  # 或其他可用镜像
```

下载后**预处理并保存为 numpy 数组**至 `data/` 目录（加载更快）：
- `data/tiny_imagenet_train_images.npy` — shape `(100000, 64, 64, 3)`，uint8（灰度图已复制通道扩展为 3 通道）
- `data/tiny_imagenet_train_labels.npy` — shape `(100000,)`，int
- `data/tiny_imagenet_test_images.npy` — shape `(10000, 64, 64, 3)`，uint8
- `data/tiny_imagenet_test_labels.npy` — shape `(10000,)`，int

**验证：** 统计训练集图片数 = 100,000（200 类 × 500 张），测试集（原 val）图片数 = 10,000；随机抽取 5 张图片显示尺寸为 64×64×3。

---

### Step 1.4 — 生成 CIFAR-10 固定验证集索引

从 CIFAR-10 的 50,000 训练样本中，使用固定随机种子（seed=42）进行**分层采样（stratified sampling）**，抽取 5,000 个索引作为验证集（每类恰好 500 个），其余 45,000 个作为训练集。使用 `sklearn.model_selection.train_test_split(stratify=y, test_size=5000, random_state=42)` 实现。将两组索引分别保存为：

- `data/splits/cifar10_val_indices.npy`
- `data/splits/cifar10_train_indices.npy`

**验证：**
1. 加载两个索引文件，确认长度分别为 45,000 和 5,000。
2. 确认两组索引无交集（交集为空集）。
3. 确认两组索引的并集 = {0, 1, ..., 49999}。
4. 确认验证集中各类别样本数恰好均匀（每类 500 个，因为是分层采样）。

---

### Step 1.5 — 生成 CIFAR-100 固定验证集索引

与 Step 1.4 相同逻辑（分层采样），但针对 CIFAR-100（同一固定种子 seed=42）。保存为：

- `data/splits/cifar100_val_indices.npy`
- `data/splits/cifar100_train_indices.npy`

**验证：**
1. 长度分别为 45,000 和 5,000。
2. 无交集，并集完整。
3. 验证集中每类恰好 50 个样本（100 类，分层采样保证）。

---

### Step 1.6 — 生成 Tiny-ImageNet 固定验证集索引

与 CIFAR 保持一致的 protocol：从 100,000 训练样本中通过**分层采样**划分验证集，官方 val 集作为 test 集。使用固定随机种子（seed=42）分层抽取 10,000 个索引作为验证集（每类恰好 50 个），其余 90,000 个作为训练集。将两组索引分别保存为：

- `data/splits/tiny_imagenet_train_indices.npy`
- `data/splits/tiny_imagenet_val_indices.npy`

使用以下映射：

- **训练集** = HuggingFace `train` split 中 90,000 张（按索引）
- **验证集** = HuggingFace `train` split 中 10,000 张（按索引）
- **测试集** = HuggingFace `valid` split（10,000 张）

流程：每 epoch 在 val（来自 train）上评估 → 选 best checkpoint → 最终在 test（官方 val）上评估并报告。与 CIFAR 的 protocol 完全一致，无信息泄漏。

**验证：**
1. 加载两个索引文件，确认长度分别为 90,000 和 10,000。
2. 确认两组索引无交集，并集 = {0, 1, ..., 99999}。
3. 确认验证集中每类恰好 50 个样本（200 类，分层采样保证）。
4. 测试集图片数 = 10,000。确认类别标签范围为 0-199。

---

### Step 1.7 — 计算各数据集的 mean 和 std

对每个数据集的训练集（不含验证集）计算通道级 mean 和 std（在像素值归一化到 [0,1] 之后）。**Tiny-ImageNet 中的灰度图需先复制通道扩展为 3 通道（RGB），再参与 mean/std 统计。** 结果保存至 `configs/dataset_stats.yaml`，格式如下：

```yaml
cifar10:
  mean: [R, G, B]
  std:  [R, G, B]
cifar100:
  mean: [R, G, B]
  std:  [R, G, B]
tiny_imagenet:
  mean: [R, G, B]
  std:  [R, G, B]
```

**验证：**
1. 每个 mean 值在 [0, 1] 范围内。
2. 每个 std 值在 (0, 0.5] 范围内。
3. CIFAR-10 和 CIFAR-100 的 mean/std 不完全相同（因为内容分布不同）。
4. 用计算出的 mean/std 对训练集做标准化后，检查结果的均值接近 0、标准差接近 1。

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
  # 注意：PyTorch 的 weight_decay 实现为 L2 正则化；Keras 为解耦权重衰减。
  # 各框架使用原生实现，在报告中说明差异。

lr:
  base_lr: 0.1
  base_batch_size: 256
  scheduler: cosine
  eta_min: 0.0                # cosine decay 衰减到 0
  warmup_epochs: 5
  warmup_start_lr: 0.0        # warmup 从 LR=0 线性增长到 base_lr
  warmup_mode: per_step        # warmup 按 iteration（per-step）线性插值，非 per-epoch

training:
  batch_size: 256
  gradient_accumulation_steps: 1  # 若 OOM 则增大此值，保持有效 batch_size=256
  loss: cross_entropy

epochs:
  cifar10: 200
  cifar100: 200
  tiny_imagenet: 100

augmentation:
  cifar:
    random_crop:
      size: 32
      padding: 4
    random_horizontal_flip: true
  tiny_imagenet:
    random_crop:
      size: 64
      padding: 8
    random_horizontal_flip: true

input_size:
  cifar: 32
  tiny_imagenet: 64
```

**验证：** 用 YAML 解析器加载该文件，确认所有字段与实验文档一致，无解析错误。

---

### Step 2.2 — 创建实验矩阵配置文件

创建 `configs/experiment_matrix.yaml`：

```yaml
stage1:
  datasets: [cifar10, cifar100]
  models: [resnet50, mobilenetv2]
  precisions: [fp32]
  seeds: [42, 123, 456]
  frameworks: [keras, pytorch]

stage2:
  datasets: [cifar10, cifar100, tiny_imagenet]
  models: [resnet50, mobilenetv2, convnext_tiny]
  precisions: [fp32, amp]
  seeds: [42, 123, 456, 789, 1024]
  frameworks: [keras, pytorch]
```

**验证：**
1. Stage 1 组合数 = 2 datasets × 2 models × 1 precision × 3 seeds × 2 frameworks = 24。
2. Stage 2 组合数 = 3 × 3 × 2 × 5 × 2 = 180。

---

## Phase 3: 模型构建

> **权重初始化策略：** PyTorch 默认使用 Kaiming/He 初始化，Keras 默认使用 Glorot/Xavier 初始化。两者不同可能影响收敛速度，但不影响最终性能对比的公平性。**各框架使用其默认初始化方案**，不手动对齐，在报告中说明该差异。

### Step 3.1 — 实现 PyTorch 版 MobileNetV2

在 `src/pytorch/models.py` 中实现 MobileNetV2。要求：

- 使用 torchvision 预定义架构（`torchvision.models.mobilenet_v2`），但不加载预训练权重。
- **CIFAR (32×32) 首层修改：** 将首层 Conv 的 stride 从 2 改为 1（即 3×3 kernel、stride=1、padding=1）。MobileNetV2 标准结构无显式 MaxPool 层，仅需修改首层 stride 即可防止特征图过早缩小。
- **Tiny-ImageNet (64×64) 首层：** 保持默认结构（3×3 stride=2，64→32），属于"温和"下采样，无需修改。
- 修改最后的分类层（classifier head）以匹配数据集类别数（10 / 100 / 200）。

**验证：**
1. 创建模型实例，分别用 `(1, 3, 32, 32)` 和 `(1, 3, 64, 64)` 的随机张量做前向传播，确认输出形状分别为 `(1, num_classes)`。
2. 对 num_classes = 10, 100, 200 各测试一次，均无报错。
3. 打印模型参数总数，确认在合理范围内。

---

### Step 3.2 — 实现 Keras 版 MobileNetV2

在 `src/keras/models.py` 中实现。要求与 PyTorch 版架构尽可能对齐：

- 使用 `tf.keras.applications.MobileNetV2`，不加载预训练权重（`weights=None`）。
- 修改 `input_shape` 以适配 32×32 和 64×64。
- **CIFAR (32×32) 首层修改：** 与 PyTorch 版对应，将首层 Conv 的 stride 从 2 改为 1（3×3 kernel、stride=1、padding='same'）。MobileNetV2 无显式 MaxPool 层，仅需修改首层 stride。
- **Tiny-ImageNet (64×64) 首层：** 保持默认结构（3×3 stride=2，64→32），无需修改。
- 修改分类头以匹配类别数。

**验证：**
1. 同 Step 3.1，测试前向传播，确认输出形状正确。
2. 打印参数总数，与 PyTorch 版对比，差异应小于 1%（允许 BN 层等微小差异）。

---

### Step 3.3 — 实现 PyTorch 版 ResNet-50

在 `src/pytorch/models.py` 中添加 ResNet-50。使用 `torchvision.models.resnet50`，不加载预训练权重。

- **CIFAR (32×32) 首层修改：** 将 conv1 改为 3×3 kernel、stride=1、padding=1，去掉 maxpool 层。
- **Tiny-ImageNet (64×64) 首层修改：** 保留 7×7 stride=2 的 conv1（64→32），但**去掉 maxpool 层**（保持 32×32），后续 layer2/layer3/layer4 按标准 ResNet 继续下采样。去掉 maxpool 后少了一次 /2，最终 avgpool 前的特征图为 4×4（而非标准的 7×7），这是预期行为。避免 64→16 的过早压缩。
- 修改 fc 层以匹配类别数。

**验证：** 同 Step 3.1 的验证方法。额外确认 Tiny-ImageNet 输入时，avgpool 前特征图尺寸为 4×4。

---

### Step 3.4 — 实现 Keras 版 ResNet-50

在 `src/keras/models.py` 中添加。使用 `tf.keras.applications.ResNet50`，不加载预训练权重。

- **CIFAR (32×32) 首层修改：** 与 PyTorch 版对应，将首层改为 3×3 kernel、stride=1，去掉 maxpool。
- **Tiny-ImageNet (64×64) 首层修改：** 与 PyTorch 版对应，保留 7×7 stride=2 conv1（64→32），去掉 maxpool 层。最终 avgpool 前特征图为 4×4（与 PyTorch 版一致）。
- 修改分类头以匹配类别数。

**验证：** 同 Step 3.2 的验证方法，参数总数与 PyTorch 版对比。

---

### Step 3.5 — 实现 PyTorch 版 ConvNeXt-Tiny

在 `src/pytorch/models.py` 中添加 ConvNeXt-Tiny。使用 `torchvision.models.convnext_tiny`，不加载预训练权重。

- **CIFAR (32×32) 首层修改：** 将首个 stem 卷积改为 3×3 kernel、stride=1、padding=1，去掉下采样（原始 4×4 stride=4 会把 32×32 压缩到 8×8）。
- **Tiny-ImageNet (64×64) 首层修改：** 将 stem 从默认 4×4 stride=4 改为 **2×2 stride=2**（64→32），后续每个 stage 的 downsample（/2）保持不变，最终仍到 2×2。对小分辨率更友好，且改动仅在第一层，易于跨框架对齐。
- 修改分类头以匹配类别数。

**验证：** 同 Step 3.1。

---

### Step 3.6 — 实现 Keras 版 ConvNeXt-Tiny

在 `src/keras/models.py` 中添加。使用 `tf.keras.applications.ConvNeXtTiny`（TF 2.10+ 内置），不加载预训练权重。

- **CIFAR (32×32) 首层修改：** 与 PyTorch 版对应，修改 stem 卷积为 3×3 stride=1，去掉下采样。
- **Tiny-ImageNet (64×64) 首层修改：** 与 PyTorch 版对应，将 stem 改为 2×2 stride=2（64→32）。
- 修改分类头以匹配类别数。

> **注意：** `tf.keras.applications.ConvNeXtTiny` 与 `torchvision.models.convnext_tiny` 在实现细节上可能存在微小差异（如 LayerScale 初始值、Stochastic Depth rate），导致参数数不完全一致。采用各框架官方实现，在报告中记录并说明该差异。

**验证：** 同 Step 3.2。ConvNeXt-Tiny 的参数差异允许放宽到 < 5%（因架构实现差异），其他模型仍要求 < 1%。

---

### Step 3.7 — 参数对齐总表

创建 `results/param_comparison.csv`，包含以下列：

| model | input_size | num_classes | pytorch_params | keras_params | diff_pct |

填入所有 3 模型 × 3 类别数 × 2 输入尺寸 的组合。

**验证：** MobileNetV2 和 ResNet-50 的 `diff_pct` < 1%；ConvNeXt-Tiny 的 `diff_pct` < 5%（因架构官方实现差异，在报告中说明）。

---

## Phase 4: 数据加载与预处理管线

### Step 4.1 — 实现 PyTorch 数据加载器

在 `src/pytorch/data.py` 中实现数据加载函数，接受参数：dataset 名称、split（train/val/test）、配置对象。功能：

- 根据 `data/splits/` 中的索引文件划分 train/val。
- 应用标准化（使用 Step 1.7 计算的 mean/std）。
- 训练集：RandomCrop(padding=4) + RandomHorizontalFlip。
- 验证集/测试集：无数据增强，仅标准化。
- 返回 DataLoader，batch_size=256，train 时 shuffle=True。
- **DataLoader 并行设置：** `num_workers=8`, `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2`。
- **`drop_last` 策略：** 仅训练集使用 `drop_last=True`（保证每个 micro-batch 大小一致）；验证集和测试集使用 `drop_last=False`（确保所有样本都参与评估，不丢弃尾部 batch）。
- **灰度图处理：** Tiny-ImageNet 中部分图片为灰度单通道，加载时复制通道扩展为 3 通道（RGB）。

**验证：**
1. 加载 CIFAR-10 训练集，确认样本数 = 45,000。
2. 加载 CIFAR-10 验证集，确认样本数 = 5,000。
3. 取一个 batch，确认形状为 `(256, 3, 32, 32)`。
4. 检查标准化后的像素值范围大致在 [-3, 3] 之间。
5. 对 CIFAR-100 和 Tiny-ImageNet 重复以上检查。
6. 对 Tiny-ImageNet 抽查灰度图样本，确认已正确扩展为 3 通道。

---

### Step 4.2 — 实现 Keras 数据加载器

在 `src/keras/data.py` 中实现，功能完全对应 Step 4.1。使用 `tf.data.Dataset` API。

- **数据管线设置：** `dataset.map(..., num_parallel_calls=tf.data.AUTOTUNE)`, `dataset.prefetch(tf.data.AUTOTUNE)`。训练集加 `dataset.shuffle(...)`。
- **`drop_remainder` 策略：** 仅训练集使用 `dataset.batch(batch_size, drop_remainder=True)`；验证集和测试集使用 `dataset.batch(batch_size, drop_remainder=False)`（确保所有样本都参与评估）。
- **灰度图处理：** 同 PyTorch 版，灰度图复制通道扩展为 3 通道。

**验证：**
1. 与 PyTorch 版相同的样本数检查。
2. 取一个 batch，确认形状为 `(256, 32, 32, 3)`（注意 Keras 通道在最后）。
3. **关键一致性测试：** 对同一张图片（固定索引，无增强），分别用两个管线加载并标准化后，比较像素值——两者之差的绝对值最大值应 < 1e-5。
4. 对 Tiny-ImageNet 灰度图样本，确认已正确扩展为 3 通道。

---

## Phase 5: 训练循环

### Step 5.1 — 实现 PyTorch 训练脚本

在 `src/pytorch/train.py` 中实现完整训练循环。要求：

- 从命令行参数接收：`--dataset`, `--model`, `--precision`, `--seed`, `--config`。
- 设置随机种子（Python, NumPy, PyTorch, CUDA）。
- 初始化模型、优化器（SGD, momentum=0.9, nesterov=True, weight_decay=1e-4）。
- 学习率调度：Cosine decay（eta_min=0）+ 5 epochs 线性 warmup（从 LR=0 **按 step/iteration** 线性增长到 base_lr）。**Cosine decay 从 warmup 结束后开始，`T_max = total_steps - warmup_steps`**，即 cosine 只覆盖 warmup 后的剩余 steps。
- 支持 FP32 和 AMP（使用 `torch.cuda.amp`）。
- **CUDA 确定性设置：** `torch.backends.cudnn.deterministic=True`，`torch.backends.cudnn.benchmark=False`，确保同 seed 结果完全一致。
- **支持 gradient accumulation：** 从配置文件读取 `gradient_accumulation_steps`，若 > 1 则分步累积梯度，确保有效 batch size 始终为 256。**累积时 loss 需除以 `gradient_accumulation_steps`**（因为 `CrossEntropyLoss` 默认 `reduction='mean'`，累积多个 micro-batch 的梯度需做缩放）。
- 每个 epoch 记录：train_loss, train_accuracy, val_loss, val_accuracy, val_macro_f1, epoch_time_seconds, learning_rate。
  - **`epoch_time_seconds`：** 仅计训练时间（从 epoch 训练开始到训练结束），**不含** validation 评估时间。这样 `images_per_sec` 指标才能准确反映训练吞吐量。
  - **`learning_rate`：** 记录该 epoch **最后一个 step** 的学习率（因为 per-step warmup/cosine 下 LR 在单个 epoch 内会变化）。
- **Best checkpoint 选择依据：** 基于 **val_accuracy** 选出最佳 epoch，保存该 checkpoint 至 `logs/checkpoints/pytorch_{dataset}_{model}_{precision}_seed{seed}.pt`。Checkpoint 文件**保留不删除**，便于事后复现和调试。（注意：val_accuracy 和 val_macro_f1 的最佳 epoch 可能不同，统一以 val_accuracy 为准。）
- 将逐 epoch 日志保存为 CSV 至 `logs/pytorch_{dataset}_{model}_{precision}_seed{seed}.csv`。
- **训练结束后：** 加载基于 val_accuracy 选出的 best checkpoint，在 test 集上评估，同时记录 test_accuracy 和 test_macro_f1，保存为 `logs/pytorch_{dataset}_{model}_{precision}_seed{seed}_test.json`。

**验证：**
1. 用 CIFAR-10 + ResNet-50 + FP32 + seed=42 运行 3 个 epoch。
2. 确认 CSV 日志文件生成，包含正确列名和 3 行数据。
3. 确认 train_loss 在 3 个 epoch 中呈下降趋势。
4. 确认 epoch_time_seconds 为正数且合理（预计每 epoch 几十秒）。
5. 确认 val_macro_f1 在 [0, 1] 范围内。
6. 确认 learning_rate 列存在且第一个 epoch 的 LR 接近 0。
7. 确认 test 评估结果文件已生成，包含 test_accuracy 和 test_macro_f1。

---

### Step 5.2 — 实现 Keras 训练脚本

在 `src/keras/train.py` 中实现，功能完全对应 Step 5.1。**使用 `tf.GradientTape` 自定义训练循环**（而非 `model.fit()` + Callback），以便精确控制 gradient accumulation 和 per-step LR warmup，与 PyTorch 训练循环结构对齐。要求：

- 使用相同的命令行参数接口。
- 设置随机种子（Python, NumPy, TensorFlow）。
- 设置 `tf.config.experimental.enable_op_determinism()` 以确保确定性（对应 PyTorch 的 cudnn.deterministic=True）。
- 使用 `tf.keras.optimizers.SGD` 配合相同超参数（weight_decay 使用 Keras 原生解耦实现）。
- 实现 Cosine decay（eta_min=0）+ warmup 的自定义 LR schedule（从 LR=0 **按 step/iteration** 线性增长到 base_lr）。**Cosine decay 从 warmup 结束后开始，`T_max = total_steps - warmup_steps`**，与 PyTorch 版完全一致。
- AMP 模式通过 `tf.keras.mixed_precision` 实现。
- **支持 gradient accumulation：** 同 PyTorch 版，确保有效 batch size 始终为 256。**累积时 loss 需除以 `gradient_accumulation_steps`。**
- 每个 epoch 记录与 Step 5.1 相同的指标（含 learning_rate）。
  - **`epoch_time_seconds`：** 仅计训练时间，不含 validation 评估时间（与 PyTorch 版一致）。
  - **`learning_rate`：** 记录该 epoch **最后一个 step** 的学习率（与 PyTorch 版一致）。
- **Best checkpoint 选择依据：** 与 PyTorch 版一致，基于 **val_accuracy** 选出最佳 epoch。Checkpoint 保存至 `logs/checkpoints/keras_{dataset}_{model}_{precision}_seed{seed}.h5`，**保留不删除**。
- **训练结束后：** 加载基于 val_accuracy 选出的 best checkpoint，在 test 集上评估，同时记录 test_accuracy 和 test_macro_f1。
- 日志保存至 `logs/keras_{dataset}_{model}_{precision}_seed{seed}.csv`。

**验证：** 同 Step 5.1 的验证方法，但使用 Keras 脚本。

---

### Step 5.3 — 验证学习率调度一致性

分别运行 PyTorch 和 Keras 脚本 10 个 epoch，在每个 epoch 开始时打印当前学习率。

**验证：** 两个框架在每个 step 的学习率差异 < 1e-6。特别检查：
1. Step 0（warmup 阶段）：LR 从 0 开始。
2. Warmup 阶段（前 5 个 epoch 的所有 step）：LR 按 step 线性增长，warmup 结束时达到 0.1。
3. Warmup 后（cosine 阶段）：LR 按余弦衰减至 0（eta_min=0）。
4. 最后一个 epoch 的最后一个 step：LR 接近 0。

---

### Step 5.4 — 验证 AMP 模式正常工作

分别用两个框架运行 CIFAR-10 + ResNet-50 + AMP 模式 3 个 epoch。

**验证：**
1. 无 NaN 或 Inf 出现在 loss 或 accuracy 中。
2. 与 FP32 模式相比，每 epoch 时间有所减少（或至少不显著增加）。
3. GPU 显存使用量在 AMP 模式下不大于 FP32 模式。

---

## Phase 6: 结果汇总脚本

### Step 6.1 — 实现单次运行结果提取脚本

创建 `scripts/extract_run_summary.py`，读取单个 epoch 级 CSV 日志，输出一行汇总数据：

- framework, dataset, model, precision, seed
- acc_best（最高验证准确率）
- f1_best（最高验证 Macro-F1）
- test_accuracy（best checkpoint 在 test 集上的准确率）
- test_macro_f1（best checkpoint 在 test 集上的 Macro-F1）
- epoch_best（达到最佳准确率的 epoch 编号）
- time_per_epoch_avg（所有 epoch 的平均训练时间，不含 val）
- time_per_epoch_avg_late（**后 50% epoch** 的平均训练时间，例如 200 epoch 取 epoch 101-200；100 epoch 取 epoch 51-100。后半段更能反映稳态性能，排除初期 warmup、JIT 编译等干扰）
- images_per_sec_avg（计算方式：`num_batches × batch_size / epoch_time_seconds`，即 DataLoader 实际迭代的样本数，因 `drop_last=True` 可能略少于完整训练集大小）
- images_per_sec_avg_late（后 50% epoch，同上计算方式）

**验证：** 对 Step 5.1 生成的 3-epoch 日志运行，确认输出字段完整、数值合理。

---

### Step 6.2 — 实现全量结果汇总脚本

创建 `scripts/aggregate_results.py`，扫描 `logs/` 下所有 CSV 文件，对每个文件调用 Step 6.1 的逻辑，将结果追加到 `results/results.csv` 中。

**验证：** 用几个测试日志文件运行，确认 `results.csv` 行数 = 日志文件数，列名与实验文档一致。

---

## Phase 7: Stage 1 — 管线验证

### Step 7.1 — 创建 Stage 1 启动脚本

创建 `scripts/run_stage1.sh`，自动遍历 Stage 1 的所有组合：

- 框架：keras, pytorch
- 数据集：cifar10, cifar100
- 模型：resnet50, mobilenetv2
- 精度：fp32
- 种子：42, 123, 456

共 24 次运行。每次运行调用对应框架的训练脚本。

**验证：** 用 `bash -n scripts/run_stage1.sh` 检查语法无误。打印出将要执行的 24 条命令列表，确认覆盖所有组合。

---

### Step 7.2 — 执行 Stage 1 全部运行

运行 `scripts/run_stage1.sh`。

**验证：**
1. `logs/` 目录下生成 24 个 CSV 文件。
2. 每个文件有正确的 epoch 数（CIFAR 为 200 行）。
3. 无任何运行以 NaN loss 结束。
4. 所有 24 个运行的最终验证准确率在合理范围内（CIFAR-10 > 85%, CIFAR-100 > 55%）。

---

### Step 7.3 — 汇总并检查 Stage 1 结果

运行 `scripts/aggregate_results.py`，生成 `results/results_stage1.csv`。

**验证：**
1. 文件包含 24 行数据。
2. 相同配置下不同 seed 的结果不完全相同（证明 seed 机制有效）。
3. 同一 dataset-model-seed 下，Keras 和 PyTorch 的准确率差距 < 5%（若差距过大，需排查实现差异）。

---

## Phase 8: Stage 2 — 正式实验

### Step 8.1 — 创建 Stage 2 启动脚本

创建 `scripts/run_stage2.sh`，覆盖全部 180 次运行。增加以下功能：

- **复用 Stage 1 已完成的运行：** 自动跳过已完成的运行（检查对应日志文件是否存在且行数正确）。Stage 1 中 seed=[42,123,456] 与 Stage 2 重叠的运行不会重复执行。
- 将每次运行的 stdout/stderr 重定向至 `logs/console/` 目录。
- 运行结束后自动调用汇总脚本。

**验证：** 同 Step 7.1，确认 180 条命令列表覆盖所有组合。

---

### Step 8.2 — 执行 Stage 2 全部运行

运行 `scripts/run_stage2.sh`（预计耗时较长）。

**验证：**
1. `logs/` 目录下共有 180 个 CSV 文件（含 Stage 1 复用的运行，新增文件补齐至 180 总数）。
2. CIFAR 文件每个 200 行，Tiny-ImageNet 文件每个 100 行。
3. 无任何运行以 NaN loss 结束。

---

### Step 8.3 — 汇总 Stage 2 结果

运行汇总脚本，生成 `results/results_stage2.csv`。

**验证：** 文件包含 180 行，所有必需列均有有效值。

---

## Phase 9: 统计分析

### Step 9.1 — 实现统计检验脚本

创建 `analysis/statistical_tests.py`。对每个 (dataset, model, precision) 组合：

1. 收集该组合下 Keras 的 5 个 seed 结果和 PyTorch 的 5 个 seed 结果。
2. **按 seed 配对：** 同一 seed 下的 Keras 结果与 PyTorch 结果形成一对（共 5 对）。
3. 对 test_accuracy 执行：
   - 配对 Wilcoxon 符号秩检验 → p-value
   - 配对 t 检验 → p-value
   - Cohen's d 效应量
   - Cliff's delta 效应量
   - 差值 Δ = mean(PyTorch) − mean(Keras)
3. 对 test_macro_f1 重复以上检验。
4. 对 images_per_sec_avg 重复以上检验。

结果保存至 `analysis/statistical_results.csv`。

**验证：**
1. 输出文件行数 = (dataset, model, precision) 组合数 × 指标数 = 18 × 3 = 54 行。
2. 所有 p-value 在 [0, 1] 范围内。
3. 所有 Cohen's d 为有限数。
4. Cliff's delta 在 [-1, 1] 范围内。

---

### Step 9.2 — 生成显著性汇总表

创建 `analysis/significance_summary.py`，读取统计结果，生成易读的汇总表：

- 标注哪些组合在 α=0.05 下存在显著差异。
- 标注效应量大小（小 / 中 / 大）。

输出为 `analysis/significance_summary.csv` 和 `analysis/significance_summary.md`（Markdown 表格）。

**验证：** 打开 Markdown 文件，确认表格格式正确，内容与 CSV 一致。

---

## Phase 10: 可视化

### Step 10.1 — 绘制训练曲线

创建 `plots/plot_training_curves.py`，为每个 (dataset, model, precision) 组合绘制：

- 子图 1：训练 loss vs. epoch（两个框架各 5 条线 + 均值粗线）。
- 子图 2：验证 accuracy vs. epoch。
- 子图 3：验证 Macro-F1 vs. epoch。
- 子图 4：学习率 vs. epoch。

保存为 `plots/curves_{dataset}_{model}_{precision}.png`。

**验证：** 生成 18 张图片（3 datasets × 3 models × 2 precisions）。打开每张图确认：
1. 有清晰的图例区分 Keras 和 PyTorch。
2. 训练 loss 呈下降趋势。
3. 学习率曲线呈 warmup + cosine decay 形状。

---

### Step 10.2 — 绘制框架对比柱状图

创建 `plots/plot_comparison_bars.py`，绘制：

- 准确率对比柱状图（按 dataset-model 分组，每组两个柱状，含误差棒）。
- 训练速度对比柱状图（images/sec）。

分别为 FP32 和 AMP 各生成一张图。

**验证：** 生成 4 张图（2 指标 × 2 精度模式）。确认误差棒存在且合理。

---

## Phase 11: 最终交付物整理

### Step 11.1 — 整理配置文件

确认 `configs/` 目录包含：
- `train_config.yaml`
- `experiment_matrix.yaml`
- `dataset_stats.yaml`

**验证：** 列出文件并用 YAML 解析器验证每个文件格式正确。

---

### Step 11.2 — 整理原始日志

确认 `logs/` 目录包含所有 180 个 CSV 日志文件，命名格式统一。

**验证：** 统计文件数 = 180，所有文件非空。

---

### Step 11.3 — 整理汇总结果表

确认 `results/results_stage2.csv` 完整且字段与实验文档一致。

**验证：** 加载 CSV，检查 180 行 × 12+ 列（含 test_accuracy 和 test_macro_f1），无缺失值。

---

### Step 11.4 — 整理统计分析输出

确认 `analysis/` 目录包含统计检验结果和显著性汇总。

**验证：** 文件存在且内容非空。

---

### Step 11.5 — 整理训练曲线和对比图

确认 `plots/` 目录包含所有生成的图片。

**验证：** 共计 22 张图片（18 训练曲线 + 4 对比柱状图）。

---

### Step 11.6 — 编写复现说明

创建 `README.md`，包含以下章节：

1. 项目简介
2. 环境安装步骤（指向 `requirements_lock.txt`）
3. 数据准备步骤
4. 如何运行单次训练
5. 如何运行完整实验
6. 如何生成统计分析和图表
7. 目录结构说明

**验证：** 按照 README 的步骤，在新环境中至少能成功运行一次单独的训练（3 个 epoch 的短测试）。

---

## 检查清单（总览）

| Phase | 步骤数 | 关键输出 |
|-------|--------|---------|
| 0 - 环境准备 | 4 | 目录结构、`environment.txt`、依赖锁定 |
| 1 - 数据准备 | 7 | 3 个数据集 + 索引文件 + `dataset_stats.yaml` |
| 2 - 配置文件 | 2 | `train_config.yaml`、`experiment_matrix.yaml` |
| 3 - 模型构建 | 7 | 6 个模型实现 + 参数对齐表 |
| 4 - 数据管线 | 2 | PyTorch 和 Keras 数据加载器 |
| 5 - 训练循环 | 4 | 两套训练脚本 + LR/AMP 验证 |
| 6 - 结果汇总 | 2 | 汇总脚本 |
| 7 - Stage 1 | 3 | 24 次运行 + 管线验证 |
| 8 - Stage 2 | 3 | 180 次运行 + 完整结果 |
| 9 - 统计分析 | 2 | 统计检验 + 显著性汇总 |
| 10 - 可视化 | 2 | 训练曲线 + 对比图 |
| 11 - 交付整理 | 6 | 完整交付物 + README |
| **合计** | **44** | |
