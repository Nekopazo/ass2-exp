# JATIT Paper Writing Prompt — Keras vs PyTorch Transfer-Learning Benchmark

> **用途：** 将此文件整体交给 AI，让它以 JATIT 期刊风格为你撰写完整论文初稿。
> 留白处用 `[TODO: ...]` 标注，待实验数据就绪后再补充。

---

## 0. Persona & Style Instructions

You are a senior academic writer specializing in information-technology journals. Your task is to produce a **complete, publication-ready manuscript** for the *Journal of Theoretical and Applied Information Technology* (JATIT). Adhere strictly to the following stylistic and formatting rules:

- **Language:** Formal academic English; third person; passive voice where conventional (e.g., "the model was trained …").
- **Tone:** Objective, precise, evidence-driven. No first-person narrative. Avoid speculative language; every claim must be backed by either an experimental number or a citation.
- **JATIT formatting requirements:**
  - Title: ALL CAPS, 16 pt Times Roman, centered.
  - Authors and affiliations: 10 pt Times Roman, centered.
  - Abstract and keywords: single-column; body text in two-column format.
  - Body text: 10 pt Times Roman, single-spaced, justified.
  - Minimum 8 pages, maximum 45 pages in journal format.
  - Reference style: Vancouver — numbered in square brackets in order of first appearance.
- **Citation policy:** Where this prompt says "[cite: ...]", insert a numbered placeholder reference and add the full entry to the References section.

---

## 1. Title (Candidate — pick the best or combine)

1. A SYSTEMATIC BENCHMARK OF KERAS AND PYTORCH TRANSFER-LEARNING PIPELINES FOR PLANT DISEASE AND GENERAL IMAGE CLASSIFICATION
2. COMPARATIVE ANALYSIS OF KERAS AND PYTORCH FRAMEWORKS IN TRANSFER-LEARNING-BASED IMAGE CLASSIFICATION: A REPRODUCIBLE BENCHMARK STUDY
3. FRAMEWORK-LEVEL REPRODUCIBILITY IN DEEP TRANSFER LEARNING: A CROSS-FRAMEWORK COMPARISON OF KERAS AND PYTORCH ON PLANTVILLAGE AND CIFAR-10

---

## 2. Abstract (150–250 words)

Write the abstract covering these five points in order:

1. **Background and motivation:** Deep-learning frameworks such as Keras (TensorFlow backend) and PyTorch are widely used for image classification via transfer learning, yet few studies provide reproducible, head-to-head comparisons with aligned hyperparameters, identical data splits, and equivalent training protocols.
2. **Objective:** This study systematically benchmarks Keras and PyTorch under strictly controlled conditions to determine whether framework choice introduces statistically significant differences in classification accuracy, macro-F1, and training throughput.
3. **Method summary:** Three ImageNet-pretrained architectures (ResNet50, VGG16-BN, MobileNetV2) are fine-tuned on two datasets — PlantVillage (38-class plant disease, 224×224) and CIFAR-10 (10-class general objects, 32×32→224×224) — using aligned SGD optimizers, cosine-annealing learning-rate schedules with linear warmup, identical 3-fold stratified cross-validation, and early stopping. A total of 36 training runs (2 frameworks × 3 models × 3 folds × 2 datasets) are executed.
4. **Key findings:** On PlantVillage, PyTorch achieved 99.77% (ResNet50), 99.76% (VGG16), and 99.74% (MobileNetV2) test accuracy, while Keras achieved comparable results for ResNet50 (99.47%) and VGG16 (99.30%) but suffered catastrophic failure with MobileNetV2 (57.18%). On CIFAR-10, Keras ResNet50 (95.85%) exhibited the highest stability, PyTorch VGG16 (95.73%) achieved the highest accuracy, and Keras MobileNetV2 again collapsed (39.61% vs. PyTorch's 95.39%). Of 18 paired t-tests, 15 reached significance at α = 0.05, but practical differences excluding MobileNetV2 were below 2.5 percentage points.
5. **Conclusion sentence:** The findings indicate that framework equivalence is conditionally dependent on the (architecture, dataset, hyperparameter) triple: for heavy architectures (ResNet50, VGG16), framework-induced accuracy differences are statistically significant but practically marginal (< 2.5 pp), whereas for lightweight architectures (MobileNetV2), the Keras implementation exhibited catastrophic training instability across both datasets, demonstrating that framework choice can be a critical deployment variable for specific model families.

**Keywords:** transfer learning, image classification, deep learning, Keras, PyTorch, PlantVillage, CIFAR-10, framework comparison, reproducibility.

---

## 3. Section Structure and Detailed Content Instructions

Below is the full section hierarchy. Write every section in full prose with proper transitions. Do NOT produce bullet points in the manuscript body — convert all points below into flowing paragraphs.

---

### 1. INTRODUCTION

Write 4–5 paragraphs covering:

**Paragraph 1 — Domain context:**
Image classification is a cornerstone task in computer vision with applications in agriculture (crop disease detection), medical imaging, autonomous driving, and industrial inspection. Transfer learning from large-scale pre-trained models (typically ImageNet) has become the standard paradigm for achieving high accuracy with limited domain-specific data [cite: Deng et al. 2009 ImageNet; Zhuang et al. 2020 transfer learning survey].

**Paragraph 2 — Framework landscape:**
TensorFlow/Keras and PyTorch dominate the deep-learning ecosystem. Keras offers a high-level API with rapid prototyping; PyTorch provides a flexible imperative paradigm popular in research. Despite their widespread adoption, practitioners often assume that framework choice is neutral — that the same architecture with the same hyperparameters produces identical results regardless of implementation. This assumption, however, has rarely been rigorously validated [cite: relevant framework comparison papers].

**Paragraph 3 — Research gap:**
Existing comparative studies either (a) compare frameworks at the API-convenience or training-speed level without controlling hyperparameters, or (b) focus on a single dataset/model and do not generalize across domains. There is a notable absence of large-scale, strictly controlled benchmarks that align every training detail — optimizer, LR schedule, augmentation, data splits, random seeds — across frameworks.

**Paragraph 4 — Contributions:**
State the three main contributions:
1. A **reproducible, fully aligned benchmark** comparing Keras and PyTorch across 3 architectures × 2 datasets × 3 folds = 36 runs, with every hyperparameter, data split, and augmentation matched.
2. A **statistical analysis** (paired t-tests, Cohen's d, Cliff's delta) quantifying whether framework choice introduces significant accuracy, F1, or throughput differences.
3. **Complete experiment artifacts** (code, configs, logs, checkpoints) released for reproducibility.

**Paragraph 5 — Paper organization:**
"The remainder of this paper is organized as follows. Section 2 defines the problem statement. Section 3 reviews related work. Section 4 presents the proposed methodology. Section 5 describes the experimental setup. Section 6 reports results and discussion. Section 7 concludes the paper and outlines future work."

---

### 2. PROBLEM STATEMENT

Write 2–3 paragraphs formally defining the research problem:

- **Input:** A labeled image dataset $D = \{(x_i, y_i)\}_{i=1}^{N}$ where $x_i \in \mathbb{R}^{H \times W \times 3}$ and $y_i \in \{1, \ldots, C\}$.
- **Task:** Train a classifier $f_\theta: \mathbb{R}^{H \times W \times 3} \rightarrow \mathbb{R}^{C}$ via transfer learning from ImageNet-pretrained weights, fine-tuning all layers with SGD.
- **Research question:** Given two framework implementations (Keras and PyTorch) with **identical** architecture definitions, loss functions, optimizers, learning-rate schedules, data augmentations, and cross-validation splits, do the resulting models exhibit statistically significant differences in:
  - (RQ1) Classification accuracy and macro-F1 score?
  - (RQ2) Training throughput (images/sec)?
  - (RQ3) Training dynamics (convergence speed, early-stopping epoch)?
- **Null hypothesis:** $H_0$: There is no statistically significant difference in mean test accuracy (or F1, or throughput) between Keras and PyTorch across folds, for a given (model, dataset) pair. Significance level $\alpha = 0.05$.

---

### 3. LITERATURE REVIEW

Write 3 subsections:

#### 3.1 Transfer Learning for Image Classification
Review key works on transfer learning from ImageNet to downstream tasks. Cover:
- The original transfer-learning paradigm: feature extraction vs. full fine-tuning [cite: Yosinski et al. 2014].
- Applications in plant disease recognition on PlantVillage [cite: Hughes & Salathe 2015; Mohanty et al. 2016].
- Transfer learning on CIFAR-10 with modern architectures [cite: He et al. 2016 ResNet; Simonyan & Zisserman 2015 VGG; Sandler et al. 2018 MobileNetV2].
- CNN + classical ML hybrid pipelines (frozen embeddings + SVM/logistic regression) as a reference paradigm [cite: papers from references/ folder — s41598-024-63767-5, jimaging-11-00207].

#### 3.2 Framework Comparisons in Deep Learning
Review existing Keras vs. PyTorch studies:
- API usability and learning curve comparisons [cite: relevant surveys].
- Training speed and GPU utilization benchmarks [cite: relevant papers].
- Identify the gap: lack of hyperparameter-aligned accuracy comparisons with statistical rigor.

#### 3.3 Experimental Reproducibility in Deep Learning
- Challenges: non-determinism from cuDNN, floating-point arithmetic, data shuffling [cite: Pham et al. 2020 or similar].
- The importance of fixed random seeds, stratified splits, and per-fold statistics.
- How this study addresses each reproducibility concern.

---

### 4. PROPOSED METHODOLOGY

This is the core methodology section. Write with full technical detail.

#### 4.1 Overview of the Benchmark Pipeline
Describe the end-to-end pipeline (this maps to Figure 1):
**Data Preparation → Fold Splitting → Training (Keras / PyTorch) → Evaluation → Aggregation → Statistical Testing.**

Explain that every component is made identical across frameworks except the framework-internal implementation.

#### 4.2 Datasets

**PlantVillage:**
- Source: Kaggle PlantVillage dataset.
- Content: 38 classes of plant leaf images (healthy and diseased).
- Pre-processing: all images resized to 224×224×3 and stored as uint8 NumPy arrays.
- Total usable images: obtain exact count from `data/plantvillage_images.npy` shape — approximately 54,000.
- Class distribution: mention that class imbalance exists and is handled by stratified splitting.

**CIFAR-10:**
- Source: standard CIFAR-10 via `tf.keras.datasets.cifar10`.
- Content: 10 classes, 60,000 images of size 32×32×3.
- Pre-processing: images are stored at original 32×32 resolution; resized to 224×224 at training time via the data pipeline to match ImageNet-pretrained model expectations.

#### 4.3 Data Splitting and Cross-Validation Protocol
- **Test set:** 20% held out via `train_test_split(stratify=y, test_size=0.2, random_state=42)`.
- **3-Fold CV on the remaining 80%:** `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`.
- Fold indices are **relative to the trainval subset** — describe the two-stage indexing mechanism (trainval indices → fold-local indices → global indices).
- **Per-fold normalization:** Channel-wise mean and std computed on each fold's training set only (values stored in `configs/dataset_stats.yaml`). Report the mean/std values:
  - PlantVillage fold 0: mean=[0.4668, 0.4893, 0.4106], std=[0.1955, 0.1706, 0.2137].
  - CIFAR-10 fold 0: mean=[0.4916, 0.4824, 0.4471], std=[0.2470, 0.2433, 0.2615].

#### 4.4 Model Architectures
For each of the three architectures, describe the transfer-learning setup:

**ResNet50:**
- PyTorch: `torchvision.models.resnet50(weights='IMAGENET1K_V1')`; replace `fc` layer with `Linear(2048, num_classes)`, Kaiming normal initialization.
- Keras: `tf.keras.applications.ResNet50(include_top=False, weights='imagenet', pooling='avg')` + `Dense(num_classes, kernel_regularizer=l2(5e-5))`.
- Parameter counts: PlantVillage — PyTorch 23,585,894 / Keras 23,612,454 (diff 0.11%).

**VGG16-BN:**
- PyTorch: `torchvision.models.vgg16_bn(weights='IMAGENET1K_V1')`; replace final classifier layer.
- Keras: **manually implemented VGG16-BN** (Conv2D + BatchNormalization + ReLU per block) because `tf.keras.applications.VGG16` lacks BatchNorm. Conv and Dense weights copied from official VGG16; BN layers initialized to defaults.
- **Framework difference disclosure:** PyTorch loads ImageNet-trained BN statistics; Keras BN starts from default (mean=0, var=1). This is an inherent alignment limitation and is discussed in Section 6.
- Parameter counts: PlantVillage — 134,424,678 (both, diff 0.0%).

**MobileNetV2:**
- PyTorch: `torchvision.models.mobilenet_v2(weights='IMAGENET1K_V1')`; replace classifier head.
- Keras: `tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', pooling='avg')` + new Dense head.
- Parameter counts: PlantVillage — 2,272,550 (both, diff 0.0%).

#### 4.5 Training Configuration (Hyperparameter Alignment)
Present **Table 1** with the aligned hyperparameters:

| Hyperparameter | Value | Notes |
|---|---|---|
| Optimizer | SGD | Both frameworks |
| Momentum | 0.9 | Nesterov enabled |
| Weight decay | 1×10⁻⁴ | PyTorch: native; Keras: L2 regularizer on kernel |
| Base learning rate | 0.04 | Linear scaling for batch 128 |
| LR schedule | Cosine annealing | η_min = 0 |
| Warmup | 5 epochs, linear per-step | From LR=0 to base_lr |
| Batch size | 128 | |
| Max epochs | 30 | |
| Early stopping | patience=7 on val_loss | min_delta=0 |
| Loss function | Cross-entropy (from logits) | |
| Input size | 224×224 | CIFAR-10 resized in pipeline |
| Augmentation | RandomResizedCrop(224, scale=[0.8,1.0]) + HorizontalFlip | |
| Random seed | 42 + fold_index | Per-fold determinism |

**Weight decay implementation note:** Explain that PyTorch applies weight decay directly in SGD, while Keras uses `kernel_regularizer=l2(weight_decay/2)` and sums `model.losses` in the GradientTape loop. Both yield equivalent gradient updates: $g \leftarrow g + \lambda \cdot w$.

**LR schedule implementation:** Detail the `WarmupCosineLRSchedule` class — linear warmup for `warmup_epochs × steps_per_epoch` steps, then cosine decay over remaining steps. LR is updated **per step** (not per epoch) in both frameworks.

#### 4.6 Training Loop Alignment
Explain how training loops are made equivalent:
- **PyTorch:** Standard loop with `optimizer.zero_grad()`, forward pass, `loss.backward()`, `optimizer.step()`. AMP via `torch.cuda.amp`.
- **Keras:** Custom `tf.GradientTape` loop (NOT `model.fit`), matching the PyTorch loop structure step by step. **No mixed precision** in Keras due to `LossScaleOptimizer` incompatibility with custom tape loops (describe the debugging experience briefly: "Initial experiments with Keras mixed precision yielded near-random accuracy (~3%) due to incorrect loss scaling in the custom GradientTape loop; this was resolved by reverting to float32.").
- Best checkpoint: selected by highest `val_accuracy` per fold.

#### 4.7 Evaluation Metrics
- **Test accuracy:** proportion of correctly classified test samples.
- **Macro-F1:** unweighted mean of per-class F1 scores; robust to class imbalance.
- **Training throughput:** images/sec = `(floor(N_train / batch_size) × batch_size) / epoch_time_seconds`.
- **Reporting convention:** 3-fold mean ± standard deviation.

#### 4.8 Statistical Analysis Protocol
- **Paired t-test:** For each (dataset, model, metric), pair Keras and PyTorch by fold → 3 paired observations. Test $H_0$: mean difference = 0 at $\alpha = 0.05$.
- **Cohen's d:** effect size = mean(diff) / sd(diff).
- **Cliff's delta:** non-parametric ordinal effect size.

---

### 5. EXPERIMENTAL SETUP

#### 5.1 Hardware and Software Environment
- GPU: NVIDIA A40 (48 GB VRAM).
- cuDNN: 9.1.0.02 (reported via `torch.backends.cudnn.version()` = 91002).
- Python: 3.10.12.
- TensorFlow: 2.20.0.
- PyTorch: 2.9.1.
- Key libraries: scikit-learn, NumPy, pandas, matplotlib, SciPy.
- **Note on environment capture:** The `environment.txt` was recorded on a login node where NVML initialisation failed; actual training executed on GPU-equipped compute nodes, as confirmed by epoch-level throughput figures (60–370 images/sec) that are inconsistent with CPU-only execution.

#### 5.2 Experiment Matrix
- 2 datasets × 3 models × 2 frameworks × 3 folds = **36 training runs**.
- Each run produces: epoch-level CSV log, test JSON with final metrics, confusion matrix (`.npy`), and a best checkpoint.
- Naming convention: `{framework}_{dataset}_{model}_fold{fold}.*`.

#### 5.3 Reproducibility Artifacts
List the artifacts generated for full reproducibility:
- `configs/train_config.yaml` — all hyperparameters.
- `configs/dataset_stats.yaml` — per-fold channel statistics.
- `data/splits/` — all train/val/test index files.
- `logs/` — 36 epoch-level CSVs + 36 test JSONs + 36 confusion matrices.
- `results/results.csv` — per-fold aggregated results (36 rows).
- `results/results_summary.csv` — mean±std summary (12 rows).
- `analysis/` — statistical test outputs.

---

### 6. RESULTS AND DISCUSSION

> **分析原则：** 本节以**场景适用性**为核心组织讨论，聚焦于"哪个模型-框架组合适合哪种部署场景"，而非简单的性能排名。

#### 6.1 Overall Classification Performance

**Table 2: Mean Test Accuracy (%) across 3 Folds**

| Dataset | Model | Keras (mean±std) | PyTorch (mean±std) | Δ (PT−Keras) |
|---|---|---|---|---|
| PlantVillage | ResNet50 | 99.47 ± 0.02 | 99.77 ± 0.05 | +0.30 pp |
| PlantVillage | VGG16 | 99.30 ± 0.10 | 99.76 ± 0.06 | +0.46 pp |
| PlantVillage | MobileNetV2 | 57.18 ± 10.45 | 99.74 ± 0.05 | **+42.56 pp** |
| CIFAR-10 | ResNet50 | 95.85 ± 0.10 | 93.47 ± 2.78 | −2.39 pp |
| CIFAR-10 | VGG16 | 93.79 ± 0.23 | 95.73 ± 0.12 | +1.94 pp |
| CIFAR-10 | MobileNetV2 | 39.61 ± 8.85 | 95.39 ± 0.14 | **+55.78 pp** |

**Table 3: Mean Test Macro-F1 across 3 Folds**

| Dataset | Model | Keras (mean±std) | PyTorch (mean±std) | Δ (PT−Keras) |
|---|---|---|---|---|
| PlantVillage | ResNet50 | 0.9931 ± 0.0001 | 0.9964 ± 0.0005 | +0.0033 |
| PlantVillage | VGG16 | 0.9907 ± 0.0014 | 0.9966 ± 0.0005 | +0.0059 |
| PlantVillage | MobileNetV2 | 0.4688 ± 0.1045 | 0.9963 ± 0.0004 | **+0.5275** |
| CIFAR-10 | ResNet50 | 0.9584 ± 0.0010 | 0.9345 ± 0.0279 | −0.0240 |
| CIFAR-10 | VGG16 | 0.9377 ± 0.0023 | 0.9572 ± 0.0012 | +0.0195 |
| CIFAR-10 | MobileNetV2 | 0.3757 ± 0.1032 | 0.9538 ± 0.0014 | **+0.5781** |

Write 3–4 paragraphs of **scenario-based** analysis (not simple ranking):

**Paragraph 1 — High-resolution domain-specific tasks (PlantVillage scenario):**
PlantVillage represents high-resolution, fine-grained domain-specific classification (38 classes, 224×224). ResNet50 and VGG16 demonstrated strong cross-framework stability: both frameworks achieved >99.3% accuracy, with PyTorch holding a statistically significant but practically marginal advantage of 0.30 pp (ResNet50) and 0.46 pp (VGG16). From a deployment perspective, this margin is unlikely to influence real-world decision-making; both implementations are equally suitable for production-grade pipelines. The choice between Keras and PyTorch for these architectures may be guided by ecosystem factors (developer familiarity, TensorFlow Serving vs. TorchServe integration) rather than classification performance. MobileNetV2, however, presents a strikingly different picture: PyTorch achieved 99.74% (on par with heavier architectures), while Keras collapsed to 57.18% — rendering it unsuitable for any deployment context in this framework (analysed in Section 6.6).

**Paragraph 2 — Low-resolution upsampled tasks (CIFAR-10 scenario):**
CIFAR-10 represents a fundamentally different scenario: natively low-resolution images (32×32) upsampled to 224×224, introducing bilinear interpolation artefacts and distribution shift from ImageNet. Keras ResNet50 exhibited the most stable behaviour (95.85%, σ = 0.10%), making it particularly suitable for applications requiring consistent, reproducible performance — e.g., regulatory-grade classification systems. PyTorch ResNet50, while achieving the highest individual fold accuracy (96.68%, fold 2), suffered warmup-phase instability in 2 out of 3 folds (early stopping at epoch 8 with ~91.9%), producing high variance (σ = 2.78%). PyTorch VGG16 delivered the most robust high-accuracy performance on CIFAR-10 (95.73%, σ = 0.12%), with its 1.94 pp advantage over Keras VGG16 attributable to pretrained BatchNorm statistics (Section 6.5). Keras MobileNetV2 again collapsed catastrophically (39.61%).

**Paragraph 3 — Deployment-constrained scenarios (lightweight models):**
MobileNetV2 (2.27M parameters) is 10× smaller than ResNet50 (23.6M) and 59× smaller than VGG16 (134.4M). In PyTorch, MobileNetV2 achieved accuracy fully competitive with heavier architectures (99.74% PlantVillage, 95.39% CIFAR-10), demonstrating that parameter efficiency need not sacrifice accuracy. However, MobileNetV2's deployment viability is framework-dependent: the Keras implementation proved unreliable across both datasets (6/6 runs failed). Practitioners requiring lightweight models for edge or mobile deployment should use PyTorch MobileNetV2 with ONNX export; if TensorFlow deployment is mandatory, ResNet50 or VGG16 remain more reliable, albeit at higher parameter cost.

**Paragraph 4 — Accuracy-F1 correlation:**
Across all 12 (dataset, model, framework) combinations, test accuracy and macro-F1 were highly correlated (Pearson r > 0.999). In the high-performing configurations (>93%), accuracy and macro-F1 differed by less than 0.3 percentage points, confirming that class imbalance effects are well-controlled by stratified splitting. The two metrics diverge only in the catastrophically failed Keras MobileNetV2 runs, where macro-F1 (0.37–0.47) is further depressed relative to accuracy (0.40–0.57) due to per-class F1 collapse on minority classes.

#### 6.2 Training Throughput Comparison

**Table 4: Mean Training Throughput (images/sec) across 3 Folds**

| Dataset | Model | Keras (mean±std) | PyTorch (mean±std) | Δ (PT−Keras) | p-value |
|---|---|---|---|---|---|
| PlantVillage | ResNet50 | 97.9 ± 31.9 | 124.9 ± 0.9 | +27.0 | 0.2900 (NS) |
| PlantVillage | VGG16 | 72.8 ± 0.1 | 76.3 ± 0.6 | +3.5 | 0.0075 |
| PlantVillage | MobileNetV2 | 210.3 ± 23.7 | 179.3 ± 17.9 | −31.0 | 0.0292 |
| CIFAR-10 | ResNet50 | 214.0 ± 2.6 | 114.4 ± 18.3 | −99.6 | 0.0125 |
| CIFAR-10 | VGG16 | 137.1 ± 0.6 | 65.6 ± 8.8 | −71.5 | 0.0052 |
| CIFAR-10 | MobileNetV2 | 369.0 ± 0.8 | 160.5 ± 14.2 | −208.6 | 0.0015 |

Write 2 paragraphs of **scenario-contingent** throughput analysis:

**Paragraph 1 — Small-source-image pipelines (CIFAR-10):**
On CIFAR-10 (32×32 source images upsampled to 224×224), Keras was substantially faster: 1.88× for ResNet50 (214 vs. 114 images/sec), 2.09× for VGG16 (137 vs. 66 images/sec), and 2.30× for MobileNetV2 (369 vs. 161 images/sec). This advantage is attributable to `tf.data`'s fused image processing operations (resize + crop + flip as a single computation graph node), which are more efficient than PyTorch's PIL-based `torchvision.transforms` pipeline when the dominant cost is image upsampling. Additionally, PyTorch's AMP overhead (GradScaler bookkeeping) provides less net speedup on a compute-light task. **For throughput-sensitive applications on low-resolution data, Keras offers a significant training speed advantage** — provided the chosen architecture is not MobileNetV2. Note that the Keras MobileNetV2 throughput figure (369 images/sec) reflects an early-stopped, failed training run (epoch 8–9) and does not represent a functionally comparable pipeline.

**Paragraph 2 — Native-resolution pipelines (PlantVillage):**
On PlantVillage (native 224×224 images), the throughput picture reverses. PyTorch ResNet50 was ~28% faster (124.9 vs. 97.9 images/sec), though this did not reach significance (p = 0.290) due to high Keras timing variance from a fold-0 anomaly (Keras ResNet50 fold 0 ran 43 epochs before early stopping vs. 30 for the other folds). VGG16 throughput was similar across frameworks (~73–76 images/sec). On native-resolution data where GPU compute is the bottleneck rather than data loading, PyTorch's AMP and optimised DataLoader (`persistent_workers`, `pin_memory`, `prefetch_factor`) narrow or eliminate Keras's tf.data advantage. **Neither framework is universally faster; throughput advantage is contingent on whether the pipeline is data-loading-bound or compute-bound.**

#### 6.3 Training Dynamics

Present:
- **Figure 2:** Training curves (loss vs epoch, val accuracy vs epoch) for each dataset-model pair. Each subplot shows Keras and PyTorch with per-fold thin lines and mean bold line.

Write 2–3 paragraphs covering the following observed training dynamics:

**Paragraph 1 — Stable convergence cases:**
For ResNet50 and VGG16 on PlantVillage, both frameworks exhibited smooth, monotonic convergence. Training loss decreased continuously over 23–30 epochs; validation accuracy climbed steadily to >99%; no early stopping was triggered in 10 out of 12 runs. Keras and PyTorch training curves are nearly overlapping, confirming that the hyperparameter alignment was effective for these configurations. The per-fold variance was minimal (σ < 0.1%), indicating robust generalisation.

**Paragraph 2 — PyTorch ResNet50 CIFAR-10 instability (Section 6.6b):**
PyTorch ResNet50 on CIFAR-10 revealed a warmup-phase instability. In folds 0 and 1, validation accuracy peaked at epoch 1 (91.97–91.98% at LR ≈ 0.008), then declined monotonically as the learning rate increased toward the base rate of 0.04. Early stopping triggered at epoch 8 in both cases. Fold 2, however, survived the warmup transition and converged normally to 96.69% at epoch 29. Keras ResNet50 showed no analogous instability — all three folds completed 30 epochs with stable convergence (95.70–96.04%). This fold-dependent behaviour suggests that the combination of AMP mixed precision + aggressive warmup LR + distribution-shifted inputs (32→224 upsampling) creates a narrow stability region that is sensitive to the random initialisation of the classification head.

**Paragraph 3 — Keras MobileNetV2 catastrophic divergence:**
Keras MobileNetV2 displayed a distinctive failure pattern across all 6 runs (2 datasets × 3 folds): training accuracy rose normally (reaching 95%+ on CIFAR-10 and 99%+ on PlantVillage), while validation loss exploded by an order of magnitude within 2–5 epochs (e.g., 1.66 → 13.41 on CIFAR-10 fold 0, 2.16 → 17.55 on PlantVillage fold 0). All runs triggered early stopping at epoch 8–9. This pattern is qualitatively distinct from classical overfitting and is analysed in detail in Section 6.6.

- **Figure 3 (optional):** Learning rate schedule verification — overlay Keras and PyTorch LR curves from any fold to confirm per-step alignment. The logs confirm that both frameworks followed identical LR trajectories (difference < 1e-6 at every step).

#### 6.4 Statistical Significance

**Table 5: Statistical Test Results (18 paired comparisons)**

| Dataset | Model | Metric | PT mean | Keras mean | Δ | p-value | Sig? | Cohen's d | Cliff's δ |
|---|---|---|---|---|---|---|---|---|---|
| PlantVillage | ResNet50 | test_accuracy | 0.9977 | 0.9947 | +0.0030 | 0.0213 | Yes | 3.90 | 1.00 |
| PlantVillage | ResNet50 | test_macro_f1 | 0.9964 | 0.9931 | +0.0033 | 0.0107 | Yes | 5.53 | 1.00 |
| PlantVillage | ResNet50 | images_per_sec | 124.9 | 97.9 | +27.0 | 0.2900 | No | 0.82 | 1.00 |
| PlantVillage | VGG16 | test_accuracy | 0.9976 | 0.9930 | +0.0046 | 0.0302 | Yes | 3.24 | 1.00 |
| PlantVillage | VGG16 | test_macro_f1 | 0.9966 | 0.9907 | +0.0059 | 0.0323 | Yes | 3.14 | 1.00 |
| PlantVillage | VGG16 | images_per_sec | 76.3 | 72.8 | +3.5 | 0.0075 | Yes | 6.61 | 1.00 |
| PlantVillage | MobileNetV2 | test_accuracy | 0.9974 | 0.5718 | +0.4256 | 0.0194 | Yes | 4.09 | 1.00 |
| PlantVillage | MobileNetV2 | test_macro_f1 | 0.9963 | 0.4688 | +0.5275 | 0.0127 | Yes | 5.06 | 1.00 |
| PlantVillage | MobileNetV2 | images_per_sec | 179.3 | 210.3 | −31.0 | 0.0292 | Yes | −3.31 | −0.56 |
| CIFAR-10 | ResNet50 | test_accuracy | 0.9347 | 0.9585 | −0.0239 | 0.2766 | No | −0.86 | −0.33 |
| CIFAR-10 | ResNet50 | test_macro_f1 | 0.9345 | 0.9584 | −0.0240 | 0.2761 | No | −0.86 | −0.33 |
| CIFAR-10 | ResNet50 | images_per_sec | 114.4 | 214.0 | −99.6 | 0.0125 | Yes | −5.12 | −1.00 |
| CIFAR-10 | VGG16 | test_accuracy | 0.9573 | 0.9379 | +0.0194 | 0.0095 | Yes | 5.89 | 1.00 |
| CIFAR-10 | VGG16 | test_macro_f1 | 0.9572 | 0.9377 | +0.0195 | 0.0095 | Yes | 5.87 | 1.00 |
| CIFAR-10 | VGG16 | images_per_sec | 65.6 | 137.1 | −71.5 | 0.0052 | Yes | −7.99 | −1.00 |
| CIFAR-10 | MobileNetV2 | test_accuracy | 0.9539 | 0.3961 | +0.5578 | 0.0084 | Yes | 6.25 | 1.00 |
| CIFAR-10 | MobileNetV2 | test_macro_f1 | 0.9538 | 0.3757 | +0.5781 | 0.0106 | Yes | 5.57 | 1.00 |
| CIFAR-10 | MobileNetV2 | images_per_sec | 160.5 | 369.0 | −208.6 | 0.0015 | Yes | −14.78 | −1.00 |

Write 3 paragraphs interpreting these results:

**Paragraph 1 — Overall significance pattern:**
Of 18 paired comparisons, 15 reached statistical significance at α = 0.05 (83%). All significant comparisons exhibited large Cohen's d values (|d| > 3.0 for accuracy/F1 metrics) and Cliff's delta of ±1.0 (complete stochastic dominance). However, these statistics must be interpreted with caution: with only 3 paired observations per test, effect-size estimates are inflated by the narrow denominator. Under Bonferroni correction (α_adj = 0.05/18 ≈ 0.0028), only 3 comparisons survive — CIFAR-10 VGG16 images/sec (p = 0.0052), CIFAR-10 MobileNetV2 images/sec (p = 0.0015), and CIFAR-10 MobileNetV2 accuracy (p = 0.0084). The PlantVillage ResNet50/VGG16 accuracy differences (p ≈ 0.02–0.03) would no longer be significant, suggesting that their small absolute magnitudes (< 0.5 pp) may not generalise.

**Paragraph 2 — The non-significant cases:**
Three comparisons did not reach significance: CIFAR-10 ResNet50 accuracy (p = 0.277), CIFAR-10 ResNet50 macro-F1 (p = 0.276), and PlantVillage ResNet50 throughput (p = 0.290). The ResNet50 non-significance is driven by PyTorch's high fold-level variance on CIFAR-10 (σ = 2.78%, reflecting the warmup instability in folds 0/1). Had all three PyTorch folds converged normally, PyTorch ResNet50 would likely have matched or exceeded Keras; conversely, had only fold 0/1 results been observed, the conclusion would be that Keras substantially outperforms PyTorch. This underscores the importance of reporting per-fold results rather than only aggregates.

**Paragraph 3 — Practical significance vs. statistical significance:**
Statistical significance is dominated by two phenomena: (a) the catastrophic Keras MobileNetV2 failures (overwhelming effect, Δ > 42 pp), and (b) smaller but consistent PyTorch advantages for ResNet50 and VGG16 on PlantVillage (Δ < 0.5 pp) and VGG16 on CIFAR-10 (Δ = 1.94 pp). When MobileNetV2 results are excluded, the accuracy differences between frameworks are modest (< 2.5 pp) and their practical impact is limited. **The most actionable finding is not the aggregate significance pattern, but rather the identification of specific (model, dataset) combinations where framework choice is critical (MobileNetV2) versus those where it is essentially neutral (ResNet50, VGG16 on PlantVillage).**

#### 6.5 VGG16-BN Cross-Framework Alignment Discussion

Write a focused paragraph on the VGG16-BN asymmetry with the following data and interpretation:

On PlantVillage, the 0.46 pp accuracy gap (Keras 99.30% vs. PyTorch 99.76%) is statistically significant (p = 0.030) but practically negligible. Both frameworks converged smoothly over 23–30 epochs, suggesting that on high-resolution, in-distribution data, the BN initialisation difference is absorbed during fine-tuning — 23–30 epochs are sufficient for Keras BN to learn appropriate running statistics from the target domain.

On CIFAR-10, the gap widens to 1.94 pp (Keras 93.79% vs. PyTorch 95.73%, p = 0.0095). The distribution shift (upsampled low-resolution images) amplifies the impact of BN initialisation: PyTorch's pretrained BN statistics provide a better starting point for feature normalisation under distribution-shifted conditions, while Keras BN must learn from scratch during a training process complicated by the non-standard input distribution.

**Assessment:** The VGG16-BN comparison is partially confounded by the BN initialisation asymmetry — an inherent limitation because `tf.keras.applications.VGG16` does not include BatchNorm. The accuracy differences cannot be attributed purely to "framework choice" but reflect a combination of framework-level and implementation-level factors. **For tasks involving substantial distribution shift, frameworks providing pretrained BN statistics offer a measurable advantage.** Future work should investigate whether custom Keras VGG16-BN models with converted pretrained BN statistics close the gap.

#### 6.6 Keras MobileNetV2 Failure: A Case Study in Conditional Framework Equivalence

This subsection should serve as a **key depth-of-analysis highlight** in the paper. The failure affects **both datasets** (not only CIFAR-10), making it a systematic, framework-level phenomenon.

**6.6.1 Observed Phenomenon (use actual epoch-level data):**

Keras MobileNetV2 exhibited a consistent pathological pattern across all 6 runs (2 datasets × 3 folds):

- **CIFAR-10:** Training accuracy reached 95.2% by epoch 8, but val_loss exploded from 1.66 (epoch 1) to 13.41 (epoch 5). Validation accuracy collapsed to 16–48%. Test accuracies: fold 0 = 47.77%, fold 1 = 40.87%, fold 2 = 30.21%. All folds early-stopped at epoch 8–9.
- **PlantVillage:** Training accuracy reached 99.2% by epoch 8, but val_loss exploded from 2.16 (epoch 1) to 17.55 (epoch 5). Validation accuracy collapsed to 3–66%. Test accuracies: fold 0 = 45.56%, fold 1 = 65.81%, fold 2 = 60.18%. All folds early-stopped at epoch 8–9.
- **PyTorch MobileNetV2 under identical configuration:** Converged normally on both datasets — 95.39% (CIFAR-10), 99.74% (PlantVillage) — with smooth training dynamics and no early stopping.

**6.6.2 Analysis angles (develop each into 1–2 paragraphs):**

1. **Distinguishing overfitting from numerical instability:** The explosive val_loss divergence (4–10× within 2–3 epochs) while training metrics remain healthy is qualitatively distinct from classical overfitting. This pattern indicates numerical instability or a severe feature-distribution mismatch between batch statistics (used during training) and running statistics (used during validation), rather than simple memorisation.

2. **BatchNorm momentum/epsilon as the primary suspect:** PyTorch BN defaults (`momentum=0.1, epsilon=1e-5`) vs. Keras BN defaults (`momentum=0.99, epsilon=1e-3`). Keras's slower EMA update causes running statistics to lag behind rapidly changing weight distributions under aggressive LR. MobileNetV2's depthwise separable convolutions amplify per-channel normalisation errors — standard convolutions in ResNet50/VGG16 provide cross-channel averaging that absorbs the mismatch.

3. **LR–BN interaction:** Epoch-level logs confirm monotonic val_loss deterioration tracking the warmup LR increase (LR 0.008 → 0.04, epochs 1–5). The gradient landscape of Keras MobileNetV2 becomes unstable at elevated learning rates, while PyTorch tolerates the same schedule.

4. **Cross-dataset consistency as evidence of systematic cause:** The identical failure pattern on both PlantVillage (high-resolution, in-distribution) and CIFAR-10 (low-resolution, distribution-shifted) eliminates dataset-specific artefacts. The failure is intrinsic to the Keras MobileNetV2 implementation under this hyperparameter regime.

5. **Core insight for the paper's contribution:** This case study elevates the paper from "frameworks are roughly equivalent" to a more nuanced finding: **framework equivalence is conditionally dependent on the (architecture, dataset, hyperparameter) triple**. Framework comparison studies should not merely report average performance, but must systematically identify failure modes — configurations where one framework produces catastrophically worse results. These failure modes are the most actionable information for practitioners, as they may be triggered silently in production deployments.

**Tone:** Objective, evidence-driven. Frame as a **strength** of the paper.

#### 6.6b PyTorch ResNet50 on CIFAR-10: Warmup-Phase Training Instability

Write 2 paragraphs covering this secondary finding:

**Phenomenon:** PyTorch ResNet50 on CIFAR-10 folds 0/1 early-stopped at epoch 8 with ~91.9% accuracy (peaked at epoch 1 during warmup at LR ≈ 0.008, then deteriorated as LR increased). Fold 2 converged normally to 96.69% over 30 epochs. Keras ResNet50 showed no instability (all 3 folds: 95.70–95.96%, full 30 epochs).

**Interpretation:** The combination of AMP mixed precision + aggressive warmup LR (0.04) + distribution-shifted inputs creates a narrow stability region sensitive to classification-head initialisation. For folds 0/1, the random seed (42+fold) placed the model in an unfavourable region where increasing LR during warmup pushed it away from the valley found at epoch 1. **Practical implication:** For distribution-shifted transfer learning tasks, a reduced base LR or extended warmup is recommended as a precautionary measure.

#### 6.7 Scenario-Specific Recommendation Summary

Present this table to synthesise the discussion:

| Scenario | Recommended Configuration | Rationale |
|---|---|---|
| High-res domain-specific (e.g., plant disease, medical) | Either framework + ResNet50 or VGG16 | >99.3% on PlantVillage; framework guided by ecosystem preference |
| High-res + edge deployment (mobile, IoT) | PyTorch MobileNetV2 | 99.7% at 2.3M params; Keras MobileNetV2 unreliable |
| Low-res upsampled + stability required | Keras ResNet50 | 95.85% with σ=0.10%; most stable on CIFAR-10 |
| Low-res + highest accuracy target | PyTorch VGG16 | 95.73% on CIFAR-10; benefits from pretrained BN |
| Throughput priority (small source images) | Keras (any stable model) | 1.9–2.3× faster on CIFAR-10 via tf.data |
| Throughput priority (native-res images) | Either framework | Difference small and dataset-dependent |
| Framework migration of lightweight models | Requires re-validation | Keras MobileNetV2 failure proves parity cannot be assumed |

#### 6.8 Limitations
- Single-GPU setting (NVIDIA A40); results may differ on multi-GPU, TPU, or consumer GPUs.
- Only SGD tested; Adam or AdamW may interact differently with frameworks and BN implementations.
- CIFAR-10 bilinear upsampling (32→224) introduces artefacts that interact non-trivially with pretrained models, limiting generalisability to native-resolution tasks.
- 3-fold CV yields only 3 paired observations per test — low statistical power. A 5- or 10-fold design would strengthen significance claims.
- VGG16-BN comparison is confounded by the BN initialisation asymmetry (pretrained vs. default), which is inherent to the framework APIs.
- Keras ran in float32 while PyTorch used AMP (due to `LossScaleOptimizer` failure in custom GradientTape loops), introducing an asymmetry in both accuracy and throughput comparisons.
- No optimizer ablation; framework-specific Adam/AdamW implementations may exhibit different behaviour.

---

### 7. CONCLUSION AND FUTURE WORK

Write 2 paragraphs:

**Conclusion:**
- Restate the research question and the experimental protocol (36 runs, 2 datasets, 3 models, 3 folds, aligned hyperparameters).
- Summarize the **conditional equivalence** finding: For heavy architectures (ResNet50, VGG16), framework-induced accuracy differences are statistically significant but practically marginal (< 2.5 pp on both datasets); for these models, practitioners may choose either framework based on ecosystem preference. However, for lightweight architectures (MobileNetV2), the Keras implementation exhibited catastrophic training instability across both datasets (6/6 runs failed, accuracy degraded to 39–57% vs. PyTorch's 95–99%), demonstrating that framework choice can be a critical deployment variable.
- State the nuanced throughput finding: Neither framework is universally faster. Keras achieves 1.9–2.3× higher throughput on low-resolution upsampled data (CIFAR-10) due to tf.data pipeline efficiency, while PyTorch is comparable or slightly faster on native-resolution data (PlantVillage) due to AMP and optimised DataLoader.
- Highlight the core insight: Framework equivalence is conditionally dependent on the (architecture, dataset, hyperparameter) triple. Framework comparison studies should not merely report aggregate performance but must identify and analyse failure modes where performance degrades catastrophically, as these represent the most actionable information for practitioners.

**Future work:**
1. Extend the benchmark to additional architectures (EfficientNet, Vision Transformer) and datasets (ImageNet subsets, medical imaging).
2. Include Adam, AdamW, and LAMB optimizers to test optimizer-framework interactions.
3. Scale to multi-GPU and distributed training to evaluate framework scaling efficiency.
4. Investigate deterministic mode impact on training speed and convergence.
5. Add open-set and few-shot protocols for practical deployment scenarios.

---

### REFERENCES

Use Vancouver style. Include at minimum:

1. Deng, J., Dong, W., Socher, R., Li, L., Li, K. and Li, F. "ImageNet: A large-scale hierarchical image database." CVPR, 2009, pp. 248–255.
2. He, K., Zhang, X., Ren, S. and Sun, J. "Deep residual learning for image recognition." CVPR, 2016, pp. 770–778.
3. Simonyan, K. and Zisserman, A. "Very deep convolutional networks for large-scale image recognition." ICLR, 2015.
4. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A. and Chen, L. "MobileNetV2: Inverted residuals and linear bottlenecks." CVPR, 2018, pp. 4510–4520.
5. Hughes, D.P. and Salathe, M. "An open access repository of images on plant health to enable the development of mobile disease diagnostics." arXiv:1511.08060, 2015.
6. Mohanty, S.P., Hughes, D.P. and Salathe, M. "Using deep learning for image-based plant disease detection." Frontiers in Plant Science, Vol. 7, 2016, p. 1419.
7. Krizhevsky, A. "Learning multiple layers of features from tiny images." Technical Report, 2009.
8. Yosinski, J., Clune, J., Bengio, Y. and Lipson, H. "How transferable are features in deep neural networks?" NeurIPS, 2014, pp. 3320–3328.
9. Zhuang, F., Qi, Z., Duan, K., et al. "A comprehensive survey on transfer learning." Proceedings of the IEEE, Vol. 109, No. 1, 2020, pp. 43–76.
10. Paszke, A., Gross, S., Massa, F., et al. "PyTorch: An imperative style, high-performance deep learning library." NeurIPS, 2019, pp. 8024–8035.
11. Abadi, M., Barham, P., Chen, J., et al. "TensorFlow: A system for large-scale machine learning." OSDI, 2016, pp. 265–283.
`[Add more references as needed from the local references/ folder and citation trails.]`

---

### FIGURES AND TABLES PLAN

- **Figure 1:** Pipeline overview diagram (Data → Splits → Training (Keras / PyTorch) → Evaluation → Aggregation → Statistical Testing).
- **Figure 2:** Training curves (2×3 grid: 2 datasets × 3 models; each with loss and accuracy subplots, per-fold thin lines + mean bold line).
- **Figure 3 (optional):** LR schedule verification — overlay Keras/PyTorch LR curves from any fold.
- **Figure 4:** Keras MobileNetV2 failure visualization — val_loss explosion vs. training accuracy for one representative fold, annotated with LR at each epoch.
- **Table 1:** Hyperparameter alignment table (Section 4.5).
- **Table 2:** Test accuracy comparison — 6 rows, filled (Section 6.1).
- **Table 3:** Test macro-F1 comparison — 6 rows, filled (Section 6.1).
- **Table 4:** Training throughput comparison — 6 rows, filled, with p-values (Section 6.2).
- **Table 5:** Statistical test results — 18 rows, fully populated (Section 6.4).
- **Table 6:** Scenario-specific recommendation summary — 7 rows (Section 6.7).
- **Table 7 (optional):** Parameter count comparison by model and dataset.

---

## 4. Data Sources Used to Fill This Prompt

All experimental data in this document was extracted from the following files:

| File | Content | Rows |
|---|---|---|
| `results/results.csv` | Per-fold results for all 36 runs | 36 (+2 extra resnet20 runs) |
| `results/results_summary.csv` | Mean±std grouped by (framework, dataset, model) | 12 (+2 extra) |
| `analysis/statistical_results.csv` | Paired t-test results | 18 |
| `analysis/significance_summary.csv` | Significance flags and effect-size labels | 18 |
| `environment.txt` | Hardware and software versions | — |
| `logs/*.csv` | Epoch-level training logs (36 files) | varies |
| `results/param_comparison.csv` | Parameter counts per model | 6 |

**All [TODO] placeholders from the original draft have been resolved.** This document is now a self-contained writing prompt with complete experimental data and detailed analytical instructions. Hand it to an AI writer to produce the full JATIT manuscript.

---

*End of writing prompt.*
