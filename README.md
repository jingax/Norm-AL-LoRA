# NormAL LoRA: What is the Perfect Size?

This repository provides the official implementation of **NormAL LoRA** (Norm Adaptive Localised LoRA), introduced in the paper:

> 📄 **NormAL LoRA: What is the Perfect Size?**

NormAL LoRA introduces a regularized and adaptive variant of LoRA tuning, leveraging intermediate norm-based clipping to guide low-rank adaptation for downstream tasks.

---

## 🔧 Features

- Supports standard LoRA fine-tuning via [PEFT](https://github.com/huggingface/peft).
- Introduces **NormAL Regularization**: Norm-based adaptive masking + clipping.
- Custom trainer `RegularizedJingaTrainer` with support for dynamic clipping strategies.
- Compatible with a wide range of datasets: NLU (GLUE), CIFAR, FOOD101, and NLG tasks.
- Visualizes rank evolution with `plot_layer_ranks`.

---

## 🧩 Installation

### 1. Clone the repo

```bash
git clone https://anonymous.4open.science/r/NormAL-LoRA-EB38.git
cd NormAL-LoRA-EB38
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download datasets
Download E2E NLG challenge dataset from [here](https://github.com/tuetschek/e2e-dataset) and place it in `./data/e2e/`
Other NLU and summerization datasets will automatically downloaded when you rn the scripts

---

## 🏁 Quick Start

### Training with NormAL LoRA

```bash
CUDA_VISIBLE_DEVICES=0 python pilot.py \
  --name test-8 \
  --task NLG \
  --model_name TinyLlama \
  --rank 8 \
  --mode jinga \
  --reg_start 0 \
  --reg_end 100 \
  --clip_stage 200 \
  --reg_w 0.001 \
  --k 150 \
  --seed 0 \
  --reset no \
  --epochs 5.0 \
  --lr 0.0001 \
  --alpha 0 \
  --loss yes \
  --batch_s 64 \
  --p 0.5 \
  --clip_strategy budget \
  --clip_mode manual \
  --weight_decay 0.01
```


### 🔧 Command-Line Arguments

| Argument            | Type     | Default     | Description                                                                 |
|---------------------|----------|-------------|-----------------------------------------------------------------------------|
| `--name`            | `str`    | `"test-8"`  | Name of the experiment (used for output folder).                            |
| `--task`            | `str`    | `"NLG"`     | Task name: e.g., `CIFAR`, `MNLI`, `NLG`, `FOOD101`.                         |
| `--model_name`      | `str`    | `"TinyLlama"`| Base model name.                                                            |
| `--rank`            | `int`    | `8`         | Maximum LoRA rank.                                                          |
| `--mode`            | `str`    | `"no-jinga"`| Use `jinga` for NormAL LoRA or `no-jinga` for vanilla training.            |
| `--reg_start`       | `int`    | `0`         | Epoch step to start norm-based regularization.                             |
| `--reg_end`         | `int`    | `100`       | Epoch step to end regularization.                                          |
| `--clip_stage`      | `int`    | `200`       | Step to perform final norm-based clipping.                                 |
| `--reg_w`           | `float`  | `0.001`     | Regularization weight for NormAL loss.                                     |
| `--k`               | `int`    | `150`       | Top-k layers (by norm) to retain (budget).                |
| `--seed`            | `int`    | `0`         | Random seed.                                                                |
| `--reset`           | `str`    | `"no"`      | Whether to reset mask post-clipping (`yes` / `no`).                        |
| `--epochs`          | `float`  | `5.0`       | Number of training epochs.                                                 |
| `--lr`              | `float`  | `0.0001`    | Learning rate.                                                              |
| `--alpha`           | `int`    | `0`         | LoRA alpha scaling factor. Leave as 0 for `2 \times rank`                                                |
| `--loss`            | `str`    | `"yes"`     | Whether to log and save training losses (`yes` / `no`).                    |
| `--batch_s`         | `int`    | `64`        | Per-device training batch size.                                            |
| `--p`               | `float`  | `0.5`       | Penalty power for masking weights (used in NormAL logic).                  |
| `--clip_strategy`   | `str`    | `"budget"`  | Strategy for clipping LoRA masks: `budget` or `zeros`.                     |
| `--clip_mode`       | `str`    | `"manual"`  | When to trigger clipping: `manual` or `auto`.                              |
| `--weight_decay`    | `float`  | `0.01`      | Weight decay for optimizer.                                                |

### Evaluating NormAL LoRA
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_metrics.py --exp nlg --name test-8
```
## 🖥 Hardware Requirements

> ✅ **Use single-GPU** setup via `CUDA_VISIBLE_DEVICES=0`. Multi-GPU support is **not fully tested**.


## 🧪 Output Structure

After training, the following will be saved in `./outputs/<name>`:

- `config.yaml` — Run configuration
- `classifier.pth` — Classifier head weights
- `pooler.pth` — Pooler weights (for some tasks)
- `losses.json` — Per-batch training loss
- `pytorch_model.bin` — Final LoRA-adapted model
- `layer_rank_plot.png` — Visualization of layer-wise rank usage

---

## 📈 Visualizations

The framework supports visual tracking of LoRA rank evolution via:

```python
from utils.display import plot_layer_ranks
```

This helps diagnose how much of the rank budget was effectively utilized post-regularization.

---

## 📄 Citation

If you use this code or ideas from the NormAL LoRA paper, please cite us:

```bibtex
@article{normal-lora2025,
  title={NormAL LoRA: What is the Perfect Size?},
  author={Anonymous},
  journal={XXX},
  year={2025}
}
