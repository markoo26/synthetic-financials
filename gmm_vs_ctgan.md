# GMM vs CTGAN in `ydata-synthetic`

## Overview

Both **GMM** (Gaussian Mixture Model) and **CTGAN** (Conditional Tabular GAN) are used to generate synthetic **tabular data**, but they come from fundamentally different families of models — one is a classical statistical approach, the other is a deep learning approach.

---

## GMM — Gaussian Mixture Model

### What it is
GMM is a **probabilistic statistical model** that assumes the data is generated from a mixture of several Gaussian (normal) distributions. It does not involve neural networks.

### How it works
1. Fits `K` Gaussian components to the training data
2. Each component captures a cluster or mode in the data distribution
3. New samples are drawn by picking a component (weighted by probability) and sampling from its Gaussian

```python
from ydata_synthetic.synthesizers.regular import GMM

synth = GMM()
synth.fit(real_data, num_cols, cat_cols)
samples = synth.sample(1000)
```

### Strengths
- ✅ Fast to train — no GPU needed
- ✅ Deterministic and reproducible
- ✅ Works well on small datasets
- ✅ Interpretable — you can inspect the learned components
- ✅ No hyperparameter tuning required

### Weaknesses
- ❌ Assumes data follows Gaussian distributions — poor fit for complex, multi-modal data
- ❌ Struggles with highly correlated features
- ❌ Cannot model complex conditional relationships between columns
- ❌ Less realistic outputs for high-dimensional or heterogeneous data

---

## CTGAN — Conditional Tabular GAN

### What it is
CTGAN is a **deep learning model** based on Generative Adversarial Networks, specifically designed for tabular data. It was introduced in the paper *"Modeling Tabular Data using Conditional GAN"* (Xu et al., NeurIPS 2019).

### How it works
1. A **Generator** network learns to produce fake rows of data
2. A **Discriminator** network learns to distinguish real vs. fake rows
3. The **conditional** part means the generator is trained to handle class imbalance by conditioning on discrete columns
4. Uses **mode-specific normalization** to handle non-Gaussian continuous columns

```python
from ydata_synthetic.synthesizers.regular import RegularSynthesizer
from ydata_synthetic.synthesizers import ModelParameters, TrainParameters

ctgan_args = ModelParameters(batch_size=500, lr=2e-4, noise_dim=128, layers_dim=128)
train_args = TrainParameters(epochs=500)

synth = RegularSynthesizer(modelname='ctgan', model_parameters=ctgan_args)
synth.fit(real_data, train_args, num_cols=num_cols, cat_cols=cat_cols)
samples = synth.sample(1000)
```

### Strengths
- ✅ Handles complex, non-Gaussian distributions
- ✅ Models intricate correlations between columns
- ✅ Designed specifically for mixed-type tabular data (numeric + categorical)
- ✅ Handles class imbalance via conditional sampling
- ✅ Produces more realistic synthetic data on complex datasets

### Weaknesses
- ❌ Slow to train — benefits from GPU
- ❌ Requires hyperparameter tuning (epochs, batch size, learning rate)
- ❌ Can suffer from **mode collapse** (generator gets stuck producing similar samples)
- ❌ Needs more data to train effectively
- ❌ Less interpretable — black box model

---

## Side-by-Side Comparison

| Feature | GMM | CTGAN |
|---|---|---|
| **Model type** | Statistical | Deep Learning (GAN) |
| **Training speed** | Very fast | Slow |
| **GPU required** | No | Recommended |
| **Data size needed** | Small datasets OK | Needs more data |
| **Distribution assumption** | Gaussian | None (learned) |
| **Handles complex correlations** | Limited | Yes |
| **Handles class imbalance** | No | Yes (conditional) |
| **Interpretability** | High | Low |
| **Hyperparameter tuning** | Minimal | Required |
| **Risk of mode collapse** | No | Yes |
| **Best for** | Simple, low-dim data | Complex, real-world tabular data |

---

## When to Use Which

### Use GMM when:
- You have a **small dataset** (< 1,000 rows)
- You need a **quick baseline** or proof of concept
- Your data is relatively **simple and low-dimensional**
- You don't have GPU resources
- **Interpretability** matters

### Use CTGAN when:
- You have a **large, complex dataset**
- Your data has **mixed types** (numeric + categorical)
- You need to **preserve intricate correlations** between features
- You have **class imbalance** in categorical columns
- Data realism is more important than training speed

---

## Quick Rule of Thumb

> Start with **GMM** to get a fast baseline, then switch to **CTGAN** if the statistical fidelity of the synthetic data isn't good enough for your use case.
