# Distributed Greedy Image Coreset Selection (DGICS)

## Overview

Modern multimodal training pipelines face a major scalability problem: datasets are growing exponentially faster than the ability to efficiently train on them.

While working with large-scale multimodal training, I designed and implemented **Distributed Greedy Image Coreset Selection (DGICS)** — a distributed data selection system that significantly improves dataset quality while reducing redundant samples during training.

This project focused on improving training efficiency and downstream benchmark performance under strict hardware constraints.

---

# Problem

Training large multimodal models at scale is extremely resource intensive.

### Key Challenges

* A 2B parameter vision-language model required:

  * **~2 hours per epoch**
  * On only **30k samples**
  * Using an **NVIDIA A100 GPU**

* Hugging Face later released:

  * **HuggingFaceM4/Docmatix**
  * A dataset containing nearly **1 million multimodal samples**
  * Total dataset size approaching **1 TB**

This created several major bottlenecks:

1. Impossible to fully load into memory
2. Extremely expensive to train end-to-end
3. Large amounts of duplicated or highly similar image samples
4. High computational waste during fine-tuning

The challenge became:

> How can we intelligently select the most informative training samples from massive multimodal datasets while remaining computationally feasible?

---

# Solution: Distributed Greedy Image Coreset Selection (DGICS)

DGICS is a distributed coreset selection pipeline designed for scalable multimodal training.

The core idea:

> Instead of training on every image sample, dynamically select a smaller but more diverse and informative subset.

This reduces redundancy while preserving training signal quality.

---

# Architecture

## Stage 1 — Distributed Dataset Sharding

The dataset is distributed across multiple workers.

Each worker:

* Loads only a local shard
* Randomizes data during each pass
* Extracts candidate embeddings
* Contributes high-value samples into a shared candidate pool

This enables scalable processing without requiring the entire dataset to be loaded simultaneously.

---

## Stage 2 — Local Greedy Candidate Selection

Each worker performs greedy diversity-based selection locally.

The objective:

* Maximize sample diversity
* Reduce near-duplicate images
* Preserve informative multimodal examples

This dramatically reduces redundancy before global aggregation.

---

## Stage 3 — Global Greedy Selection

After local candidate pools are generated:

* All candidate subsets are merged
* A second global greedy selection step is applied
* The final coreset is produced

This ensures diversity both:

* Within local shards
* Across the entire distributed dataset

---

# System Configuration

## Training Setup

### Dataset

* 40k selected samples from:

  * HuggingFaceM4/Docmatix

### Model

* IBM Granite Vision 3.3 2B
* LoRA fine-tuning

### Infrastructure

* 2 T4 GPU nodes
* 16 CPU cores per GPU
* 8 distributed workers
* 3 CPU, 0.25 GPU cores allocated per worker, 

---

# Evaluation

## Similarity Reduction

The effectiveness of DGICS was evaluated using pairwise similarity metrics.

Lower similarity indicates:

* Higher diversity
* Less duplicated information
* Better training efficiency

---

## Without Coreset Selection

| Metric            | Score  |
| ----------------- | ------ |
| Mean Similarity   | 0.5999 |
| Median Similarity | 0.6087 |

---

## With DGICS Coreset Selection

| Metric            | Score  |
| ----------------- | ------ |
| Mean Similarity   | 0.4392 |
| Median Similarity | 0.4416 |

---

## Result

DGICS substantially reduced dataset redundancy.

### Improvements

* ~26.8% reduction in mean similarity
* ~27.5% reduction in median similarity
* Increased dataset diversity
* More efficient use of limited compute resources

---

# Benchmark Performance

## MMMU Pro Results

| Model                                             | Score     |
| ------------------------------------------------- | --------- |
| ibm-granite/granite-vision-3.3-2b                 | 1.79%     |
| Issactoto/granite-vision-3.3-2b-enhanced-first40k | 6.65%     |
| Issactoto/granite-vision-3.3-2b-enhanced-coreset  | **7.51%** |

---

# Key Outcome

The DGICS-selected dataset outperformed:

* The original base model
* A non-coreset fine-tuned baseline

Despite:

* Severe hardware limitations
* Only a single training epoch
* Training performed on a single A100-class environment

This demonstrates that:

> Better data selection can outperform simply increasing dataset size.

---

# Technical Highlights

## Core Skills Demonstrated

* Distributed systems design
* Scalable multimodal data processing
* Efficient large-scale training pipelines
* Greedy optimization algorithms
* Representation diversity analysis
* LoRA fine-tuning
* TPU/GPU resource optimization
* Vision-language model training

---

# Why This Matters

Large multimodal datasets are becoming too expensive to fully train on.

DGICS provides a practical framework for:

* Training under constrained compute budgets
* Reducing redundant data processing
* Improving data efficiency
* Scaling multimodal pipelines economically

This approach becomes increasingly valuable as datasets continue growing into multi-terabyte scale.

---

# Final Takeaway

Distributed Greedy Image Coreset Selection (DGICS) demonstrates that intelligent dataset selection can:

* Improve model quality
* Reduce redundancy
* Lower training cost
* Enable scalable multimodal learning

Even under limited hardware conditions, strategic data engineering can produce measurable benchmark gains.
