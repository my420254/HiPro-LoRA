# HiPro-LoRA: Mitigating Feature Degeneration in Low-Resource Adaptation via Hierarchical Pooling and Prototype Regularization

------

## Overview

HiPro-LoRA addresses the challenge of fine-tuning pre-trained language models on **severely imbalanced text classification** tasks under **low-resource settings**. The method combines two complementary components:

- **AHSP (Adaptive Hierarchical State Pooling)**: A gate-controlled pooling mechanism that dynamically fuses CLS-token, mean-pooling, max-pooling, and attention-weighted representations, adaptively selecting the most discriminative sentence embedding for each sample.
- **TPMB (Tail-aware Prototype Memory Bank)**: A contrastive learning module that maintains per-class prototype queues, applies tail-class gradient scaling, and uses temperature-controlled prototype-to-sample contrastive loss to improve the geometric separability of minority-class embeddings.

Experiments are conducted on three benchmarks spanning Chinese social media, English fine-grained sentiment, and English Twitter sentiment, under two imbalanced low-resource training regimes per dataset.

------

## Datasets

All three datasets used in this paper are **publicly available** and can be downloaded automatically via the Hugging Face `datasets` library. No manual download or license agreement is required.

To rigorously simulate low-resource and long-tailed scenarios, we constructed imbalanced subsets from the original benchmarks using **Stratified Random Sampling**. Unlike random cropping, this strategy preserves the intrinsic semantic difficulty of each task while artificially introducing class imbalance. Crucially, all **Validation Sets are strictly class-balanced** to ensure unbiased Macro-F1 evaluation — preventing models from gaming the metric by predicting majority classes.

------

### SMP2020-EWECT — Chinese Emotion Classification

| Property          | Details                                                      |
| ----------------- | ------------------------------------------------------------ |
| **Task**          | 6-class emotion classification                               |
| **Language**      | Chinese (Simplified), Weibo social media posts               |
| **Source**        | SMP2020-EWECT shared task, "General" subset (Zhang et al., 2020) |
| **Availability**  | ✅ Publicly available via Hugging Face                        |
| **HF Identifier** | `Um1neko/smp2020`                                            |
| **Load Command**  | `load_dataset("Um1neko/smp2020")`                            |
| **Backbone**      | `hfl/chinese-macbert-base`                                   |

**Class definitions and roles:**

| Class ID | Emotion   | Role     | Config A (N=1000) Train | Config B (N=2000) Train |
| -------- | --------- | -------- | ----------------------- | ----------------------- |
| 3        | Neutral   | **Head** | 450                     | 1,000                   |
| 2        | Fear      | **Head** | 250                     | 500                     |
| 0        | Angry     | Mid      | 120                     | 200                     |
| 4        | Happy     | Mid      | 100                     | 150                     |
| 1        | Sad       | **Tail** | 60                      | 100                     |
| 5        | Surprise  | **Tail** | **20**                  | **50**                  |
| —        | **Total** | —        | **1,000**               | **2,000**               |

**Validation set (fixed, balanced):** 50 samples × 6 classes = **300 samples**

- **Imbalance Ratio (Max/Min):** 22.5 : 1 in Config A (Neutral 450 vs. Surprise 20)
- The Surprise class (Class 5) contains only 20 training samples in the hardest configuration, representing a severe tail-class scarcity challenge in Chinese social media emotion recognition.

------

### SST-5 — Fine-Grained English Sentiment Analysis

| Property          | Details                                           |
| ----------------- | ------------------------------------------------- |
| **Task**          | 5-class fine-grained sentiment classification     |
| **Language**      | English, movie review sentences                   |
| **Source**        | Stanford Sentiment Treebank (Socher et al., 2013) |
| **Availability**  | ✅ Publicly available via Hugging Face             |
| **HF Identifier** | `SetFit/sst5`                                     |
| **Load Command**  | `load_dataset("SetFit/sst5")`                     |
| **Backbone**      | `roberta-base`                                    |

**Class definitions and roles:**

| Class ID | Sentiment     | Role     | Config A (N=1150) Train | Config B (N=2300) Train |
| -------- | ------------- | -------- | ----------------------- | ----------------------- |
| 4        | Very Positive | **Head** | 600                     | 1,200                   |
| 3        | Positive      | **Head** | 300                     | 600                     |
| 2        | Neutral       | Mid      | 150                     | 300                     |
| 1        | Negative      | **Tail** | 80                      | 150                     |
| 0        | Very Negative | **Tail** | **20**                  | **50**                  |
| —        | **Total**     | —        | **1,150**               | **2,300**               |

**Validation set (fixed, balanced):** 80 samples × 5 classes = **400 samples**

- **Imbalance Ratio (Max/Min):** 30 : 1 in Config A (Very Positive 600 vs. Very Negative 20)
- This configuration tests a model's ability to maintain fine-grained boundaries between adjacent sentiment classes (e.g., Negative vs. Very Negative) when the minority class has extremely few training examples.

------

### TweetEval — Twitter Sentiment Classification

| Property          | Details                                                    |
| ----------------- | ---------------------------------------------------------- |
| **Task**          | 3-class sentiment classification                           |
| **Language**      | English, Twitter/social media informal text                |
| **Source**        | TweetEval benchmark (Barbieri et al., 2020)                |
| **Availability**  | ✅ Publicly available via Hugging Face                      |
| **HF Identifier** | `tweet_eval` / `cardiffnlp/tweet_eval`, subset `sentiment` |
| **Load Command**  | `load_dataset("tweet_eval", "sentiment")`                  |
| **Backbone**      | `roberta-base`                                             |

**Class definitions and roles:**

| Class ID | Sentiment | Role     | Config A (N=1000) Train | Config B (N=2000) Train |
| -------- | --------- | -------- | ----------------------- | ----------------------- |
| 1        | Neutral   | **Head** | 600                     | 1,200                   |
| 2        | Positive  | **Tail** | **300**                 | **600**                 |
| 0        | Negative  | **Tail** | **100**                 | **200**                 |
| —        | **Total** | —        | **1,000**               | **2,000**               |

**Validation set (fixed, balanced):** 50 samples × 3 classes = **150 samples**

- **Imbalance Ratio (Head vs. minority Tail class):** 6 : 1 in Config A (Neutral 600 vs. Negative 100)
- This creates a **"One-Head, Two-Tails"** scenario where the Neutral class dominates as background noise, while both Positive and Negative are treated as Tail targets. Note that the two tail classes themselves are also imbalanced (Positive 300 vs. Negative 100), requiring the model to handle compound imbalance simultaneously.

------

## Pre-trained Models

The following backbone models are loaded directly from Hugging Face Hub at runtime:

| Model                         | HF Identifier                                 | Used For                            |
| ----------------------------- | --------------------------------------------- | ----------------------------------- |
| Chinese MacBERT               | `hfl/chinese-macbert-base`                    | SMP2020 experiments                 |
| RoBERTa-base                  | `roberta-base`                                | SST-5 and TweetEval experiments     |
| Qwen2.5-7B-Instruct           | `Qwen/Qwen2.5-7B-Instruct`                    | LLM baseline (zero-shot & few-shot) |
| Llama-3.1-8B-Instruct (4-bit) | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | LLM baseline (zero-shot & few-shot) |

------

## Repository Structure

> **Note:** Large intermediate files (per-run embedding arrays `viz_data_*/`, gate trajectory logs `gate_logs_*/`, and HuggingFace repo manifests `__huggingface_repos__.json`) have been excluded from this repository due to GitHub file size limits. The files present here are sufficient to verify all reported metrics and regenerate all paper figures.

```
imageCodeAndResult/
│
├── create_Image.ipynb              # Visualization notebook — regenerates all paper figures
├── llm-result.md                   # Aggregated LLM baseline results (Qwen & Llama, all prompting strategies)
├── sensitivity_add_result.md       # Aggregated hyperparameter sensitivity results (all datasets)
├── Times_New_Roman.ttf             # Font file required for figure rendering
│
├── smp2020_main/                   # SMP2020 main experiment results
│   ├── smp2020_main.csv            # Raw per-run results (one row per method × N × seed)
│   ├── smp2020_main_Summary.md     # Aggregated Mean±Std summary (used by visualization)
│   └── smp2020_sensitivity.csv     # Initial sensitivity sweep raw results
│
├── smp2020_fine_grained_ablation/  # SMP2020 AHSP pooling ablation results
│   ├── smp2020_fine_grained_ablation.csv
│   └── smp2020_fine_grained_ablation_Summary.md
│
├── smp2020_add_sensitivity/        # SMP2020 extended sensitivity sweep results
│   ├── smp2020_ours_final.csv      # HiPro-LoRA reference scores at selected hyperparameters
│   └── smp2020_sensitivity_add.csv # Full sweep results across M, γ, τ, β
│
├── sst5_main/                      # SST-5 main experiment results
│   ├── sst5_main.csv
│   ├── sst5_main_Summary.md
│   └── sst5_sensitivity.csv
│
├── sst5_fine_grained_ablation/     # SST-5 AHSP pooling ablation results
│   ├── sst5_fine_grained_ablation.csv
│   └── sst5_fine_grained_ablation_Summary.md
│
├── sst5_add_sensitivity/           # SST-5 extended sensitivity sweep results
│   ├── sst5_ours_final.csv
│   └── sst5_sensitivity_add.csv
│
├── tweeteval_main/                 # TweetEval main experiment results
│   ├── tweeteval_main.csv
│   ├── tweeteval_main_Summary.md
│   └── tweeteval_sensitivity.csv
│
├── tweeteval_fine_grained_ablation/  # TweetEval AHSP pooling ablation results
│   ├── tweeteval_fine_grained_ablation.csv
│   └── tweeteval_fine_grained_ablation_Summary.md
│
└── tweeteval_add_sensitivity/      # TweetEval extended sensitivity sweep results
    ├── tweeteval_ours_final.csv
    └── tweeteval_sensitivity_add.csv
```

------

## Notebooks

All training and evaluation code is provided as self-contained Jupyter notebooks. Each notebook loads data directly from Hugging Face, runs all experiments with the specified random seeds, and saves results to the CSV and Markdown files listed above.

### Main Experiments

| Notebook               | Dataset   | Backbone             | Description                                                  |
| ---------------------- | --------- | -------------------- | ------------------------------------------------------------ |
| `smp2020_main.ipynb`   | SMP2020   | chinese-macbert-base | Trains all baselines + HiPro-LoRA variants at N=1000 and N=2000; 5 random seeds each |
| `sst5_main.ipynb`      | SST-5     | roberta-base         | Trains all baselines + HiPro-LoRA variants at N=1150 and N=2300; 5 random seeds each |
| `tweeteval_main.ipynb` | TweetEval | roberta-base         | Trains all baselines + HiPro-LoRA variants at N=1000 and N=2000; 5 random seeds each |

**Methods compared in main experiments**: LoRA-Vanilla, LoRA-Balanced, LoRA-Focal, LoRA-LDAM, LoRA-LogitAdj, DoRA-Balanced, Full-FineTuning, w/o TPMB, w/o AHSP, **HiPro-LoRA (Ours)**.

### Adversarial LoRA Baseline

| Notebook                   | Dataset   | Description                                     |
| -------------------------- | --------- | ----------------------------------------------- |
| `smp2020_LoRA_Adv.ipynb`   | SMP2020   | Trains LoRA-Adv (adversarial training baseline) |
| `sst5_LoRA_Adv.ipynb`      | SST-5     | Trains LoRA-Adv (adversarial training baseline) |
| `tweeteval_LoRA_Adv.ipynb` | TweetEval | Trains LoRA-Adv (adversarial training baseline) |

### AHSP Fine-Grained Pooling Ablation

| Notebook                                | Dataset   | Description                                                  |
| --------------------------------------- | --------- | ------------------------------------------------------------ |
| `smp2020_fine_grained_ablation.ipynb`   | SMP2020   | Tests 7 AHSP pooling variants: OnlyMean, OnlyMax, OnlyAttn, Mean+Attn, Max+Mean, Max+Attn, Full Tri-Fusion |
| `sst5_fine_grained_ablation.ipynb`      | SST-5     | Same 7 AHSP variants                                         |
| `tweeteval_fine_grained_ablation.ipynb` | TweetEval | Same 7 AHSP variants                                         |

### Hyperparameter Sensitivity

| Notebook                          | Dataset   | Description                                                  |
| --------------------------------- | --------- | ------------------------------------------------------------ |
| `smp2020_sensitivity_add.ipynb`   | SMP2020   | Sweeps over Memory Size M, Tail Weight γ, Temperature τ, Loss Weight β; 5 seeds per config |
| `sst5_sensitivity_add.ipynb`      | SST-5     | Same 4-parameter sensitivity sweep                           |
| `tweeteval_sensitivity_add.ipynb` | TweetEval | Same 4-parameter sensitivity sweep                           |

### LLM Baselines

| Notebook             | Model                 | Description                                                  |
| -------------------- | --------------------- | ------------------------------------------------------------ |
| `Qwen-no-cot.ipynb`  | Qwen2.5-7B-Instruct   | Zero-Shot and Balanced 1-Shot inference without Chain-of-Thought; 4-bit quantization |
| `Qwen-cot.ipynb`     | Qwen2.5-7B-Instruct   | Zero-Shot CoT and Balanced 1-Shot CoT inference              |
| `Llama-no-cot.ipynb` | Llama-3.1-8B-Instruct | Zero-Shot and Balanced 1-Shot inference without Chain-of-Thought; 4-bit quantization |
| `Llama-cot.ipynb`    | Llama-3.1-8B-Instruct | Zero-Shot CoT and Balanced 1-Shot CoT inference              |

> LLM notebooks use 4-bit quantization via `bitsandbytes` and evaluate on all three datasets. Results are aggregated into `llm-result.md`.

------

## Result Files

### `*_Summary.md` — Aggregated Metric Tables

Each `*_Summary.md` file contains a Markdown table aggregated across 5 random seeds, with columns including `Method`, `N`, `Macro-F1 (Mean±Std)`, `Train_Time_Sec (Mean±Std)`, and `Peak_Memory_MB (Mean±Std)`. These files are read directly by `create_Image.ipynb` to generate paper figures.

### `*_main.csv` — Raw Per-Run Results

Stores one row per (method × N × seed) with full per-run metrics including Macro-F1, Tail-F1, training time, and peak memory. Used for statistical significance testing and detailed analysis.

### `*_sensitivity_add.csv` — Sensitivity Sweep Results

Long-format table with columns `Dataset`, `Type` (MemSize / TailWeight / Temperature / LossWeight), `Value`, and one column per random seed. Read by `plot_sensitivity()` in `create_Image.ipynb` to generate Figure 6.

### `*_fine_grained_ablation.csv` / `*_fine_grained_ablation_Summary.md`

Results for the 7-variant AHSP pooling ablation study, in the same raw and aggregated formats as the main experiment files.

### `llm-result.md`

Aggregated LLM inference results covering all four prompting conditions (Zero-Shot No-CoT, Balanced 1-Shot No-CoT, Zero-Shot CoT, Balanced 1-Shot CoT) for both Qwen-2.5-7B and Llama-3.1-8B across all three datasets.

### `sensitivity_add_result.md`

Combined sensitivity results across all three datasets in a single reference table, used alongside the per-dataset CSV files for the sensitivity analysis section.

------

## Visualization

All paper figures are generated by running `create_Image.ipynb`. Place the notebook in the same directory as the result folders and run all cells. The figures are saved to a `figures/` subdirectory that will be created automatically on first run.

| Paper Figure | Output File                      | Description                                                  |
| ------------ | -------------------------------- | ------------------------------------------------------------ |
| Figure 4     | `figures/figure4_llm.pdf`        | Grouped bar chart comparing Qwen-2.5-7B and Llama-3.1-8B (No-CoT variants) vs. HiPro-LoRA across 3 datasets |
| Figure 5     | `figures/figure5_efficiency.pdf` | Bubble scatter plot of training time vs. peak memory; bubble size = Macro-F1 |
| Figure 6     | `figures/figure6_sens.pdf`       | 2×2 sensitivity line plots for M, γ, τ, β across 3 datasets; gold stars mark selected optimal configs |

**Font**: The notebook loads `Times_New_Roman.ttf` from the repository root. If not found, it falls back to the system serif font automatically.

------

## Environment & Dependencies

All notebooks were developed and tested on a single **NVIDIA Tesla P100 (16 GB)** GPU.

```bash
pip install torch==2.6.0+cu124
pip install transformers==4.53.3
pip install peft==0.16.0
pip install datasets==4.4.1
pip install bitsandbytes==0.43.0   # LLM 4-bit quantization (NF4)
pip install scikit-learn==1.2.2
pip install pandas==2.2.3
pip install numpy==1.26.4
pip install matplotlib==3.7.2
pip install seaborn==0.12.2
pip install tqdm==4.67.1
```

Python 3.9+ recommended.

- Fine-tuning notebooks: batch size 32–64 (with gradient accumulation where noted); ≥8 GB VRAM.
- LLM inference notebooks: 4-bit quantized models; ≥16 GB VRAM recommended.

------

## Reproducibility Notes

- All experiments use **5 fixed random seeds**: `[45, 123, 789, 1001, 2024]`. Seeds control dataset sub-sampling, model weight initialization, and training data shuffling.
- All reported Macro-F1 scores are **mean ± std over 5 seeds** evaluated on the full original test split.
- LoRA adapters target the **Query, Key, and Value** projection matrices with rank **r=16**, scaling factor **α=32**, and dropout 0.1.
- Training runs for a maximum of 20 epochs with step-based evaluation every 10–15 steps and early stopping patience of **8 evaluation intervals**.
- To mitigate class imbalance, all methods except Vanilla LoRA use cost-sensitive cross-entropy with class weights inversely proportional to training label frequencies.

------

