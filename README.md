# HiPro-LoRA Strict Evaluation Package

This directory contains the strict held-out evaluation package for HiPro-LoRA. It includes the experiment runners, strict result CSVs, figure-generation script, final table/figure data packages, and LLM baseline records used to reproduce the reported results.

The internal experiment name `LoRA-Ours` corresponds to the paper-facing method name `HiPro-LoRA`.

## Contents

```text
HiPro-loRA/
├── table2_strict/                  # Main strict held-out results (Table 2)
├── table3_strict/                  # Fine-grained ablation results (Table 3)
├── sensitivity_gate_strict/        # Sensitivity & gate-dynamics (Figure 2, Figure 5)
│   ├── sst5_sensitivity.py         #   SST-5 sensitivity sweeps → results/sst5_sensitivity.csv
│   ├── smp2020_sensitivity.py      #   SMP2020 sensitivity sweeps → results/smp2020_sensitivity.csv
│   ├── tweeteval_sensitivity.py    #   TweetEval sensitivity sweeps → results/tweeteval_sensitivity.csv
│   ├── sst5_gate.py                #   SST-5 gate-dynamics trace
│   ├── smp2020_gate.py             #   SMP2020 gate-dynamics trace
│   ├── tweeteval_gate.py           #   TweetEval gate-dynamics trace
│   └── results/                    #   Generated sensitivity & gate CSV output
├── llm_baselines/                  # Strict few-shot LLM baseline runners and results
├── run_scripts/                    # Unified experiment entry points
├── figures/                        # Generated paper figures
├── paper_source/                   # LaTeX source files
├── Table2_summary.md               # Table 2 data package
├── Table3_final.md                 # Table 3 data package
├── Figure2_gate_dynamics_data.md   # Figure 2 data package
├── Figure3_llm_comparison_data.md  # Figure 3 data package
├── Figure4_efficiency_data.md      # Figure 4 data package
├── Figure5_sensitivity_data.md     # Figure 5 data package
└── create_Image.py                 # Figure generation script
```

## Environment

Run commands from `mainCode`:

```bash
cd <PROJECT_ROOT>/HiPro-LoRA/mainCode
source <CONDA_INSTALL>/etc/profile.d/conda.sh
conda activate hipro-lora
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

For deterministic single-task launches, optionally set:

```bash
export PYTHONHASHSEED=45
```

## Dependencies

The current project environment is:

```text
Python 3.10.20
PyTorch 2.7.0+cu128
Transformers 4.53.3
Datasets 4.4.1
PEFT 0.16.0
scikit-learn 1.7.2
pandas 2.3.3
NumPy 2.2.5
Matplotlib 3.10.9
Seaborn 0.12.2
Accelerate 1.13.0
bitsandbytes 0.49.2
tqdm 4.67.1
```

Core package groups:

- Training and model loading: `torch`, `transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`
- Metrics and data processing: `scikit-learn`, `pandas`, `numpy`
- Figure generation: `matplotlib`, `seaborn`
- Progress and utility support: `tqdm`

## Dataset Sources

All datasets are loaded through Hugging Face `datasets`.

| Dataset | Hugging Face source | Split usage |
|---|---|---|
| SMP2020 | `Um1neko/smp2020` | The script uses the HF `train` split as the training pool, creates a stratified 20% validation split with seed `42`, and evaluates on the HF `test` split. |
| SST-5 | `SetFit/sst5` | Table2/Table3: official `train`/`validation`/`test` splits. Sensitivity/gate: stratified 20% validation split from `train` with seed `42`. All validation/test subsets are class-balanced (80 per class) with seed `42`. |
| TweetEval | `tweet_eval`, config `sentiment`; fallback `cardiffnlp/tweet_eval`, config `sentiment` | The script uses the official `train`, `validation`, and `test` splits. Validation and test subsets are class-balanced with seed `42`. |

The low-resource long-tail training subsets are sampled from the training pool with the configured seeds. The held-out test subsets are never used for model selection.

Backbone models:

| Dataset | Backbone |
|---|---|
| SMP2020 | `hfl/chinese-macbert-base` |
| SST-5 | `roberta-base` |
| TweetEval | `roberta-base` |

## Main Entry Points

```bash
python -u HiPro-loRA/run_scripts/run_table2_strict.py
python -u HiPro-loRA/run_scripts/run_table3_strict.py
python -u HiPro-loRA/run_scripts/run_sensitivity_gate_strict.py
python -u HiPro-loRA/run_scripts/run_figures.py
python -u HiPro-loRA/run_scripts/run_all.py
```

GPU scheduling can be controlled with:

```bash
export HIPRO_RUN_GPUS=0,1,2,3
export HIPRO_RUN_GPU_SLOTS=0:2,1:2,2:3,3:3
```

`run_all.py` executes the strict experiment groups sequentially and can overwrite existing result CSVs. Use it only when a full reproduction is intended.

## Starting the Project

Use the following workflow for normal use:

```bash
cd <PROJECT_ROOT>/HiPro-LoRA/mainCode
source <CONDA_INSTALL>/etc/profile.d/conda.sh
conda activate hipro-lora
python -u HiPro-loRA/run_scripts/run_figures.py
```

To reproduce all strict experimental results, run:

```bash
python -u HiPro-loRA/run_scripts/run_all.py
```

To rerun only a specific result group, use the corresponding script from `run_scripts/`.

## Final Result Files

The main strict result tables are stored in:

```text
table2_strict/results/*_table2_strict_results.csv
table3_strict/results/*_table3_strict_results.csv
```

Sensitivity and gate results are generated by the scripts under `sensitivity_gate_strict/`:

```text
sensitivity_gate_strict/results/sst5_sensitivity.csv
sensitivity_gate_strict/results/smp2020_sensitivity.csv
sensitivity_gate_strict/results/tweeteval_sensitivity.csv
```

The figure data packages are:

```text
Figure2_gate_dynamics_data.md
Figure3_llm_comparison_data.md
Figure4_efficiency_data.md
Figure5_sensitivity_data.md
```

Generated figures are written to:

```text
figures/
```

The strict LLM baseline summary and raw records are:

```text
llm_baselines/LLM_BASELINE_SUMMARY.md
llm_baselines/few_shot_results/llm_fewshot_results.csv
llm_baselines/run_logs/
```

## Figure Generation

To rebuild paper figures from the strict result files:

```bash
python -u HiPro-loRA/run_scripts/run_figures.py
```

The figure script reads strict CSVs and the final data packages. It does not use legacy mixed-protocol outputs.
