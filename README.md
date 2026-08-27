# HiPro-LoRA Strict Evaluation Package

## 中文项目介绍

HiPro-LoRA 是我完成的一篇参数高效微调方向工作，论文已被 **ECML-PKDD** 接收并提交最终版。ECML-PKDD 是机器学习与知识发现方向的重要国际会议，国内通常按 **CCF B 类会议** 识别，在算法研发、NLP、机器学习岗位里有比较高的项目辨识度。

这个仓库重点展示两件事：一是方法本身围绕低资源长尾情感分析做了 LoRA 结构和训练策略改造；二是我把论文实验从“能跑出结果”整理成了严格 held-out evaluation package，保证验证集、测试集、低资源采样、类别均衡测试和 LLM baseline 的口径清晰。

## 我解决的问题

长尾低资源场景下，普通 LoRA 微调容易出现两个问题：

- 少数类样本少，低秩增量参数容易优先拟合头部类别；
- 验证集选择、测试集报告和低资源采样如果口径不严，容易把方法提升和数据划分收益混在一起。

HiPro-LoRA 的重点是让 PEFT 方法在长尾类别上更稳定，同时让实验协议足够严格，能经得起面试和论文审稿追问。

## 面试展示重点

- **会议含金量**：ECML-PKDD 属于机器学习/数据挖掘方向的国际会议，CCF B 分类，能证明工作不只是课程实验，而是完整科研闭环。
- **方法能力**：围绕 LoRA 在低资源长尾分布下的特征表达瓶颈，做结构、门控/原型感知和训练约束上的改造。
- **评测严谨性**：显式区分 validation model selection 和 held-out test reporting，保留 Table/Figure 的最终数据包，避免结果口径混乱。
- **工程能力**：提供统一 run scripts、GPU slot 调度、LLM baseline runner、figure generation、严格 CSV 输出，方便复现和排查。
- **可讲难点**：低资源采样随机性、尾部类别 F1 波动、LLM baseline 成本、不同数据集 backbone 差异、审稿阶段对实验协议的可解释性要求。

## 技术关键词

`PyTorch` · `Transformers` · `PEFT` · `LoRA` · `Long-tail Learning` · `Low-resource NLP` · `LLM Baseline` · `Reproducible Evaluation`

This repository packages the strict held-out evaluation protocol for **HiPro-LoRA**, a PEFT framework for low-resource long-tailed sentiment analysis.

## Why this repo matters

Most low-resource sentiment papers mix validation selection and final reporting too loosely. This package keeps the protocol explicit:

- validation is used for model selection
- final scores are reported on disjoint class-balanced test subsets
- all strict CSVs and figure data are stored separately

## My contribution

- Designed the method and strict evaluation protocol
- Built the result tables, sensitivity sweeps, and gate-dynamics traces
- Organized the LLM baseline comparison package
- Prepared the final table / figure data for the paper

## Main result

Across the strict held-out settings, HiPro-LoRA achieves the best Tail-F1 in 5 of 6 configurations and the best Macro-F1 in 4 of 6, while staying competitive in the remaining settings.

## Status

ECML-PKDD accepted, final version submitted.

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
