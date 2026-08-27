# HiPro-LoRA 严格评测代码包

该目录保存 HiPro-LoRA 的实验复现材料，包括主实验 runner、严格结果 CSV、图表生成脚本、表格/图表数据包和 LLM baseline 记录。内部实验名 `LoRA-Ours` 对应论文中的方法名 `HiPro-LoRA`。

## 目录结构

```text
HiPro-loRA/
├── table2_strict/                  # Table 2 主实验严格 held-out 结果
├── table3_strict/                  # Table 3 细粒度消融实验
├── sensitivity_gate_strict/        # Figure 2 / Figure 5 敏感性与 gate 动态
│   ├── sst5_sensitivity.py
│   ├── smp2020_sensitivity.py
│   ├── tweeteval_sensitivity.py
│   ├── sst5_gate.py
│   ├── smp2020_gate.py
│   ├── tweeteval_gate.py
│   └── results/
├── llm_baselines/                  # 严格 few-shot LLM baseline
├── run_scripts/                    # 统一实验入口
├── figures/                        # 生成后的论文图表
├── paper_source/                   # 论文源码材料
├── Table2_summary.md               # Table 2 数据包
├── Table3_final.md                 # Table 3 数据包
├── Figure2_gate_dynamics_data.md   # Figure 2 数据包
├── Figure3_llm_comparison_data.md  # Figure 3 数据包
├── Figure4_efficiency_data.md      # Figure 4 数据包
├── Figure5_sensitivity_data.md     # Figure 5 数据包
└── create_Image.py                 # 图表生成脚本
```

## 环境建议

```bash
source <CONDA_INSTALL>/etc/profile.d/conda.sh
conda activate hipro-lora
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=false
export NVIDIA_TF32_OVERRIDE=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

可选确定性设置：

```bash
export PYTHONHASHSEED=45
```

主要依赖版本：

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

## 数据集

所有数据集通过 Hugging Face `datasets` 加载。

| 数据集 | Hugging Face 来源 | 划分策略 |
| --- | --- | --- |
| SMP2020 | `Um1neko/smp2020` | HF `train` 作为训练池，seed `42` 分层划出 20% validation，HF `test` 用于最终评测 |
| SST-5 | `SetFit/sst5` | Table2/Table3 使用官方 `train` / `validation` / `test`；敏感性和 gate 实验从 `train` 分层划分 validation |
| TweetEval | `tweet_eval` / `cardiffnlp/tweet_eval` | 使用官方 `train` / `validation` / `test`，validation/test 子集按类别均衡采样 |

低资源长尾训练子集只从训练池采样。最终 held-out test subsets 不参与模型选择。

## Backbone

| 数据集 | Backbone |
| --- | --- |
| SMP2020 | `hfl/chinese-macbert-base` |
| SST-5 | `roberta-base` |
| TweetEval | `roberta-base` |

## 主要入口

```bash
python -u HiPro-loRA/run_scripts/run_table2_strict.py
python -u HiPro-loRA/run_scripts/run_table3_strict.py
python -u HiPro-loRA/run_scripts/run_sensitivity_gate_strict.py
python -u HiPro-loRA/run_scripts/run_figures.py
python -u HiPro-loRA/run_scripts/run_all.py
```

GPU 调度：

```bash
export HIPRO_RUN_GPUS=0,1,2,3
export HIPRO_RUN_GPU_SLOTS=0:2,1:2,2:3,3:3
```

`run_all.py` 会顺序执行严格实验组，并可能覆盖已有结果 CSV。只需要生成论文图表时，优先运行 `run_figures.py`。

## 结果文件

```text
table2_strict/results/*_table2_strict_results.csv
table3_strict/results/*_table3_strict_results.csv
sensitivity_gate_strict/results/sst5_sensitivity.csv
sensitivity_gate_strict/results/smp2020_sensitivity.csv
sensitivity_gate_strict/results/tweeteval_sensitivity.csv
llm_baselines/few_shot_results/llm_fewshot_results.csv
```

图表数据包：

```text
Figure2_gate_dynamics_data.md
Figure3_llm_comparison_data.md
Figure4_efficiency_data.md
Figure5_sensitivity_data.md
```

生成图表：

```bash
python -u HiPro-loRA/run_scripts/run_figures.py
```

图表脚本读取严格 CSV 和最终数据包，不使用旧版混合协议输出。
