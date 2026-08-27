# HiPro-LoRA：面向低资源长尾情感分析的层次化原型 LoRA

HiPro-LoRA 是一套面向低资源长尾情感分析的参数高效微调方法，论文已被 **ECML-PKDD** 接收并提交最终版。ECML-PKDD 是机器学习与知识发现方向的重要国际会议，国内通常按 **CCF B 类会议** 识别，能够体现该工作在机器学习、NLP 和数据挖掘方向的完整科研价值。

本仓库整理了 HiPro-LoRA 的严格 held-out evaluation package，包括主实验、消融实验、门控动态分析、敏感性分析、LLM baseline、论文图表生成脚本和最终数据包。

## 研究问题

低资源长尾情感分析同时存在两类困难：

- 数据层面：尾部类别样本少、表达变化大，低资源采样会进一步放大类别不均衡；
- 模型层面：普通 LoRA 的低秩增量参数容易优先拟合头部类别，尾部类别决策边界不稳定。

HiPro-LoRA 的目标是在不显著增加部署成本的前提下，增强 LoRA 对尾部类别的表示能力，并通过严格评测协议确保提升来自方法本身，而不是数据划分或模型选择口径。

## 方法亮点

- **层次化表示建模**：针对长尾低资源场景中单层表示不稳定的问题，引入更细粒度的语义融合机制；
- **原型感知约束**：围绕尾部类别构建更稳定的类别中心，改善少数类特征聚集性；
- **门控动态分析**：不仅报告最终指标，还保留训练过程中 gate 行为和敏感性曲线，增强方法可解释性；
- **严格 held-out protocol**：显式区分 validation model selection 和 final test reporting，避免把验证集选择收益混入最终结果；
- **LLM baseline 对齐**：提供少样本大模型 baseline，用于比较轻量 PEFT 与 7B 级模型在成本和效果上的差异。

## 实验价值

在严格 held-out 设置中，HiPro-LoRA 在 6 个配置里取得 5 个 Tail-F1 最优和 4 个 Macro-F1 最优，其余设置也保持竞争力。该结果说明，面向长尾低资源任务时，结构化 PEFT 设计可以比单纯扩大模型规模更高效。

内部实验名 `LoRA-Ours` 对应论文中的 `HiPro-LoRA` 方法名。

## 仓库结构

```text
.
├── README.md
└── code/
    ├── run_scripts/                    # 统一实验入口
    ├── table2_strict/                  # 主实验严格 held-out 结果
    ├── table3_strict/                  # 细粒度消融结果
    ├── sensitivity_gate_strict/        # 敏感性分析与 gate 动态
    ├── llm_baselines/                  # 大模型 few-shot baseline
    ├── figures/                        # 论文图表
    ├── paper_source/                   # 论文源码材料
    ├── Table2_summary.md               # Table 2 数据包
    ├── Table3_final.md                 # Table 3 数据包
    ├── Figure2_gate_dynamics_data.md   # Figure 2 数据包
    ├── Figure3_llm_comparison_data.md  # Figure 3 数据包
    ├── Figure4_efficiency_data.md      # Figure 4 数据包
    └── Figure5_sensitivity_data.md     # Figure 5 数据包
```

## 数据集与 backbone

| 数据集 | 来源 | Backbone |
| --- | --- | --- |
| SMP2020 | `Um1neko/smp2020` | `hfl/chinese-macbert-base` |
| SST-5 | `SetFit/sst5` | `roberta-base` |
| TweetEval | `tweet_eval` / `cardiffnlp/tweet_eval` | `roberta-base` |

训练子集按低资源长尾设置采样，验证集用于模型选择，最终指标在 held-out class-balanced test subsets 上报告。

## 运行方式

```bash
cd code
python -u HiPro-loRA/run_scripts/run_figures.py
```

复现完整严格实验：

```bash
cd code
python -u HiPro-loRA/run_scripts/run_all.py
```

按实验组单独运行：

```bash
python -u HiPro-loRA/run_scripts/run_table2_strict.py
python -u HiPro-loRA/run_scripts/run_table3_strict.py
python -u HiPro-loRA/run_scripts/run_sensitivity_gate_strict.py
```

多 GPU 调度可通过环境变量控制：

```bash
export HIPRO_RUN_GPUS=0,1,2,3
export HIPRO_RUN_GPU_SLOTS=0:2,1:2,2:3,3:3
```

## 结果文件

```text
code/table2_strict/results/*_table2_strict_results.csv
code/table3_strict/results/*_table3_strict_results.csv
code/sensitivity_gate_strict/results/*_sensitivity.csv
code/llm_baselines/few_shot_results/llm_fewshot_results.csv
```

## 项目状态

- 论文状态：ECML-PKDD 已接收，最终版已提交；
- 代码状态：严格评测包、图表脚本和结果数据已整理；
- 许可协议：MIT License。
