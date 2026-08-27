# Figure 4 Data: Efficiency Comparison

Purpose: data package for rebuilding the paper efficiency figure.

No new experiment is required. This figure uses Table 2 strict result CSVs and only aggregates completed five-seed runs.

Hardware: each profiling run is measured on a single NVIDIA RTX PRO 6000 GPU.

## Figure Mapping

| Paper item | Current generated figure | Meaning |
|---|---|---|
| Figure 4 | `figures/figure5_efficiency.pdf` | Train time vs peak memory, marker size by Macro-F1 |

## Source Files

- `table2_strict/results/smp2020_main_table2_strict_results.csv`
- `table2_strict/results/smp2020_lora_adv_table2_strict_results.csv`
- `table2_strict/results/sst5_main_table2_strict_results.csv`
- `table2_strict/results/sst5_lora_adv_table2_strict_results.csv`
- `table2_strict/results/tweeteval_main_table2_strict_results.csv`
- `table2_strict/results/tweeteval_lora_adv_table2_strict_results.csv`

## Plotting Columns

Use `Train Time mean (s)` as x-axis, `Peak Memory mean (MB)` as y-axis, and `Macro-F1 mean` for marker size. The plotted methods match `create_Image.py`: `LoRA-Ours`, `LoRA-Adv`, `DoRA-Balanced`, `LoRA-Balanced`, `Full-FineTuning`.

## Direct Plot Data

| Dataset   | N    | Method          | Macro-F1 mean | Macro-F1 std | Train Time mean (s) | Train Time std (s) | Peak Memory mean (MB) | Peak Memory std (MB) | Params mean (M) |
| --------- | ---- | --------------- | ------------- | ------------ | ------------------- | ------------------ | --------------------- | -------------------- | --------------- |
| SMP2020   | 1000 | LoRA-Ours       | 0.699337      | 0.015215     | 139.689             | 18.759             | 2108.100              | 1.140                | 6.898           |
| SMP2020   | 1000 | LoRA-Adv        | 0.691062      | 0.028104     | 145.382             | 14.790             | 3547.146              | 0.000                | 0.889           |
| SMP2020   | 1000 | DoRA-Balanced   | 0.691043      | 0.026464     | 121.926             | 24.692             | 2484.473              | 0.000                | 0.917           |
| SMP2020   | 1000 | LoRA-Balanced   | 0.689804      | 0.031687     | 96.510              | 18.716             | 2012.618              | 0.000                | 0.889           |
| SMP2020   | 1000 | Full-FineTuning | 0.667072      | 0.012600     | 98.702              | 15.683             | 3499.792              | 1.036                | 102.272         |
| TweetEval | 1000 | LoRA-Ours       | 0.751062      | 0.028999     | 86.091              | 16.981             | 2118.450              | 146.002              | 6.894           |
| TweetEval | 1000 | LoRA-Adv        | 0.734303      | 0.033942     | 96.126              | 13.065             | 3522.491              | 296.424              | 0.887           |
| TweetEval | 1000 | DoRA-Balanced   | 0.734185      | 0.025120     | 84.186              | 10.601             | 2500.221              | 197.612              | 0.915           |
| TweetEval | 1000 | LoRA-Balanced   | 0.734480      | 0.025425     | 60.680              | 8.488              | 2041.584              | 149.061              | 0.887           |
| TweetEval | 1000 | Full-FineTuning | 0.718805      | 0.025406     | 73.771              | 9.868              | 3270.877              | 182.351              | 124.648         |
| SST-5     | 1150 | LoRA-Ours       | 0.477806      | 0.021815     | 98.104              | 15.152             | 2208.612              | 141.795              | 6.897           |
| SST-5     | 1150 | LoRA-Adv        | 0.455497      | 0.032540     | 127.369             | 22.274             | 3805.800              | 284.744              | 0.889           |
| SST-5     | 1150 | DoRA-Balanced   | 0.454652      | 0.010537     | 110.350             | 14.030             | 2688.483              | 190.898              | 0.916           |
| SST-5     | 1150 | LoRA-Balanced   | 0.457083      | 0.033929     | 93.476              | 10.600             | 2183.091              | 143.013              | 0.889           |
| SST-5     | 1150 | Full-FineTuning | 0.431931      | 0.014821     | 71.135              | 10.816             | 3435.278              | 180.778              | 124.649         |
