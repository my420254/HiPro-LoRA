# Figure 7 Data: Table2-Aligned Gate Dynamics

Purpose: data package for rebuilding the paper gate-dynamics figure. This uses the Table2-aligned `LoRA-Ours` gate rerun, not the older `gate_configA` files.

No new experiment is required. These rows are collected from completed strict results.

## Important Note About Apparent Missing Values

All three datasets have the expected five completed seed runs (`45`, `123`, `789`, `1001`, `2024`). No rerun is required for the paper figure.

The raw aligned curve has no missing `trace_index` within each dataset. The apparent blanks occur only at the tail of some curves because different seeds stopped/logged at slightly different lengths. When `n_seeds=1`, `gate_std` is undefined. For the paper figure, use the `Common Five-Seed Curve Data` section below, where every point has all five seeds and a valid standard deviation.

## Figure Mapping

| Paper item | Current source figure/file | Meaning |
|---|---|---|
| Figure 7 | `sensitivity_gate_strict/gate_table2_aligned/results/gate_table2_aligned_curve_by_index.csv` | AHSP gate scalar `sigma(lambda)` curve, mean +/- std over five seeds |

## Source Files

- `sensitivity_gate_strict/gate_table2_aligned/results/gate_table2_aligned_curve_by_index.csv`
- `sensitivity_gate_strict/gate_table2_aligned/results/gate_table2_aligned_gate_summary.csv`
- `sensitivity_gate_strict/gate_table2_aligned/results/gate_table2_aligned_final_by_seed.csv`
- `sensitivity_gate_strict/gate_table2_aligned/results/gate_table2_aligned_metrics_internal.csv`

## Curve Coverage

| Dataset   | N    | All curve points | Common five-seed points | Use for paper          |
| --------- | ---- | ---------------- | ----------------------- | ---------------------- |
| SMP2020   | 1000 | 31               | 23                      | common five-seed curve |
| SST-5     | 1150 | 31               | 22                      | common five-seed curve |
| TweetEval | 1000 | 23               | 17                      | common five-seed curve |

## Seed Coverage Check

| Dataset   | N    | Seeds present              | Completed seed runs | Need rerun? |
| --------- | ---- | -------------------------- | ------------------- | ----------- |
| SMP2020   | 1000 | 45, 123, 789, 1001, 2024   | 5/5                 | No          |
| SST-5     | 1150 | 45, 123, 789, 1001, 2024   | 5/5                 | No          |
| TweetEval | 1000 | 45, 123, 789, 1001, 2024   | 5/5                 | No          |

## Plotting Columns

Use `trace_index` or `mean_epoch` as x-axis. Use `gate_mean` as y-axis and `gate_std` for the shaded band. For the final paper figure, use only rows where `n_seeds=5`.

## Gate Summary

| Dataset   | N    | Seeds | Gate init             | Gate final            | Delta                 | Mean min | Mean max |
| --------- | ---- | ----- | --------------------- | --------------------- | --------------------- | -------- | -------- |
| SMP2020   | 1000 | 5     | 0.119203 +/- 0.000000 | 0.122384 +/- 0.001767 | 0.003181 +/- 0.001767 | 0.119023 | 0.124156 |
| SST-5     | 1150 | 5     | 0.119203 +/- 0.000000 | 0.122930 +/- 0.001668 | 0.003727 +/- 0.001668 | 0.119182 | 0.125559 |
| TweetEval | 1000 | 5     | 0.010987 +/- 0.000000 | 0.011962 +/- 0.000280 | 0.000975 +/- 0.000280 | 0.010987 | 0.012526 |

## Final Gate By Seed

| Dataset   | N    | Seed | Gate init | Gate final | Delta    | Gate min | Gate max |
| --------- | ---- | ---- | --------- | ---------- | -------- | -------- | -------- |
| SMP2020   | 1000 | 45   | 0.119203  | 0.121164   | 0.001961 | 0.119203 | 0.122896 |
| SMP2020   | 1000 | 123  | 0.119203  | 0.123696   | 0.004493 | 0.119203 | 0.125839 |
| SMP2020   | 1000 | 789  | 0.119203  | 0.123942   | 0.004739 | 0.119203 | 0.126183 |
| SMP2020   | 1000 | 1001 | 0.119203  | 0.123217   | 0.004014 | 0.119203 | 0.124503 |
| SMP2020   | 1000 | 2024 | 0.119203  | 0.119903   | 0.000700 | 0.118301 | 0.121360 |
| SST-5     | 1150 | 45   | 0.119203  | 0.123883   | 0.004680 | 0.119099 | 0.126551 |
| SST-5     | 1150 | 123  | 0.119203  | 0.120785   | 0.001582 | 0.119203 | 0.122774 |
| SST-5     | 1150 | 789  | 0.119203  | 0.122079   | 0.002876 | 0.119203 | 0.125428 |
| SST-5     | 1150 | 1001 | 0.119203  | 0.125138   | 0.005935 | 0.119203 | 0.128011 |
| SST-5     | 1150 | 2024 | 0.119203  | 0.122767   | 0.003564 | 0.119203 | 0.125031 |
| TweetEval | 1000 | 45   | 0.010987  | 0.012205   | 0.001218 | 0.010987 | 0.012732 |
| TweetEval | 1000 | 123  | 0.010987  | 0.011867   | 0.000880 | 0.010987 | 0.012397 |
| TweetEval | 1000 | 789  | 0.010987  | 0.011704   | 0.000717 | 0.010987 | 0.012275 |
| TweetEval | 1000 | 1001 | 0.010987  | 0.011723   | 0.000736 | 0.010987 | 0.012614 |
| TweetEval | 1000 | 2024 | 0.010987  | 0.012311   | 0.001324 | 0.010987 | 0.012612 |

## Common Five-Seed Curve Data

| Dataset   | N    | trace_index | gate_mean | gate_std | gate_min | gate_max | mean_epoch | n_seeds |
| --------- | ---- | ----------- | --------- | -------- | -------- | -------- | ---------- | ------- |
| SMP2020   | 1000 | 0           | 0.119203  | 0.000000 | 0.119203 | 0.119203 | 0.000      | 5       |
| SMP2020   | 1000 | 1           | 0.119327  | 0.000010 | 0.119312 | 0.119340 | 0.625      | 5       |
| SMP2020   | 1000 | 2           | 0.119810  | 0.000093 | 0.119651 | 0.119889 | 1.250      | 5       |
| SMP2020   | 1000 | 3           | 0.120282  | 0.000595 | 0.119759 | 0.121007 | 1.875      | 5       |
| SMP2020   | 1000 | 4           | 0.121197  | 0.001511 | 0.119314 | 0.122631 | 2.500      | 5       |
| SMP2020   | 1000 | 5           | 0.121473  | 0.001931 | 0.118667 | 0.122968 | 3.125      | 5       |
| SMP2020   | 1000 | 6           | 0.121904  | 0.002220 | 0.118468 | 0.123796 | 3.750      | 5       |
| SMP2020   | 1000 | 7           | 0.121950  | 0.002263 | 0.118382 | 0.124293 | 4.375      | 5       |
| SMP2020   | 1000 | 8           | 0.121959  | 0.002345 | 0.118301 | 0.124184 | 5.000      | 5       |
| SMP2020   | 1000 | 9           | 0.121842  | 0.002171 | 0.118329 | 0.123613 | 5.625      | 5       |
| SMP2020   | 1000 | 10          | 0.121807  | 0.002027 | 0.118570 | 0.123453 | 6.250      | 5       |
| SMP2020   | 1000 | 11          | 0.121873  | 0.002033 | 0.118675 | 0.123670 | 6.875      | 5       |
| SMP2020   | 1000 | 12          | 0.121716  | 0.002154 | 0.118356 | 0.123636 | 7.500      | 5       |
| SMP2020   | 1000 | 13          | 0.121994  | 0.002179 | 0.118596 | 0.123942 | 8.125      | 5       |
| SMP2020   | 1000 | 14          | 0.121991  | 0.002201 | 0.118705 | 0.124325 | 8.750      | 5       |
| SMP2020   | 1000 | 15          | 0.122297  | 0.002179 | 0.119099 | 0.124639 | 9.375      | 5       |
| SMP2020   | 1000 | 16          | 0.122401  | 0.002235 | 0.119083 | 0.124977 | 10.000     | 5       |
| SMP2020   | 1000 | 17          | 0.122504  | 0.002272 | 0.119199 | 0.125200 | 10.625     | 5       |
| SMP2020   | 1000 | 18          | 0.122712  | 0.002358 | 0.119223 | 0.125415 | 11.250     | 5       |
| SMP2020   | 1000 | 19          | 0.122843  | 0.002379 | 0.119365 | 0.125432 | 11.875     | 5       |
| SMP2020   | 1000 | 20          | 0.123171  | 0.002418 | 0.119601 | 0.125822 | 12.500     | 5       |
| SMP2020   | 1000 | 21          | 0.123443  | 0.002416 | 0.119903 | 0.126183 | 13.125     | 5       |
| SMP2020   | 1000 | 22          | 0.123228  | 0.001995 | 0.120034 | 0.125026 | 13.625     | 5       |
| SST-5     | 1150 | 0           | 0.119203  | 0.000000 | 0.119203 | 0.119203 | 0.000      | 5       |
| SST-5     | 1150 | 1           | 0.119262  | 0.000017 | 0.119247 | 0.119288 | 0.556      | 5       |
| SST-5     | 1150 | 2           | 0.119628  | 0.000130 | 0.119471 | 0.119773 | 1.111      | 5       |
| SST-5     | 1150 | 3           | 0.120061  | 0.000260 | 0.119604 | 0.120253 | 1.667      | 5       |
| SST-5     | 1150 | 4           | 0.119744  | 0.000257 | 0.119374 | 0.120093 | 2.222      | 5       |
| SST-5     | 1150 | 5           | 0.119544  | 0.000241 | 0.119267 | 0.119876 | 2.778      | 5       |
| SST-5     | 1150 | 6           | 0.119663  | 0.000418 | 0.119174 | 0.120177 | 3.333      | 5       |
| SST-5     | 1150 | 7           | 0.120170  | 0.000968 | 0.119099 | 0.121661 | 3.889      | 5       |
| SST-5     | 1150 | 8           | 0.120779  | 0.001072 | 0.119870 | 0.122472 | 4.444      | 5       |
| SST-5     | 1150 | 9           | 0.120904  | 0.001028 | 0.119568 | 0.122407 | 5.000      | 5       |
| SST-5     | 1150 | 10          | 0.121207  | 0.001268 | 0.119698 | 0.123109 | 5.556      | 5       |
| SST-5     | 1150 | 11          | 0.121301  | 0.001156 | 0.119815 | 0.122943 | 6.111      | 5       |
| SST-5     | 1150 | 12          | 0.121511  | 0.001371 | 0.119801 | 0.123489 | 6.667      | 5       |
| SST-5     | 1150 | 13          | 0.121623  | 0.001378 | 0.119570 | 0.123334 | 7.222      | 5       |
| SST-5     | 1150 | 14          | 0.121939  | 0.001332 | 0.119914 | 0.123485 | 7.778      | 5       |
| SST-5     | 1150 | 15          | 0.122158  | 0.001366 | 0.120010 | 0.123512 | 8.333      | 5       |
| SST-5     | 1150 | 16          | 0.122390  | 0.001388 | 0.120177 | 0.123560 | 8.889      | 5       |
| SST-5     | 1150 | 17          | 0.122752  | 0.001649 | 0.120047 | 0.124074 | 9.444      | 5       |
| SST-5     | 1150 | 18          | 0.123094  | 0.001749 | 0.120215 | 0.124494 | 10.000     | 5       |
| SST-5     | 1150 | 19          | 0.123416  | 0.001858 | 0.120369 | 0.124969 | 10.556     | 5       |
| SST-5     | 1150 | 20          | 0.123732  | 0.001980 | 0.120489 | 0.125428 | 11.111     | 5       |
| SST-5     | 1150 | 21          | 0.123376  | 0.001908 | 0.120785 | 0.125469 | 11.556     | 5       |
| TweetEval | 1000 | 0           | 0.010987  | 0.000000 | 0.010987 | 0.010987 | 0.000      | 5       |
| TweetEval | 1000 | 1           | 0.011014  | 0.000009 | 0.011002 | 0.011022 | 0.938      | 5       |
| TweetEval | 1000 | 2           | 0.011087  | 0.000025 | 0.011058 | 0.011113 | 1.875      | 5       |
| TweetEval | 1000 | 3           | 0.011116  | 0.000046 | 0.011059 | 0.011179 | 2.812      | 5       |
| TweetEval | 1000 | 4           | 0.011303  | 0.000084 | 0.011183 | 0.011384 | 3.750      | 5       |
| TweetEval | 1000 | 5           | 0.011449  | 0.000110 | 0.011288 | 0.011540 | 4.688      | 5       |
| TweetEval | 1000 | 6           | 0.011554  | 0.000113 | 0.011389 | 0.011647 | 5.625      | 5       |
| TweetEval | 1000 | 7           | 0.011633  | 0.000124 | 0.011455 | 0.011733 | 6.562      | 5       |
| TweetEval | 1000 | 8           | 0.011715  | 0.000128 | 0.011523 | 0.011832 | 7.500      | 5       |
| TweetEval | 1000 | 9           | 0.011792  | 0.000132 | 0.011595 | 0.011937 | 8.438      | 5       |
| TweetEval | 1000 | 10          | 0.011905  | 0.000135 | 0.011710 | 0.012043 | 9.375      | 5       |
| TweetEval | 1000 | 11          | 0.012004  | 0.000162 | 0.011790 | 0.012207 | 10.312     | 5       |
| TweetEval | 1000 | 12          | 0.012095  | 0.000177 | 0.011867 | 0.012327 | 11.250     | 5       |
| TweetEval | 1000 | 13          | 0.012189  | 0.000180 | 0.011952 | 0.012417 | 12.188     | 5       |
| TweetEval | 1000 | 14          | 0.012282  | 0.000184 | 0.012055 | 0.012524 | 13.125     | 5       |
| TweetEval | 1000 | 15          | 0.012369  | 0.000184 | 0.012152 | 0.012614 | 14.062     | 5       |
| TweetEval | 1000 | 16          | 0.012126  | 0.000396 | 0.011704 | 0.012572 | 14.625     | 5       |

## Raw Tail Traceability

Variable-length raw tail points are intentionally omitted from this paper data file because they include rows with `n_seeds < 5` and undefined `gate_std`. For audit traceability, use `sensitivity_gate_strict/gate_table2_aligned/results/gate_table2_aligned_curve_by_index.csv`. The paper figure should use only the `Common Five-Seed Curve Data` section above.
