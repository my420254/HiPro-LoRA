# Figure 3 Data: Strict LLM Comparison

Purpose: data package for rebuilding the paper LLM-comparison figure. Figure 3 is the large-model comparison under the strict held-out protocol.

No new experiment is required. The strict LLM matrix is already complete: 2 models x 3 datasets x 4 shot settings x 2 prompt styles = 48 rows.

## Fairness Note

For Table 2 strict, `N=1000/2000` (or `N=1150/2300` for SST-5) changes only the fine-tuning training subset size. The held-out test subset is fixed by dataset, class-balanced, and sampled from the official test split with seed `42`. The LLM runner uses the same official test split, same per-class test size, and seed `42`. Therefore the best strict HiPro-LoRA row from Table 2 can be compared to the existing strict LLM rows without rerunning LLMs. The figure should label this as `HiPro-LoRA best strict Table 2 result vs strict LLM few-shot on the same held-out test`, not as an equal-training-budget comparison.

## Figure Mapping

| Paper item | Current generated figure | Meaning |
|---|---|---|
| Figure 3 | `figures/figure4_llm.pdf` | Full strict LLM few-shot curves with the best strict HiPro-LoRA result shown as a reference line |

## Source Files

- `llm_baselines/few_shot_results/llm_fewshot_results.csv`
- `llm_baselines/LLM_BASELINE_SUMMARY.md`
- `table2_strict/results/smp2020_main_table2_strict_results.csv`
- `table2_strict/results/sst5_main_table2_strict_results.csv`
- `table2_strict/results/tweeteval_main_table2_strict_results.csv`
- `llm_baselines/run_logs/`

## Plot Design

The figure uses one panel per dataset. Within each panel, all strict LLM rows are plotted as shot-wise curves:

- x-axis: shot count (`0`, `1`, `3`, `5`)
- color: LLM backbone (`Qwen`, `Llama`)
- marker/line style: prompt style (`No CoT`, `CoT`)
- orange horizontal line: best strict Table 2 `LoRA-Ours` / HiPro-LoRA result for that dataset

This shows all 48 strict LLM results while keeping the HiPro-LoRA comparison readable.

## HiPro-LoRA Reference Lines

| Dataset   | HiPro-LoRA N | HiPro-LoRA Macro-F1 | HiPro-LoRA % | Qwen best strict LLM Method | Qwen best strict LLM Macro-F1 | Qwen best strict LLM % | Llama best strict LLM Method | Llama best strict LLM Macro-F1 | Llama best strict LLM % |
| --------- | ------------ | ------------------- | ------------ | --------------------------- | ----------------------------- | ---------------------- | ---------------------------- | ------------------------------ | ----------------------- |
| SMP2020   | 2000         | 0.703167            | 70.32        | Zero-Shot CoT               | 0.642363                      | 64.24                  | Balanced 1-Shot (No CoT)     | 0.563698                       | 56.37                   |
| SST-5     | 2300         | 0.512315            | 51.23        | Balanced 3-Shot (No CoT)    | 0.559433                      | 55.94                  | Balanced 1-Shot CoT          | 0.527417                       | 52.74                   |
| TweetEval | 1000         | 0.751062            | 75.11        | Balanced 3-Shot CoT         | 0.735711                      | 73.57                  | Balanced 1-Shot CoT          | 0.702304                       | 70.23                   |

## Verdict Against Best Strict LLM

| Dataset   | HiPro-LoRA N | HiPro-LoRA Macro-F1 | Best strict LLM                                | Best LLM Macro-F1 | Margin Ours-LLM | Outcome    |
| --------- | ------------ | ------------------- | ---------------------------------------------- | ----------------- | --------------- | ---------- |
| SMP2020   | 2000         | 0.703167            | Qwen2.5-7B-Instruct / Zero-Shot CoT            | 0.642363          | +0.060804       | Ours leads |
| SST-5     | 2300         | 0.512315            | Qwen2.5-7B-Instruct / Balanced 3-Shot (No CoT) | 0.559433          | -0.047118       | LLM leads  |
| TweetEval | 1000         | 0.751062            | Qwen2.5-7B-Instruct / Balanced 3-Shot CoT      | 0.735711          | +0.015351       | Ours leads |

## Full Strict LLM Source Rows

| Dataset   | Model                               | Method                   | Macro-F1 | Accuracy | Inference ms | Peak MB  | Params M | Protocol               |
| --------- | ----------------------------------- | ------------------------ | -------- | -------- | ------------ | -------- | -------- | ---------------------- |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot (No CoT) | 0.563698 | 0.580000 | 111.017      | 5665.271 | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot CoT      | 0.538536 | 0.543333 | 2890.603     | 5672.133 | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 3-Shot (No CoT) | 0.545939 | 0.573333 | 112.368      | 5815.481 | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 3-Shot CoT      | 0.487760 | 0.490000 | 3235.204     | 5822.060 | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 5-Shot (No CoT) | 0.554083 | 0.583333 | 143.712      | 5924.686 | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 5-Shot CoT      | 0.527531 | 0.543333 | 2339.353     | 5931.264 | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot (No CoT)       | 0.437706 | 0.453333 | 74.105       | 5619.932 | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot CoT            | 0.403853 | 0.420000 | 2843.023     | 5626.276 | 4540.600 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | Balanced 1-Shot (No CoT) | 0.633873 | 0.646667 | 84.483       | 5500.433 | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | Balanced 1-Shot CoT      | 0.615696 | 0.620000 | 1454.759     | 5503.219 | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | Balanced 3-Shot (No CoT) | 0.508967 | 0.533333 | 86.580       | 5577.522 | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | Balanced 3-Shot CoT      | 0.634366 | 0.636667 | 1255.199     | 5590.235 | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | Balanced 5-Shot (No CoT) | 0.486412 | 0.526667 | 109.446      | 5646.732 | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | Balanced 5-Shot CoT      | 0.626374 | 0.636667 | 1278.530     | 5667.438 | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | Zero-Shot (No CoT)       | 0.575081 | 0.593333 | 58.862       | 5470.860 | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | Zero-Shot CoT            | 0.642363 | 0.643333 | 1586.813     | 5475.425 | 4352.972 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot (No CoT) | 0.487317 | 0.520000 | 74.954       | 5634.312 | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot CoT      | 0.527417 | 0.537500 | 2256.949     | 5637.695 | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 3-Shot (No CoT) | 0.446999 | 0.470000 | 89.419       | 5699.919 | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 3-Shot CoT      | 0.525307 | 0.532500 | 2317.387     | 5704.232 | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 5-Shot (No CoT) | 0.359314 | 0.395000 | 105.724      | 5757.152 | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 5-Shot CoT      | 0.526627 | 0.532500 | 2364.180     | 5760.661 | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot (No CoT)       | 0.373446 | 0.420000 | 71.144       | 5598.149 | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot CoT            | 0.501670 | 0.510000 | 2306.006     | 5601.532 | 4540.600 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | Balanced 1-Shot (No CoT) | 0.549135 | 0.555000 | 84.355       | 5487.116 | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | Balanced 1-Shot CoT      | 0.478211 | 0.505000 | 1489.689     | 5490.597 | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | Balanced 3-Shot (No CoT) | 0.559433 | 0.575000 | 71.168       | 5524.415 | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | Balanced 3-Shot CoT      | 0.488039 | 0.507500 | 1449.966     | 5536.817 | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | Balanced 5-Shot (No CoT) | 0.498340 | 0.527500 | 88.832       | 5564.980 | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | Balanced 5-Shot CoT      | 0.481933 | 0.502500 | 1541.376     | 5577.495 | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | Zero-Shot (No CoT)       | 0.533729 | 0.537500 | 57.468       | 5460.669 | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | Zero-Shot CoT            | 0.468174 | 0.492500 | 1344.914     | 5463.119 | 4352.972 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot (No CoT) | 0.519980 | 0.586667 | 71.334       | 5617.464 | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot CoT      | 0.702304 | 0.700000 | 2505.230     | 5620.722 | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 3-Shot (No CoT) | 0.522294 | 0.613333 | 80.209       | 5667.463 | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 3-Shot CoT      | 0.678443 | 0.680000 | 2763.946     | 5670.972 | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 5-Shot (No CoT) | 0.584915 | 0.633333 | 91.231       | 5713.515 | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 5-Shot CoT      | 0.652228 | 0.653333 | 2894.897     | 5718.023 | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot (No CoT)       | 0.426705 | 0.460000 | 83.435       | 5592.001 | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot CoT            | 0.694595 | 0.706667 | 2315.316     | 5595.260 | 4540.600 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | Balanced 1-Shot (No CoT) | 0.720157 | 0.726667 | 73.519       | 5476.571 | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | Balanced 1-Shot CoT      | 0.716129 | 0.726667 | 1286.149     | 5478.610 | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | Balanced 3-Shot (No CoT) | 0.690341 | 0.686667 | 68.204       | 5502.274 | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | Balanced 3-Shot CoT      | 0.735711 | 0.740000 | 1180.434     | 5516.115 | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | Balanced 5-Shot (No CoT) | 0.690165 | 0.693333 | 79.253       | 5535.097 | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | Balanced 5-Shot CoT      | 0.726589 | 0.733333 | 1189.597     | 5548.121 | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | Zero-Shot (No CoT)       | 0.678864 | 0.700000 | 56.670       | 5456.459 | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | Zero-Shot CoT            | 0.702140 | 0.720000 | 1249.020     | 5458.939 | 4352.972 | balanced-held-out-test |
