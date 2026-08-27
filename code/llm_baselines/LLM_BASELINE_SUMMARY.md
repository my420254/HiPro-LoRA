# Strict LLM Baseline Summary

Generated: 2026-06-02 05:23:34 UTC

## Scope

This file is the single summary for the strict LLM baseline rerun. LLM rows come only from `llm_baselines/few_shot_results/llm_fewshot_results.csv` after filtering `Protocol == balanced-held-out-test`. The main-result rows are recomputed from `table2_strict/results/*_main_table2_strict_results.csv`.

The internal method name `LoRA-Ours` is the paper-facing `HiPro-LoRA` result.

## Key Conclusions

- The strict LLM matrix is complete: Qwen and Llama across 3 datasets, 0/1/3/5-shot, and CoT/No-CoT.
- The archived mixed-protocol LLM markdown should not be used for the final LLM claim because it contains old `original-baseline` rows and is not a single strict official-test protocol.
- With the selected main rows, HiPro-LoRA leads all strict LLM baselines on SMP2020, TweetEval; strict LLM baselines lead on SST-5.
- With the best main `LoRA-Ours` N per dataset, HiPro-LoRA leads on SMP2020, TweetEval; strict LLM baselines lead on SST-5.

## Final Verdict Table

| Dataset   | Selected Ours N | Selected Ours Macro-F1 | Best Ours N | Best Ours Macro-F1 | Best Strict LLM                                | Best LLM Macro-F1 | Selected Margin | Best-Ours Margin | Strict Conclusion   |
| --------- | --------------- | ---------------------- | ----------- | ------------------ | ---------------------------------------------- | ----------------- | --------------- | ---------------- | ------------------- |
| SMP2020   | 1000            | 0.699337               | 2000        | 0.703167           | Qwen2.5-7B-Instruct / Zero-Shot CoT            | 0.642363          | +0.056974       | +0.060804        | Ours leads all LLMs |
| SST-5     | 1150            | 0.480854               | 2300        | 0.512315           | Qwen2.5-7B-Instruct / Balanced 3-Shot (No CoT) | 0.559433          | -0.078579       | -0.047118        | Strict LLM leads    |
| TweetEval | 1000            | 0.751062               | 1000        | 0.751062           | Qwen2.5-7B-Instruct / Balanced 3-Shot CoT      | 0.735711          | +0.015351       | +0.015351        | Ours leads all LLMs |

## TweetEval Key Check

These are the three Qwen rows that must not exceed the current main HiPro-LoRA result. They do not exceed it under the current Table 2 result CSV.

| Dataset   | LLM Setting                           | Method                   | HiPro-LoRA Macro-F1 | LLM Macro-F1 | Delta HiPro-LLM | Verdict    |
| --------- | ------------------------------------- | ------------------------ | ------------------- | ------------ | --------------- | ---------- |
| TweetEval | Qwen2.5-7B-Instruct / 1-shot / No CoT | Balanced 1-Shot (No CoT) | 0.751062            | 0.720157     | +0.030906       | Ours leads |
| TweetEval | Qwen2.5-7B-Instruct / 3-shot / CoT    | Balanced 3-Shot CoT      | 0.751062            | 0.735711     | +0.015351       | Ours leads |
| TweetEval | Qwen2.5-7B-Instruct / 5-shot / CoT    | Balanced 5-Shot CoT      | 0.751062            | 0.726589     | +0.024473       | Ours leads |

## Coverage

| Item                        | Value                                                    |
| --------------------------- | -------------------------------------------------------- |
| Datasets                    | SMP2020, SST-5, TweetEval                                |
| Models                      | Qwen2.5-7B-Instruct, Meta-Llama-3.1-8B-Instruct-bnb-4bit |
| Shots                       | 0-shot, 1-shot, 3-shot, 5-shot                           |
| Prompt styles               | No CoT, CoT                                              |
| Strict rows                 | 48 / 48                                                  |
| Missing strict combinations | None                                                     |

## Compared And Not Compared

| Scope                                                            | Status                            | Interpretation                                                                                                   |
| ---------------------------------------------------------------- | --------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Current strict 0/1/3/5-shot x CoT/No-CoT x 2 models x 3 datasets | Compared now                      | All 48 strict combinations are present in the CSV and compared below.                                            |
| Original 0/1-shot LLM rows                                       | Rerun/replaced                    | The old rows used a legacy validation-style/original-baseline protocol and should not be mixed with strict rows. |
| Original 3/5-shot LLM rows                                       | Not covered in the old manuscript | These are new strict held-out results.                                                                           |

## Main LoRA-Ours / HiPro-LoRA Rows Used For Comparison

| Dataset   | N    | Internal Method | Paper Label | Best (Seed/F1)    | Macro-F1            | Accuracy            | Train_Time_Sec         | Inference_Time_Sec  | Peak_Memory_MB           | Params_M            |
| --------- | ---- | --------------- | ----------- | ----------------- | ------------------- | ------------------- | ---------------------- | ------------------- | ------------------------ | ------------------- |
| SMP2020   | 1000 | LoRA-Ours       | HiPro-LoRA  | Seed 45: 0.7183   | 0.6993 (+/- 0.0152) | 0.6980 (+/- 0.0157) | 139.6887 (+/- 18.7591) | 1.0339 (+/- 0.1176) | 2108.1004 (+/- 1.1397)   | 6.8983 (+/- 0.0000) |
| SST-5     | 1150 | LoRA-Ours       | HiPro-LoRA  | Seed 1001: 0.5155 | 0.4809 (+/- 0.0223) | 0.4865 (+/- 0.0201) | 130.0519 (+/- 15.5012) | 1.3157 (+/- 0.2444) | 2257.2590 (+/- 141.1107) | 6.8968 (+/- 0.0000) |
| TweetEval | 1000 | LoRA-Ours       | HiPro-LoRA  | Seed 789: 0.7947  | 0.7511 (+/- 0.0290) | 0.7533 (+/- 0.0309) | 86.0906 (+/- 16.9814)  | 0.4044 (+/- 0.0407) | 2118.4495 (+/- 146.0020) | 6.8937 (+/- 0.0000) |

## Best Main LoRA-Ours Rows By Dataset

| Dataset   | N    | Internal Method | Paper Label | Best (Seed/F1)    | Macro-F1            | Accuracy            | Train_Time_Sec         | Inference_Time_Sec  | Peak_Memory_MB           | Params_M            |
| --------- | ---- | --------------- | ----------- | ----------------- | ------------------- | ------------------- | ---------------------- | ------------------- | ------------------------ | ------------------- |
| SMP2020   | 2000 | LoRA-Ours       | HiPro-LoRA  | Seed 789: 0.7121  | 0.7032 (+/- 0.0127) | 0.7000 (+/- 0.0115) | 142.4497 (+/- 11.7171) | 0.8989 (+/- 0.1189) | 2079.8941 (+/- 2.4595)   | 6.8983 (+/- 0.0000) |
| SST-5     | 2300 | LoRA-Ours       | HiPro-LoRA  | Seed 1001: 0.5377 | 0.5123 (+/- 0.0169) | 0.5150 (+/- 0.0183) | 169.6591 (+/- 19.5144) | 1.4670 (+/- 0.2419) | 2347.1638 (+/- 108.2246) | 6.8968 (+/- 0.0000) |
| TweetEval | 1000 | LoRA-Ours       | HiPro-LoRA  | Seed 789: 0.7947  | 0.7511 (+/- 0.0290) | 0.7533 (+/- 0.0309) | 86.0906 (+/- 16.9814)  | 0.4044 (+/- 0.0407) | 2118.4495 (+/- 146.0020) | 6.8937 (+/- 0.0000) |

## Rows Where A Strict LLM Beats The Selected Main HiPro-LoRA Row

| Dataset | Ours N | Ours Macro-F1 | Model                               | Shot   | Prompt | LLM Macro-F1 | Delta Ours-LLM | Outcome   | LLM Accuracy | LLM Time ms |
| ------- | ------ | ------------- | ----------------------------------- | ------ | ------ | ------------ | -------------- | --------- | ------------ | ----------- |
| SST-5   | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 0-shot | No CoT | 0.533729     | -0.052876      | LLM leads | 0.537500     | 57.468      |
| SST-5   | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 1-shot | No CoT | 0.549135     | -0.068281      | LLM leads | 0.555000     | 84.355      |
| SST-5   | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 3-shot | No CoT | 0.559433     | -0.078579      | LLM leads | 0.575000     | 71.168      |
| SST-5   | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 3-shot | CoT    | 0.488039     | -0.007186      | LLM leads | 0.507500     | 1449.966    |
| SST-5   | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 5-shot | No CoT | 0.498340     | -0.017487      | LLM leads | 0.527500     | 88.832      |
| SST-5   | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 5-shot | CoT    | 0.481933     | -0.001079      | LLM leads | 0.502500     | 1541.376    |
| SST-5   | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | CoT    | 0.501670     | -0.020817      | LLM leads | 0.510000     | 2306.006    |
| SST-5   | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | No CoT | 0.487317     | -0.006463      | LLM leads | 0.520000     | 74.954      |
| SST-5   | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | CoT    | 0.527417     | -0.046564      | LLM leads | 0.537500     | 2256.949    |
| SST-5   | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | CoT    | 0.525307     | -0.044453      | LLM leads | 0.532500     | 2317.387    |
| SST-5   | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | CoT    | 0.526627     | -0.045774      | LLM leads | 0.532500     | 2364.180    |

## Strict LLM Result Table

| Dataset   | Model                               | Shot   | Prompt | Method                   | Macro-F1 | Accuracy | Inference_Time_ms | Peak_Memory_MB | Params_M | Protocol               |
| --------- | ----------------------------------- | ------ | ------ | ------------------------ | -------- | -------- | ----------------- | -------------- | -------- | ---------------------- |
| SMP2020   | Qwen2.5-7B-Instruct                 | 0-shot | No CoT | Zero-Shot (No CoT)       | 0.575081 | 0.593333 | 58.862            | 5470.860       | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | 0-shot | CoT    | Zero-Shot CoT            | 0.642363 | 0.643333 | 1586.813          | 5475.425       | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | 1-shot | No CoT | Balanced 1-Shot (No CoT) | 0.633873 | 0.646667 | 84.483            | 5500.433       | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | 1-shot | CoT    | Balanced 1-Shot CoT      | 0.615696 | 0.620000 | 1454.759          | 5503.219       | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | 3-shot | No CoT | Balanced 3-Shot (No CoT) | 0.508967 | 0.533333 | 86.580            | 5577.522       | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | 3-shot | CoT    | Balanced 3-Shot CoT      | 0.634366 | 0.636667 | 1255.199          | 5590.235       | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | 5-shot | No CoT | Balanced 5-Shot (No CoT) | 0.486412 | 0.526667 | 109.446           | 5646.732       | 4352.972 | balanced-held-out-test |
| SMP2020   | Qwen2.5-7B-Instruct                 | 5-shot | CoT    | Balanced 5-Shot CoT      | 0.626374 | 0.636667 | 1278.530          | 5667.438       | 4352.972 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | No CoT | Zero-Shot (No CoT)       | 0.437706 | 0.453333 | 74.105            | 5619.932       | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | CoT    | Zero-Shot CoT            | 0.403853 | 0.420000 | 2843.023          | 5626.276       | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | No CoT | Balanced 1-Shot (No CoT) | 0.563698 | 0.580000 | 111.017           | 5665.271       | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | CoT    | Balanced 1-Shot CoT      | 0.538536 | 0.543333 | 2890.603          | 5672.133       | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | No CoT | Balanced 3-Shot (No CoT) | 0.545939 | 0.573333 | 112.368           | 5815.481       | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | CoT    | Balanced 3-Shot CoT      | 0.487760 | 0.490000 | 3235.204          | 5822.060       | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | No CoT | Balanced 5-Shot (No CoT) | 0.554083 | 0.583333 | 143.712           | 5924.686       | 4540.600 | balanced-held-out-test |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | CoT    | Balanced 5-Shot CoT      | 0.527531 | 0.543333 | 2339.353          | 5931.264       | 4540.600 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | 0-shot | No CoT | Zero-Shot (No CoT)       | 0.533729 | 0.537500 | 57.468            | 5460.669       | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | 0-shot | CoT    | Zero-Shot CoT            | 0.468174 | 0.492500 | 1344.914          | 5463.119       | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | 1-shot | No CoT | Balanced 1-Shot (No CoT) | 0.549135 | 0.555000 | 84.355            | 5487.116       | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | 1-shot | CoT    | Balanced 1-Shot CoT      | 0.478211 | 0.505000 | 1489.689          | 5490.597       | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | 3-shot | No CoT | Balanced 3-Shot (No CoT) | 0.559433 | 0.575000 | 71.168            | 5524.415       | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | 3-shot | CoT    | Balanced 3-Shot CoT      | 0.488039 | 0.507500 | 1449.966          | 5536.817       | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | 5-shot | No CoT | Balanced 5-Shot (No CoT) | 0.498340 | 0.527500 | 88.832            | 5564.980       | 4352.972 | balanced-held-out-test |
| SST-5     | Qwen2.5-7B-Instruct                 | 5-shot | CoT    | Balanced 5-Shot CoT      | 0.481933 | 0.502500 | 1541.376          | 5577.495       | 4352.972 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | No CoT | Zero-Shot (No CoT)       | 0.373446 | 0.420000 | 71.144            | 5598.149       | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | CoT    | Zero-Shot CoT            | 0.501670 | 0.510000 | 2306.006          | 5601.532       | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | No CoT | Balanced 1-Shot (No CoT) | 0.487317 | 0.520000 | 74.954            | 5634.312       | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | CoT    | Balanced 1-Shot CoT      | 0.527417 | 0.537500 | 2256.949          | 5637.695       | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | No CoT | Balanced 3-Shot (No CoT) | 0.446999 | 0.470000 | 89.419            | 5699.919       | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | CoT    | Balanced 3-Shot CoT      | 0.525307 | 0.532500 | 2317.387          | 5704.232       | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | No CoT | Balanced 5-Shot (No CoT) | 0.359314 | 0.395000 | 105.724           | 5757.152       | 4540.600 | balanced-held-out-test |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | CoT    | Balanced 5-Shot CoT      | 0.526627 | 0.532500 | 2364.180          | 5760.661       | 4540.600 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | 0-shot | No CoT | Zero-Shot (No CoT)       | 0.678864 | 0.700000 | 56.670            | 5456.459       | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | 0-shot | CoT    | Zero-Shot CoT            | 0.702140 | 0.720000 | 1249.020          | 5458.939       | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | 1-shot | No CoT | Balanced 1-Shot (No CoT) | 0.720157 | 0.726667 | 73.519            | 5476.571       | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | 1-shot | CoT    | Balanced 1-Shot CoT      | 0.716129 | 0.726667 | 1286.149          | 5478.610       | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | 3-shot | No CoT | Balanced 3-Shot (No CoT) | 0.690341 | 0.686667 | 68.204            | 5502.274       | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | 3-shot | CoT    | Balanced 3-Shot CoT      | 0.735711 | 0.740000 | 1180.434          | 5516.115       | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | 5-shot | No CoT | Balanced 5-Shot (No CoT) | 0.690165 | 0.693333 | 79.253            | 5535.097       | 4352.972 | balanced-held-out-test |
| TweetEval | Qwen2.5-7B-Instruct                 | 5-shot | CoT    | Balanced 5-Shot CoT      | 0.726589 | 0.733333 | 1189.597          | 5548.121       | 4352.972 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | No CoT | Zero-Shot (No CoT)       | 0.426705 | 0.460000 | 83.435            | 5592.001       | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | CoT    | Zero-Shot CoT            | 0.694595 | 0.706667 | 2315.316          | 5595.260       | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | No CoT | Balanced 1-Shot (No CoT) | 0.519980 | 0.586667 | 71.334            | 5617.464       | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | CoT    | Balanced 1-Shot CoT      | 0.702304 | 0.700000 | 2505.230          | 5620.722       | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | No CoT | Balanced 3-Shot (No CoT) | 0.522294 | 0.613333 | 80.209            | 5667.463       | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | CoT    | Balanced 3-Shot CoT      | 0.678443 | 0.680000 | 2763.946          | 5670.972       | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | No CoT | Balanced 5-Shot (No CoT) | 0.584915 | 0.633333 | 91.231            | 5713.515       | 4540.600 | balanced-held-out-test |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | CoT    | Balanced 5-Shot CoT      | 0.652228 | 0.653333 | 2894.897          | 5718.023       | 4540.600 | balanced-held-out-test |

## Ours Vs Every Strict LLM Row

| Dataset   | Ours N | Ours Macro-F1 | Model                               | Shot   | Prompt | LLM Macro-F1 | Delta Ours-LLM | Outcome    | LLM Accuracy | LLM Time ms |
| --------- | ------ | ------------- | ----------------------------------- | ------ | ------ | ------------ | -------------- | ---------- | ------------ | ----------- |
| SMP2020   | 1000   | 0.699337      | Qwen2.5-7B-Instruct                 | 0-shot | No CoT | 0.575081     | +0.124255      | Ours leads | 0.593333     | 58.862      |
| SMP2020   | 1000   | 0.699337      | Qwen2.5-7B-Instruct                 | 0-shot | CoT    | 0.642363     | +0.056974      | Ours leads | 0.643333     | 1586.813    |
| SMP2020   | 1000   | 0.699337      | Qwen2.5-7B-Instruct                 | 1-shot | No CoT | 0.633873     | +0.065464      | Ours leads | 0.646667     | 84.483      |
| SMP2020   | 1000   | 0.699337      | Qwen2.5-7B-Instruct                 | 1-shot | CoT    | 0.615696     | +0.083641      | Ours leads | 0.620000     | 1454.759    |
| SMP2020   | 1000   | 0.699337      | Qwen2.5-7B-Instruct                 | 3-shot | No CoT | 0.508967     | +0.190370      | Ours leads | 0.533333     | 86.580      |
| SMP2020   | 1000   | 0.699337      | Qwen2.5-7B-Instruct                 | 3-shot | CoT    | 0.634366     | +0.064971      | Ours leads | 0.636667     | 1255.199    |
| SMP2020   | 1000   | 0.699337      | Qwen2.5-7B-Instruct                 | 5-shot | No CoT | 0.486412     | +0.212925      | Ours leads | 0.526667     | 109.446     |
| SMP2020   | 1000   | 0.699337      | Qwen2.5-7B-Instruct                 | 5-shot | CoT    | 0.626374     | +0.072963      | Ours leads | 0.636667     | 1278.530    |
| SMP2020   | 1000   | 0.699337      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | No CoT | 0.437706     | +0.261630      | Ours leads | 0.453333     | 74.105      |
| SMP2020   | 1000   | 0.699337      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | CoT    | 0.403853     | +0.295484      | Ours leads | 0.420000     | 2843.023    |
| SMP2020   | 1000   | 0.699337      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | No CoT | 0.563698     | +0.135639      | Ours leads | 0.580000     | 111.017     |
| SMP2020   | 1000   | 0.699337      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | CoT    | 0.538536     | +0.160801      | Ours leads | 0.543333     | 2890.603    |
| SMP2020   | 1000   | 0.699337      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | No CoT | 0.545939     | +0.153398      | Ours leads | 0.573333     | 112.368     |
| SMP2020   | 1000   | 0.699337      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | CoT    | 0.487760     | +0.211577      | Ours leads | 0.490000     | 3235.204    |
| SMP2020   | 1000   | 0.699337      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | No CoT | 0.554083     | +0.145254      | Ours leads | 0.583333     | 143.712     |
| SMP2020   | 1000   | 0.699337      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | CoT    | 0.527531     | +0.171806      | Ours leads | 0.543333     | 2339.353    |
| SST-5     | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 0-shot | No CoT | 0.533729     | -0.052876      | LLM leads  | 0.537500     | 57.468      |
| SST-5     | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 0-shot | CoT    | 0.468174     | +0.012679      | Ours leads | 0.492500     | 1344.914    |
| SST-5     | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 1-shot | No CoT | 0.549135     | -0.068281      | LLM leads  | 0.555000     | 84.355      |
| SST-5     | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 1-shot | CoT    | 0.478211     | +0.002642      | Ours leads | 0.505000     | 1489.689    |
| SST-5     | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 3-shot | No CoT | 0.559433     | -0.078579      | LLM leads  | 0.575000     | 71.168      |
| SST-5     | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 3-shot | CoT    | 0.488039     | -0.007186      | LLM leads  | 0.507500     | 1449.966    |
| SST-5     | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 5-shot | No CoT | 0.498340     | -0.017487      | LLM leads  | 0.527500     | 88.832      |
| SST-5     | 1150   | 0.480854      | Qwen2.5-7B-Instruct                 | 5-shot | CoT    | 0.481933     | -0.001079      | LLM leads  | 0.502500     | 1541.376    |
| SST-5     | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | No CoT | 0.373446     | +0.107407      | Ours leads | 0.420000     | 71.144      |
| SST-5     | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | CoT    | 0.501670     | -0.020817      | LLM leads  | 0.510000     | 2306.006    |
| SST-5     | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | No CoT | 0.487317     | -0.006463      | LLM leads  | 0.520000     | 74.954      |
| SST-5     | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | CoT    | 0.527417     | -0.046564      | LLM leads  | 0.537500     | 2256.949    |
| SST-5     | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | No CoT | 0.446999     | +0.033854      | Ours leads | 0.470000     | 89.419      |
| SST-5     | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | CoT    | 0.525307     | -0.044453      | LLM leads  | 0.532500     | 2317.387    |
| SST-5     | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | No CoT | 0.359314     | +0.121540      | Ours leads | 0.395000     | 105.724     |
| SST-5     | 1150   | 0.480854      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | CoT    | 0.526627     | -0.045774      | LLM leads  | 0.532500     | 2364.180    |
| TweetEval | 1000   | 0.751062      | Qwen2.5-7B-Instruct                 | 0-shot | No CoT | 0.678864     | +0.072198      | Ours leads | 0.700000     | 56.670      |
| TweetEval | 1000   | 0.751062      | Qwen2.5-7B-Instruct                 | 0-shot | CoT    | 0.702140     | +0.048922      | Ours leads | 0.720000     | 1249.020    |
| TweetEval | 1000   | 0.751062      | Qwen2.5-7B-Instruct                 | 1-shot | No CoT | 0.720157     | +0.030906      | Ours leads | 0.726667     | 73.519      |
| TweetEval | 1000   | 0.751062      | Qwen2.5-7B-Instruct                 | 1-shot | CoT    | 0.716129     | +0.034934      | Ours leads | 0.726667     | 1286.149    |
| TweetEval | 1000   | 0.751062      | Qwen2.5-7B-Instruct                 | 3-shot | No CoT | 0.690341     | +0.060721      | Ours leads | 0.686667     | 68.204      |
| TweetEval | 1000   | 0.751062      | Qwen2.5-7B-Instruct                 | 3-shot | CoT    | 0.735711     | +0.015351      | Ours leads | 0.740000     | 1180.434    |
| TweetEval | 1000   | 0.751062      | Qwen2.5-7B-Instruct                 | 5-shot | No CoT | 0.690165     | +0.060897      | Ours leads | 0.693333     | 79.253      |
| TweetEval | 1000   | 0.751062      | Qwen2.5-7B-Instruct                 | 5-shot | CoT    | 0.726589     | +0.024473      | Ours leads | 0.733333     | 1189.597    |
| TweetEval | 1000   | 0.751062      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | No CoT | 0.426705     | +0.324357      | Ours leads | 0.460000     | 83.435      |
| TweetEval | 1000   | 0.751062      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-shot | CoT    | 0.694595     | +0.056467      | Ours leads | 0.706667     | 2315.316    |
| TweetEval | 1000   | 0.751062      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | No CoT | 0.519980     | +0.231082      | Ours leads | 0.586667     | 71.334      |
| TweetEval | 1000   | 0.751062      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 1-shot | CoT    | 0.702304     | +0.048759      | Ours leads | 0.700000     | 2505.230    |
| TweetEval | 1000   | 0.751062      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | No CoT | 0.522294     | +0.228769      | Ours leads | 0.613333     | 80.209      |
| TweetEval | 1000   | 0.751062      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 3-shot | CoT    | 0.678443     | +0.072619      | Ours leads | 0.680000     | 2763.946    |
| TweetEval | 1000   | 0.751062      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | No CoT | 0.584915     | +0.166148      | Ours leads | 0.633333     | 91.231      |
| TweetEval | 1000   | 0.751062      | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 5-shot | CoT    | 0.652228     | +0.098834      | Ours leads | 0.653333     | 2894.897    |

## Time, Memory, And Parameter Accounting

- `Inference_Time_ms` in the LLM table is the average per-sample generation latency measured by `run_llm_fewshot.py`.
- `Inference_Time_Sec` in the main table is the classifier inference-time field from the main strict summaries.
- These timing fields are useful cost indicators, but they are not the same system-level measurement. CoT generation is expected to be slower than direct classification.
- `Peak_Memory_MB` and `Params_M` are reported from each runner and should be interpreted under each runner's model-loading path and quantization settings.

## Strict Entry Points

| File                     | Role                                                                      |
| ------------------------ | ------------------------------------------------------------------------- |
| run_llm_fewshot.py       | Single strict runner for one or more models/datasets/shots/prompt styles. |
| run_llm_aligned_all.py   | Batch launcher for the complete aligned strict matrix.                    |
| summarize_llm_results.py | Generates this summary without requiring the external `tabulate` package. |

Old per-model scripts from the non-strict workflow are not kept in this strict baseline directory. The three files above are the only strict LLM entry points.

## Figure Script Note

`../create_Image.py` uses the strict CSV at `llm_baselines/few_shot_results/llm_fewshot_results.csv` for the LLM comparison figure. Archived mixed-protocol outputs are not result sources.
