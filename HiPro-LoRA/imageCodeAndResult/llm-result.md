| Dataset   | Model                               | Method                   | Macro-F1 | Accuracy | Inference_Time_ms | Peak_Memory_MB | Params_M    |
| --------- | ----------------------------------- | ------------------------ | -------- | -------- | ----------------- | -------------- | ----------- |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-Shot CoT               | 0.289526 | 0.306667 | 11227.885655      | 5617.280762    | 4540.60032  |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot CoT      | 0.535362 | 0.546667 | 9461.083745       | 5782.325195    | 4540.60032  |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-Shot CoT               | 0.151759 | 0.205000 | 11462.186949      | 5596.766113    | 4540.60032  |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot CoT      | 0.472344 | 0.467500 | 7867.084808       | 5664.228027    | 4540.60032  |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | 0-Shot CoT               | 0.246421 | 0.360000 | 11587.022618      | 5588.638184    | 4540.60032  |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot CoT      | 0.292107 | 0.393333 | 11943.231964      | 5622.144043    | 4540.60032  |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot (No CoT)       | 0.457400 | 0.496667 | 707.076400        | 5614.976074    | 4540.60032  |
| SMP2020   | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot (No CoT) | 0.540008 | 0.576667 | 1160.322696       | 5656.403320    | 4540.60032  |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot (No CoT)       | 0.423725 | 0.467500 | 631.764177        | 5598.456543    | 4540.60032  |
| SST-5     | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot (No CoT) | 0.534590 | 0.550000 | 998.432533        | 5627.129395    | 4540.60032  |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Zero-Shot (No CoT)       | 0.393809 | 0.433333 | 587.555526        | 5591.074219    | 4540.60032  |
| TweetEval | Meta-Llama-3.1-8B-Instruct-bnb-4bit | Balanced 1-Shot (No CoT) | 0.551597 | 0.600000 | 887.748928        | 5607.974609    | 4540.60032  |
| SMP2020   | Qwen2.5-7B-Instruct                 | 0-Shot CoT               | 0.366608 | 0.383333 | 10904.704456      | 5626.213867    | 4352.972288 |
| SMP2020   | Qwen2.5-7B-Instruct                 | Balanced 1-Shot CoT      | 0.608536 | 0.610000 | 7183.455234       | 5680.214355    | 4352.972288 |
| SST-5     | Qwen2.5-7B-Instruct                 | 0-Shot CoT               | 0.170038 | 0.217500 | 11598.404477      | 5620.231445    | 4352.972288 |
| SST-5     | Qwen2.5-7B-Instruct                 | Balanced 1-Shot CoT      | 0.436023 | 0.435000 | 10915.523761      | 5670.977051    | 4352.972288 |
| TweetEval | Qwen2.5-7B-Instruct                 | 0-Shot CoT               | 0.270171 | 0.340000 | 11472.599413      | 5614.765625    | 4352.972288 |
| TweetEval | Qwen2.5-7B-Instruct                 | Balanced 1-Shot CoT      | 0.543931 | 0.540000 | 10439.191292      | 5639.547363    | 4352.972288 |
| SMP2020   | Qwen2.5-7B-Instruct                 | Zero-Shot (No CoT)       | 0.595514 | 0.613333 | 578.711477        | 5627.056641    | 4352.972288 |
| SMP2020   | Qwen2.5-7B-Instruct                 | Balanced 1-Shot (No CoT) | 0.652116 | 0.660000 | 922.128348        | 5653.681152    | 4352.972288 |
| SST-5     | Qwen2.5-7B-Instruct                 | Zero-Shot (No CoT)       | 0.547008 | 0.545000 | 551.801170        | 5622.377930    | 4352.972288 |
| SST-5     | Qwen2.5-7B-Instruct                 | Balanced 1-Shot (No CoT) | 0.550520 | 0.555000 | 829.432578        | 5642.103516    | 4352.972288 |
| TweetEval | Qwen2.5-7B-Instruct                 | Zero-Shot (No CoT)       | 0.629461 | 0.646667 | 531.885117        | 5617.536133    | 4352.972288 |
| TweetEval | Qwen2.5-7B-Instruct                 | Balanced 1-Shot (No CoT) | 0.660009 | 0.666667 | 698.912423        | 5631.334473    | 4352.972288 |