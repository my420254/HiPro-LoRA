# Table 3 Final

Fine-grained AHSP component ablation under the strict held-out test protocol.
Values are Macro-F1 (mean ± std over 5 seeds). Bold marks the best result in each column.

## SST-5

| Method | N=1150 | N=2300 |
| --- | --- | --- |
| HiPro-LoRA (Full) | **0.4778 ± 0.0218** | 0.5123 ± 0.0169 |
| AHSP-OnlyMax | 0.4628 ± 0.0300 | 0.3435 ± 0.2385 |
| AHSP-OnlyMean | 0.4688 ± 0.0271 | 0.5000 ± 0.0224 |
| AHSP-OnlyAttn | 0.4643 ± 0.0273 | **0.5153 ± 0.0383** |
| AHSP-Max+Mean | 0.4616 ± 0.0248 | 0.4172 ± 0.1842 |
| AHSP-Max+Attn | 0.4667 ± 0.0295 | 0.4209 ± 0.1865 |
| AHSP-Mean+Attn | 0.4647 ± 0.0202 | 0.5013 ± 0.0260 |
| No-AHSP (NoHSP) | 0.4534 ± 0.0238 | 0.2376 ± 0.2303 |
| NoMem | 0.4689 ± 0.0231 | 0.5056 ± 0.0211 |

## TweetEval

| Method | N=1000 | N=2000 |
| --- | --- | --- |
| HiPro-LoRA (Full) | **0.7511 ± 0.0290** | **0.7401 ± 0.0279** |
| AHSP-OnlyMax | 0.7373 ± 0.0457 | 0.7139 ± 0.0330 |
| AHSP-OnlyMean | 0.7449 ± 0.0485 | 0.7088 ± 0.0312 |
| AHSP-OnlyAttn | 0.7296 ± 0.0411 | 0.7167 ± 0.0290 |
| AHSP-Max+Mean | 0.7265 ± 0.0251 | 0.7192 ± 0.0228 |
| AHSP-Max+Attn | 0.7448 ± 0.0259 | 0.7184 ± 0.0249 |
| AHSP-Mean+Attn | 0.7386 ± 0.0137 | 0.7145 ± 0.0312 |
| No-AHSP (NoHSP) | 0.7307 ± 0.0461 | 0.7198 ± 0.0168 |
| NoMem | 0.7258 ± 0.0118 | 0.7158 ± 0.0202 |

## SMP2020

| Method | N=1000 | N=2000 |
| --- | --- | --- |
| HiPro-LoRA (Full) | **0.6993 ± 0.0152** | 0.7032 ± 0.0127 |
| AHSP-OnlyMax | 0.6906 ± 0.0193 | 0.6960 ± 0.0065 |
| AHSP-OnlyMean | 0.6880 ± 0.0227 | 0.6798 ± 0.0098 |
| AHSP-OnlyAttn | 0.6835 ± 0.0267 | 0.6961 ± 0.0144 |
| AHSP-Max+Mean | 0.6960 ± 0.0157 | **0.7134 ± 0.0091** |
| AHSP-Max+Attn | 0.6910 ± 0.0206 | 0.7115 ± 0.0063 |
| AHSP-Mean+Attn | 0.6687 ± 0.0191 | 0.6867 ± 0.0179 |
| No-AHSP (NoHSP) | 0.6887 ± 0.0117 | 0.6884 ± 0.0169 |
| NoMem | 0.6774 ± 0.0161 | 0.6996 ± 0.0110 |
