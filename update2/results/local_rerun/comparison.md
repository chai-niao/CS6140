# Runs side by side (test split, all pairs)

| run | backbone | trainable params | lambda_sw / lambda_dur | steps | train min | mean F1 | sigma | F1 at 0.5 | dur acc | macro F1 by type | F1 tag | F1 inter | F1 intra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive baseline (Update 1) | P(switch \| token) |  |  |  |  | 0.644 | 0.068 |  | 0.617 |  |  |  |  |
| Model A (XLM-R) | xlm-roberta-base | 277M | 1 / 1 | 5948 | 10.2 | 0.784 | 0.070 | 0.713 | 0.694 | 0.786 | 0.843 | 0.834 | 0.680 |
| Model B (Qwen2.5-0.5B) | Qwen/Qwen2.5-0.5B | 494M | 1 / 1 | 5978 | 59.1 | 0.807 | 0.063 | 0.743 | 0.704 | 0.809 | 0.860 | 0.854 | 0.711 |
