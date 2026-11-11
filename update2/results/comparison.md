# Runs side by side (test split, all pairs)

| run | backbone | trainable params | lambda_sw / lambda_dur | steps | train min | mean F1 | sigma | F1 at 0.5 | dur acc | macro F1 by type | F1 tag | F1 inter | F1 intra |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| naive baseline (Update 1) | P(switch \| token) |  |  |  |  | 0.644 | 0.068 |  | 0.617 |  |  |  |  |
| Model A (XLM-R) | xlm-roberta-base | 277M | 1 / 1 | 5948 | 40.6 | 0.784 | 0.070 | 0.714 | 0.694 | 0.786 | 0.842 | 0.833 | 0.682 |
| Model B (Qwen2.5-0.5B) | Qwen/Qwen2.5-0.5B | 494M | 1 / 1 | 5978 | 129.2 | 0.805 | 0.064 | 0.745 | 0.704 | 0.807 | 0.859 | 0.853 | 0.710 |
