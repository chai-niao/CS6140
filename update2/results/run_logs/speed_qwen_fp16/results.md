# speed_qwen_fp16: test split

| pair | switch_rate_% | threshold | P | R | F1 | F1_at_0.5 | dur_acc | F1_macro_by_type |
|---|---|---|---|---|---|---|---|---|
| Hindi_eng | 24.7 | 0.7 | 0.614 | 0.7424 | 0.6722 | 0.6367 | 0.6091 | 0.6693 |
| Spanish_eng | 12.4 | 0.7 | 0.8113 | 0.7938 | 0.8025 | 0.7606 | 0.6677 | 0.7825 |

Mean Anticipatory F1 = 0.7373, sigma = 0.0651 (min 0.672 Hindi_eng, max 0.802 Spanish_eng); mean F1 at threshold 0.5 = 0.6986; duration accuracy at true switches = 0.6384; macro F1 by switch type = 0.7259.
