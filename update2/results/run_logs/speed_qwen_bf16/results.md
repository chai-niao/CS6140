# speed_qwen_bf16: test split

| pair | switch_rate_% | threshold | P | R | F1 | F1_at_0.5 | dur_acc | F1_macro_by_type |
|---|---|---|---|---|---|---|---|---|
| Hindi_eng | 24.7 | 0.65 | 0.5986 | 0.8 | 0.6848 | 0.6395 | 0.6076 | 0.6808 |
| Spanish_eng | 12.4 | 0.7 | 0.8307 | 0.8 | 0.815 | 0.7631 | 0.6738 | 0.7965 |

Mean Anticipatory F1 = 0.7499, sigma = 0.0651 (min 0.685 Hindi_eng, max 0.815 Spanish_eng); mean F1 at threshold 0.5 = 0.7013; duration accuracy at true switches = 0.6407; macro F1 by switch type = 0.7387.
