# speed_xlmr_fp16: test split

| pair | switch_rate_% | threshold | P | R | F1 | F1_at_0.5 | dur_acc | F1_macro_by_type |
|---|---|---|---|---|---|---|---|---|
| Hindi_eng | 24.7 | 0.5 | 0.3372 | 0.7076 | 0.4567 | 0.4567 | 0.4727 | 0.4537 |
| Spanish_eng | 12.4 | 0.85 | 0.7755 | 0.7015 | 0.7367 | 0.6352 | 0.5846 | 0.7062 |

Mean Anticipatory F1 = 0.5967, sigma = 0.1400 (min 0.457 Hindi_eng, max 0.737 Spanish_eng); mean F1 at threshold 0.5 = 0.5459; duration accuracy at true switches = 0.5287; macro F1 by switch type = 0.5799.
