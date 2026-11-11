# speed_xlmr_bf16: test split

| pair | switch_rate_% | threshold | P | R | F1 | F1_at_0.5 | dur_acc | F1_macro_by_type |
|---|---|---|---|---|---|---|---|---|
| Hindi_eng | 24.7 | 0.55 | 0.3716 | 0.6076 | 0.4612 | 0.4563 | 0.4561 | 0.4567 |
| Spanish_eng | 12.4 | 0.8 | 0.777 | 0.7077 | 0.7407 | 0.6703 | 0.5692 | 0.71 |

Mean Anticipatory F1 = 0.6009, sigma = 0.1398 (min 0.461 Hindi_eng, max 0.741 Spanish_eng); mean F1 at threshold 0.5 = 0.5633; duration accuracy at true switches = 0.5127; macro F1 by switch type = 0.5834.
