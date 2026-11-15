# Loss weighting (Model A, 2,000 training conversations per pair, test split)

| run | lambda_switch | lambda_duration | class weights | mean F1 | sigma | F1 at 0.5 | dur acc | macro F1 by type | final switch loss | final duration loss |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 0 | yes | 0.754 | 0.072 | 0.685 | n/a | 0.756 | 0.310 | n/a |
| 0.5 | 1 | 0.5 | yes | 0.745 | 0.071 | 0.683 | 0.659 | 0.747 | 0.321 | 0.815 |
| 1 | 1 | 1 | yes | 0.735 | 0.074 | 0.682 | 0.661 | 0.738 | 0.336 | 0.815 |
| 2 | 1 | 2 | yes | 0.730 | 0.077 | 0.674 | 0.664 | 0.732 | 0.345 | 0.810 |
| no class weights | 1 | 1 | no | 0.733 | 0.074 | 0.717 | 0.683 | 0.735 | 0.202 | 0.764 |
| duration only | 0 | 1 | yes | n/a | n/a | n/a | 0.657 | n/a | n/a | 0.839 |
