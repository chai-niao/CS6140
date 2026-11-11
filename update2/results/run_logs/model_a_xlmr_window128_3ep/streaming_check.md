Streaming consistency on 5 conversations (xlmr, fp32, cuda):
| conv_id | words | subwords | max |P_stream - P_window| |
|---|---|---|---|
| Arabic_eng_4610 | 38 | 65 | 4.62e-07 |
| Arabic_eng_5845 | 67 | 100 | 2.83e-07 |
| Arabic_eng_446 | 67 | 114 | 1.04e-06 |
| Arabic_eng_806 | 33 | 50 | 4.47e-07 |
| Arabic_eng_4251 | 66 | 111 | 6.56e-07 |

worst difference 1.04e-06: identical up to floating point noise
