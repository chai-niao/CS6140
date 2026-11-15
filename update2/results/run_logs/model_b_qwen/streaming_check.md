Streaming consistency on 5 conversations (qwen, fp32, cuda):
| conv_id | words | subwords | max |P_stream - P_window| |
|---|---|---|---|
| Arabic_eng_4610 | 38 | 77 | 3.35e-07 |
| Arabic_eng_5845 | 67 | 97 | 2.61e-07 |
| Arabic_eng_446 | 67 | 128 | 1.91e-06 |
| Arabic_eng_806 | 33 | 52 | 2.21e-06 |
| Arabic_eng_4251 | 66 | 111 | 2.95e-06 |

worst difference 2.95e-06: identical up to floating point noise
