Streaming consistency on 5 conversations (qwen, fp32, cuda):
| conv_id | words | subwords | max |P_stream - P_window| |
|---|---|---|---|
| Arabic_eng_4610 | 38 | 77 | 2.61e-07 |
| Arabic_eng_5845 | 67 | 97 | 8.46e-07 |
| Arabic_eng_446 | 67 | 128 | 1.70e-06 |
| Arabic_eng_806 | 33 | 52 | 5.36e-06 |
| Arabic_eng_4251 | 66 | 111 | 1.25e-06 |

worst difference 5.36e-06: identical up to floating point noise
