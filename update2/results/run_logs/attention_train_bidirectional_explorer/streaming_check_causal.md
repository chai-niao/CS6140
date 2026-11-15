Streaming consistency (xlmr, fp32, cuda): 45 conversations, 2 random + the longest of each of 15 pairs; windows of 256 subwords, stride 128; the streaming prefix is capped at the window that scores the word.
| conv_id | words | subwords | windows | longest prefix | max |P_stream - P_window| |
|---|---|---|---|---|---|
| Arabic_eng_300 | 42 | 71 | 1 | 69 | 2.09e-06 |
| Arabic_eng_6062 | 14 | 22 | 1 | 21 | 2.03e-06 |
| Arabic_eng_7789 | 117 | 172 | 1 | 168 | 1.26e-05 |
| Hindi_eng_5700 | 67 | 103 | 1 | 102 | 2.92e-06 |
| Hindi_eng_2174 | 63 | 78 | 1 | 77 | 1.45e-06 |
| Hindi_eng_5339 | 230 | 308 | 2 | 257 | 1.22e-05 |
| Korean_eng_3465 | 103 | 153 | 1 | 152 | 5.90e-06 |
| Korean_eng_2681 | 108 | 168 | 1 | 164 | 1.54e-05 |
| Korean_eng_5159 | 116 | 195 | 1 | 193 | 5.77e-06 |
| Russian_eng_2139 | 45 | 69 | 1 | 64 | 2.44e-06 |
| Russian_eng_7054 | 17 | 26 | 1 | 25 | 1.73e-06 |
| Russian_eng_2897 | 161 | 244 | 1 | 242 | 7.60e-06 |
| Chinese_eng_5806 | 30 | 45 | 1 | 43 | 1.91e-06 |
| Chinese_eng_5623 | 56 | 72 | 1 | 71 | 9.09e-07 |
| Chinese_eng_4259 | 165 | 206 | 1 | 204 | 5.66e-06 |
| Can_Eng_66763 | 57 | 82 | 1 | 81 | 1.25e-06 |
| Can_Eng_23921 | 65 | 110 | 1 | 109 | 4.82e-06 |
| Can_Eng_60363 | 182 | 257 | 2 | 256 | 4.44e-06 |
| Japanese_eng_3879 | 83 | 119 | 1 | 118 | 5.07e-06 |
| Japanese_eng_7702 | 86 | 112 | 1 | 111 | 5.36e-06 |
| Japanese_eng_5397 | 219 | 329 | 2 | 257 | 9.28e-06 |
| Spanish_eng_4715 | 80 | 114 | 1 | 113 | 4.44e-06 |
| Spanish_eng_5878 | 43 | 61 | 1 | 60 | 1.71e-06 |
| Spanish_eng_3625 | 207 | 282 | 2 | 256 | 9.75e-06 |
| French_eng_4720 | 69 | 106 | 1 | 105 | 4.65e-06 |
| French_eng_5185 | 63 | 99 | 1 | 98 | 3.64e-06 |
| French_eng_7900 | 166 | 239 | 1 | 238 | 3.87e-06 |
| German_Eng_5140 | 28 | 38 | 1 | 37 | 2.68e-06 |
| German_Eng_79 | 95 | 128 | 1 | 127 | 8.70e-06 |
| German_Eng_671 | 151 | 217 | 1 | 215 | 8.29e-06 |
| Italian_eng_2005 | 83 | 107 | 1 | 106 | 1.94e-06 |
| Italian_eng_2983 | 16 | 23 | 1 | 22 | 1.91e-06 |
| Italian_eng_6606 | 165 | 222 | 1 | 220 | 1.95e-06 |
| Malay_eng_7810 | 44 | 62 | 1 | 61 | 9.54e-07 |
| Malay_eng_944 | 16 | 18 | 1 | 17 | 6.56e-07 |
| Malay_eng_3406 | 225 | 589 | 4 | 257 | 6.32e-06 |
| Philippines_eng_5479 | 59 | 87 | 1 | 86 | 3.78e-06 |
| Philippines_eng_7994 | 34 | 60 | 1 | 59 | 3.61e-06 |
| Philippines_eng_7810 | 169 | 237 | 1 | 236 | 6.79e-06 |
| Vietnamese_eng_169 | 60 | 67 | 1 | 66 | 1.55e-06 |
| Vietnamese_eng_2279 | 92 | 106 | 1 | 105 | 5.07e-06 |
| Vietnamese_eng_4726 | 270 | 310 | 2 | 257 | 7.61e-06 |
| German_French_5007 | 41 | 80 | 1 | 79 | 5.90e-07 |
| German_French_2360 | 85 | 114 | 1 | 114 | 1.85e-06 |
| German_French_2784 | 172 | 249 | 1 | 248 | 3.22e-06 |

6 of 45 conversations span more than one window. Worst difference 1.54e-05: identical up to floating point noise
