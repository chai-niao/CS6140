Streaming consistency (qwen, fp32, cuda): 45 conversations, 2 random + the longest of each of 15 pairs; windows of 256 subwords, stride 128; the streaming prefix is capped at the window that scores the word.
| conv_id | words | subwords | windows | longest prefix | max |P_stream - P_window| |
|---|---|---|---|---|---|
| Arabic_eng_300 | 42 | 74 | 1 | 72 | 3.28e-06 |
| Arabic_eng_6062 | 14 | 26 | 1 | 24 | 1.19e-06 |
| Arabic_eng_7789 | 117 | 178 | 1 | 175 | 6.71e-07 |
| Hindi_eng_5700 | 67 | 229 | 1 | 227 | 1.61e-06 |
| Hindi_eng_2174 | 63 | 169 | 1 | 167 | 1.07e-06 |
| Hindi_eng_5339 | 230 | 658 | 5 | 256 | 1.51e-06 |
| Korean_eng_3465 | 103 | 181 | 1 | 179 | 1.46e-06 |
| Korean_eng_2681 | 108 | 185 | 1 | 180 | 1.34e-06 |
| Korean_eng_5159 | 116 | 204 | 1 | 202 | 2.38e-06 |
| Russian_eng_2139 | 45 | 96 | 1 | 87 | 6.56e-07 |
| Russian_eng_7054 | 17 | 33 | 1 | 31 | 9.91e-07 |
| Russian_eng_2897 | 161 | 302 | 2 | 255 | 1.78e-06 |
| Chinese_eng_5806 | 30 | 42 | 1 | 40 | 3.58e-07 |
| Chinese_eng_5623 | 56 | 69 | 1 | 67 | 9.54e-07 |
| Chinese_eng_4259 | 165 | 189 | 1 | 186 | 2.35e-06 |
| Can_Eng_66763 | 57 | 87 | 1 | 85 | 1.13e-06 |
| Can_Eng_23921 | 65 | 110 | 1 | 108 | 1.01e-06 |
| Can_Eng_60363 | 182 | 286 | 2 | 256 | 1.67e-06 |
| Japanese_eng_3879 | 83 | 111 | 1 | 109 | 1.13e-06 |
| Japanese_eng_7702 | 86 | 112 | 1 | 110 | 4.38e-06 |
| Japanese_eng_5397 | 219 | 323 | 2 | 256 | 5.72e-06 |
| Spanish_eng_4715 | 80 | 111 | 1 | 109 | 8.64e-07 |
| Spanish_eng_5878 | 43 | 63 | 1 | 61 | 1.37e-06 |
| Spanish_eng_3625 | 207 | 295 | 2 | 256 | 6.56e-07 |
| French_eng_4720 | 69 | 111 | 1 | 109 | 1.31e-06 |
| French_eng_5185 | 63 | 97 | 1 | 95 | 1.67e-06 |
| French_eng_7900 | 166 | 247 | 1 | 245 | 8.34e-07 |
| German_Eng_5140 | 28 | 37 | 1 | 35 | 1.88e-06 |
| German_Eng_79 | 95 | 149 | 1 | 147 | 1.37e-06 |
| German_Eng_671 | 151 | 230 | 1 | 228 | 1.59e-06 |
| Italian_eng_2005 | 83 | 121 | 1 | 119 | 1.01e-06 |
| Italian_eng_2983 | 16 | 24 | 1 | 22 | 2.38e-07 |
| Italian_eng_6606 | 165 | 278 | 2 | 256 | 1.94e-06 |
| Malay_eng_7810 | 44 | 100 | 1 | 98 | 8.94e-07 |
| Malay_eng_944 | 16 | 27 | 1 | 25 | 7.23e-07 |
| Malay_eng_3406 | 225 | 492 | 3 | 255 | 1.73e-06 |
| Philippines_eng_5479 | 59 | 107 | 1 | 104 | 7.75e-07 |
| Philippines_eng_7994 | 34 | 58 | 1 | 56 | 1.13e-06 |
| Philippines_eng_7810 | 169 | 278 | 2 | 256 | 1.40e-06 |
| Vietnamese_eng_169 | 60 | 68 | 1 | 66 | 3.22e-06 |
| Vietnamese_eng_2279 | 92 | 107 | 1 | 105 | 7.60e-07 |
| Vietnamese_eng_4726 | 270 | 312 | 2 | 256 | 5.30e-06 |
| German_French_5007 | 41 | 88 | 1 | 86 | 1.07e-06 |
| German_French_2360 | 85 | 127 | 1 | 126 | 7.30e-07 |
| German_French_2784 | 172 | 287 | 2 | 256 | 2.06e-06 |

10 of 45 conversations span more than one window. Worst difference 5.72e-06: identical up to floating point noise
