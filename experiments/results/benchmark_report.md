# HUDS vs Random Sampling Benchmark Report

## Overview

- **Scenarios:** 9
- **Strategies:** huds, random
- **Total data points:** 148

## v2i_a_simple (vector_to_image)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 49 | 0.1376 | -0.2745 |
| 1 | 113 | 0.4440 | -0.0257 |
| 2 | 177 | 0.5825 | 0.2322 |
| 3 | 241 | 0.6565 | 0.4502 |
| 4 | 305 | 0.7097 | 0.6159 |
| 5 | 369 | 0.7439 | 0.7277 |

**Final step comparison:**

- **HUDS**: R2=0.7439, labeled=369, time=12.1s
- **RANDOM**: R2=0.7277, labeled=369, time=10.7s

## v2i_b_medium (vector_to_image)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 99 | 0.3706 | 0.0951 |
| 1 | 163 | 0.7076 | 0.2750 |
| 2 | 227 | 0.7980 | 0.3924 |
| 3 | 291 | 0.8284 | 0.4476 |
| 4 | 355 | 0.8481 | 0.4606 |
| 5 | 419 | 0.8655 | 0.4648 |
| 6 | 483 | 0.8845 | 0.4682 |
| 7 | 547 | 0.8922 | 0.4716 |
| 8 | 611 | 0.9013 | 0.4767 |

**Final step comparison:**

- **HUDS**: R2=0.9013, labeled=611, time=22.1s
- **RANDOM**: R2=0.4767, labeled=611, time=23.1s

## v2i_c_complex (vector_to_image)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 199 | 0.2030 | 0.2996 |
| 1 | 263 | 0.2158 | 0.4366 |
| 2 | 327 | 0.2800 | 0.4692 |
| 3 | 391 | N/A | 0.4916 |
| 4 | 455 | N/A | 0.5058 |
| 5 | 519 | N/A | 0.5141 |
| 6 | 583 | N/A | 0.5232 |
| 7 | 647 | N/A | 0.5302 |
| 8 | 711 | N/A | 0.5359 |
| 9 | 775 | N/A | 0.5396 |
| 10 | 839 | N/A | 0.7094 |

**Final step comparison:**

- **RANDOM**: R2=0.7094, labeled=839, time=172.7s

## v2ts_a_simple (vector_to_time_series)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 49 | 0.0044 | -0.0329 |
| 1 | 113 | 0.1418 | 0.1927 |
| 2 | 177 | 0.1278 | 0.2608 |
| 3 | 241 | 0.2323 | 0.5325 |
| 4 | 305 | 0.4495 | 0.7130 |
| 5 | 369 | 0.6060 | 0.7993 |

**Final step comparison:**

- **HUDS**: R2=0.6060, labeled=369, time=3.4s
- **RANDOM**: R2=0.7993, labeled=369, time=3.6s

## v2ts_b_medium (vector_to_time_series)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 99 | 0.0463 | 0.0339 |
| 1 | 163 | 0.1039 | 0.0941 |
| 2 | 227 | 0.1438 | 0.1863 |
| 3 | 291 | 0.2075 | 0.1496 |
| 4 | 355 | 0.2218 | 0.2103 |
| 5 | 419 | 0.2592 | 0.2341 |
| 6 | 483 | 0.2817 | 0.3130 |
| 7 | 547 | 0.2846 | 0.3474 |
| 8 | 611 | 0.2861 | 0.3239 |

**Final step comparison:**

- **HUDS**: R2=0.2861, labeled=611, time=8.9s
- **RANDOM**: R2=0.3239, labeled=611, time=9.1s

## v2ts_c_complex (vector_to_time_series)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 199 | 0.0361 | 0.0514 |
| 1 | 263 | 0.0639 | 0.0870 |
| 2 | 327 | 0.0954 | 0.0554 |
| 3 | 391 | 0.0648 | 0.0995 |
| 4 | 455 | 0.1044 | 0.1491 |
| 5 | 519 | 0.1247 | 0.1816 |
| 6 | 583 | 0.1321 | 0.1855 |
| 7 | 647 | 0.1903 | 0.2091 |
| 8 | 711 | 0.1997 | 0.1985 |
| 9 | 775 | 0.2064 | 0.2095 |
| 10 | 839 | 0.2146 | 0.2215 |

**Final step comparison:**

- **HUDS**: R2=0.2146, labeled=839, time=24.9s
- **RANDOM**: R2=0.2215, labeled=839, time=12.2s

## v2v_a_simple (vector_to_vector)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 49 | 0.8047 | 0.2845 |
| 1 | 113 | 0.9644 | 0.9112 |
| 2 | 177 | 0.9898 | 0.9720 |
| 3 | 241 | 0.9915 | 0.9913 |
| 4 | 305 | 0.9952 | 0.9933 |
| 5 | 369 | 0.9954 | 0.9943 |

**Final step comparison:**

- **HUDS**: R2=0.9954, labeled=369, time=1.3s
- **RANDOM**: R2=0.9943, labeled=369, time=1.0s

## v2v_b_medium (vector_to_vector)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 99 | 0.8616 | 0.8513 |
| 1 | 163 | 0.9493 | 0.9538 |
| 2 | 227 | 0.9745 | 0.9720 |
| 3 | 291 | 0.9802 | 0.9793 |
| 4 | 355 | 0.9840 | 0.9849 |
| 5 | 419 | 0.9875 | 0.9881 |
| 6 | 483 | 0.9895 | 0.9894 |
| 7 | 547 | 0.9907 | 0.9914 |
| 8 | 611 | 0.9913 | 0.9918 |

**Final step comparison:**

- **HUDS**: R2=0.9913, labeled=611, time=4.0s
- **RANDOM**: R2=0.9918, labeled=611, time=1.9s

## v2v_c_complex (vector_to_vector)

| Step | labeled | HUDS R2 | RANDOM R2 |
|------|---------||------||------|
| 0 | 199 | 0.8756 | 0.8616 |
| 1 | 263 | 0.9148 | 0.8951 |
| 2 | 327 | 0.9363 | 0.9284 |
| 3 | 391 | 0.9502 | 0.9438 |
| 4 | 455 | 0.9586 | 0.9556 |
| 5 | 519 | 0.9685 | 0.9640 |
| 6 | 583 | 0.9702 | 0.9684 |
| 7 | 647 | 0.9778 | 0.9742 |
| 8 | 711 | 0.9794 | 0.9758 |
| 9 | 775 | 0.9816 | 0.9794 |
| 10 | 839 | 0.9837 | 0.9829 |

**Final step comparison:**

- **HUDS**: R2=0.9837, labeled=839, time=11.2s
- **RANDOM**: R2=0.9829, labeled=839, time=17.0s

## Sample Efficiency Analysis

Steps needed to reach R2 >= 0.5:

| Scenario | Random | HUDS |

|----------|--------|------|

| v2i_a_simple | 4 (305 samples) | 2 (177 samples) |
| v2i_b_medium | N/A | 1 (163 samples) |
| v2i_c_complex | 4 (455 samples) | N/A |
| v2ts_a_simple | 3 (241 samples) | 5 (369 samples) |
| v2ts_b_medium | N/A | N/A |
| v2ts_c_complex | N/A | N/A |
| v2v_a_simple | 1 (113 samples) | 0 (49 samples) |
| v2v_b_medium | 0 (99 samples) | 0 (99 samples) |
| v2v_c_complex | 0 (199 samples) | 0 (199 samples) |