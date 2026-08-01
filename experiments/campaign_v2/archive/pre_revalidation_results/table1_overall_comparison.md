| Policy | Avg Reward | Avg Quality | Avg Latency | Avg Iterations | Match Rate | 95% CI (reward) | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|
| 1. Heuristic Policy (Baseline) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | n/a | - | - |
| 2. Cold-Start LinUCB (alpha=0.25) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=0.5) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=1.0) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=2.0) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | 1.0000 | 0.0000 | False |
| 3. Sequential Learning LinUCB (alpha=1.0, canonical) | 0.5559 | 0.6644 | 1.4052 | 1.6354 | 0.0823 | [0.4658, 0.6226] | <0.0001 | -5.0725 | True |
