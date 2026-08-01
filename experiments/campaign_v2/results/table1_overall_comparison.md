| Policy | Avg Reward | Avg Quality | Avg Latency | Avg Iterations | Match Rate | 95% CI (reward) | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|
| 1. Heuristic Policy (Baseline) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] | n/a | - | - |
| 2. Cold-Start LinUCB (alpha=0.25) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=0.5) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=1.0) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] | 1.0000 | 0.0000 | False |
| 2. Cold-Start LinUCB (alpha=2.0) | 0.7951 | 0.7857 | 1.1136 | 0.9263 | 0.4017 | [0.7583, 0.8273] | 1.0000 | 0.0000 | False |
| 3. Sequential Learning LinUCB (alpha=1.0, canonical) | 0.7653 | 0.7525 | 1.1128 | 0.9994 | 0.0072 | [0.4740, 0.9616] | 0.5561 | -0.2168 | False |
