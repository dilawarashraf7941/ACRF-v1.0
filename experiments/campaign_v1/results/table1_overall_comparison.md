| Experiment | Avg Reward | Avg Quality | Avg Latency | Avg Iterations | Match Rate | 95% CI (reward) | vs | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|---|
| 1. Baseline (HeuristicPolicy) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | - | n/a | - | - |
| 2. LinUCB alpha=0 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 3. LinUCB alpha=0.25 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 4. LinUCB alpha=0.5 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 5. LinUCB alpha=1.0 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 6. LinUCB alpha=2.0 | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 10. Full ACRF (LinUCB alpha=1.0) | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Baseline | 1.0000 | 0.0000 | False |
| 7. Random Critic | 0.6591 | 0.6959 | 1.3214 | 1.3125 | 0.2220 | [0.5885, 0.7254] | Full ACRF | <0.0001 | -3.9881 | True |
| 8. Reduced Context | 0.7864 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7652, 0.8120] | Full ACRF | 1.0000 | 0.0000 | False |
| 9. Quality-only Reward | 0.7768 | 0.7768 | 1.0640 | 1.0096 | 0.3931 | [0.7621, 0.7888] | Full ACRF | 0.0003 | -0.7498 | True |
