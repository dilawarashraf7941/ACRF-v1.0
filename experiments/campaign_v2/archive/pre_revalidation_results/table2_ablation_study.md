| Ablation | Baseline Reward (Full ACRF) | Candidate Reward | Reward Diff | Quality Diff | Latency Diff | Iteration Diff | Match Rate | Winner | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4. Random Critic | 0.7864 | 0.6591 | -0.1273 | -0.0810 | +0.2574 | +0.3030 | 0.2220 | LinUCBPolicy | <0.0001 | -3.9881 | True |
| 5. Reduced Context Ablation | 0.7864 | 0.7864 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.3931 | tie | 1.0000 | 0.0000 | False |
| 6. Quality-only Reward Ablation | 0.7864 | 0.7768 | -0.0096 | +0.0000 | +0.0000 | +0.0000 | 0.3931 | LinUCBPolicy | 0.0003 | -0.7498 | True |
