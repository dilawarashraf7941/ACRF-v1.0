| Ablation | Baseline Reward (Full ACRF) | Candidate Reward | Reward Diff | Quality Diff | Latency Diff | Iteration Diff | Match Rate | Winner | p-value | Effect Size | Significant |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4. Random Critic | 0.7951 | 0.6916 | -0.1035 | -0.0722 | +0.1936 | +0.2584 | 0.2280 | LinUCBPolicy | <0.0001 | -4.6491 | True |
| 5. Reduced Context Ablation | 0.7951 | 0.7951 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.4017 | tie | 1.0000 | 0.0000 | False |
| 6. Quality-only Reward Ablation | 0.7951 | 0.7857 | -0.0094 | +0.0000 | +0.0000 | +0.0000 | 0.4017 | LinUCBPolicy | 0.0016 | -0.6372 | True |
