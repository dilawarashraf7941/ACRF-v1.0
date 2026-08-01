# Ablation Study Report

## Summary Table

| Ablation Type | Baseline Reward | Candidate Reward | Reward Diff | Significant | p-value |
|---|---|---|---|---|---|
| reduced_context_features | 0.7951 | 0.7951 | +0.0000 | False | 1.0000 |
| alternative_reward_definitions | 0.7951 | 0.7857 | -0.0094 | True | 0.0016 |
| random_critic_selection | 0.7951 | 0.6916 | -0.1035 | True | 0.0000 |

## Ranking (by candidate reward, highest first)

1. **reduced_context_features** (LinUCBPolicy): candidate_reward = 0.7951
2. **alternative_reward_definitions** (LinUCBPolicy): candidate_reward = 0.7857
3. **random_critic_selection** (RandomCriticPolicy): candidate_reward = 0.6916

## Best Configuration

- **reduced_context_features** (LinUCBPolicy): candidate_reward = 0.7951, reward_difference = +0.0000
- No statistically significant difference between LinUCBPolicy (baseline) and LinUCBPolicy (candidate) for the 'reduced_context_features' ablation (p=1.0000).

## Worst Configuration

- **random_critic_selection** (RandomCriticPolicy): candidate_reward = 0.6916, reward_difference = -0.1035
- LinUCBPolicy performed significantly better in the 'random_critic_selection' ablation (p=0.0000, effect size=-4.6491).

## Key Observations

- 3 ablation(s) evaluated; 2 showed a statistically significant reward difference.
- 0 ablation(s) improved reward over their baseline; 2 regressed.
- Largest regression: **random_critic_selection** (-0.1035 reward).
