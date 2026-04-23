# Experiment Stats Summary
## Scenario: low

- **greedy**: mean total_reward=-174.0 (std=758.4, n=20)
- **greedy_sequential**: mean total_reward=-174.0 (std=758.4, n=20)
  - vs greedy: t-test p=nan, Wilcoxon p=nan, Cohen's d=0.000
- **nearest**: mean total_reward=-543.0 (std=2366.9, n=20)
  - vs greedy: t-test p=0.3299, Wilcoxon p=0.3173, Cohen's d=-0.229
- **random**: mean total_reward=-567.0 (std=2471.5, n=20)
  - vs greedy: t-test p=0.3299, Wilcoxon p=0.3173, Cohen's d=-0.229
- **rl**: mean total_reward=-174.0 (std=758.4, n=20)
  - vs greedy: t-test p=nan, Wilcoxon p=nan, Cohen's d=0.000

## Scenario: medium

- **greedy**: mean total_reward=-68202.0 (std=22125.3, n=20)
- **greedy_sequential**: mean total_reward=-73257.0 (std=24751.0, n=20)
  - vs greedy: t-test p=0.0775, Wilcoxon p=0.0121, Cohen's d=-0.428
- **nearest**: mean total_reward=-118185.0 (std=43063.0, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.941
- **random**: mean total_reward=-121449.0 (std=40334.4, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-2.222
- **rl**: mean total_reward=-73554.0 (std=28314.0, n=20)
  - vs greedy: t-test p=0.1576, Wilcoxon p=0.1429, Cohen's d=-0.338

## Scenario: high

- **greedy**: mean total_reward=-868671.0 (std=119835.9, n=20)
- **greedy_sequential**: mean total_reward=-842037.0 (std=125283.5, n=20)
  - vs greedy: t-test p=0.0062, Wilcoxon p=0.0083, Cohen's d=0.705
- **nearest**: mean total_reward=-1014177.0 (std=127458.6, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.851
- **random**: mean total_reward=-936033.0 (std=118180.6, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.371
- **rl**: mean total_reward=-833388.0 (std=126203.9, n=20)
  - vs greedy: t-test p=0.0025, Wilcoxon p=0.0020, Cohen's d=0.797

## Scenario: congested

- **greedy**: mean total_reward=-3507267.0 (std=118334.6, n=20)
- **greedy_sequential**: mean total_reward=-3478248.0 (std=110537.7, n=20)
  - vs greedy: t-test p=0.0072, Wilcoxon p=0.0064, Cohen's d=0.691
- **nearest**: mean total_reward=-3640305.0 (std=126524.4, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.568
- **random**: mean total_reward=-3440574.0 (std=104912.6, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=1.598
- **rl**: mean total_reward=-3387717.0 (std=105751.0, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=2.670

