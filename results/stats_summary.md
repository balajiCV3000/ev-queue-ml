# Experiment Stats Summary
## Scenario: low

- **greedy**: mean total_reward=-192.0 (std=836.9, n=20)
- **greedy_sequential**: mean total_reward=-192.0 (std=836.9, n=20)
  - vs greedy: t-test p=nan, Wilcoxon p=nan, Cohen's d=0.000
- **nearest**: mean total_reward=-561.0 (std=2445.3, n=20)
  - vs greedy: t-test p=0.3299, Wilcoxon p=0.3173, Cohen's d=-0.229
- **random**: mean total_reward=-612.0 (std=2667.6, n=20)
  - vs greedy: t-test p=0.3299, Wilcoxon p=0.3173, Cohen's d=-0.229
- **rl**: mean total_reward=-192.0 (std=836.9, n=20)
  - vs greedy: t-test p=nan, Wilcoxon p=nan, Cohen's d=0.000

## Scenario: medium

- **greedy**: mean total_reward=-72051.0 (std=22995.5, n=20)
- **greedy_sequential**: mean total_reward=-77997.0 (std=26258.3, n=20)
  - vs greedy: t-test p=0.0370, Wilcoxon p=0.0083, Cohen's d=-0.515
- **nearest**: mean total_reward=-120486.0 (std=43700.8, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.919
- **random**: mean total_reward=-132360.0 (std=43371.2, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-2.346
- **rl**: mean total_reward=-76704.0 (std=28262.5, n=20)
  - vs greedy: t-test p=0.1725, Wilcoxon p=0.1054, Cohen's d=-0.325

## Scenario: high

- **greedy**: mean total_reward=-897834.0 (std=120974.4, n=20)
- **greedy_sequential**: mean total_reward=-880164.0 (std=127941.4, n=20)
  - vs greedy: t-test p=0.0596, Wilcoxon p=0.0583, Cohen's d=0.460
- **nearest**: mean total_reward=-1029720.0 (std=128290.1, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.717
- **random**: mean total_reward=-998580.0 (std=124657.0, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.989
- **rl**: mean total_reward=-965673.0 (std=113744.9, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.517

## Scenario: congested

- **greedy**: mean total_reward=-3600270.0 (std=115686.8, n=20)
- **greedy_sequential**: mean total_reward=-3604656.0 (std=108809.2, n=20)
  - vs greedy: t-test p=0.6515, Wilcoxon p=0.8124, Cohen's d=-0.105
- **nearest**: mean total_reward=-3695412.0 (std=124607.2, n=20)
  - vs greedy: t-test p=0.0000, Wilcoxon p=0.0000, Cohen's d=-1.202
- **random**: mean total_reward=-3611007.0 (std=104804.6, n=20)
  - vs greedy: t-test p=0.3236, Wilcoxon p=0.3488, Cohen's d=-0.232
- **rl**: mean total_reward=-3617085.0 (std=109103.9, n=20)
  - vs greedy: t-test p=0.1242, Wilcoxon p=0.1769, Cohen's d=-0.369

