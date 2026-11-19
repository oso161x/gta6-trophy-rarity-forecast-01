# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oso161x/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/oso161x/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI PlayStation 5 trophy, the percent of the players Exophase tracks who will have earned it one week after launch (2026-11-26 00:00 UTC), as five quantiles per trophy (5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How it works

A trophy's earn rate is a ratio, `rate = 100 * earners / players_tracked`, and the two counts move for different reasons: new players joining dilute every rate, while players progressing through the game raise the earners. Modelling the ratio directly confuses the two motions, so `main.py` models the counts separately and divides:

- **Log-time fits.** `log players` and `log earners` per trophy are each fit by least squares against `log(1 + hours since launch)` over the case's recent distinct reads (Exophase refreshes a game's counts hours apart, so most reads repeat the last refresh — only distinct states carry signal). Log-time is the key stabiliser: launch-week growth decelerates as the player base matures, and a fit in log-time extrapolates that deceleration instead of compounding a 24-hour slope unchanged for 4.5 days. A linear-time fit was tried first and scored ~3x worse in backtests.
- **Shrinkage.** Each trophy's earners slope is shrunk toward the players slope (a rate-flat prior), more so the fewer distinct reads the trophy has.
- **Flat-history fallback.** When a case's history is a single frozen state (the launch-night practice cases), no slope exists to fit, so the program uses the refresh structure the competition's own rules publish: Exophase refreshes a game's counts hours apart, and the first refresh took players tracked 638 → 3,868 while trophies earned went ~861 → 4,276, so a typical trophy's rate fell to ~0.30x. For a short horizon (≤ 24 h) it predicts one refresh per ~4 h, each multiplying a typical trophy's rate by 0.30 with the dilution fading as launch growth fades, tilted by the trophy's current rate (earlier/easier trophies grow their earners faster, so dilute less), with a wide spread. This path never fires on a long horizon or a history with real variation.
- **Quantiles.** The median in logit space plus/minus z·spread, where the spread combines a base term, a sqrt(horizon) term and the trophy's fit residual.

Standard library only, no network, deterministic (same case → same bytes), ~0.05 s per case (limit 8 s).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory (`{"schema": "makefun/gta6-trophies-case/v1", ...}`) and writes `<id>.out` (`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`) to the output directory. The kit ships two practice cases (`input/p00.in`, `input/p01.in` in the competition kit).

## Validation (pinball loss on the logit scale, lower is better)

- **Real launch-night data.** Forecasting the live Exophase page at 2026-11-19 13:40 UTC from the flat launch-night history (cut 03:30 UTC): **0.313** vs the house persistence baseline's **2.387** (typical miss 0.08 points, 90% coverage 0.95). The refresh prior is what closes the gap: the page's first refresh diluted most rates to ~5% of their launch-night values.
- **Synthetic final cases** (Nov 21 11:30 cut → Nov 26 00:00 target, simulated player adoption + per-trophy progress curves calibrated to the live 13:40 snapshot, three player-growth scenarios): **0.22–0.32** vs persistence **0.95–0.98** and the house curve baseline **0.51–0.56**.

See `main.py`'s docstring for the full method and the assumptions behind the flat-history prior.

---

Entered on [make.fun](https://make.fun/?ref=4s86eynw&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
