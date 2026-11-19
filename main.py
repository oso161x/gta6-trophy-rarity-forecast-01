"""gta6-trophy-rarity-forecast-01: forecast each GTA VI PS5 trophy's earn rate on Exophase
at the case target, as five quantiles (5/25/50/75/95%), scored by pinball loss on the logit scale.

Method (Python 3.13 stdlib only, no network, deterministic):
  A rate is a ratio: rate_i(t) = 100 * E_i(t) / P(t), where P is the players Exophase tracks
  and E_i the tracked players who have earned trophy i. Modelling the ratio directly
  confuses the two motions (new players dilute every rate; players progressing raise
  the earners). So this program models the two counts separately and divides:
    - log P(t) and log E_i(t) are each fit by least squares against log(1 + hours
      since launch) over the case's recent distinct reads (Exophase refreshes a
      game's counts hours apart, so most reads repeat the last refresh: only
      distinct states carry signal). Log-time is the key stabiliser: launch-week
      growth decelerates as the player base matures, and a fit in log-time
      extrapolates that deceleration instead of compounding a 24-hour slope
      unchanged for 4.5 days (a linear-time fit explodes: in backtests it scored
      3x worse);
    - the per-trophy slope is shrunk toward the players slope (a rate-flat prior),
      more so the fewer distinct reads the trophy has;
    - the forecast rate is 100 * E*(target) / P*(target), capped so earners never
      exceed players.
  Quantiles are the median in logit space plus/minus z_tau * s_i, where s_i combines
  a base spread, a sqrt(horizon) term and the trophy's fit residual: an honest spread
  for how far a 4.5-day forecast can miss.

  Flat-history fallback (the launch-night practice cases): when the case's history is
  a single frozen state, no slope exists to fit, so the program uses the refresh
  structure the competition's own rules publish: Exophase refreshes a game's counts
  hours apart; the first refresh took players tracked 638 -> 3,868 (x6.06) while
  trophies earned went ~861 -> 4,276, so a typical trophy's rate fell to ~0.30x.
  For a short horizon (<= 24 h) the program therefore predicts one refresh per
  ~4 h, each multiplying a typical trophy's rate by 0.30 with the dilution decaying
  (x1.5 per refresh) as launch growth fades, tilted by the trophy's current rate
  (higher-rate = earlier/easier trophies grow their earners faster, so dilute less),
  with a wide spread. This path never fires on a long horizon or a history with
  real variation: there the data-driven fit above rules.

Usage (as the grader runs it): python3 -P -s /bot/main.py, with MAKEFUN_INPUT_DIR
and MAKEFUN_OUTPUT_DIR standing in for /input and /output outside the sandbox.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- tuning constants (fixed in advance; see README for how they were chosen) ---
WINDOW_HOURS = 24.0        # recent window for the log-time fits
SHRINK_N = 3.0             # slope shrinkage: w = n / (n + SHRINK_N)
PLAYERS_SHRINK = 1.0       # players slope used as fitted (log-time already damps it)
SPREAD_BASE = 0.30         # logit spread floor
SPREAD_HORIZON = 0.30      # logit spread per sqrt(day) of horizon
REFRESH_HOURS = 4.0        # assumed Exophase refresh cadence (flat-history path)
REFRESH_FIRST = 0.30       # typical rate factor at the first refresh (rules)
REFRESH_DECAY = 1.5        # dilution fades by this per refresh
REFRESH_TILT = 0.5         # tilt exponent on (rate / mean rate)
REFRESH_SPREAD = 0.8       # spread coefficient on the flat-history path
REFRESH_MAX_HOURS = 24.0   # flat-history path only fires on short horizons


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def inv_logit(l: float) -> float:
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s: str) -> float:
    """ISO-8601 timestamp -> hours since 2026-11-19T00:00:00Z."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    epoch = datetime(2026, 11, 19, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds() / 3600.0


def ols(xs, ys):
    """Least-squares slope/intercept of y on x; returns (slope, intercept, resid_std)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / max(n - 2, 1)) if n > 2 else 0.0
    return slope, intercept, resid


def quantiles(median_rate: float, spread: float):
    lg = logit(median_rate)
    out = []
    prev = None
    for z in Z:
        q = round4(clamp(inv_logit(lg + z * spread)))
        q = float(q)
        if prev is not None and q < prev:
            q = prev
        out.append(q)
        prev = q
    return out


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    horizon_h = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0

    # Distinct states only: Exophase refreshes hours apart, so consecutive identical
    # reads repeat one refresh and carry no signal.
    states = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if not states or states[-1][0] != key:
            states.append((key, read))
    last = history[-1]
    last_rates = last["rates"]
    last_p = last["players_tracked"]

    if len(states) <= 1 and horizon_h <= REFRESH_MAX_HOURS:
        return forecast_flat(case, name, last_rates, horizon_h, horizon_days)
    return forecast_trend(case, name, states, last, last_rates, last_p, horizon_h, horizon_days)


def forecast_flat(case, name, last_rates, horizon_h, horizon_days):
    """Single frozen state, short horizon: expect Exophase refreshes (see module docstring)."""
    n_refresh = max(1, min(6, round(horizon_h / REFRESH_HOURS)))
    factor = 1.0
    for k in range(1, n_refresh + 1):
        factor *= min(1.0, REFRESH_FIRST * (REFRESH_DECAY ** (k - 1)))
    present = [r for r in last_rates if r is not None and r < 99.99]
    mean_r = sum(present) / len(present) if present else 0.885
    spread = REFRESH_SPREAD * math.sqrt(max(horizon_days, 1e-9))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last_rates[i]
        if r is None:
            r = 0.01
        if r >= 99.99:
            median = 99.99
        else:
            tilt = (max(r, 0.01) / mean_r) ** REFRESH_TILT
            median = clamp(r * min(1.0, factor * tilt))
        rows.append({"id": trophy["id"], "q": quantiles(median, spread)})
    return {"trophies": rows}


def forecast_trend(case, name, states, last, last_rates, last_p, horizon_h, horizon_days):
    """Data-driven path: separate log-time fits of players and earners."""
    times = [parse_time(read["at"]) for _, read in states]
    t_last = times[-1]
    # Recent window; widen until it holds at least 3 distinct states.
    win = [j for j, t in enumerate(times) if t >= t_last - WINDOW_HOURS]
    if len(win) < 3:
        win = list(range(len(times)))
    xs = [math.log1p(max(times[j], 0.0)) for j in win]
    x_last = math.log1p(max(t_last, 0.0))
    x_target = math.log1p(max(t_last + horizon_h, 0.0))
    dx = x_target - x_last

    log_p = [math.log(max(read["players_tracked"], 1)) for _, read in states]
    b_p, _, _ = ols(xs, [log_p[j] for j in win])
    b_p *= PLAYERS_SHRINK
    log_p_star = math.log(max(last_p, 1)) + b_p * dx
    p_star = math.exp(log_p_star)

    n_win = len(win)
    w_shrink = n_win / (n_win + SHRINK_N)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        # Earners per read in the window (0.5 continuity where the rate is 0.00).
        xs_e, ys_e = [], []
        for k, j in enumerate(win):
            read = states[j][1]
            r = read["rates"][i]
            if r is None:
                continue
            e = max(r * read["players_tracked"] / 100.0, 0.5)
            xs_e.append(xs[k])
            ys_e.append(math.log(e))
        r_last = last_rates[i]
        if r_last is None:
            # Trophy absent from the last read: use its last seen rate.
            r_last = next((read["rates"][i] for _, read in reversed(states)
                           if read["rates"][i] is not None), 0.01)
        if len(xs_e) >= 2:
            b_e, _, resid = ols(xs_e, ys_e)
            b_e = w_shrink * b_e + (1.0 - w_shrink) * b_p
        else:
            b_e, resid = b_p, 0.0
        e_last = max(r_last * last_p / 100.0, 0.5)
        log_e_star = math.log(e_last) + b_e * dx
        e_star = min(math.exp(log_e_star), 0.999 * p_star)
        if r_last >= 99.99:
            median = 99.99
        else:
            median = clamp(100.0 * e_star / p_star)
        spread = math.sqrt(SPREAD_BASE ** 2 + (SPREAD_HORIZON * math.sqrt(horizon_days)) ** 2
                           + min(resid, 1.0) ** 2)
        rows.append({"id": trophy["id"], "q": quantiles(median, spread)})
    return {"trophies": rows}


def fallback_forecast(case: dict) -> dict:
    """A valid file no matter what: last rate, starter spread. Only used if the model raises."""
    last = case["history"][-1] if case.get("history") else {"rates": [0.01] * len(case["trophies"])}
    spread = 0.5 * math.sqrt(max(case["horizon_minutes"], 0) / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        r = last["rates"][i] if last["rates"][i] is not None else 0.01
        rows.append({"id": trophy["id"], "q": quantiles(clamp(r), spread)})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            forecast = forecast_case(case, path.name)
        except Exception:
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
