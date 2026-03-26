"""
ScannrAI TP/SL Optimizer
========================
Sweeps TP/SL multiplier grid across the full 14,201-signal dataset
to find configurations where Profit Factor > 1.0.

Architecture:
  1. Generate all signals once (reuses historical_backtest.py)
  2. For each signal, store the forward N-bar price path (highs/lows)
  3. Sweep TP/SL multipliers against stored paths (no refetching)
  4. Compute metrics per segment (market × direction × regime)
  5. Return optimal parameters with statistical backing

The key insight: signal ENTRY points are fixed. Only the EXIT
parameters (TP/SL distance, max hold time) change during optimization.
This makes the sweep ~1000x faster than re-running the full backtest.
"""

import math
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np

from historical_backtest import (
    calc_rsi, calc_ema, calc_macd, calc_bb, calc_atr, calc_adx,
    detect_regime, detect_squeeze, generate_signal_at_bar,
    fetch_daily_data, ASSET_YAHOO_SYMBOLS, ASSET_MARKET
)

logger = logging.getLogger("tpsl-optimizer")


def generate_signals_with_paths(
    asset: str, fmp_key: str, years: int = 10,
    max_forward_bars: int = 40,
    min_bars_between: int = 5
) -> List[Dict]:
    """
    Generate signals and store forward price paths for each.
    The forward path allows re-evaluating different TP/SL without refetching data.
    """
    market = ASSET_MARKET.get(asset, "futures")
    data = fetch_daily_data(asset, fmp_key, years)
    if not data or len(data.get("closes", [])) < 100:
        return []

    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    volumes = data["volumes"]
    dates = data["dates"]
    total_bars = len(closes)

    signals = []
    prev_squeeze = "none"
    last_signal_bar = -min_bars_between

    for bar_idx in range(50, total_bars - max_forward_bars):
        if bar_idx - last_signal_bar < min_bars_between:
            continue

        sig = generate_signal_at_bar(
            closes, highs, lows, volumes, bar_idx, asset, market, prev_squeeze
        )

        # Update squeeze
        if bar_idx > 0:
            c_w = closes[:bar_idx + 1]
            h_w = highs[:bar_idx + 1]
            l_w = lows[:bar_idx + 1]
            if len(c_w) >= 20:
                prev_squeeze = detect_squeeze(c_w, h_w, l_w, prev_squeeze)

        if sig is None:
            continue

        # Store forward price path (the next N bars of highs/lows/closes)
        fwd_end = min(bar_idx + 1 + max_forward_bars, total_bars)
        sig["forward_highs"] = highs[bar_idx + 1:fwd_end]
        sig["forward_lows"] = lows[bar_idx + 1:fwd_end]
        sig["forward_closes"] = closes[bar_idx + 1:fwd_end]
        sig["date"] = dates[bar_idx] if bar_idx < len(dates) else ""
        sig["market"] = market

        signals.append(sig)
        last_signal_bar = bar_idx

    logger.info(f"[SIGNALS] {asset}: {len(signals)} signals with forward paths")
    return signals


def evaluate_tpsl(
    signals: List[Dict],
    tp_mult: float,
    sl_mult: float,
    max_bars: int = 20,
    min_consensus: int = 2,
    rsi_min: float = 0,
    rsi_max: float = 100,
) -> Dict:
    """
    Evaluate a specific TP/SL configuration against pre-generated signals.
    Returns metrics: WR, PF, EV, avg_win, avg_loss, sample_size.
    """
    wins = 0
    losses = 0
    expired = 0
    win_pnls = []
    loss_pnls = []
    all_pnls = []  # Ordered sequence for drawdown calc

    for sig in signals:
        # Apply filters
        if sig["agreeing"] < min_consensus:
            continue
        if sig["rsi"] < rsi_min or sig["rsi"] > rsi_max:
            continue

        entry = sig["entry_price"]
        atr = sig["atr"]
        is_long = sig["signal_type"] == "BUY"

        tp_dist = atr * tp_mult
        sl_dist = atr * sl_mult

        if is_long:
            tp_price = entry + tp_dist
            sl_price = entry - sl_dist
        else:
            tp_price = entry - tp_dist
            sl_price = entry + sl_dist

        fwd_h = sig["forward_highs"]
        fwd_l = sig["forward_lows"]
        fwd_c = sig["forward_closes"]

        outcome = "expired"
        pnl_pct = 0

        bars_to_check = min(max_bars, len(fwd_h))

        for i in range(bars_to_check):
            if is_long:
                # Check SL first (conservative)
                if fwd_l[i] <= sl_price:
                    outcome = "lost"
                    pnl_pct = (sl_price - entry) / entry * 100
                    break
                if fwd_h[i] >= tp_price:
                    outcome = "won"
                    pnl_pct = (tp_price - entry) / entry * 100
                    break
            else:
                if fwd_h[i] >= sl_price:
                    outcome = "lost"
                    pnl_pct = (entry - sl_price) / entry * 100
                    break
                if fwd_l[i] <= tp_price:
                    outcome = "won"
                    pnl_pct = (entry - tp_price) / entry * 100
                    break

        if outcome == "expired" and bars_to_check > 0:
            last_close = fwd_c[bars_to_check - 1]
            pnl_pct = ((last_close - entry) / entry * 100) if is_long else ((entry - last_close) / entry * 100)

        if outcome == "won":
            wins += 1
            win_pnls.append(pnl_pct)
            all_pnls.append(pnl_pct)
        elif outcome == "lost":
            losses += 1
            loss_pnls.append(abs(pnl_pct))
            all_pnls.append(pnl_pct)
        else:
            expired += 1
            all_pnls.append(pnl_pct)

    resolved = wins + losses
    if resolved < 10:
        return {"valid": False, "resolved": resolved}

    wr = wins / resolved
    avg_win = float(np.mean(win_pnls)) if win_pnls else 0
    avg_loss = float(np.mean(loss_pnls)) if loss_pnls else 0
    total_win_pnl = sum(win_pnls)
    total_loss_pnl = sum(loss_pnls)
    pf = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float("inf")
    ev = wr * avg_win - (1 - wr) * avg_loss
    total_pnl = sum(all_pnls)

    # Max consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for p in all_pnls:
        if p < 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    # Max drawdown (cumulative)
    cumulative = np.cumsum(all_pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = cumulative - running_max
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

    # Wilson CI for win rate
    z = 1.96
    p_hat = wr
    denom = 1 + z * z / resolved
    center = (p_hat + z * z / (2 * resolved)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * resolved)) / resolved) / denom
    ci_low = max(0, center - spread)
    ci_high = min(1, center + spread)

    return {
        "valid": True,
        "tp_mult": tp_mult,
        "sl_mult": sl_mult,
        "max_bars": max_bars,
        "min_consensus": min_consensus,
        "signals": wins + losses + expired,
        "resolved": resolved,
        "won": wins,
        "lost": losses,
        "expired": expired,
        "win_rate": round(wr, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "profit_factor": round(float(pf), 4) if pf != float("inf") else 999,
        "ev_per_trade": round(ev, 4),
        "total_pnl_pct": round(total_pnl, 2),
        "max_consecutive_losses": max_consec_loss,
        "max_drawdown_pct": round(max_dd, 2),
        "rr_ratio": round(tp_mult / sl_mult, 2),
    }


def run_optimization(
    all_signals: List[Dict],
    segment_name: str = "all",
    tp_range: List[float] = None,
    sl_range: List[float] = None,
    max_bars_range: List[int] = None,
    consensus_range: List[int] = None,
) -> Dict:
    """
    Run full parameter grid search for a signal segment.
    Returns top N parameter sets ranked by objective function.
    """
    if tp_range is None:
        tp_range = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    if sl_range is None:
        sl_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    if max_bars_range is None:
        max_bars_range = [5, 10, 15, 20, 30]
    if consensus_range is None:
        consensus_range = [2, 3, 4, 5]

    total_combos = len(tp_range) * len(sl_range) * len(max_bars_range) * len(consensus_range)
    logger.info(f"[OPTIMIZE] {segment_name}: {len(all_signals)} signals, {total_combos} parameter combinations")

    results = []
    evaluated = 0

    for min_cons in consensus_range:
        for max_bars in max_bars_range:
            for tp_m in tp_range:
                for sl_m in sl_range:
                    r = evaluate_tpsl(
                        all_signals,
                        tp_mult=tp_m,
                        sl_mult=sl_m,
                        max_bars=max_bars,
                        min_consensus=min_cons,
                    )
                    evaluated += 1
                    if r["valid"] and r["resolved"] >= 30:
                        results.append(r)

            if evaluated % 200 == 0:
                logger.info(f"[OPTIMIZE] {segment_name}: {evaluated}/{total_combos} evaluated, {len(results)} valid")

    logger.info(f"[OPTIMIZE] {segment_name}: Done. {len(results)} valid results from {evaluated} combos")

    if not results:
        return {"segment": segment_name, "error": "No valid parameter sets found", "signals": len(all_signals)}

    # Rank by objective function:
    # Primary: PF > 1.0 (profitable)
    # Secondary: EV per trade (maximize)
    # Tertiary: Sample size (prefer more signals)
    # Penalty: max consecutive losses > 10
    def score_result(r):
        pf_bonus = 50 if r["profit_factor"] > 1.0 else 0
        pf_bonus += 25 if r["profit_factor"] > 1.2 else 0
        pf_bonus += 25 if r["profit_factor"] > 1.5 else 0
        ev_score = r["ev_per_trade"] * 100  # Scale up
        sample_bonus = min(20, r["resolved"] / 10)  # Up to 20 pts for 200+ signals
        consec_penalty = -30 if r["max_consecutive_losses"] > 10 else 0
        dd_penalty = max(-20, r["max_drawdown_pct"] / 5)  # Penalize deep drawdowns
        return pf_bonus + ev_score + sample_bonus + consec_penalty + dd_penalty

    for r in results:
        r["_score"] = round(score_result(r), 2)

    results.sort(key=lambda r: r["_score"], reverse=True)

    # Also find: best PF with ≥100 resolved, best EV with ≥50 resolved
    pf_candidates = [r for r in results if r["resolved"] >= 100]
    ev_candidates = [r for r in results if r["resolved"] >= 50 and r["profit_factor"] > 1.0]

    best_pf = max(pf_candidates, key=lambda r: r["profit_factor"]) if pf_candidates else None
    best_ev = max(ev_candidates, key=lambda r: r["ev_per_trade"]) if ev_candidates else None

    # Profitable parameter sets (PF > 1.0 with ≥30 samples)
    profitable = [r for r in results if r["profit_factor"] > 1.0]

    return {
        "segment": segment_name,
        "total_signals": len(all_signals),
        "combos_evaluated": evaluated,
        "valid_results": len(results),
        "profitable_configs": len(profitable),
        "top_20": results[:20],
        "best_overall": results[0] if results else None,
        "best_pf_100plus": best_pf,
        "best_ev_50plus": best_ev,
        "profitable_summary": {
            "count": len(profitable),
            "avg_pf": round(float(np.mean([r["profit_factor"] for r in profitable])), 4) if profitable else 0,
            "avg_ev": round(float(np.mean([r["ev_per_trade"] for r in profitable])), 4) if profitable else 0,
            "avg_wr": round(float(np.mean([r["win_rate"] for r in profitable])), 4) if profitable else 0,
        }
    }


def run_full_optimization(fmp_key: str, years: int = 10) -> Dict:
    """
    Run complete TP/SL optimization across all 37 assets.
    
    1. Fetch data + generate signals for all assets
    2. Segment by market × direction
    3. Also segment by regime
    4. Find optimal parameters for each segment
    5. Return actionable configuration
    """
    start_time = time.time()
    all_signals = []
    asset_counts = {}
    errors = []

    assets = list(ASSET_YAHOO_SYMBOLS.keys())
    logger.info(f"[FULL-OPT] Starting optimization for {len(assets)} assets, {years} years")

    for asset in assets:
        try:
            sigs = generate_signals_with_paths(asset, fmp_key, years, max_forward_bars=40)
            if sigs:
                all_signals.extend(sigs)
                asset_counts[asset] = len(sigs)
            else:
                errors.append(asset)
        except Exception as e:
            logger.error(f"[FULL-OPT] {asset}: {e}")
            errors.append(asset)
        time.sleep(0.3)  # Yahoo rate limit

    logger.info(f"[FULL-OPT] Generated {len(all_signals)} signals from {len(asset_counts)} assets")

    # Segment signals
    crypto_buy = [s for s in all_signals if s["market"] == "crypto" and s["signal_type"] == "BUY"]
    crypto_sell = [s for s in all_signals if s["market"] == "crypto" and s["signal_type"] == "SELL"]
    futures_buy = [s for s in all_signals if s["market"] == "futures" and s["signal_type"] == "BUY"]
    futures_sell = [s for s in all_signals if s["market"] == "futures" and s["signal_type"] == "SELL"]

    # Regime segments (futures only, largest pool)
    futures_trending_buy = [s for s in futures_buy if s["regime"] in ("trending_up", "trending_down")]
    futures_ranging_buy = [s for s in futures_buy if s["regime"] in ("ranging", "mean_reverting")]
    futures_trending_sell = [s for s in futures_sell if s["regime"] in ("trending_up", "trending_down")]
    futures_ranging_sell = [s for s in futures_sell if s["regime"] in ("ranging", "mean_reverting")]

    segments = {
        "crypto_buy": crypto_buy,
        "crypto_sell": crypto_sell,
        "futures_buy": futures_buy,
        "futures_sell": futures_sell,
        "futures_trending_buy": futures_trending_buy,
        "futures_ranging_buy": futures_ranging_buy,
        "futures_trending_sell": futures_trending_sell,
        "futures_ranging_sell": futures_ranging_sell,
        "all": all_signals,
    }

    logger.info(f"[FULL-OPT] Segments: " + ", ".join(f"{k}={len(v)}" for k, v in segments.items()))

    # Run optimization for each segment
    results = {}
    for seg_name, seg_signals in segments.items():
        if len(seg_signals) < 30:
            results[seg_name] = {"error": f"Too few signals ({len(seg_signals)})", "signals": len(seg_signals)}
            continue

        logger.info(f"[FULL-OPT] Optimizing {seg_name} ({len(seg_signals)} signals)...")
        results[seg_name] = run_optimization(seg_signals, seg_name)

    elapsed = round(time.time() - start_time, 1)

    # Build deployment recommendation
    deployment = build_deployment_config(results)

    return {
        "total_signals": len(all_signals),
        "assets_processed": len(asset_counts),
        "errors": errors,
        "segments": {k: len(v) for k, v in segments.items()},
        "optimization_results": results,
        "deployment_config": deployment,
        "computation_time_s": elapsed,
    }


def build_deployment_config(results: Dict) -> Dict:
    """
    Extract the best actionable parameters from optimization results.
    These get deployed to AppConfig for the live scanner.
    """
    config = {
        "version": "2.0",
        "calibration_source": "14,201 signals, 10yr backtest, TP/SL grid search",
    }

    # For each key segment, extract the best profitable config
    for seg_name in ["crypto_buy", "crypto_sell", "futures_buy", "futures_sell",
                     "futures_trending_buy", "futures_ranging_buy",
                     "futures_trending_sell", "futures_ranging_sell"]:
        seg = results.get(seg_name, {})
        best = seg.get("best_overall")
        if best and best.get("profit_factor", 0) > 0.8:
            config[seg_name] = {
                "tp_mult": best["tp_mult"],
                "sl_mult": best["sl_mult"],
                "max_bars": best["max_bars"],
                "min_consensus": best["min_consensus"],
                "profit_factor": best["profit_factor"],
                "win_rate": best["win_rate"],
                "ev_per_trade": best["ev_per_trade"],
                "resolved": best["resolved"],
            }

    return config
