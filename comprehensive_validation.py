"""
ScannrAI Comprehensive Validation Suite
========================================
Addresses ALL gaps in the basic walk-forward:

1. Transaction Cost Modeling
   - Futures: $1.25/side commission + 1 tick slippage per side
   - Crypto: 0.075% taker fee per side (Binance tier)
   - Applied to EVERY trade in PF/EV calculations

2. Parameter Sensitivity Analysis
   - Tests ±20% perturbation around optimal TP/SL
   - If PF degrades >30%, edge is FRAGILE
   - If PF degrades <15%, edge is ROBUST

3. Full-Grid Rolling Walk-Forward
   - Uses the complete 1,440-combo grid, not the 180-combo shortcut
   - Each fold gets the same rigor as the full optimization

4. Multi-Cutoff Stability
   - Tests train/test splits at 2015, 2017, 2019, 2021, 2023
   - If verdict is consistent across all splits → truly robust
   - If verdict changes → result is cutoff-dependent

5. Regime-Conditional Testing
   - Tags each signal with market regime at time of signal
   - Evaluates PF separately in bull, bear, sideways, and crisis periods
   - Uses VIX proxy (ATR regime) to classify market state
"""

import math
import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

from tpsl_optimizer import evaluate_tpsl, run_optimization, generate_signals_with_paths
from historical_backtest import ASSET_YAHOO_SYMBOLS, ASSET_MARKET

logger = logging.getLogger("comprehensive-validation")


# ═══════════════════════════════════════════════════════════════
# 1. TRANSACTION COST MODELING
# ═══════════════════════════════════════════════════════════════

# Realistic cost assumptions (as % of notional)
# Futures micros (ES/NQ): ~$1.25 commission + 1 tick slippage per side
# On $27,500 notional (MES at 5500): $2.50 RT / $27,500 = 0.009% per side
# Total round-trip for futures: ~0.02% of notional
FUTURES_COST_PCT_PER_SIDE = 0.01  # 0.01% per side (commission + slippage)

# Crypto (Binance taker): 0.075% per side
CRYPTO_COST_PCT_PER_SIDE = 0.075  # 0.075% per side

def apply_transaction_costs(result: Dict, market: str) -> Dict:
    """
    Adjust backtest metrics for realistic transaction costs.
    Costs are applied as percentage of entry price per round-trip.
    """
    if not result.get("valid"):
        return result

    cost_per_side = CRYPTO_COST_PCT_PER_SIDE if market == "crypto" else FUTURES_COST_PCT_PER_SIDE
    cost_pct = cost_per_side * 2  # Round-trip

    # Adjust each trade's P&L by subtracting costs
    adj_won = result["won"]
    adj_lost = result["lost"]
    avg_win = result["avg_win_pct"] - cost_pct   # Winners shrink
    avg_loss = result["avg_loss_pct"] + cost_pct  # Losers grow

    # Some marginal winners become losers after costs
    # Estimate: if avg_win was close to cost, some fraction flip
    if result["avg_win_pct"] > 0 and cost_pct > 0:
        # Rough model: assume win P&L is normally distributed around avg_win
        # Fraction that flips = P(win_pnl < cost) ≈ cost / (2 × avg_win) for small costs
        flip_fraction = min(0.15, cost_pct / (2 * result["avg_win_pct"]))
        flipped = int(adj_won * flip_fraction)
        adj_won -= flipped
        adj_lost += flipped

    adj_resolved = adj_won + adj_lost
    if adj_resolved == 0:
        return {**result, "cost_adjusted": True, "cost_pct_rt": round(cost_pct, 4),
                "adj_profit_factor": 0, "adj_win_rate": 0, "adj_ev_per_trade": 0}

    adj_wr = adj_won / adj_resolved
    total_win_pnl = adj_won * max(0, avg_win)
    total_loss_pnl = adj_lost * avg_loss
    adj_pf = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 999
    adj_ev = adj_wr * max(0, avg_win) - (1 - adj_wr) * avg_loss

    return {
        **result,
        "cost_adjusted": True,
        "cost_pct_round_trip": round(cost_pct, 4),
        "adj_profit_factor": round(float(adj_pf), 4),
        "adj_win_rate": round(adj_wr, 4),
        "adj_ev_per_trade": round(adj_ev, 4),
        "adj_avg_win_pct": round(max(0, avg_win), 4),
        "adj_avg_loss_pct": round(avg_loss, 4),
        "winners_flipped_to_losers": flipped if 'flipped' in dir() else 0,
        "raw_profit_factor": result["profit_factor"],
        "raw_win_rate": result["win_rate"],
    }


# ═══════════════════════════════════════════════════════════════
# 2. PARAMETER SENSITIVITY ANALYSIS
# ═══════════════════════════════════════════════════════════════

def parameter_sensitivity(
    signals: List[Dict],
    center_tp: float,
    center_sl: float,
    center_bars: int,
    center_cons: int,
    perturbation: float = 0.20,  # ±20%
    steps: int = 5,
) -> Dict:
    """
    Test how sensitive PF is to small changes in TP/SL.
    Sweep TP and SL independently around the optimal center point.
    """
    results = {"center": {}, "tp_sweep": [], "sl_sweep": [], "bars_sweep": []}

    # Center evaluation
    center = evaluate_tpsl(signals, center_tp, center_sl, center_bars, center_cons)
    if not center.get("valid"):
        return {"error": "Center params invalid", "center": center}
    results["center"] = {
        "tp": center_tp, "sl": center_sl, "bars": center_bars,
        "pf": center["profit_factor"], "wr": center["win_rate"], "ev": center["ev_per_trade"],
    }
    center_pf = center["profit_factor"]

    # TP sweep (SL fixed)
    for i in range(steps * 2 + 1):
        factor = 1 - perturbation + (perturbation * 2 * i / (steps * 2))
        tp_test = round(center_tp * factor, 3)
        if tp_test <= 0:
            continue
        r = evaluate_tpsl(signals, tp_test, center_sl, center_bars, center_cons)
        if r.get("valid"):
            pf = r["profit_factor"]
            decay = (center_pf - pf) / center_pf if center_pf > 0 else 0
            results["tp_sweep"].append({
                "tp": tp_test, "pf": pf, "wr": r["win_rate"],
                "ev": r["ev_per_trade"], "pf_decay": round(decay, 4),
            })

    # SL sweep (TP fixed)
    for i in range(steps * 2 + 1):
        factor = 1 - perturbation + (perturbation * 2 * i / (steps * 2))
        sl_test = round(center_sl * factor, 3)
        if sl_test <= 0:
            continue
        r = evaluate_tpsl(signals, center_tp, sl_test, center_bars, center_cons)
        if r.get("valid"):
            pf = r["profit_factor"]
            decay = (center_pf - pf) / center_pf if center_pf > 0 else 0
            results["sl_sweep"].append({
                "sl": sl_test, "pf": pf, "wr": r["win_rate"],
                "ev": r["ev_per_trade"], "pf_decay": round(decay, 4),
            })

    # Max hold bars sweep
    for bars_test in [3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        r = evaluate_tpsl(signals, center_tp, center_sl, bars_test, center_cons)
        if r.get("valid"):
            pf = r["profit_factor"]
            decay = (center_pf - pf) / center_pf if center_pf > 0 else 0
            results["bars_sweep"].append({
                "bars": bars_test, "pf": pf, "wr": r["win_rate"],
                "ev": r["ev_per_trade"], "pf_decay": round(decay, 4),
            })

    # Sensitivity verdict
    tp_pfs = [x["pf"] for x in results["tp_sweep"]]
    sl_pfs = [x["pf"] for x in results["sl_sweep"]]
    all_pfs = tp_pfs + sl_pfs

    if all_pfs:
        min_pf = min(all_pfs)
        max_decay = (center_pf - min_pf) / center_pf if center_pf > 0 else 1.0
        avg_pf = float(np.mean(all_pfs))
        avg_decay = (center_pf - avg_pf) / center_pf if center_pf > 0 else 1.0

        if max_decay < 0.15:
            verdict = "VERY_ROBUST"
            explanation = f"Max PF decay {max_decay:.1%} within ±{perturbation:.0%} perturbation. Edge survives parameter uncertainty."
        elif max_decay < 0.30:
            verdict = "ROBUST"
            explanation = f"Max PF decay {max_decay:.1%}. Edge degrades gracefully."
        elif max_decay < 0.50:
            verdict = "MODERATE"
            explanation = f"Max PF decay {max_decay:.1%}. Edge is somewhat sensitive to exact parameters."
        else:
            verdict = "FRAGILE"
            explanation = f"Max PF decay {max_decay:.1%}. Edge depends heavily on exact TP/SL values — likely won't survive live trading."
    else:
        verdict = "UNTESTABLE"
        explanation = "Could not evaluate perturbations."
        max_decay = avg_decay = 0

    results["verdict"] = verdict
    results["explanation"] = explanation
    results["max_pf_decay"] = round(max_decay, 4) if all_pfs else None
    results["avg_pf_decay"] = round(avg_decay, 4) if all_pfs else None
    results["min_pf_in_range"] = round(min(all_pfs), 4) if all_pfs else None

    return results


# ═══════════════════════════════════════════════════════════════
# 3. MULTI-CUTOFF STABILITY
# ═══════════════════════════════════════════════════════════════

def multi_cutoff_validation(
    all_signals: List[Dict],
    cutoffs: List[str] = None,
) -> Dict:
    """
    Test if the walk-forward verdict is stable across different train/test splits.
    """
    if cutoffs is None:
        cutoffs = ["2015-01-01", "2017-01-01", "2019-01-01", "2021-01-01", "2023-01-01"]

    from walk_forward import walk_forward_validate

    results = {}
    verdicts = []

    # Use streamlined grid for speed across many cutoffs
    tp_range = [1.0, 1.25, 1.5, 2.0, 3.0]
    sl_range = [1.0, 1.5, 2.0, 3.0]
    bars_range = [5, 10, 20]
    cons_range = [2, 3, 4]

    for cutoff in cutoffs:
        logger.info(f"[MULTI-CUTOFF] Testing cutoff={cutoff}...")
        wf = walk_forward_validate(
            all_signals, train_cutoff=cutoff,
            tp_range=tp_range, sl_range=sl_range,
            max_bars_range=bars_range, consensus_range=cons_range,
        )

        verdict = wf.get("verdict", "ERROR")
        verdicts.append(verdict)

        train_n = wf.get("train_signals", 0)
        test_n = wf.get("test_signals", 0)
        train_pf = wf.get("train_performance", {}).get("profit_factor", 0)
        test_pf = wf.get("test_performance", {}).get("profit_factor", 0)

        results[cutoff] = {
            "verdict": verdict,
            "train_signals": train_n,
            "test_signals": test_n,
            "train_pf": train_pf,
            "test_pf": test_pf if wf.get("test_performance", {}).get("profit_factor") else None,
            "pf_decay": wf.get("pf_decay", None),
            "best_params": wf.get("best_train_params"),
        }

    # Stability assessment
    strong_count = sum(1 for v in verdicts if v in ("STRONG_EDGE", "ROBUST_EDGE"))
    profitable_count = sum(1 for v in verdicts if v not in ("CURVE_FIT", "NO_EDGE"))
    total = len(verdicts)

    if strong_count >= total * 0.8:
        stability = "VERY_STABLE"
        explanation = f"{strong_count}/{total} cutoffs show STRONG_EDGE. Result is cutoff-independent."
    elif profitable_count >= total * 0.7:
        stability = "STABLE"
        explanation = f"{profitable_count}/{total} cutoffs profitable. Edge is mostly stable."
    elif profitable_count >= total * 0.5:
        stability = "UNSTABLE"
        explanation = f"Only {profitable_count}/{total} cutoffs profitable. Result depends on which years are in train vs test."
    else:
        stability = "CUTOFF_DEPENDENT"
        explanation = f"Only {profitable_count}/{total} cutoffs profitable. Verdict changes with train/test split — not a real edge."

    # Check if parameters are consistent across cutoffs
    params_list = [r.get("best_params") for r in results.values() if r.get("best_params")]
    if params_list:
        tp_vals = [p.get("tp_mult", 0) for p in params_list]
        sl_vals = [p.get("sl_mult", 0) for p in params_list]
        tp_consistent = float(np.std(tp_vals)) < 0.5
        sl_consistent = float(np.std(sl_vals)) < 0.5
    else:
        tp_consistent = sl_consistent = False

    return {
        "stability": stability,
        "explanation": explanation,
        "cutoffs_tested": total,
        "strong_edge_count": strong_count,
        "profitable_count": profitable_count,
        "param_consistency": {
            "tp_consistent": tp_consistent,
            "sl_consistent": sl_consistent,
            "tp_values": [r.get("best_params", {}).get("tp_mult") for r in results.values()],
            "sl_values": [r.get("best_params", {}).get("sl_mult") for r in results.values()],
        },
        "cutoff_results": results,
    }


# ═══════════════════════════════════════════════════════════════
# 4. REGIME-CONDITIONAL TESTING
# ═══════════════════════════════════════════════════════════════

def classify_market_regime_at_signal(signal: Dict) -> str:
    """
    Classify the market regime at the time of signal generation.
    Uses the signal's own regime field + additional heuristics.
    """
    regime = signal.get("regime", "ranging")
    rsi = signal.get("rsi", 50)
    atr = signal.get("atr", 1)
    price = signal.get("entry_price", 1)

    # ATR as % of price → volatility proxy
    vol_pct = (atr / price * 100) if price > 0 else 1

    if regime in ("trending_up",) and vol_pct < 2.0:
        return "bull_low_vol"
    elif regime in ("trending_up",) and vol_pct >= 2.0:
        return "bull_high_vol"
    elif regime in ("trending_down",) and vol_pct < 2.0:
        return "bear_low_vol"
    elif regime in ("trending_down",) and vol_pct >= 2.0:
        return "bear_high_vol"  # Crisis-like
    elif regime == "high_volatility":
        return "crisis"
    elif regime == "mean_reverting":
        return "mean_reverting"
    else:
        return "sideways"


def regime_conditional_test(
    signals: List[Dict],
    tp_mult: float = 1.5,
    sl_mult: float = 3.0,
    max_bars: int = 5,
    min_consensus: int = 3,
) -> Dict:
    """
    Evaluate performance separately in each market regime.
    """
    # Classify each signal
    regime_groups = defaultdict(list)
    for s in signals:
        regime = classify_market_regime_at_signal(s)
        regime_groups[regime].append(s)

    results = {}
    for regime, sigs in regime_groups.items():
        r = evaluate_tpsl(sigs, tp_mult, sl_mult, max_bars, min_consensus)
        if r.get("valid"):
            results[regime] = {
                "signals": len(sigs),
                "resolved": r["resolved"],
                "win_rate": r["win_rate"],
                "profit_factor": r["profit_factor"],
                "ev_per_trade": r["ev_per_trade"],
                "avg_win": r["avg_win_pct"],
                "avg_loss": r["avg_loss_pct"],
            }
        else:
            results[regime] = {"signals": len(sigs), "resolved": r.get("resolved", 0), "insufficient": True}

    # Overall assessment
    profitable_regimes = [k for k, v in results.items() if v.get("profit_factor", 0) > 1.0]
    total_regimes = [k for k, v in results.items() if not v.get("insufficient")]

    if len(profitable_regimes) >= len(total_regimes) * 0.8:
        verdict = "ALL_REGIME_EDGE"
        explanation = f"Profitable in {len(profitable_regimes)}/{len(total_regimes)} regimes. Edge works in all market conditions."
    elif len(profitable_regimes) >= len(total_regimes) * 0.5:
        verdict = "PARTIAL_REGIME_EDGE"
        explanation = f"Profitable in {len(profitable_regimes)}/{len(total_regimes)} regimes. Edge is regime-dependent."
    else:
        verdict = "REGIME_DEPENDENT"
        explanation = f"Only profitable in {len(profitable_regimes)}/{len(total_regimes)} regimes. Performance varies heavily by market condition."

    return {
        "verdict": verdict,
        "explanation": explanation,
        "profitable_regimes": len(profitable_regimes),
        "total_regimes": len(total_regimes),
        "regime_results": dict(sorted(results.items(), key=lambda x: x[1].get("profit_factor", 0), reverse=True)),
    }


# ═══════════════════════════════════════════════════════════════
# 5. COMPREHENSIVE VALIDATION (combines all tests)
# ═══════════════════════════════════════════════════════════════

def run_comprehensive_validation(fmp_key: str, years: int = 20) -> Dict:
    """
    The most rigorous possible validation suite.
    Runs ALL tests: walk-forward, transaction costs, sensitivity,
    multi-cutoff stability, and regime-conditional analysis.
    """
    start_time = time.time()
    logger.info(f"[COMPREHENSIVE] Starting {years}-year comprehensive validation...")

    # Step 1: Generate all signals
    all_signals = []
    asset_counts = {}
    for asset in ASSET_YAHOO_SYMBOLS.keys():
        try:
            sigs = generate_signals_with_paths(asset, fmp_key, years, max_forward_bars=40)
            if sigs:
                all_signals.extend(sigs)
                asset_counts[asset] = len(sigs)
        except Exception as e:
            logger.error(f"[COMPREHENSIVE] {asset}: {e}")
        time.sleep(0.3)

    logger.info(f"[COMPREHENSIVE] Generated {len(all_signals)} signals from {len(asset_counts)} assets")

    if len(all_signals) < 500:
        return {"error": f"Too few signals: {len(all_signals)}"}

    # Segment
    futures = [s for s in all_signals if s["market"] == "futures"]
    crypto = [s for s in all_signals if s["market"] == "crypto"]
    futures_buy = [s for s in futures if s["signal_type"] == "BUY"]
    futures_sell = [s for s in futures if s["signal_type"] == "SELL"]

    # Step 2: Walk-forward with FULL grid (the deployed params)
    from walk_forward import walk_forward_validate, rolling_walk_forward

    logger.info(f"[COMPREHENSIVE] Running walk-forward validation...")
    wf_all = walk_forward_validate(all_signals, train_cutoff="2020-01-01")
    wf_futures_buy = walk_forward_validate(futures_buy, train_cutoff="2020-01-01") if len(futures_buy) >= 200 else {"error": "insufficient"}
    wf_futures_sell = walk_forward_validate(futures_sell, train_cutoff="2020-01-01") if len(futures_sell) >= 200 else {"error": "insufficient"}

    logger.info(f"[COMPREHENSIVE] Running rolling walk-forward...")
    rolling = rolling_walk_forward(all_signals, window_years=4, test_years=1)

    # Step 3: Transaction cost analysis
    logger.info(f"[COMPREHENSIVE] Applying transaction costs...")
    # Evaluate with optimal params
    raw_futures = evaluate_tpsl(futures, tp_mult=1.5, sl_mult=3.0, max_bars=5, min_consensus=3)
    raw_crypto = evaluate_tpsl(crypto, tp_mult=2.5, sl_mult=2.0, max_bars=15, min_consensus=2)
    cost_futures = apply_transaction_costs(raw_futures, "futures")
    cost_crypto = apply_transaction_costs(raw_crypto, "crypto")

    # Step 4: Parameter sensitivity
    logger.info(f"[COMPREHENSIVE] Running parameter sensitivity...")
    sensitivity_futures = parameter_sensitivity(futures, center_tp=1.5, center_sl=3.0, center_bars=5, center_cons=3)
    sensitivity_crypto = parameter_sensitivity(crypto, center_tp=2.5, center_sl=2.0, center_bars=15, center_cons=2)

    # Step 5: Multi-cutoff stability
    logger.info(f"[COMPREHENSIVE] Running multi-cutoff stability...")
    # Determine valid cutoffs based on available data
    dates = sorted(set(s.get("date", "")[:4] for s in all_signals if s.get("date")))
    min_year = int(min(dates)) if dates else 2006
    cutoffs = []
    for y in range(min_year + 4, min_year + years - 2, 2):
        cutoffs.append(f"{y}-01-01")
    multi_cutoff = multi_cutoff_validation(all_signals, cutoffs=cutoffs[:5])

    # Step 6: Regime-conditional analysis
    logger.info(f"[COMPREHENSIVE] Running regime-conditional analysis...")
    regime_futures = regime_conditional_test(futures, tp_mult=1.5, sl_mult=3.0, max_bars=5, min_consensus=3)
    regime_crypto = regime_conditional_test(crypto, tp_mult=2.5, sl_mult=2.0, max_bars=15, min_consensus=2)

    elapsed = round(time.time() - start_time, 1)

    # ═══════════════════════════════════════════════════════════
    # FINAL VERDICT (combines all tests)
    # ═══════════════════════════════════════════════════════════
    scores = {
        "walk_forward": 0,
        "transaction_costs": 0,
        "sensitivity": 0,
        "multi_cutoff": 0,
        "regime": 0,
    }

    # Walk-forward score (0-25)
    wf_verdict = wf_all.get("verdict", "CURVE_FIT")
    if wf_verdict == "STRONG_EDGE": scores["walk_forward"] = 25
    elif wf_verdict == "MARGINAL_EDGE": scores["walk_forward"] = 15
    elif wf_verdict == "WEAK_EDGE": scores["walk_forward"] = 5

    # Rolling walk-forward bonus (0-15)
    rolling_verdict = rolling.get("verdict", "NO_EDGE")
    if rolling_verdict == "ROBUST_EDGE": scores["walk_forward"] += 15
    elif rolling_verdict == "MODERATE_EDGE": scores["walk_forward"] += 8

    # Transaction costs (0-20)
    futures_adj_pf = cost_futures.get("adj_profit_factor", 0)
    if futures_adj_pf > 1.5: scores["transaction_costs"] = 20
    elif futures_adj_pf > 1.2: scores["transaction_costs"] = 15
    elif futures_adj_pf > 1.0: scores["transaction_costs"] = 10
    elif futures_adj_pf > 0.8: scores["transaction_costs"] = 3

    # Sensitivity (0-20)
    sens_verdict = sensitivity_futures.get("verdict", "FRAGILE")
    if sens_verdict == "VERY_ROBUST": scores["sensitivity"] = 20
    elif sens_verdict == "ROBUST": scores["sensitivity"] = 15
    elif sens_verdict == "MODERATE": scores["sensitivity"] = 8

    # Multi-cutoff (0-10)
    mc_stability = multi_cutoff.get("stability", "CUTOFF_DEPENDENT")
    if mc_stability == "VERY_STABLE": scores["multi_cutoff"] = 10
    elif mc_stability == "STABLE": scores["multi_cutoff"] = 7
    elif mc_stability == "UNSTABLE": scores["multi_cutoff"] = 3

    # Regime (0-10)
    regime_verdict = regime_futures.get("verdict", "REGIME_DEPENDENT")
    if regime_verdict == "ALL_REGIME_EDGE": scores["regime"] = 10
    elif regime_verdict == "PARTIAL_REGIME_EDGE": scores["regime"] = 5

    total_score = sum(scores.values())

    if total_score >= 80:
        final_verdict = "PRODUCTION_READY"
        final_explanation = f"Score {total_score}/100. System passed all validation tests. Safe to deploy with real capital."
    elif total_score >= 60:
        final_verdict = "DEPLOY_WITH_CAUTION"
        final_explanation = f"Score {total_score}/100. Most tests pass but some weaknesses. Use reduced position sizes."
    elif total_score >= 40:
        final_verdict = "PAPER_TRADE_ONLY"
        final_explanation = f"Score {total_score}/100. Significant gaps in validation. Paper trade first."
    else:
        final_verdict = "DO_NOT_TRADE"
        final_explanation = f"Score {total_score}/100. Validation failed. Do not risk real capital."

    return {
        "final_verdict": final_verdict,
        "final_explanation": final_explanation,
        "validation_score": total_score,
        "score_breakdown": scores,
        "total_signals": len(all_signals),
        "years": years,
        "assets": len(asset_counts),
        "walk_forward": {
            "all": {k: v for k, v in wf_all.items() if k != "top5_on_test"},
            "futures_buy": {k: v for k, v in wf_futures_buy.items() if k != "top5_on_test"} if isinstance(wf_futures_buy, dict) else wf_futures_buy,
            "futures_sell": {k: v for k, v in wf_futures_sell.items() if k != "top5_on_test"} if isinstance(wf_futures_sell, dict) else wf_futures_sell,
        },
        "rolling_walk_forward": rolling,
        "transaction_costs": {
            "futures": {k: v for k, v in cost_futures.items() if k != "ci_95"},
            "crypto": {k: v for k, v in cost_crypto.items() if k != "ci_95"},
        },
        "parameter_sensitivity": {
            "futures": sensitivity_futures,
            "crypto": sensitivity_crypto,
        },
        "multi_cutoff_stability": multi_cutoff,
        "regime_analysis": {
            "futures": regime_futures,
            "crypto": regime_crypto,
        },
        "computation_time_s": elapsed,
    }
