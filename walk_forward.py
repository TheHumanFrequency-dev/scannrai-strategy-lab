"""
ScannrAI Walk-Forward Validator + Timeframe Converter
=====================================================

Part 1: Walk-Forward Validation
  Splits 10yr data into train (2016-2022) and test (2023-2026).
  Optimizes TP/SL on train, evaluates on test WITHOUT re-tuning.
  If PF holds >1.0 on test, the edge is real. If it collapses, we curve-fit.
  
  Also runs ROLLING walk-forward:
    Fold 1: train 2016-2019, test 2020
    Fold 2: train 2017-2020, test 2021
    Fold 3: train 2018-2021, test 2022
    Fold 4: train 2019-2022, test 2023
    Fold 5: train 2020-2023, test 2024
    Fold 6: train 2021-2024, test 2025

Part 2: Timeframe Conversion
  The optimizer found optimal parameters on DAILY bars.
  The live scanner runs on HOURLY bars.
  This module converts daily parameters to hourly equivalents.
  
  Key relationships:
    - Daily ATR ≈ Hourly ATR × √(N_hours_per_day)
    - TP/SL multipliers are in ATR units, so they're timeframe-independent
    - BUT "5 bars max hold" means 5 DAYS on daily, 5 HOURS on hourly
    - The daily "5 bar" hold = ~32-35 hourly bars (6.5 trading hours/day × 5)
    - So hourly max_hold should be ~33 bars, not 5
"""

import math
import time
import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict

import numpy as np

from historical_backtest import (
    calc_rsi, calc_ema, calc_macd, calc_bb, calc_atr, calc_adx,
    detect_regime, detect_squeeze, generate_signal_at_bar,
    fetch_daily_data, ASSET_YAHOO_SYMBOLS, ASSET_MARKET
)
from tpsl_optimizer import (
    generate_signals_with_paths, evaluate_tpsl, run_optimization
)

logger = logging.getLogger("walk-forward")


# ═══════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════

def split_signals_by_date(signals: List[Dict], cutoff_date: str) -> tuple:
    """Split signals into train (before cutoff) and test (after cutoff)."""
    train = [s for s in signals if s.get("date", "") < cutoff_date]
    test = [s for s in signals if s.get("date", "") >= cutoff_date]
    return train, test


def walk_forward_validate(
    all_signals: List[Dict],
    train_cutoff: str = "2023-01-01",
    tp_range: List[float] = None,
    sl_range: List[float] = None,
    max_bars_range: List[int] = None,
    consensus_range: List[int] = None,
) -> Dict:
    """
    Walk-forward validation: optimize on train period, evaluate on test period.
    
    The test period NEVER participates in parameter selection.
    If PF > 1.0 on test, the edge generalizes. If not, it's curve-fitting.
    """
    start_time = time.time()

    if tp_range is None:
        tp_range = [0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    if sl_range is None:
        sl_range = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    if max_bars_range is None:
        max_bars_range = [5, 10, 15, 20, 30]
    if consensus_range is None:
        consensus_range = [2, 3, 4, 5]

    train_signals, test_signals = split_signals_by_date(all_signals, train_cutoff)

    logger.info(f"[WF] Split: {len(train_signals)} train (before {train_cutoff}), {len(test_signals)} test (after)")

    if len(train_signals) < 100 or len(test_signals) < 30:
        return {
            "error": f"Insufficient signals: train={len(train_signals)}, test={len(test_signals)}",
            "train_count": len(train_signals),
            "test_count": len(test_signals),
        }

    # Step 1: Optimize on TRAIN data only
    logger.info(f"[WF] Optimizing on train set ({len(train_signals)} signals)...")
    train_result = run_optimization(
        train_signals, "train",
        tp_range=tp_range, sl_range=sl_range,
        max_bars_range=max_bars_range, consensus_range=consensus_range,
    )

    best_train = train_result.get("best_overall")
    if not best_train:
        return {"error": "No valid parameters found on train set", "train_result": train_result}

    # Step 2: Evaluate best train params on TEST data (no re-optimization!)
    logger.info(f"[WF] Evaluating on test set: TP={best_train['tp_mult']} SL={best_train['sl_mult']} "
                f"bars={best_train['max_bars']} cons≥{best_train['min_consensus']}")

    test_result = evaluate_tpsl(
        test_signals,
        tp_mult=best_train["tp_mult"],
        sl_mult=best_train["sl_mult"],
        max_bars=best_train["max_bars"],
        min_consensus=best_train["min_consensus"],
    )

    # Step 3: Also evaluate top 5 train configs on test (robustness check)
    top5_test_results = []
    for i, cfg in enumerate(train_result.get("top_20", [])[:5]):
        tr = evaluate_tpsl(
            test_signals,
            tp_mult=cfg["tp_mult"],
            sl_mult=cfg["sl_mult"],
            max_bars=cfg["max_bars"],
            min_consensus=cfg["min_consensus"],
        )
        if tr.get("valid"):
            tr["train_rank"] = i + 1
            tr["train_pf"] = cfg["profit_factor"]
            tr["train_wr"] = cfg["win_rate"]
            top5_test_results.append(tr)

    # Step 4: Evaluate the "universal" config (TP=1.5, SL=3, bars=5, cons≥3)
    universal_test = evaluate_tpsl(test_signals, tp_mult=1.5, sl_mult=3.0, max_bars=5, min_consensus=3)
    universal_train = evaluate_tpsl(train_signals, tp_mult=1.5, sl_mult=3.0, max_bars=5, min_consensus=3)

    # Step 5: Evaluate old params on both (baseline comparison)
    old_test = evaluate_tpsl(test_signals, tp_mult=2.0, sl_mult=1.2, max_bars=20, min_consensus=2)
    old_train = evaluate_tpsl(train_signals, tp_mult=2.0, sl_mult=1.2, max_bars=20, min_consensus=2)

    elapsed = round(time.time() - start_time, 1)

    # Verdict
    test_pf = test_result.get("profit_factor", 0) if test_result.get("valid") else 0
    train_pf = best_train.get("profit_factor", 0)
    pf_decay = (train_pf - test_pf) / train_pf if train_pf > 0 else 1.0

    if test_pf >= 1.5:
        verdict = "STRONG_EDGE"
        explanation = f"PF {test_pf:.2f} on unseen data. Edge is real and robust."
    elif test_pf >= 1.0:
        verdict = "MARGINAL_EDGE"
        explanation = f"PF {test_pf:.2f} on unseen data. Profitable but thin margin. Be cautious with position sizing."
    elif test_pf >= 0.8:
        verdict = "WEAK_EDGE"
        explanation = f"PF {test_pf:.2f} on unseen data. Nearly break-even. Edge may not survive trading costs."
    else:
        verdict = "CURVE_FIT"
        explanation = f"PF {test_pf:.2f} on unseen data (vs {train_pf:.2f} in-sample). The optimization curve-fit to historical patterns that don't persist."

    return {
        "verdict": verdict,
        "explanation": explanation,
        "train_cutoff": train_cutoff,
        "train_signals": len(train_signals),
        "test_signals": len(test_signals),
        "best_train_params": {
            "tp_mult": best_train["tp_mult"],
            "sl_mult": best_train["sl_mult"],
            "max_bars": best_train["max_bars"],
            "min_consensus": best_train["min_consensus"],
        },
        "train_performance": {
            "profit_factor": train_pf,
            "win_rate": best_train["win_rate"],
            "ev_per_trade": best_train["ev_per_trade"],
            "resolved": best_train["resolved"],
        },
        "test_performance": {
            "profit_factor": test_result.get("profit_factor", 0) if test_result.get("valid") else None,
            "win_rate": test_result.get("win_rate", 0) if test_result.get("valid") else None,
            "ev_per_trade": test_result.get("ev_per_trade", 0) if test_result.get("valid") else None,
            "resolved": test_result.get("resolved", 0),
            "won": test_result.get("won", 0),
            "lost": test_result.get("lost", 0),
            "max_drawdown": test_result.get("max_drawdown_pct", 0) if test_result.get("valid") else None,
            "max_consecutive_losses": test_result.get("max_consecutive_losses", 0) if test_result.get("valid") else None,
        },
        "pf_decay": round(pf_decay, 4),
        "top5_on_test": top5_test_results,
        "universal_config": {
            "params": {"tp": 1.5, "sl": 3.0, "bars": 5, "cons": 3},
            "train": {
                "pf": universal_train.get("profit_factor", 0) if universal_train.get("valid") else None,
                "wr": universal_train.get("win_rate", 0) if universal_train.get("valid") else None,
                "resolved": universal_train.get("resolved", 0),
            },
            "test": {
                "pf": universal_test.get("profit_factor", 0) if universal_test.get("valid") else None,
                "wr": universal_test.get("win_rate", 0) if universal_test.get("valid") else None,
                "resolved": universal_test.get("resolved", 0),
            },
        },
        "old_config_baseline": {
            "params": {"tp": 2.0, "sl": 1.2, "bars": 20, "cons": 2},
            "train": {
                "pf": old_train.get("profit_factor", 0) if old_train.get("valid") else None,
                "wr": old_train.get("win_rate", 0) if old_train.get("valid") else None,
                "resolved": old_train.get("resolved", 0),
            },
            "test": {
                "pf": old_test.get("profit_factor", 0) if old_test.get("valid") else None,
                "wr": old_test.get("win_rate", 0) if old_test.get("valid") else None,
                "resolved": old_test.get("resolved", 0),
            },
        },
        "computation_time_s": elapsed,
    }


def rolling_walk_forward(
    all_signals: List[Dict],
    window_years: int = 4,
    test_years: int = 1,
) -> Dict:
    """
    Rolling walk-forward: slide a train/test window across the full dataset.
    Each fold optimizes on train window, evaluates on the next test window.
    
    With 10 years of data and 4yr train + 1yr test:
      Fold 1: train 2016-2019, test 2020
      Fold 2: train 2017-2020, test 2021
      ...
      Fold 6: train 2021-2024, test 2025
    """
    start_time = time.time()
    
    # Get date range from signals
    dates = sorted(set(s.get("date", "")[:4] for s in all_signals if s.get("date")))
    if len(dates) < window_years + test_years:
        return {"error": f"Need {window_years + test_years}+ years, have {len(dates)}"}
    
    years = sorted(set(dates))
    logger.info(f"[ROLLING-WF] Years available: {years}")
    
    folds = []
    # Use simplified TP/SL grid for speed (rolling has many folds)
    tp_range = [1.0, 1.25, 1.5, 2.0, 3.0]
    sl_range = [1.0, 1.5, 2.0, 3.0]
    bars_range = [5, 10, 20]
    cons_range = [2, 3, 4]
    
    for i in range(len(years) - window_years - test_years + 1):
        train_start = years[i]
        train_end = years[i + window_years - 1]
        test_year = years[i + window_years]
        
        train_cutoff = f"{int(test_year)}-01-01"
        test_end = f"{int(test_year) + test_years}-01-01"
        
        train_sigs = [s for s in all_signals 
                      if s.get("date", "") >= f"{train_start}-01-01" 
                      and s.get("date", "") < train_cutoff]
        test_sigs = [s for s in all_signals
                     if s.get("date", "") >= train_cutoff
                     and s.get("date", "") < test_end]
        
        logger.info(f"[ROLLING-WF] Fold {i+1}: train {train_start}-{train_end} ({len(train_sigs)}), "
                     f"test {test_year} ({len(test_sigs)})")
        
        if len(train_sigs) < 50 or len(test_sigs) < 20:
            folds.append({
                "fold": i + 1,
                "train_period": f"{train_start}-{train_end}",
                "test_period": test_year,
                "error": f"Insufficient signals: train={len(train_sigs)}, test={len(test_sigs)}",
            })
            continue
        
        # Optimize on train
        train_opt = run_optimization(
            train_sigs, f"fold_{i+1}_train",
            tp_range=tp_range, sl_range=sl_range,
            max_bars_range=bars_range, consensus_range=cons_range,
        )
        
        best = train_opt.get("best_overall")
        if not best:
            folds.append({
                "fold": i + 1,
                "train_period": f"{train_start}-{train_end}",
                "test_period": test_year,
                "error": "No valid params on train",
            })
            continue
        
        # Evaluate on test
        test_eval = evaluate_tpsl(
            test_sigs,
            tp_mult=best["tp_mult"],
            sl_mult=best["sl_mult"],
            max_bars=best["max_bars"],
            min_consensus=best["min_consensus"],
        )
        
        fold_result = {
            "fold": i + 1,
            "train_period": f"{train_start}-{train_end}",
            "test_period": test_year,
            "train_signals": len(train_sigs),
            "test_signals": len(test_sigs),
            "best_params": {
                "tp": best["tp_mult"],
                "sl": best["sl_mult"],
                "bars": best["max_bars"],
                "cons": best["min_consensus"],
            },
            "train_pf": best["profit_factor"],
            "train_wr": best["win_rate"],
            "test_pf": test_eval.get("profit_factor", 0) if test_eval.get("valid") else None,
            "test_wr": test_eval.get("win_rate", 0) if test_eval.get("valid") else None,
            "test_ev": test_eval.get("ev_per_trade", 0) if test_eval.get("valid") else None,
            "test_resolved": test_eval.get("resolved", 0),
        }
        folds.append(fold_result)
    
    # Aggregate
    valid_folds = [f for f in folds if f.get("test_pf") is not None]
    profitable_folds = [f for f in valid_folds if f["test_pf"] > 1.0]
    
    avg_test_pf = float(np.mean([f["test_pf"] for f in valid_folds])) if valid_folds else 0
    avg_train_pf = float(np.mean([f["train_pf"] for f in valid_folds])) if valid_folds else 0
    avg_decay = (avg_train_pf - avg_test_pf) / avg_train_pf if avg_train_pf > 0 else 1.0
    
    # Check parameter stability (do different folds find similar optimal params?)
    if valid_folds:
        tp_vals = [f["best_params"]["tp"] for f in valid_folds]
        sl_vals = [f["best_params"]["sl"] for f in valid_folds]
        tp_std = float(np.std(tp_vals))
        sl_std = float(np.std(sl_vals))
        param_stable = tp_std < 0.5 and sl_std < 0.5
    else:
        tp_std = sl_std = 0
        param_stable = False
    
    if len(profitable_folds) >= len(valid_folds) * 0.7 and avg_test_pf > 1.2:
        verdict = "ROBUST_EDGE"
        explanation = (f"{len(profitable_folds)}/{len(valid_folds)} folds profitable. "
                       f"Avg test PF={avg_test_pf:.2f}. Edge is consistent across time periods.")
    elif len(profitable_folds) >= len(valid_folds) * 0.5 and avg_test_pf > 1.0:
        verdict = "MODERATE_EDGE"
        explanation = (f"{len(profitable_folds)}/{len(valid_folds)} folds profitable. "
                       f"Avg test PF={avg_test_pf:.2f}. Edge exists but varies by market regime.")
    elif avg_test_pf > 0.9:
        verdict = "WEAK_EDGE"
        explanation = (f"Only {len(profitable_folds)}/{len(valid_folds)} folds profitable. "
                       f"Avg test PF={avg_test_pf:.2f}. Edge is fragile.")
    else:
        verdict = "NO_EDGE"
        explanation = (f"Only {len(profitable_folds)}/{len(valid_folds)} folds profitable. "
                       f"Avg test PF={avg_test_pf:.2f}. Optimization was curve-fitting.")
    
    elapsed = round(time.time() - start_time, 1)
    
    return {
        "verdict": verdict,
        "explanation": explanation,
        "total_folds": len(folds),
        "valid_folds": len(valid_folds),
        "profitable_folds": len(profitable_folds),
        "avg_train_pf": round(avg_train_pf, 4),
        "avg_test_pf": round(avg_test_pf, 4),
        "avg_pf_decay": round(avg_decay, 4),
        "param_stability": {
            "tp_std": round(tp_std, 3),
            "sl_std": round(sl_std, 3),
            "stable": param_stable,
        },
        "folds": folds,
        "computation_time_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════
# TIMEFRAME CONVERSION (DAILY → HOURLY)
# ═══════════════════════════════════════════════════════════════

def convert_daily_to_hourly(daily_config: Dict) -> Dict:
    """
    Convert daily-optimized parameters to hourly equivalents.
    
    Key relationships for futures (6.5 trading hours/day):
      - Daily "5 bar hold" = 5 days = ~33 hourly bars
      - TP/SL multipliers are in ATR units — ATR is computed per-timeframe
        so the multipliers transfer directly (1.5× hourly_ATR ≈ 1.5× daily_ATR scaled)
      - BUT the R:R dynamics change: hourly has more noise, more false triggers
      - Conservative approach: keep TP/SL multipliers, scale hold time
    
    For crypto (24/7 markets):
      - Daily "5 bar hold" = 5 days = ~120 hourly bars (24h × 5)
      - But realistically we want faster exits on hourly
      - Use 24-48 hourly bars (1-2 days equivalent)
    """
    hourly_config = {
        "version": "2.0-hourly",
        "conversion_method": "direct_multiplier_transfer_with_hold_scaling",
        "notes": [
            "TP/SL multipliers transfer directly (ATR is timeframe-specific)",
            "Max hold bars scaled: daily_bars × hours_per_day",
            "Consensus gates unchanged (computed per scan, not per bar)",
            "IMPORTANT: These are theoretical conversions. Live hourly performance",
            "may differ due to: intraday noise patterns, market microstructure,",
            "overnight gaps (futures), and session-specific liquidity variations.",
            "Monitor live performance and recalibrate after 200+ hourly signals.",
        ],
    }
    
    futures_hours_per_day = 6.5  # Regular trading hours
    crypto_hours_per_day = 24.0  # 24/7
    
    for seg_name, seg_params in daily_config.items():
        if not isinstance(seg_params, dict) or "tp" not in seg_params:
            continue
        
        is_crypto = "crypto" in seg_name
        hours_per_day = crypto_hours_per_day if is_crypto else futures_hours_per_day
        
        daily_bars = seg_params.get("bars", 5)
        # Scale hold time, but cap to avoid holding forever on hourly
        hourly_bars = min(int(daily_bars * hours_per_day), 48 if is_crypto else 33)
        
        hourly_config[seg_name] = {
            "tp_mult": seg_params["tp"],      # Transfers directly
            "sl_mult": seg_params["sl"],      # Transfers directly
            "max_hold_bars": hourly_bars,     # Scaled from daily
            "min_consensus": seg_params.get("min_cons", 3),  # Unchanged
            "daily_equivalent_bars": daily_bars,
            "conversion_factor": round(hours_per_day, 1),
            # Expected performance (apply ~20% decay for timeframe noise)
            "expected_pf_range": [
                round(seg_params.get("pf", 1.0) * 0.65, 2),
                round(seg_params.get("pf", 1.0) * 0.85, 2),
            ],
            "expected_wr_range": [
                round(seg_params.get("wr", 0.5) * 0.85, 3),
                round(seg_params.get("wr", 0.5) * 0.95, 3),
            ],
        }
    
    return hourly_config


def generate_scanner_tpsl_config(optimization_results: Dict, walk_forward_results: Dict) -> Dict:
    """
    Generate the final scanner configuration based on optimization AND validation.
    
    If walk-forward validates the edge → use optimized parameters
    If walk-forward shows curve-fitting → use conservative defaults
    """
    deploy_config = optimization_results.get("deployment_config", {})
    wf_verdict = walk_forward_results.get("verdict", "NO_EDGE")
    
    if wf_verdict in ("STRONG_EDGE", "ROBUST_EDGE"):
        # Full confidence: deploy optimized params with hourly conversion
        hourly = convert_daily_to_hourly(deploy_config)
        hourly["confidence"] = "HIGH"
        hourly["deployment_reason"] = f"Walk-forward verdict: {wf_verdict}"
        return hourly
    
    elif wf_verdict in ("MARGINAL_EDGE", "MODERATE_EDGE"):
        # Partial confidence: deploy optimized params but with tighter risk
        hourly = convert_daily_to_hourly(deploy_config)
        # Tighten SL by 20% (less room for error on marginal edge)
        for seg_name, params in hourly.items():
            if isinstance(params, dict) and "sl_mult" in params:
                params["sl_mult"] = round(params["sl_mult"] * 0.8, 2)
        hourly["confidence"] = "MEDIUM"
        hourly["deployment_reason"] = f"Walk-forward verdict: {wf_verdict}. SL tightened 20%."
        return hourly
    
    elif wf_verdict in ("WEAK_EDGE",):
        # Low confidence: use very conservative parameters
        conservative = {
            "version": "2.0-conservative",
            "confidence": "LOW",
            "deployment_reason": f"Walk-forward verdict: {wf_verdict}. Using conservative defaults.",
        }
        for seg in ["crypto_buy", "futures_buy", "futures_sell"]:
            conservative[seg] = {
                "tp_mult": 1.0,    # Very tight TP
                "sl_mult": 2.0,    # Moderate SL
                "max_hold_bars": 20,
                "min_consensus": 4,  # High bar
            }
        return conservative
    
    else:
        # No edge: minimal config, basically paused
        return {
            "version": "2.0-paused",
            "confidence": "NONE",
            "deployment_reason": f"Walk-forward verdict: {wf_verdict}. Edge not validated.",
            "recommendation": "Do not trade live. Re-examine signal generation logic.",
        }
