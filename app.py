"""
ScannrAI Strategy Lab — Railway Microservice
=============================================
Signal calibration, backtesting, Monte Carlo simulation, and asset qualification.

All endpoints are SYNCHRONOUS — results return immediately in the response.
599 signals takes <1 second to process. No async needed.

Endpoints:
  GET  /health         — Health check
  POST /backtest       — Full signal analysis + Platt recalibration
  POST /validate       — Walk-forward validation (train/test split)
  POST /monte-carlo    — Monte Carlo equity curve simulation
  POST /asset-qualify  — Per-asset win rate + qualification
  GET  /results        — Returns last backtest results (cached in memory)

Deploy: Push to GitHub → Connect Railway → auto-deploys via Dockerfile.
"""

import os
import json
import math
import logging
import time
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from flask import Flask, request, jsonify, Response

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("strategy-lab")


# ── CORS ──────────────────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    """Allow cross-origin requests from any origin (Base44 apps, scannrai.com, etc.)."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Max-Age"] = "3600"
    return response


@app.route("/health", methods=["OPTIONS"])
@app.route("/backtest", methods=["OPTIONS"])
@app.route("/results", methods=["OPTIONS"])
@app.route("/validate", methods=["OPTIONS"])
@app.route("/monte-carlo", methods=["OPTIONS"])
@app.route("/asset-qualify", methods=["OPTIONS"])
@app.route("/historical-backtest/asset", methods=["OPTIONS"])
@app.route("/historical-backtest/full", methods=["OPTIONS"])
@app.route("/historical-backtest/results", methods=["OPTIONS"])
def handle_preflight():
    """Handle CORS preflight requests."""
    return "", 204

# ── In-memory cache for /results ──────────────────────────────
_last_results: Dict[str, Any] = {}

# ── Auth ──────────────────────────────────────────────────────
API_KEY = os.environ.get("SCANNRAI_API_KEY", "")

def check_auth() -> bool:
    if not API_KEY:
        return True
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    return token == API_KEY.strip()


# ── Statistical Helpers ───────────────────────────────────────

def wilson_ci(wins: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for binomial proportion."""
    if total == 0:
        return (0.0, 0.0)
    p = wins / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0, center - spread), min(1, center + spread))


def platt_fit(signals: List[Dict]) -> Dict[str, float]:
    """
    Fit Platt scaling parameters via gradient descent.
    Maps raw confidence → calibrated probability.
    P(win | score) = 1 / (1 + exp(A * score + B))
    """
    resolved = [(s["confidence"], 1 if s["status"] == "won" else 0) 
                for s in signals if s.get("confidence") is not None]
    
    if len(resolved) < 30:
        return {"A": 0.0, "B": 0.0, "n": len(resolved), "error": "insufficient_data"}
    
    scores = np.array([r[0] for r in resolved])
    labels = np.array([r[1] for r in resolved])
    
    # Normalize scores to [0, 1]
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-6:
        return {"A": 0.0, "B": 0.0, "n": len(resolved), "error": "no_score_variance"}
    s_norm = (scores - s_min) / (s_max - s_min)
    
    # Gradient descent for Platt parameters
    A, B = -1.0, 0.0
    lr = 0.01
    for _ in range(2000):
        z = A * s_norm + B
        z = np.clip(z, -30, 30)
        p = 1.0 / (1.0 + np.exp(z))
        grad_A = np.mean((p - labels) * s_norm)
        grad_B = np.mean(p - labels)
        A -= lr * grad_A
        B -= lr * grad_B
    
    # Calculate calibration error (ECE)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    z_all = A * s_norm + B
    z_all = np.clip(z_all, -30, 30)
    pred_probs = 1.0 / (1.0 + np.exp(z_all))
    
    ece = 0.0
    bin_data = []
    for i in range(n_bins):
        mask = (pred_probs >= bin_edges[i]) & (pred_probs < bin_edges[i + 1])
        if mask.sum() > 0:
            avg_pred = pred_probs[mask].mean()
            avg_actual = labels[mask].mean()
            bin_count = int(mask.sum())
            ece += abs(avg_pred - avg_actual) * bin_count / len(labels)
            bin_data.append({
                "bin": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                "count": bin_count,
                "predicted": round(float(avg_pred), 4),
                "actual": round(float(avg_actual), 4),
                "gap": round(float(abs(avg_pred - avg_actual)), 4)
            })
    
    return {
        "A": round(float(A), 6),
        "B": round(float(B), 6),
        "score_min": round(float(s_min), 2),
        "score_max": round(float(s_max), 2),
        "n": len(resolved),
        "ece": round(float(ece), 4),
        "calibration_bins": bin_data
    }


def kelly_criterion(win_rate: float, avg_win_pct: float, avg_loss_pct: float) -> Dict[str, float]:
    """Kelly Criterion position sizing."""
    if avg_loss_pct == 0 or avg_win_pct == 0:
        return {"full_kelly": 0, "half_kelly": 0, "quarter_kelly": 0}
    
    b = avg_win_pct / avg_loss_pct  # win/loss ratio
    q = 1 - win_rate
    full = (win_rate * b - q) / b if b > 0 else 0
    full = max(0, min(full, 1))  # Clamp [0, 1]
    
    return {
        "full_kelly": round(full * 100, 2),
        "half_kelly": round(full * 50, 2),
        "quarter_kelly": round(full * 25, 2),
        "win_loss_ratio": round(b, 4),
        "edge": round(win_rate * b - q, 4)
    }


# ── Core Analysis Functions ───────────────────────────────────

def analyze_signals(signals: List[Dict]) -> Dict[str, Any]:
    """Comprehensive signal analysis."""
    start = time.time()
    
    won = [s for s in signals if s["status"] == "won"]
    lost = [s for s in signals if s["status"] == "lost"]
    total = len(signals)
    n_won = len(won)
    n_lost = len(lost)
    
    if total == 0:
        return {"error": "no_signals", "count": 0}
    
    wr = n_won / total
    ci_low, ci_high = wilson_ci(n_won, total)
    
    # P&L stats
    pnls = [s.get("outcome_pnl_pct", 0) or 0 for s in signals]
    win_pnls = [s.get("outcome_pnl_pct", 0) or 0 for s in won]
    loss_pnls = [abs(s.get("outcome_pnl_pct", 0) or 0) for s in lost]
    
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    profit_factor = (sum(win_pnls) / sum(loss_pnls)) if loss_pnls and sum(loss_pnls) > 0 else float("inf")
    
    # By market
    by_market = defaultdict(lambda: {"won": 0, "lost": 0, "pnls": []})
    for s in signals:
        m = s.get("market", "unknown")
        by_market[m]["won" if s["status"] == "won" else "lost"] += 1
        by_market[m]["pnls"].append(s.get("outcome_pnl_pct", 0) or 0)
    
    market_stats = {}
    for m, d in by_market.items():
        mt = d["won"] + d["lost"]
        mwr = d["won"] / mt if mt > 0 else 0
        mci = wilson_ci(d["won"], mt)
        market_stats[m] = {
            "won": d["won"], "lost": d["lost"], "total": mt,
            "win_rate": round(mwr, 4),
            "ci_95": [round(mci[0], 4), round(mci[1], 4)],
            "avg_pnl": round(float(np.mean(d["pnls"])), 4) if d["pnls"] else 0,
            "total_pnl": round(float(sum(d["pnls"])), 4)
        }
    
    # By signal type
    by_type = defaultdict(lambda: {"won": 0, "lost": 0})
    for s in signals:
        t = s.get("signal_type", "UNKNOWN")
        by_type[t]["won" if s["status"] == "won" else "lost"] += 1
    
    type_stats = {}
    for t, d in by_type.items():
        tt = d["won"] + d["lost"]
        type_stats[t] = {
            "won": d["won"], "lost": d["lost"], "total": tt,
            "win_rate": round(d["won"] / tt, 4) if tt > 0 else 0
        }
    
    # By RSI zone
    rsi_zones = [
        ("RSI < 25", 0, 25), ("RSI 25-30", 25, 30), ("RSI 30-35", 30, 35),
        ("RSI 35-45", 35, 45), ("RSI 45-55", 45, 55), ("RSI 55-65", 55, 65),
        ("RSI 65-70", 65, 70), ("RSI 70-75", 70, 75), ("RSI > 75", 75, 101)
    ]
    rsi_stats = []
    for label, lo, hi in rsi_zones:
        zone_sigs = [s for s in signals if s.get("rsi") is not None and lo <= s["rsi"] < hi]
        zw = len([s for s in zone_sigs if s["status"] == "won"])
        zt = len(zone_sigs)
        if zt >= 5:
            zci = wilson_ci(zw, zt)
            rsi_stats.append({
                "zone": label, "won": zw, "total": zt,
                "win_rate": round(zw / zt, 4),
                "ci_95": [round(zci[0], 4), round(zci[1], 4)]
            })
    
    # By asset
    by_asset = defaultdict(lambda: {"won": 0, "lost": 0, "pnls": []})
    for s in signals:
        a = s.get("coin", "?")
        by_asset[a]["won" if s["status"] == "won" else "lost"] += 1
        by_asset[a]["pnls"].append(s.get("outcome_pnl_pct", 0) or 0)
    
    asset_stats = {}
    for a, d in by_asset.items():
        at = d["won"] + d["lost"]
        awr = d["won"] / at if at > 0 else 0
        asset_stats[a] = {
            "won": d["won"], "lost": d["lost"], "total": at,
            "win_rate": round(awr, 4),
            "avg_pnl": round(float(np.mean(d["pnls"])), 4) if d["pnls"] else 0,
            "qualified": at >= 10 and awr >= 0.40
        }
    
    # Factor attribution
    factor_wins = defaultdict(int)
    factor_total = defaultdict(int)
    for s in signals:
        for f in (s.get("key_factors") or []):
            factor_total[f] += 1
            if s["status"] == "won":
                factor_wins[f] += 1
    
    factor_stats = {}
    for f in factor_total:
        ft = factor_total[f]
        fw = factor_wins[f]
        if ft >= 10:
            fwr = fw / ft
            fci = wilson_ci(fw, ft)
            factor_stats[f] = {
                "won": fw, "total": ft,
                "win_rate": round(fwr, 4),
                "ci_95": [round(fci[0], 4), round(fci[1], 4)],
                "edge_vs_baseline": round(fwr - wr, 4)
            }
    
    # Platt scaling
    platt = platt_fit(signals)
    
    # Kelly criterion
    kelly = kelly_criterion(wr, float(avg_win), float(avg_loss))
    
    # Confidence bucket analysis
    conf_buckets = [
        ("50-60", 50, 60), ("60-70", 60, 70), ("70-80", 70, 80),
        ("80-90", 80, 90), ("90-100", 90, 101)
    ]
    conf_stats = []
    for label, lo, hi in conf_buckets:
        bucket = [s for s in signals if s.get("confidence") is not None and lo <= s["confidence"] < hi]
        bw = len([s for s in bucket if s["status"] == "won"])
        bt = len(bucket)
        if bt >= 5:
            bci = wilson_ci(bw, bt)
            conf_stats.append({
                "bucket": label, "won": bw, "total": bt,
                "win_rate": round(bw / bt, 4),
                "ci_95": [round(bci[0], 4), round(bci[1], 4)]
            })
    
    elapsed = round(time.time() - start, 3)
    
    return {
        "summary": {
            "total_signals": total,
            "won": n_won,
            "lost": n_lost,
            "win_rate": round(wr, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "avg_win_pct": round(float(avg_win), 4),
            "avg_loss_pct": round(float(avg_loss), 4),
            "profit_factor": round(float(profit_factor), 4) if profit_factor != float("inf") else "inf",
            "total_pnl_pct": round(float(sum(pnls)), 4),
            "sharpe_approx": round(float(np.mean(pnls) / np.std(pnls)) if np.std(pnls) > 0 else 0, 4),
            "max_drawdown_pct": round(float(min(np.minimum.accumulate(np.cumsum(pnls)) - np.cumsum(pnls))), 4) if pnls else 0
        },
        "by_market": market_stats,
        "by_signal_type": type_stats,
        "by_rsi_zone": rsi_stats,
        "by_asset": dict(sorted(asset_stats.items(), key=lambda x: x[1]["total"], reverse=True)),
        "by_confidence": conf_stats,
        "factor_attribution": dict(sorted(factor_stats.items(), key=lambda x: x[1]["edge_vs_baseline"], reverse=True)),
        "platt_scaling": platt,
        "kelly_criterion": kelly,
        "computation_time_s": elapsed,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


def walk_forward_validate(signals: List[Dict], n_folds: int = 3) -> Dict[str, Any]:
    """Walk-forward validation: train on period N, test on period N+1."""
    # Sort by created_date
    sorted_sigs = sorted(signals, key=lambda s: s.get("created_date", ""))
    fold_size = len(sorted_sigs) // n_folds
    
    if fold_size < 20:
        return {"error": "insufficient_data", "min_required": n_folds * 20, "actual": len(sorted_sigs)}
    
    folds = []
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = start_idx + fold_size if i < n_folds - 1 else len(sorted_sigs)
        fold_sigs = sorted_sigs[start_idx:end_idx]
        won = len([s for s in fold_sigs if s["status"] == "won"])
        total = len(fold_sigs)
        ci = wilson_ci(won, total)
        
        # Date range
        dates = [s.get("created_date", "") for s in fold_sigs if s.get("created_date")]
        
        folds.append({
            "fold": i + 1,
            "start": dates[0] if dates else None,
            "end": dates[-1] if dates else None,
            "total": total,
            "won": won,
            "win_rate": round(won / total, 4) if total > 0 else 0,
            "ci_95": [round(ci[0], 4), round(ci[1], 4)]
        })
    
    # Check if edge persists across folds
    win_rates = [f["win_rate"] for f in folds]
    edge_stable = max(win_rates) - min(win_rates) < 0.15  # <15% variation
    
    # Train Platt on fold 1, test on fold 2, validate on fold 3
    train_platt = platt_fit(sorted_sigs[:fold_size])
    test_platt = platt_fit(sorted_sigs[fold_size:2*fold_size])
    
    return {
        "n_folds": n_folds,
        "fold_size": fold_size,
        "folds": folds,
        "win_rate_variance": round(float(np.std(win_rates)), 4),
        "edge_stable": edge_stable,
        "verdict": "STABLE — edge persists across time periods" if edge_stable else "UNSTABLE — possible curve fit",
        "train_platt": train_platt,
        "test_platt": test_platt,
        "platt_drift": round(abs(train_platt.get("A", 0) - test_platt.get("A", 0)), 4)
    }


def monte_carlo_sim(signals: List[Dict], n_sims: int = 1000, account_size: float = 50000) -> Dict[str, Any]:
    """Monte Carlo simulation of equity curves."""
    pnls = [s.get("outcome_pnl_pct", 0) or 0 for s in signals]
    if not pnls:
        return {"error": "no_pnl_data"}
    
    n_trades = len(pnls)
    pnl_array = np.array(pnls)
    
    # Simulate n_sims equity curves by resampling trades
    np.random.seed(42)
    final_equities = []
    max_drawdowns = []
    
    for _ in range(n_sims):
        sampled = np.random.choice(pnl_array, size=n_trades, replace=True)
        equity = account_size
        peak = equity
        max_dd = 0
        
        for pnl_pct in sampled:
            equity *= (1 + pnl_pct / 100)
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        
        final_equities.append(equity)
        max_drawdowns.append(max_dd * 100)
    
    final_arr = np.array(final_equities)
    dd_arr = np.array(max_drawdowns)
    
    return {
        "account_size": account_size,
        "n_simulations": n_sims,
        "n_trades": n_trades,
        "final_equity": {
            "median": round(float(np.median(final_arr)), 2),
            "p5": round(float(np.percentile(final_arr, 5)), 2),
            "p25": round(float(np.percentile(final_arr, 25)), 2),
            "p75": round(float(np.percentile(final_arr, 75)), 2),
            "p95": round(float(np.percentile(final_arr, 95)), 2),
            "mean": round(float(np.mean(final_arr)), 2),
            "std": round(float(np.std(final_arr)), 2)
        },
        "max_drawdown_pct": {
            "median": round(float(np.median(dd_arr)), 2),
            "p5": round(float(np.percentile(dd_arr, 5)), 2),
            "p95": round(float(np.percentile(dd_arr, 95)), 2),
            "worst": round(float(np.max(dd_arr)), 2)
        },
        "ruin_probability": round(float(np.mean(final_arr < account_size * 0.5)), 4),
        "profit_probability": round(float(np.mean(final_arr > account_size)), 4),
        "double_probability": round(float(np.mean(final_arr > account_size * 2)), 4)
    }


def asset_qualify(signals: List[Dict], min_signals: int = 10, min_wr: float = 0.40) -> Dict[str, Any]:
    """Qualify assets based on statistical evidence."""
    by_asset = defaultdict(list)
    for s in signals:
        by_asset[s.get("coin", "?")].append(s)
    
    qualified = []
    disqualified = []
    insufficient = []
    
    for asset, sigs in sorted(by_asset.items()):
        won = len([s for s in sigs if s["status"] == "won"])
        total = len(sigs)
        wr = won / total if total > 0 else 0
        ci = wilson_ci(won, total)
        pnls = [s.get("outcome_pnl_pct", 0) or 0 for s in sigs]
        
        entry = {
            "asset": asset,
            "market": sigs[0].get("market", "unknown"),
            "won": won,
            "lost": total - won,
            "total": total,
            "win_rate": round(wr, 4),
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
            "avg_pnl": round(float(np.mean(pnls)), 4) if pnls else 0,
            "total_pnl": round(float(sum(pnls)), 4)
        }
        
        if total < min_signals:
            entry["reason"] = f"Only {total} signals (need {min_signals}+)"
            insufficient.append(entry)
        elif wr < min_wr:
            entry["reason"] = f"Win rate {wr:.1%} below {min_wr:.0%} threshold"
            disqualified.append(entry)
        elif ci[0] < min_wr * 0.8:
            entry["reason"] = f"CI lower bound {ci[0]:.1%} too close to threshold"
            entry["status"] = "probation"
            qualified.append(entry)
        else:
            entry["status"] = "qualified"
            qualified.append(entry)
    
    return {
        "thresholds": {"min_signals": min_signals, "min_win_rate": min_wr},
        "qualified": sorted(qualified, key=lambda x: x["win_rate"], reverse=True),
        "disqualified": sorted(disqualified, key=lambda x: x["win_rate"]),
        "insufficient_data": insufficient,
        "summary": {
            "qualified_count": len(qualified),
            "disqualified_count": len(disqualified),
            "insufficient_count": len(insufficient)
        }
    }


# ── Flask Routes ──────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "scannrai-strategy-lab",
        "version": "2.0.0",
        "endpoints": ["/health", "/backtest", "/results", "/validate", "/monte-carlo", "/asset-qualify"],
        "last_backtest": _last_results.get("timestamp"),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })


@app.route("/backtest", methods=["POST"])
def backtest():
    """Full signal analysis + Platt recalibration."""
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body required. Send {\"signals\": [...]}"}), 400
        
        signals = data.get("signals", [])
        if not signals:
            return jsonify({"error": "No signals provided. Send {\"signals\": [{...}, ...]}"}), 400
        
        # Filter to resolved signals only
        resolved = [s for s in signals if s.get("status") in ("won", "lost")]
        if len(resolved) < 10:
            return jsonify({"error": f"Need 10+ resolved signals, got {len(resolved)}"}), 400
        
        logger.info(f"[BACKTEST] Analyzing {len(resolved)} resolved signals")
        results = analyze_signals(resolved)
        
        # Cache for /results endpoint
        global _last_results
        _last_results = results
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"[BACKTEST ERROR] {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/results", methods=["GET"])
def results():
    """Return last backtest results. Always returns valid JSON."""
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    if not _last_results:
        return jsonify({
            "status": "no_results",
            "message": "No backtest has been run yet. POST to /backtest with your signal data first.",
            "usage": {
                "method": "POST",
                "url": "/backtest",
                "body": "{\"signals\": [{\"status\": \"won\", \"confidence\": 75, \"rsi\": 28, ...}, ...]}"
            }
        })
    
    return jsonify(_last_results)


@app.route("/validate", methods=["POST"])
def validate():
    """Walk-forward validation."""
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        if not data or not data.get("signals"):
            return jsonify({"error": "Send {\"signals\": [...], \"n_folds\": 3}"}), 400
        
        resolved = [s for s in data["signals"] if s.get("status") in ("won", "lost")]
        n_folds = data.get("n_folds", 3)
        
        logger.info(f"[VALIDATE] {len(resolved)} signals, {n_folds} folds")
        results = walk_forward_validate(resolved, n_folds)
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"[VALIDATE ERROR] {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/monte-carlo", methods=["POST"])
def monte_carlo():
    """Monte Carlo equity curve simulation."""
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        if not data or not data.get("signals"):
            return jsonify({"error": "Send {\"signals\": [...], \"n_sims\": 1000, \"account_size\": 50000}"}), 400
        
        resolved = [s for s in data["signals"] if s.get("status") in ("won", "lost")]
        n_sims = min(data.get("n_sims", 1000), 10000)
        account_size = data.get("account_size", 50000)
        
        logger.info(f"[MONTE-CARLO] {len(resolved)} signals, {n_sims} sims, ${account_size}")
        results = monte_carlo_sim(resolved, n_sims, account_size)
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"[MONTE-CARLO ERROR] {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/asset-qualify", methods=["POST"])
def asset_qualify_route():
    """Per-asset qualification analysis."""
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        if not data or not data.get("signals"):
            return jsonify({"error": "Send {\"signals\": [...], \"min_signals\": 10, \"min_wr\": 0.40}"}), 400
        
        resolved = [s for s in data["signals"] if s.get("status") in ("won", "lost")]
        min_sigs = data.get("min_signals", 10)
        min_wr = data.get("min_wr", 0.40)
        
        logger.info(f"[ASSET-QUALIFY] {len(resolved)} signals, min={min_sigs}, wr={min_wr}")
        results = asset_qualify(resolved, min_sigs, min_wr)
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"[ASSET-QUALIFY ERROR] {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


# ── Historical Backtest (10-year deep calibration) ────────────

from historical_backtest import run_historical_backtest, run_full_backtest, run_backtest_from_data, ASSET_FMP_SYMBOLS, ASSET_MARKET, ASSET_YAHOO_SYMBOLS

# In-memory cache for long-running full backtest
_historical_results: Dict[str, Any] = {}


@app.route("/historical-backtest/asset", methods=["POST"])
def historical_backtest_asset():
    """
    Run 10-year historical backtest for a SINGLE asset.
    Fetches daily OHLCV from FMP, runs scanner logic bar-by-bar, tracks outcomes.
    
    Takes ~5-15 seconds per asset.
    
    Request: {"asset": "ES", "fmp_key": "...", "years": 10, "max_bars": 20}
    Response: Full signal history with outcomes and summary stats.
    """
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Send {\"asset\": \"ES\", \"fmp_key\": \"...\"}"}), 400
        
        asset = data.get("asset")
        fmp_key = data.get("fmp_key", os.environ.get("FMP_API_KEY", ""))
        years = data.get("years", 10)
        max_bars = data.get("max_bars", 20)
        
        if not asset:
            return jsonify({"error": "Missing 'asset' field"}), 400
        if not fmp_key:
            return jsonify({"error": "Missing 'fmp_key' field or FMP_API_KEY env var"}), 400
        if asset not in ASSET_FMP_SYMBOLS and asset not in ASSET_YAHOO_SYMBOLS:
            return jsonify({"error": f"Unknown asset: {asset}", "available": list(ASSET_YAHOO_SYMBOLS.keys())}), 400
        
        logger.info(f"[HIST-BT] Starting {asset} ({years}yr, max_bars={max_bars})")
        result = run_historical_backtest(asset, fmp_key, years, max_bars)
        
        # Strip full signal list if too large (keep summary + first 200 signals)
        if "signals" in result and len(result["signals"]) > 200:
            result["signals_truncated"] = True
            result["signals_total"] = len(result["signals"])
            result["signals"] = result["signals"][:200]
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"[HIST-BT ERROR] {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/historical-backtest/full", methods=["POST"])
def historical_backtest_full():
    """
    Run 10-year backtest across ALL assets. This is the comprehensive calibration run.
    
    Takes ~3-8 minutes (37 assets × 5-15s each + FMP rate limits).
    Runs synchronously — Railway has a 5-minute timeout on free tier,
    so this may need to be split into batches on free plans.
    
    Request: {"fmp_key": "...", "years": 10, "assets": ["ES","NQ",...] (optional)}
    Response: Aggregated results with per-asset breakdown.
    """
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Send {\"fmp_key\": \"...\", \"years\": 10}"}), 400
        
        fmp_key = data.get("fmp_key", os.environ.get("FMP_API_KEY", ""))
        years = data.get("years", 10)
        assets = data.get("assets", list(ASSET_FMP_SYMBOLS.keys()))
        
        if not fmp_key:
            return jsonify({"error": "Missing 'fmp_key'"}), 400
        
        logger.info(f"[FULL-BT] Starting {len(assets)} assets, {years} years")
        result = run_full_backtest(assets, fmp_key, years)
        
        # Cache for /results endpoint
        global _historical_results
        _historical_results = result
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"[FULL-BT ERROR] {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/historical-backtest/results", methods=["GET"])
def historical_backtest_results():
    """Return last full historical backtest results."""
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    if not _historical_results:
        return jsonify({"status": "no_results", "message": "No historical backtest has been run yet."})
    
    return jsonify(_historical_results)


@app.route("/historical-backtest/analyze", methods=["POST", "OPTIONS"])
def historical_backtest_analyze():
    """
    Run backtest on PRE-FETCHED OHLCV data sent by the browser.
    This bypasses Railway→FMP connectivity issues.
    
    The browser fetches FMP data directly (which works), then sends
    the OHLCV arrays here for signal generation + outcome tracking.
    
    Request: {
        "asset": "ES",
        "market": "futures",
        "closes": [5000.0, 5010.5, ...],
        "highs": [5020.0, ...],
        "lows": [4990.0, ...],
        "volumes": [100000, ...],
        "dates": ["2016-01-04", ...],   (optional, for labeling)
        "max_bars": 20
    }
    """
    if request.method == "OPTIONS":
        return "", 204
    if not check_auth():
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Send {asset, market, closes, highs, lows, volumes}"}), 400
        
        asset = data.get("asset", "UNKNOWN")
        market = data.get("market", ASSET_MARKET.get(asset, "futures"))
        closes = data.get("closes", [])
        highs = data.get("highs", [])
        lows = data.get("lows", [])
        volumes = data.get("volumes", [])
        dates = data.get("dates", [f"bar_{i}" for i in range(len(closes))])
        max_bars = data.get("max_bars", 20)
        
        if len(closes) < 100:
            return jsonify({"error": f"Need 100+ bars, got {len(closes)}"}), 400
        
        ohlcv = {"closes": closes, "highs": highs, "lows": lows, "volumes": volumes, "dates": dates}
        
        logger.info(f"[ANALYZE] {asset} with {len(closes)} pre-fetched bars")
        result = run_backtest_from_data(asset, market, ohlcv, max_bars)
        
        if "signals" in result and len(result.get("signals", [])) > 200:
            result["signals_truncated"] = True
            result["signals_total"] = len(result["signals"])
            result["signals"] = result["signals"][:200]
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"[ANALYZE ERROR] {str(e)}", exc_info=True)
        return jsonify({"error": str(e), "type": type(e).__name__}), 500


@app.route("/test-fmp", methods=["POST", "OPTIONS"])
def test_fmp():
    """Diagnostic: test if Railway can reach FMP."""
    if request.method == "OPTIONS":
        return "", 204
    
    import requests as req
    data = request.get_json() or {}
    fmp_key = data.get("fmp_key", os.environ.get("FMP_API_KEY", ""))
    symbol = data.get("symbol", "SPY")
    
    results = {}
    
    # Test 1: Can we reach FMP at all?
    try:
        r = req.get(f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={fmp_key}", timeout=10)
        results["quote_status"] = r.status_code
        results["quote_body"] = r.text[:300]
    except Exception as e:
        results["quote_error"] = str(e)
    
    # Test 2: Historical data endpoint
    try:
        r = req.get(f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from=2024-01-01&apikey={fmp_key}", timeout=15)
        results["historical_status"] = r.status_code
        body = r.json() if r.status_code == 200 else r.text[:300]
        if isinstance(body, dict):
            results["historical_keys"] = list(body.keys())
            hist = body.get("historical", [])
            results["historical_bars"] = len(hist)
            if hist:
                results["historical_sample"] = hist[0]
        else:
            results["historical_body"] = str(body)[:300]
    except Exception as e:
        results["historical_error"] = str(e)
    
    return jsonify(results)


# ── Error handlers ────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "not_found",
        "endpoints": [
            "/health", "/backtest", "/results", "/validate", "/monte-carlo", "/asset-qualify",
            "/historical-backtest/asset", "/historical-backtest/full", "/historical-backtest/results"
        ]
    }), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "internal_server_error", "message": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Strategy Lab on port {port}")
    app.run(host="0.0.0.0", port=port)
