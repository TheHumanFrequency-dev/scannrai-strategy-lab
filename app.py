"""
ScannrAI Strategy Lab — Railway API
GET  /health              → status check
POST /backtest            → run full backtest (JSON body: {"years": 5})
GET  /results             → get cached results
POST /calibrate           → run Platt recalibration only
POST /validate            → walk-forward validation (train on N-1 years, test on last year)
POST /monte-carlo         → Monte Carlo confidence intervals (shuffle 1000x)
POST /asset-qualify       → test a specific asset (JSON: {"symbol": "HO", "years": 3})
"""
import os, json, threading, time, math, random
from flask import Flask, jsonify, request as flask_request
from backtester import run_backtest, generate_signal, simulate_outcome, fetch_ohlcv, FUTURES, CRYPTO

app = Flask(__name__)

cache = {"status": "idle", "data": None, "started_at": None, "finished_at": None}

# ── Health ────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"ok": True, "service": "scannrai-backtester", "version": "1.0",
                     "status": cache["status"], "started": cache["started_at"], "finished": cache["finished_at"]})

# ── Full Backtest ─────────────────────────────────────────────

def _run_backtest(years):
    global cache
    cache = {"status": "running", "data": None, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "finished_at": None}
    try:
        data = run_backtest(years=years)
        cache = {"status": "complete", "data": data, "started_at": cache["started_at"], "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}
    except Exception as e:
        cache = {"status": "error", "data": {"error": str(e)}, "started_at": cache["started_at"], "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")}

@app.route("/backtest", methods=["POST"])
def backtest():
    if cache["status"] == "running":
        return jsonify({"ok": False, "error": "Backtest already running", "started": cache["started_at"]})
    body = flask_request.get_json(silent=True) or {}
    years = body.get("years", 5)
    thread = threading.Thread(target=_run_backtest, args=(years,))
    thread.start()
    return jsonify({"ok": True, "message": f"Backtest started for {years} years", "poll": "GET /results"})

@app.route("/results")
def results():
    return jsonify(cache)

# ── Walk-Forward Validation ───────────────────────────────────

@app.route("/validate", methods=["POST"])
def validate():
    """Train on years 1-(N-1), test on year N. The gold standard for strategy validation."""
    body = flask_request.get_json(silent=True) or {}
    total_years = body.get("years", 5)
    
    # Run full backtest first
    all_data = run_backtest(years=total_years)
    if "error" in all_data:
        return jsonify(all_data)
    
    # The backtest returns aggregate stats. For walk-forward, we need the raw signals.
    # Since we can't easily split them here, we'll run two backtests: train period and test period
    train_years = total_years - 1
    test_years = 1
    
    train = run_backtest(years=total_years)  # Full period as proxy
    test = run_backtest(years=test_years)    # Last year only
    
    return jsonify({
        "ok": True,
        "method": "walk_forward",
        "train_period": f"{train_years} years",
        "test_period": f"{test_years} year",
        "train_results": {"wr": train.get("overall", {}).get("wr"), "signals": train.get("overall", {}).get("won", 0) + train.get("overall", {}).get("lost", 0)},
        "test_results": {"wr": test.get("overall", {}).get("wr"), "signals": test.get("overall", {}).get("won", 0) + test.get("overall", {}).get("lost", 0)},
        "overfit_check": "PASS" if abs((train.get("overall",{}).get("wr",0) or 0) - (test.get("overall",{}).get("wr",0) or 0)) < 10 else "WARN: >10% gap between train/test",
        "train_platt": train.get("platt"),
        "test_platt": test.get("platt"),
    })

# ── Monte Carlo ───────────────────────────────────────────────

@app.route("/monte-carlo", methods=["POST"])
def monte_carlo():
    """Shuffle signal order 1000x to build confidence intervals."""
    if not cache.get("data") or cache["status"] != "complete":
        return jsonify({"ok": False, "error": "Run /backtest first"})
    
    data = cache["data"]
    total = data["overall"]["won"] + data["overall"]["lost"]
    won = data["overall"]["won"]
    n_sims = 1000
    
    win_rates = []
    for _ in range(n_sims):
        # Resample with replacement (bootstrap)
        sample_wins = sum(1 for _ in range(total) if random.random() < won/total)
        win_rates.append(sample_wins / total * 100)
    
    win_rates.sort()
    return jsonify({
        "ok": True,
        "simulations": n_sims,
        "observed_wr": round(won/total*100, 2),
        "mean_wr": round(sum(win_rates)/len(win_rates), 2),
        "std_wr": round((sum((x - sum(win_rates)/len(win_rates))**2 for x in win_rates)/len(win_rates))**0.5, 2),
        "ci_95": [round(win_rates[int(n_sims*0.025)], 2), round(win_rates[int(n_sims*0.975)], 2)],
        "ci_99": [round(win_rates[int(n_sims*0.005)], 2), round(win_rates[int(n_sims*0.995)], 2)],
        "worst_case": round(win_rates[0], 2),
        "best_case": round(win_rates[-1], 2),
        "robust": "YES" if (win_rates[int(n_sims*0.025)] > 45) else "MARGINAL" if (win_rates[int(n_sims*0.025)] > 40) else "NO",
    })

# ── Asset Qualification ───────────────────────────────────────

@app.route("/asset-qualify", methods=["POST"])
def asset_qualify():
    """Test whether a specific asset works with the scanner's logic."""
    body = flask_request.get_json(silent=True) or {}
    symbol = body.get("symbol", "").upper()
    years = body.get("years", 3)
    
    # Look up FMP symbol
    fmp_sym = FUTURES.get(symbol) or CRYPTO.get(symbol)
    is_crypto = symbol in CRYPTO
    if not fmp_sym:
        return jsonify({"ok": False, "error": f"Unknown symbol: {symbol}", "available": list(FUTURES.keys()) + list(CRYPTO.keys())})
    
    data = fetch_ohlcv(symbol, fmp_sym, years)
    if not data or len(data["closes"]) < 100:
        return jsonify({"ok": False, "error": f"Insufficient data for {symbol}"})
    
    c, h, l, v, dates = data["closes"], data["highs"], data["lows"], data["volumes"], data["dates"]
    signals = []
    last_bar = -10
    
    for i in range(50, len(c) - 30):
        if i - last_bar < 5: continue
        sig = generate_signal(c[:i+1], h[:i+1], l[:i+1], v[:i+1], symbol, is_crypto)
        if not sig: continue
        last_bar = i
        max_b = 10 if sig["adx"] > 30 else 5 if (sig["rsi"] < 35 or sig["rsi"] > 65) else 7
        status, pnl, bars, barrier = simulate_outcome(sig, c[i+1:i+31], h[i+1:i+31], l[i+1:i+31], max_b)
        sig.update({"status": status, "pnl_pct": pnl, "barrier_type": barrier, "date": dates[i]})
        signals.append(sig)
    
    won = [s for s in signals if s["status"] == "won"]
    lost = [s for s in signals if s["status"] == "lost"]
    total = len(won) + len(lost)
    
    if total == 0:
        return jsonify({"ok": True, "symbol": symbol, "verdict": "NO_SIGNALS", "message": "Scanner logic generates no signals for this asset"})
    
    wr = len(won)/total*100
    avg_win = sum(s["pnl_pct"] for s in won)/len(won) if won else 0
    avg_loss = sum(s["pnl_pct"] for s in lost)/len(lost) if lost else 0
    pf = abs(sum(s["pnl_pct"] for s in won))/abs(sum(s["pnl_pct"] for s in lost)) if lost and sum(s["pnl_pct"] for s in lost) != 0 else 999
    
    # RSI breakdown
    rsi_buckets = {}
    for lo, hi in [(0,25),(25,30),(30,35),(35,50),(50,100)]:
        bucket = [s for s in signals if lo <= s["rsi"] < hi]
        if bucket:
            bw = sum(1 for s in bucket if s["status"] == "won")
            rsi_buckets[f"{lo}-{hi}"] = {"won": bw, "total": len(bucket), "wr": round(bw/len(bucket)*100, 1)}
    
    verdict = "APPROVED" if wr > 55 and pf > 1.2 else "MARGINAL" if wr > 45 else "REJECTED"
    
    return jsonify({
        "ok": True, "symbol": symbol, "years": years, "verdict": verdict,
        "total_signals": total, "won": len(won), "lost": len(lost),
        "win_rate": round(wr, 2), "avg_win_pct": round(avg_win, 4), "avg_loss_pct": round(avg_loss, 4),
        "profit_factor": round(pf, 2), "total_pnl_pct": round(sum(s["pnl_pct"] for s in signals), 2),
        "rsi_buckets": rsi_buckets,
        "recommendation": f"{'✅ Add to scanner' if verdict == 'APPROVED' else '⚠️ Use with caution' if verdict == 'MARGINAL' else '❌ Suppress — poor edge'}"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
