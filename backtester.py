"""
ScannrAI Historical Backtester v1.0
Replays scanner logic against 5+ years of OHLCV data.
Returns structured JSON for calibration.
"""
import json, math, time, os
from datetime import datetime
from urllib.request import urlopen, Request

FMP_KEY = os.environ.get("FMP_API_KEY", "NDvaRigH7VqZ4OtMh16pkAFu4l1ULNXW")

FUTURES = {
    "ES": "ES=F", "NQ": "NQ=F", "YM": "YM=F", "RTY": "RTY=F",
    "CL": "CL=F", "NG": "NG=F", "GC": "GC=F", "SI": "SI=F",
    "HG": "HG=F", "ZB": "ZB=F", "ZN": "ZN=F",
    "ZC": "ZC=F", "ZW": "ZW=F", "ZS": "ZS=F",
}
CRYPTO = {"BTC": "BTCUSD", "ETH": "ETHUSD", "SOL": "SOLUSD"}

# ── Indicators (mirrors scanner exactly) ──────────────────────

def calc_rsi(closes, period=14):
    if len(closes) < period + 1: return 50.0
    ag = al = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i-1]
        if d > 0: ag += d
        else: al -= d
    ag /= period; al /= period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i-1]
        ag = (ag * (period - 1) + max(0, d)) / period
        al = (al * (period - 1) + max(0, -d)) / period
    if al == 0 and ag == 0: return 50.0
    if al == 0: return 100.0
    return 100 - 100 / (1 + ag / al)

def calc_ema(closes, period):
    if not closes: return 0
    e = sum(closes[:period]) / period if len(closes) >= period else closes[0]
    k = 2 / (period + 1)
    for i in range(period, len(closes)):
        e = closes[i] * k + e * (1 - k)
    return e

def calc_macd(closes):
    if len(closes) < 35:
        return {"histogram": 0, "crossover": "neutral"}
    fast = calc_ema(closes, 12)
    slow = calc_ema(closes, 26)
    macd_line = fast - slow
    # Build MACD history for signal line
    macd_vals = []
    for i in range(26, len(closes)):
        f = calc_ema(closes[:i+1], 12)
        s = calc_ema(closes[:i+1], 26)
        macd_vals.append(f - s)
    if len(macd_vals) < 9:
        return {"histogram": macd_line, "crossover": "neutral"}
    sig = sum(macd_vals[-9:]) / 9
    hist = macd_line - sig
    prev_sig = sum(macd_vals[-10:-1]) / 9 if len(macd_vals) > 9 else sig
    prev_hist = macd_vals[-2] - prev_sig if len(macd_vals) > 1 else 0
    if hist > 0 and prev_hist <= 0: cross = "bullish_cross"
    elif hist < 0 and prev_hist >= 0: cross = "bearish_cross"
    elif hist > 0: cross = "bullish"
    elif hist < 0: cross = "bearish"
    else: cross = "neutral"
    return {"histogram": hist, "crossover": cross}

def calc_bollinger(closes, period=20):
    if len(closes) < period: return {"pct": 0.5, "width": 0}
    sl = closes[-period:]
    mid = sum(sl) / period
    std = math.sqrt(sum((x - mid)**2 for x in sl) / period)
    upper = mid + 2 * std
    lower = mid - 2 * std
    pct = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    width = (upper - lower) / mid if mid > 0 else 0
    return {"pct": pct, "width": width}

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < 2: return 1
    trs = []
    for i in range(1, len(closes)):
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
    if len(trs) < period: return sum(trs)/len(trs) if trs else 1
    atr = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        atr = (atr * (period-1) + trs[i]) / period
    return atr

def calc_adx(highs, lows, closes, period=14):
    if len(closes) < period + 1: return 0
    pdms, ndms, trs = [], [], []
    for i in range(1, len(closes)):
        pdm = max(highs[i]-highs[i-1], 0)
        ndm = max(lows[i-1]-lows[i], 0)
        if pdm > ndm: ndm = 0
        elif ndm > pdm: pdm = 0
        else: pdm = ndm = 0
        trs.append(max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])))
        pdms.append(pdm); ndms.append(ndm)
    if len(trs) < period: return 0
    atr_v = sum(trs[:period])/period
    pdm_v = sum(pdms[:period])/period
    ndm_v = sum(ndms[:period])/period
    for i in range(period, len(trs)):
        atr_v = (atr_v*(period-1)+trs[i])/period
        pdm_v = (pdm_v*(period-1)+pdms[i])/period
        ndm_v = (ndm_v*(period-1)+ndms[i])/period
    pdi = (pdm_v/atr_v)*100 if atr_v > 0 else 0
    ndi = (ndm_v/atr_v)*100 if atr_v > 0 else 0
    return abs(pdi-ndi)/(pdi+ndi)*100 if (pdi+ndi) > 0 else 0

def calc_keltner_width(closes, highs, lows, period=20, mult=1.5):
    atr = calc_atr(highs, lows, closes, period)
    mid = sum(closes[-period:])/period if len(closes) >= period else closes[-1]
    return (2*mult*atr)/mid if mid > 0 else 0

# ── Signal Generation ─────────────────────────────────────────

def generate_signal(closes, highs, lows, volumes, sym, is_crypto=False):
    if len(closes) < 50: return None
    price = closes[-1]
    rsi = calc_rsi(closes)
    macd = calc_macd(closes)
    bb = calc_bollinger(closes)
    atr = calc_atr(highs, lows, closes)
    adx = calc_adx(highs, lows, closes)
    kw = calc_keltner_width(closes, highs, lows)
    squeeze_on = bb["width"] < kw
    vol_ratio = (volumes[-1] / (sum(volumes[-20:])/20)) if len(volumes) >= 20 and sum(volumes[-20:]) > 0 else 1
    ema8 = calc_ema(closes, 8); ema21 = calc_ema(closes, 21)
    ema_trend = "uptrend" if ema8 > ema21 else "downtrend"

    # 7-strategy consensus
    vwap = "BUY" if bb["pct"] < 0.15 else "SELL" if bb["pct"] > 0.85 else "NEUTRAL"
    rsi_mr = "BUY" if rsi < 30 else "SELL" if rsi > 70 else "NEUTRAL"
    regime = "BUY" if ema_trend == "uptrend" and adx > 25 else "SELL" if ema_trend == "downtrend" and adx > 25 else "NEUTRAL"
    ch1 = closes[-1] - closes[-2] if len(closes) >= 2 else 0
    of = "BUY" if ch1 > 0 and vol_ratio > 1.5 else "SELL" if ch1 < 0 and vol_ratio > 1.5 else "NEUTRAL"
    votes4 = [vwap, rsi_mr, regime, of]
    bv4 = sum(1 for v in votes4 if v == "BUY"); sv4 = sum(1 for v in votes4 if v == "SELL")
    ensemble = "BUY" if bv4 >= 3 else "SELL" if sv4 >= 3 else "NEUTRAL"
    bb_strat = "BUY" if bb["pct"] < 0.15 else "SELL" if bb["pct"] > 0.85 else "NEUTRAL"
    sq_strat = ("BUY" if macd["histogram"] > 0 else "SELL" if macd["histogram"] < 0 else "NEUTRAL") if squeeze_on else "NEUTRAL"

    strategies = [vwap, rsi_mr, regime, of, ensemble, bb_strat, sq_strat]
    buy_v = sum(1 for s in strategies if s == "BUY")
    sell_v = sum(1 for s in strategies if s == "SELL")
    agreeing = max(buy_v, sell_v)
    cons_dir = "BUY" if buy_v > sell_v else "SELL" if sell_v > buy_v else "NEUTRAL"
    if cons_dir == "NEUTRAL": return None

    # Scoring
    raw = 50
    if agreeing >= 5: raw += 15
    elif agreeing >= 4: raw += 10
    elif agreeing >= 3: raw += 5
    if adx > 35: raw += 12
    elif adx > 25: raw += 6
    if (cons_dir == "BUY" and rsi < 30) or (cons_dir == "SELL" and rsi > 70): raw += 10
    if squeeze_on: raw += 5
    if vol_ratio > 1.5: raw += 5
    score = max(0, min(100, raw))

    # Grading
    if not is_crypto and cons_dir == "BUY" and rsi < 25: grade = "A+"
    elif not is_crypto and cons_dir == "BUY" and rsi < 30: grade = "A"
    elif not is_crypto and cons_dir == "SELL" and rsi > 75: grade = "A"
    elif score >= 78 and agreeing >= 5: grade = "A+"
    elif score >= 70 and agreeing >= 4: grade = "A"
    elif score >= 62 and agreeing >= 3: grade = "B+"
    elif score >= 55: grade = "B"
    else: return None  # C/D/F = no signal

    # TP/SL
    tp_m = 2.5 if adx > 30 else 1.5 if (rsi < 35 or rsi > 65) else 1.8
    sl_m = 1.0 if adx > 30 else 0.75 if (rsi < 35 or rsi > 65) else 0.9
    is_long = cons_dir == "BUY"
    tp = price + atr*tp_m if is_long else price - atr*tp_m
    sl = price - atr*sl_m if is_long else price + atr*sl_m

    return {"sym": sym, "direction": cons_dir, "grade": grade, "score": score,
            "rsi": rsi, "adx": adx, "macd_cross": macd["crossover"], "bb_pct": bb["pct"],
            "squeeze": squeeze_on, "agreeing": agreeing, "vol_ratio": vol_ratio,
            "price": price, "tp": tp, "sl": sl, "atr": atr, "is_crypto": is_crypto}

# ── Triple Barrier Simulation ─────────────────────────────────

def simulate_outcome(sig, fut_c, fut_h, fut_l, max_bars=24):
    entry, tp, sl = sig["price"], sig["tp"], sig["sl"]
    is_long = sig["direction"] == "BUY"
    for i in range(min(len(fut_c), max_bars)):
        if is_long:
            if fut_h[i] >= tp: return ("won", ((tp-entry)/entry)*100, i+1, "tp")
            if fut_l[i] <= sl: return ("lost", ((sl-entry)/entry)*100, i+1, "sl")
        else:
            if fut_l[i] <= tp: return ("won", ((entry-tp)/entry)*100, i+1, "tp")
            if fut_h[i] >= sl: return ("lost", ((entry-sl)/entry)*100, i+1, "sl")
    # Time barrier
    if fut_c:
        ep = fut_c[min(max_bars-1, len(fut_c)-1)]
        pnl = ((ep-entry)/entry)*100 if is_long else ((entry-ep)/entry)*100
        return ("won" if pnl > 0 else "lost", pnl, max_bars, "time")
    return ("expired", 0, 0, "none")

# ── FMP Data Fetch ────────────────────────────────────────────

def fetch_ohlcv(symbol, fmp_sym, years=5):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{fmp_sym}?timeseries={years*365}&apikey={FMP_KEY}"
    try:
        req = Request(url, headers={"User-Agent": "ScannrAI/1.0"})
        resp = urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
        hist = data.get("historical", [])
        if not hist: return None
        hist.reverse()
        return {"dates": [h["date"] for h in hist],
                "opens": [h["open"] for h in hist], "highs": [h["high"] for h in hist],
                "lows": [h["low"] for h in hist], "closes": [h["close"] for h in hist],
                "volumes": [h.get("volume", 0) for h in hist]}
    except Exception as e:
        print(f"  [{symbol}] Error: {e}")
        return None

# ── Main Backtest ─────────────────────────────────────────────

def run_backtest(years=5):
    print(f"ScannrAI Backtester v1.0 | {years} years | {len(FUTURES)}F + {len(CRYPTO)}C")
    all_assets = {**{s: (v, False) for s, v in FUTURES.items()}, **{s: (v, True) for s, v in CRYPTO.items()}}
    
    # Fetch data
    ohlcv = {}
    for sym, (fmp_sym, is_crypto) in all_assets.items():
        print(f"  Fetching {sym}...", end=" ", flush=True)
        d = fetch_ohlcv(sym, fmp_sym, years)
        if d and len(d["closes"]) >= 100:
            ohlcv[sym] = (d, is_crypto)
            print(f"{len(d['closes'])} bars")
        else:
            print("SKIP")
        time.sleep(0.25)
    
    print(f"\nLoaded {len(ohlcv)} assets. Running simulation...")
    
    LOOKBACK, FORWARD, COOLDOWN = 50, 30, 5
    signals = []
    asset_stats = {}
    last_signal_bar = {}

    for sym, (data, is_crypto) in ohlcv.items():
        c, h, l, v, dates = data["closes"], data["highs"], data["lows"], data["volumes"], data["dates"]
        sw = sl_count = 0
        for i in range(LOOKBACK, len(c) - FORWARD):
            if sym in last_signal_bar and i - last_signal_bar[sym] < COOLDOWN:
                continue
            sig = generate_signal(c[:i+1], h[:i+1], l[:i+1], v[:i+1], sym, is_crypto)
            if not sig: continue
            last_signal_bar[sym] = i
            max_b = 10 if sig["adx"] > 30 else 5 if (sig["rsi"] < 35 or sig["rsi"] > 65) else 7
            status, pnl, bars, barrier = simulate_outcome(sig, c[i+1:i+1+FORWARD], h[i+1:i+1+FORWARD], l[i+1:i+1+FORWARD], max_b)
            sig.update({"status": status, "pnl_pct": pnl, "bars_held": bars, "barrier_type": barrier, "date": dates[i]})
            signals.append(sig)
            if status == "won": sw += 1
            elif status == "lost": sl_count += 1
        total = sw + sl_count
        asset_stats[sym] = {"won": sw, "lost": sl_count, "total": total, "wr": (sw/total*100) if total > 0 else 0}
        if total > 0: print(f"  {sym:5s}: {total:5d} signals | {sw}W/{sl_count}L | {sw/total*100:.1f}%")

    # ── Analysis ──
    won = [s for s in signals if s["status"] == "won"]
    lost = [s for s in signals if s["status"] == "lost"]
    total = len(won) + len(lost)
    if total == 0:
        return {"error": "No signals generated", "assets_loaded": len(ohlcv)}

    results = {
        "metadata": {"version": "1.0", "run_date": datetime.now().isoformat(), "years": years,
                      "total_signals": total, "assets": len(ohlcv)},
        "overall": {"won": len(won), "lost": len(lost), "wr": round(len(won)/total*100, 2),
                     "avg_win_pct": round(sum(s["pnl_pct"] for s in won)/len(won), 4) if won else 0,
                     "avg_loss_pct": round(sum(s["pnl_pct"] for s in lost)/len(lost), 4) if lost else 0,
                     "total_pnl_pct": round(sum(s["pnl_pct"] for s in signals), 2),
                     "profit_factor": round(abs(sum(s["pnl_pct"] for s in won))/abs(sum(s["pnl_pct"] for s in lost)), 2) if lost and sum(s["pnl_pct"] for s in lost) != 0 else 999},
        "by_market": {},
        "by_direction": {},
        "futures_buy_by_rsi": {},
        "futures_sell_by_rsi": {},
        "crypto_sell": {},
        "by_asset": asset_stats,
        "by_grade": {},
        "by_consensus": {},
        "by_barrier": {},
        "platt": {},
    }

    # By market
    for mkt, is_c in [("futures", False), ("crypto", True)]:
        ms = [s for s in signals if s["is_crypto"] == is_c]
        mw = sum(1 for s in ms if s["status"] == "won")
        results["by_market"][mkt] = {"won": mw, "total": len(ms), "wr": round(mw/len(ms)*100, 2) if ms else 0}

    # By direction
    for d in ["BUY", "SELL"]:
        ds = [s for s in signals if s["direction"] == d]
        dw = sum(1 for s in ds if s["status"] == "won")
        results["by_direction"][d] = {"won": dw, "total": len(ds), "wr": round(dw/len(ds)*100, 2) if ds else 0}

    # Futures BUY by RSI
    fut_buy = [s for s in signals if not s["is_crypto"] and s["direction"] == "BUY"]
    for lo, hi in [(0,20),(20,25),(25,30),(30,35),(35,40),(40,50),(50,100)]:
        bucket = [s for s in fut_buy if lo <= s["rsi"] < hi]
        if bucket:
            bw = sum(1 for s in bucket if s["status"] == "won")
            results["futures_buy_by_rsi"][f"{lo}-{hi}"] = {"won": bw, "total": len(bucket), "wr": round(bw/len(bucket)*100, 2)}

    # Futures SELL by RSI
    fut_sell = [s for s in signals if not s["is_crypto"] and s["direction"] == "SELL"]
    for lo, hi in [(50,60),(60,65),(65,70),(70,75),(75,80),(80,100)]:
        bucket = [s for s in fut_sell if lo <= s["rsi"] < hi]
        if bucket:
            bw = sum(1 for s in bucket if s["status"] == "won")
            results["futures_sell_by_rsi"][f"{lo}-{hi}"] = {"won": bw, "total": len(bucket), "wr": round(bw/len(bucket)*100, 2)}

    # Crypto SELL
    cry_sell = [s for s in signals if s["is_crypto"] and s["direction"] == "SELL"]
    if cry_sell:
        cw = sum(1 for s in cry_sell if s["status"] == "won")
        results["crypto_sell"] = {"won": cw, "total": len(cry_sell), "wr": round(cw/len(cry_sell)*100, 2)}

    # By grade
    for g in ["A+", "A", "B+", "B"]:
        gs = [s for s in signals if s["grade"] == g]
        if gs:
            gw = sum(1 for s in gs if s["status"] == "won")
            results["by_grade"][g] = {"won": gw, "total": len(gs), "wr": round(gw/len(gs)*100, 2)}

    # By consensus
    for c in range(3, 8):
        cs = [s for s in signals if s["agreeing"] == c]
        if cs:
            cw = sum(1 for s in cs if s["status"] == "won")
            results["by_consensus"][str(c)] = {"won": cw, "total": len(cs), "wr": round(cw/len(cs)*100, 2)}

    # By barrier type
    for bt in ["tp", "sl", "time"]:
        bs = [s for s in signals if s["barrier_type"] == bt]
        if bs:
            bw = sum(1 for s in bs if s["status"] == "won")
            results["by_barrier"][bt] = {"won": bw, "total": len(bs), "wr": round(bw/len(bs)*100, 2)}

    # Platt recalibration
    A, B = 0.3654, -0.1910
    for _ in range(10000):
        gA = gB = 0
        for s in signals:
            x = s["score"] / 100
            z = max(-500, min(500, A*x + B))
            p = max(min(1/(1+math.exp(-z)), 0.999), 0.001)
            y = 1 if s["status"] == "won" else 0
            gA += (p-y)*x; gB += (p-y)
        A -= 0.01*gA/total; B -= 0.01*gB/total
    results["platt"] = {"A": round(A, 6), "B": round(B, 6), "note": f"Calibrated from {total} historical signals"}

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {total} signals | {len(won)}W / {len(lost)}L | {len(won)/total*100:.1f}% WR")
    print(f"Profit Factor: {results['overall']['profit_factor']}")
    print(f"Platt: A={A:.6f}, B={B:.6f}")
    print(f"{'='*60}")

    return results

if __name__ == "__main__":
    import sys
    yrs = int(sys.argv[sys.argv.index("--years")+1]) if "--years" in sys.argv else 5
    r = run_backtest(yrs)
    print(json.dumps(r, indent=2))
