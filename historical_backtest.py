"""
ScannrAI Historical Backtester
==============================
Fetches 10 years of OHLCV data from FMP, runs the scanner's signal logic
bar-by-bar, tracks outcomes (TP/SL hit), and returns a comprehensive
calibration dataset.

This is the engine that was supposed to power the Strategy Lab from day one.
The 599-signal analysis was a stopgap; this produces 5,000-20,000+ signals
with tracked outcomes for statistically robust calibration.

Architecture:
  - POST /historical-backtest: Run backtest for one or more assets
  - Fetches daily OHLCV from FMP (10 years = ~2,520 bars per asset)
  - Computes RSI, MACD, BB, EMA, ADX, squeeze, regime at each bar
  - Generates signals using the same logic as sentinelScan
  - Tracks forward N bars to determine if price hit TP or SL first
  - Returns full signal dataset with outcomes

Why daily bars instead of hourly:
  - FMP free tier returns full daily history (10+ years)
  - FMP hourly is limited to recent data on free tier
  - Daily signals are more independent (less autocorrelation)
  - RSI(14) on daily = standard 14-day RSI used by most traders
  - Gives ~20-50 signals per asset × 37 assets = 700-1,800 total
  - Each signal outcome is tracked over 5-20 trading days
"""

import math
import time
import logging
import requests
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger("historical-bt")


# ═══════════════════════════════════════════════════════════════
# INDICATOR LIBRARY (faithful port from TypeScript scanner)
# ═══════════════════════════════════════════════════════════════

def calc_rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d > 0:
            avg_gain += d
        else:
            avg_loss -= d
    avg_gain /= period
    avg_loss /= period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(0, d)) / period
        avg_loss = (avg_loss * (period - 1) + max(0, -d)) / period
    if avg_loss == 0 and avg_gain == 0:
        return 50.0
    if avg_loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def calc_ema(closes: List[float], period: int) -> float:
    if not closes:
        return 0.0
    if len(closes) < period:
        return closes[-1]
    k = 2.0 / (period + 1)
    e = sum(closes[:period]) / period
    for i in range(period, len(closes)):
        e = closes[i] * k + e * (1 - k)
    return e


def calc_macd(closes: List[float]) -> Dict[str, Any]:
    if len(closes) < 35:
        return {"macd": 0, "signal": 0, "histogram": 0, "crossover": "neutral"}
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    macd_line = ema12 - ema26
    macd_series = []
    for i in range(max(26, len(closes) - 35), len(closes) + 1):
        sl = closes[:i]
        if len(sl) >= 26:
            macd_series.append(calc_ema(sl, 12) - calc_ema(sl, 26))
    sig = calc_ema(macd_series, min(9, len(macd_series))) if len(macd_series) >= 2 else macd_line
    prev = macd_series[-2] if len(macd_series) >= 2 else macd_line
    if macd_line > sig and prev <= sig:
        crossover = "bullish_cross"
    elif macd_line < sig and prev >= sig:
        crossover = "bearish_cross"
    elif macd_line > sig:
        crossover = "bullish"
    elif macd_line < sig:
        crossover = "bearish"
    else:
        crossover = "neutral"
    return {"macd": macd_line, "signal": sig, "histogram": macd_line - sig, "crossover": crossover}


def calc_bb(closes: List[float], period: int = 20) -> Dict[str, float]:
    n = min(period, len(closes))
    sl = closes[-n:]
    mid = sum(sl) / n
    std = math.sqrt(sum((x - mid) ** 2 for x in sl) / n) if n > 0 else 0
    upper = mid + 2 * std
    lower = mid - 2 * std
    pct = (closes[-1] - lower) / (upper - lower) if upper != lower else 0.5
    return {"upper": upper, "lower": lower, "mid": mid, "pct": pct, "bw": upper - lower, "std": std}


def calc_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    length = min(len(highs), len(lows), len(closes))
    if length < 2:
        return abs(highs[0] - lows[0]) if highs and lows else 1.0
    trs = []
    for i in range(1, length):
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
    p = min(period, len(trs))
    return sum(trs[-p:]) / p if p > 0 else 1.0


def calc_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    length = min(len(highs), len(lows), len(closes))
    if length < period + 1:
        return 25.0
    trs, pdms, ndms = [], [], []
    for i in range(1, length):
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        pdms.append(up if up > dn and up > 0 else 0)
        ndms.append(dn if dn > up and dn > 0 else 0)
    atr_v = sum(trs[-period:]) / period
    pdm_v = sum(pdms[-period:]) / period
    ndm_v = sum(ndms[-period:]) / period
    for i in range(period, len(trs)):
        atr_v = (atr_v * (period - 1) + trs[i]) / period
        pdm_v = (pdm_v * (period - 1) + pdms[i]) / period
        ndm_v = (ndm_v * (period - 1) + ndms[i]) / period
    pdi = (pdm_v / atr_v) * 100 if atr_v > 0 else 0
    ndi = (ndm_v / atr_v) * 100 if atr_v > 0 else 0
    denom = pdi + ndi
    if denom == 0:
        return 25.0
    return min(100, max(0, abs(pdi - ndi) / denom * 100))


def detect_regime(closes: List[float], ema9: float, ema21: float, price: float) -> str:
    window = closes[-20:] if len(closes) >= 20 else closes
    if len(window) < 2:
        return "ranging"
    rets = [(window[i] - window[i - 1]) / window[i - 1] for i in range(1, len(window))]
    vol_pct = math.sqrt(sum(r * r for r in rets) / len(rets)) * 100 if rets else 1
    if ema9 > ema21 and price > ema9 and vol_pct < 2.0:
        return "trending_up"
    if ema9 < ema21 and price < ema9 and vol_pct < 2.0:
        return "trending_down"
    if vol_pct > 3.0:
        return "high_volatility"
    if abs(ema9 - ema21) / (ema21 or 1) < 0.004:
        return "mean_reverting"
    return "ranging"


def detect_squeeze(closes: List[float], highs: List[float], lows: List[float], prev_squeeze: str = "none") -> str:
    bb = calc_bb(closes, 20)
    atr = calc_atr(highs, lows, closes, 20)
    ema20 = calc_ema(closes, min(20, len(closes)))
    kc_u = ema20 + 1.5 * atr
    kc_l = ema20 - 1.5 * atr
    is_on = bb["upper"] < kc_u and bb["lower"] > kc_l
    if is_on:
        return "squeeze_on"
    if prev_squeeze == "squeeze_on":
        mom = closes[-1] - closes[-2] if len(closes) >= 2 else 0
        return "firing_bullish" if mom > 0 else "firing_bearish"
    return "none"


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION (mirrors sentinelScan logic)
# ═══════════════════════════════════════════════════════════════

def generate_signal_at_bar(
    closes: List[float], highs: List[float], lows: List[float], volumes: List[float],
    bar_idx: int, asset: str, market: str, prev_squeeze: str = "none"
) -> Optional[Dict]:
    """
    Given OHLCV data up to bar_idx, compute indicators and determine
    if a signal would fire. Returns signal dict or None.
    """
    # Need at least 50 bars of history
    if bar_idx < 50:
        return None

    c = closes[:bar_idx + 1]
    h = highs[:bar_idx + 1]
    l = lows[:bar_idx + 1]
    v = volumes[:bar_idx + 1]
    price = c[-1]

    # Compute indicators
    rsi = calc_rsi(c)
    macd = calc_macd(c)
    bb = calc_bb(c)
    ema9 = calc_ema(c, 9)
    ema21 = calc_ema(c, 21)
    ema50 = calc_ema(c, 50)
    atr = calc_atr(h, l, c)
    adx = calc_adx(h, l, c)
    regime = detect_regime(c, ema9, ema21, price)
    squeeze = detect_squeeze(c, h, l, prev_squeeze)

    # Volume ratio
    vol_window = v[-20:] if len(v) >= 20 else v
    avg_vol = sum(vol_window) / len(vol_window) if vol_window else 1
    vol_ratio = v[-1] / avg_vol if avg_vol > 0 and v else 1.0

    # EMA trend
    if price > ema9 and ema9 > ema21 and ema21 > ema50:
        ema_trend = "strong_uptrend"
    elif price > ema9 and ema9 > ema21:
        ema_trend = "uptrend"
    elif price < ema9 and ema9 < ema21 and ema21 < ema50:
        ema_trend = "strong_downtrend"
    elif price < ema9 and ema9 < ema21:
        ema_trend = "downtrend"
    else:
        ema_trend = "flat"

    # BB position
    bb_pct = bb["pct"]
    bb_pos = "above_upper" if bb_pct > 1.0 else "upper_zone" if bb_pct > 0.8 else "below_lower" if bb_pct < 0.0 else "lower_zone" if bb_pct < 0.2 else "middle"

    # 7-strategy consensus (same as scanner)
    is_crypto = market == "crypto"
    rsi_low = 28 if is_crypto else 25
    rsi_high = 72 if is_crypto else 75

    vwap_sum = sum(c[i] * (v[i] or 1) for i in range(len(c)))
    vol_sum = sum(v[i] or 1 for i in range(len(v)))
    vwap = vwap_sum / vol_sum if vol_sum > 0 else price

    vwap_strat = "BUY" if price > vwap * 1.001 else "SELL" if price < vwap * 0.999 else "NEUTRAL"
    rsi_mr_strat = "BUY" if rsi < rsi_low else "SELL" if rsi > rsi_high else "NEUTRAL"
    regime_strat = "BUY" if regime == "trending_up" else "SELL" if regime == "trending_down" else ("BUY" if rsi < 50 else "SELL") if regime == "mean_reverting" else "NEUTRAL"
    of_strat = "BUY" if vol_ratio > 1.3 and macd["histogram"] > 0 else "SELL" if vol_ratio > 1.3 and macd["histogram"] < 0 else "NEUTRAL"
    ensemble_votes = sum(1 for s in [rsi_mr_strat, regime_strat, vwap_strat] if s == "BUY")
    ensemble_strat = "BUY" if ensemble_votes >= 3 else "SELL" if ensemble_votes == 0 else "NEUTRAL"
    bb_strat = "BUY" if bb_pct < 0.15 else "SELL" if bb_pct > 0.85 else "NEUTRAL"
    sq_strat = "BUY" if squeeze == "firing_bullish" and rsi < 55 else "SELL" if squeeze == "firing_bearish" and rsi > 45 else "NEUTRAL"

    strategies = [vwap_strat, rsi_mr_strat, regime_strat, of_strat, ensemble_strat, bb_strat, sq_strat]
    buy_votes = sum(1 for s in strategies if s == "BUY")
    sell_votes = sum(1 for s in strategies if s == "SELL")
    agreeing = max(buy_votes, sell_votes)
    cons_dir = "BUY" if buy_votes > sell_votes else "SELL" if sell_votes > buy_votes else "NEUTRAL"

    if cons_dir == "NEUTRAL":
        return None

    # Direction with RSI override
    is_long = cons_dir == "BUY"
    rsi_oversold = rsi <= rsi_low
    rsi_overbought = rsi >= rsi_high

    direction = "NEUTRAL"
    if cons_dir == "BUY" and not rsi_overbought:
        direction = "STRONG_BUY" if agreeing >= 4 else "BUY"
    elif cons_dir == "SELL" and not rsi_oversold:
        direction = "STRONG_SELL" if agreeing >= 4 else "SELL"
    if rsi_oversold and cons_dir == "SELL":
        direction = "BUY"
        is_long = True
    if rsi_overbought and cons_dir == "BUY":
        direction = "SELL"
        is_long = False

    if direction == "NEUTRAL":
        return None

    # Crypto SELL suppression (0% historical WR)
    if is_crypto and direction in ("SELL", "STRONG_SELL"):
        if not (rsi > 85):  # Only allow extreme contrarian
            return None

    # TP/SL based on regime
    if regime in ("trending_up", "trending_down"):
        sl_mult, tp_mult = 2.0, 4.0
    elif regime == "mean_reverting":
        sl_mult, tp_mult = 1.0, 1.5
    elif regime == "high_volatility":
        sl_mult, tp_mult = 2.5, 3.0
    else:
        sl_mult, tp_mult = 1.2, 2.0

    sl_dist = atr * sl_mult
    tp_dist = atr * tp_mult

    if is_long:
        sl_price = price - sl_dist
        tp_price = price + tp_dist
    else:
        sl_price = price + sl_dist
        tp_price = price - tp_dist

    # Grade (same logic as scanner)
    raw_score = 50  # Simplified — we'll compute evidence score later
    # RSI-based grade overrides (the proven edges)
    if not is_crypto and is_long and rsi < 25:
        grade = "A+"
    elif not is_crypto and is_long and rsi < 30:
        grade = "A"
    elif not is_crypto and not is_long and rsi > 75:
        grade = "A"
    elif agreeing >= 5:
        grade = "A"
    elif agreeing >= 4:
        grade = "B+"
    elif agreeing >= 3:
        grade = "B"
    else:
        grade = "C"

    # Suppression for bad assets
    if asset == "ZM" and is_long:
        grade = "F"  # suppress
    if asset == "PL" and is_long:
        grade = "F"

    if grade == "F":
        return None

    # Build key factors
    factors = []
    if squeeze in ("firing_bullish", "firing_bearish"):
        factors.append("squeeze_firing")
    if rsi < 30:
        factors.append("rsi_oversold")
    if rsi > 70:
        factors.append("rsi_overbought")
    if adx > 30:
        factors.append("strong_trend_adx")
    if ema_trend in ("strong_uptrend",) and is_long:
        factors.append("ema_aligned_long")
    if ema_trend in ("strong_downtrend",) and not is_long:
        factors.append("ema_aligned_short")
    if (regime == "trending_up" and is_long) or (regime == "trending_down" and not is_long):
        factors.append("regime_aligned")
    if not is_crypto and is_long and rsi < 30:
        factors.append("futures_buy_rsi_extreme")
    if not is_crypto and not is_long and rsi > 75:
        factors.append("futures_sell_rsi_extreme")

    return {
        "bar_idx": bar_idx,
        "price": price,
        "asset": asset,
        "market": market,
        "signal_type": "BUY" if is_long else "SELL",
        "direction": direction,
        "rsi": round(rsi, 2),
        "adx": round(adx, 1),
        "macd_crossover": macd["crossover"],
        "bb_pct": round(bb_pct, 4),
        "ema_trend": ema_trend,
        "regime": regime,
        "squeeze": squeeze,
        "atr": atr,
        "vol_ratio": round(vol_ratio, 2),
        "agreeing": agreeing,
        "cons_dir": cons_dir,
        "grade": grade,
        "key_factors": factors,
        "entry_price": price,
        "tp_price": tp_price,
        "sl_price": sl_price,
        "tp_dist_pct": round(tp_dist / price * 100, 4),
        "sl_dist_pct": round(sl_dist / price * 100, 4),
    }


# ═══════════════════════════════════════════════════════════════
# OUTCOME TRACKING
# ═══════════════════════════════════════════════════════════════

def track_outcome(
    signal: Dict, closes: List[float], highs: List[float], lows: List[float],
    max_bars: int = 20
) -> Dict:
    """
    Track forward from signal bar to see if TP or SL was hit first.
    Uses high/low (not just close) for realistic fill simulation.
    """
    bar_idx = signal["bar_idx"]
    is_long = signal["signal_type"] == "BUY"
    tp = signal["tp_price"]
    sl = signal["sl_price"]
    entry = signal["entry_price"]

    for i in range(bar_idx + 1, min(bar_idx + 1 + max_bars, len(closes))):
        high_price = highs[i]
        low_price = lows[i]

        if is_long:
            # Check SL first (more conservative — if both hit in same bar, assume loss)
            if low_price <= sl:
                pnl_pct = (sl - entry) / entry * 100
                return {"status": "lost", "outcome_bar": i, "bars_held": i - bar_idx,
                        "outcome_price": sl, "pnl_pct": round(pnl_pct, 4)}
            if high_price >= tp:
                pnl_pct = (tp - entry) / entry * 100
                return {"status": "won", "outcome_bar": i, "bars_held": i - bar_idx,
                        "outcome_price": tp, "pnl_pct": round(pnl_pct, 4)}
        else:
            # Short: SL is above, TP is below
            if high_price >= sl:
                pnl_pct = (entry - sl) / entry * 100
                return {"status": "lost", "outcome_bar": i, "bars_held": i - bar_idx,
                        "outcome_price": sl, "pnl_pct": round(pnl_pct, 4)}
            if low_price <= tp:
                pnl_pct = (entry - tp) / entry * 100
                return {"status": "won", "outcome_bar": i, "bars_held": i - bar_idx,
                        "outcome_price": tp, "pnl_pct": round(pnl_pct, 4)}

    # Expired — use last close
    last_close = closes[min(bar_idx + max_bars, len(closes) - 1)]
    pnl_pct = ((last_close - entry) / entry * 100) if is_long else ((entry - last_close) / entry * 100)
    return {"status": "expired", "outcome_bar": min(bar_idx + max_bars, len(closes) - 1),
            "bars_held": max_bars, "outcome_price": last_close, "pnl_pct": round(pnl_pct, 4)}


# ═══════════════════════════════════════════════════════════════
# FMP DATA FETCHING
# ═══════════════════════════════════════════════════════════════

# Asset → FMP symbol mapping
ASSET_FMP_SYMBOLS = {
    # Futures (FMP uses the =F suffix)
    "ES": "ES=F", "NQ": "NQ=F", "YM": "YM=F", "RTY": "RTY=F",
    "CL": "CL=F", "NG": "NG=F", "RB": "RB=F", "HO": "HO=F",
    "GC": "GC=F", "SI": "SI=F", "HG": "HG=F", "PL": "PL=F",
    "ZB": "ZB=F", "ZN": "ZN=F", "ZF": "ZF=F", "ZT": "ZT=F", "UB": "UB=F",
    "ZS": "ZS=F", "ZC": "ZC=F", "ZW": "ZW=F", "ZM": "ZM=F", "ZL": "ZL=F",
    "HE": "HE=F", "LE": "LE=F",
    "6E": "6E=F", "6J": "6J=F", "6B": "6B=F", "NKD": "NKD=F",
    # Crypto (FMP uses BTCUSD format)
    "BTC": "BTCUSD", "ETH": "ETHUSD", "SOL": "SOLUSD", "XRP": "XRPUSD",
    "BNB": "BNBUSD", "LINK": "LINKUSD", "ADA": "ADAUSD", "AVAX": "AVAXUSD", "DOGE": "DOGEUSD",
}

ASSET_MARKET = {
    "ES": "futures", "NQ": "futures", "YM": "futures", "RTY": "futures",
    "CL": "futures", "NG": "futures", "RB": "futures", "HO": "futures",
    "GC": "futures", "SI": "futures", "HG": "futures", "PL": "futures",
    "ZB": "futures", "ZN": "futures", "ZF": "futures", "ZT": "futures", "UB": "futures",
    "ZS": "futures", "ZC": "futures", "ZW": "futures", "ZM": "futures", "ZL": "futures",
    "HE": "futures", "LE": "futures",
    "6E": "futures", "6J": "futures", "6B": "futures", "NKD": "futures",
    "BTC": "crypto", "ETH": "crypto", "SOL": "crypto", "XRP": "crypto",
    "BNB": "crypto", "LINK": "crypto", "ADA": "crypto", "AVAX": "crypto", "DOGE": "crypto",
}


def fetch_fmp_daily(symbol: str, fmp_key: str, years: int = 10) -> Optional[Dict]:
    """Fetch daily OHLCV from FMP for the given symbol."""
    try:
        from_date = f"{2026 - years}-01-01"
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={from_date}&apikey={fmp_key}"
        logger.info(f"[FMP] Fetching {symbol} from {from_date}...")
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            logger.warning(f"[FMP] {symbol} HTTP {r.status_code}")
            return None
        data = r.json()
        hist = data.get("historical", [])
        if not hist:
            logger.warning(f"[FMP] {symbol} no historical data")
            return None
        # FMP returns newest first — reverse to chronological
        hist.reverse()
        closes = [bar["close"] for bar in hist if bar.get("close")]
        highs = [bar["high"] for bar in hist if bar.get("high")]
        lows = [bar["low"] for bar in hist if bar.get("low")]
        volumes = [bar.get("volume", 0) for bar in hist]
        dates = [bar["date"] for bar in hist]
        n = min(len(closes), len(highs), len(lows), len(dates))
        logger.info(f"[FMP] {symbol} loaded {n} daily bars ({dates[0] if dates else '?'} to {dates[-1] if dates else '?'})")
        return {
            "closes": closes[:n], "highs": highs[:n], "lows": lows[:n],
            "volumes": volumes[:n], "dates": dates[:n],
        }
    except Exception as e:
        logger.error(f"[FMP ERROR] {symbol}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# MAIN BACKTEST RUNNER
# ═══════════════════════════════════════════════════════════════

def run_historical_backtest(
    asset: str, fmp_key: str, years: int = 10,
    max_outcome_bars: int = 20, min_bars_between_signals: int = 5
) -> Dict:
    """
    Run complete historical backtest for one asset.
    
    Args:
        asset: Ticker symbol (e.g. "ES", "BTC")
        fmp_key: FMP API key
        years: Number of years of history
        max_outcome_bars: Max bars to track outcome
        min_bars_between_signals: Cooldown between signals to avoid clustering
    
    Returns:
        Dict with signals, outcomes, and summary stats
    """
    start_time = time.time()
    fmp_sym = ASSET_FMP_SYMBOLS.get(asset)
    market = ASSET_MARKET.get(asset, "futures")
    
    if not fmp_sym:
        return {"error": f"Unknown asset: {asset}", "asset": asset}
    
    # Fetch data
    data = fetch_fmp_daily(fmp_sym, fmp_key, years)
    if not data or len(data["closes"]) < 100:
        return {"error": f"Insufficient data for {asset}", "asset": asset,
                "bars_available": len(data["closes"]) if data else 0}
    
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    volumes = data["volumes"]
    dates = data["dates"]
    total_bars = len(closes)
    
    logger.info(f"[BACKTEST] {asset} starting: {total_bars} bars, {dates[0]} to {dates[-1]}")
    
    # Walk through bars, generate signals, track outcomes
    signals = []
    prev_squeeze = "none"
    last_signal_bar = -min_bars_between_signals  # Allow first signal immediately
    
    for bar_idx in range(50, total_bars - max_outcome_bars):
        # Cooldown: don't fire signals too close together
        if bar_idx - last_signal_bar < min_bars_between_signals:
            continue
        
        # Generate signal
        sig = generate_signal_at_bar(
            closes, highs, lows, volumes, bar_idx, asset, market, prev_squeeze
        )
        
        # Update squeeze state
        if bar_idx > 0:
            c_window = closes[:bar_idx + 1]
            h_window = highs[:bar_idx + 1]
            l_window = lows[:bar_idx + 1]
            if len(c_window) >= 20:
                prev_squeeze = detect_squeeze(c_window, h_window, l_window, prev_squeeze)
        
        if sig is None:
            continue
        
        # Track outcome
        outcome = track_outcome(sig, closes, highs, lows, max_outcome_bars)
        sig.update(outcome)
        sig["date"] = dates[bar_idx]
        sig["outcome_date"] = dates[outcome["outcome_bar"]] if outcome["outcome_bar"] < len(dates) else dates[-1]
        
        signals.append(sig)
        last_signal_bar = bar_idx
    
    # Compute summary statistics
    won = [s for s in signals if s["status"] == "won"]
    lost = [s for s in signals if s["status"] == "lost"]
    expired = [s for s in signals if s["status"] == "expired"]
    
    total = len(signals)
    resolved = len(won) + len(lost)
    win_rate = len(won) / resolved if resolved > 0 else 0
    
    # Wilson CI
    ci_low, ci_high = 0, 0
    if resolved > 0:
        z = 1.96
        p = len(won) / resolved
        denom = 1 + z * z / resolved
        center = (p + z * z / (2 * resolved)) / denom
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * resolved)) / resolved) / denom
        ci_low = max(0, center - spread)
        ci_high = min(1, center + spread)
    
    # P&L
    win_pnls = [s["pnl_pct"] for s in won]
    loss_pnls = [abs(s["pnl_pct"]) for s in lost]
    avg_win = np.mean(win_pnls) if win_pnls else 0
    avg_loss = np.mean(loss_pnls) if loss_pnls else 0
    profit_factor = sum(win_pnls) / sum(loss_pnls) if loss_pnls and sum(loss_pnls) > 0 else float("inf")
    
    # RSI zone breakdown
    rsi_zones = []
    for label, lo, hi in [("<25", 0, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35-45", 35, 45),
                           ("45-55", 45, 55), ("55-65", 55, 65), ("65-70", 65, 70), ("70-75", 70, 75), (">75", 75, 101)]:
        zone = [s for s in signals if s["status"] in ("won", "lost") and lo <= s["rsi"] < hi]
        zw = len([s for s in zone if s["status"] == "won"])
        zt = len(zone)
        if zt >= 3:
            rsi_zones.append({"zone": label, "won": zw, "total": zt, "win_rate": round(zw / zt, 4)})
    
    # By signal type
    buy_sigs = [s for s in signals if s["signal_type"] == "BUY" and s["status"] in ("won", "lost")]
    sell_sigs = [s for s in signals if s["signal_type"] == "SELL" and s["status"] in ("won", "lost")]
    buy_wr = len([s for s in buy_sigs if s["status"] == "won"]) / len(buy_sigs) if buy_sigs else 0
    sell_wr = len([s for s in sell_sigs if s["status"] == "won"]) / len(sell_sigs) if sell_sigs else 0
    
    elapsed = round(time.time() - start_time, 2)
    
    return {
        "asset": asset,
        "market": market,
        "fmp_symbol": fmp_sym,
        "total_bars": total_bars,
        "date_range": f"{dates[0]} to {dates[-1]}",
        "years": years,
        "summary": {
            "total_signals": total,
            "resolved": resolved,
            "won": len(won),
            "lost": len(lost),
            "expired": len(expired),
            "win_rate": round(win_rate, 4),
            "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            "avg_win_pct": round(float(avg_win), 4),
            "avg_loss_pct": round(float(avg_loss), 4),
            "profit_factor": round(float(profit_factor), 4) if profit_factor != float("inf") else "inf",
            "total_pnl_pct": round(sum(s["pnl_pct"] for s in signals if s["status"] in ("won", "lost")), 4),
        },
        "by_rsi_zone": rsi_zones,
        "by_signal_type": {
            "BUY": {"count": len(buy_sigs), "win_rate": round(buy_wr, 4)},
            "SELL": {"count": len(sell_sigs), "win_rate": round(sell_wr, 4)},
        },
        "signals": signals,  # Full signal list for detailed analysis
        "computation_time_s": elapsed,
    }


def run_full_backtest(assets: List[str], fmp_key: str, years: int = 10) -> Dict:
    """Run backtest across multiple assets and aggregate."""
    results = {}
    all_signals = []
    errors = []
    
    for asset in assets:
        logger.info(f"[FULL-BT] Processing {asset}...")
        result = run_historical_backtest(asset, fmp_key, years)
        if "error" in result:
            errors.append({"asset": asset, "error": result["error"]})
            continue
        results[asset] = result
        all_signals.extend(result["signals"])
        # Rate limit: FMP free tier
        time.sleep(0.5)
    
    # Aggregate stats
    resolved = [s for s in all_signals if s["status"] in ("won", "lost")]
    won = [s for s in resolved if s["status"] == "won"]
    total_resolved = len(resolved)
    overall_wr = len(won) / total_resolved if total_resolved > 0 else 0
    
    # Wilson CI
    ci_low, ci_high = 0, 0
    if total_resolved > 0:
        z = 1.96
        p = len(won) / total_resolved
        denom = 1 + z * z / total_resolved
        center = (p + z * z / (2 * total_resolved)) / denom
        spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total_resolved)) / total_resolved) / denom
        ci_low = max(0, center - spread)
        ci_high = min(1, center + spread)
    
    # Global RSI zones
    global_rsi_zones = []
    for label, lo, hi in [("<25", 0, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35-45", 35, 45),
                           ("45-55", 45, 55), ("55-65", 55, 65), ("65-70", 65, 70), ("70-75", 70, 75), (">75", 75, 101)]:
        zone = [s for s in resolved if lo <= s["rsi"] < hi]
        zw = len([s for s in zone if s["status"] == "won"])
        zt = len(zone)
        if zt >= 5:
            global_rsi_zones.append({"zone": label, "won": zw, "total": zt, "win_rate": round(zw / zt, 4)})
    
    # Per-asset qualification
    asset_stats = {}
    for asset, result in results.items():
        s = result["summary"]
        asset_stats[asset] = {
            "market": result["market"],
            "signals": s["total_signals"],
            "resolved": s["resolved"],
            "won": s["won"],
            "lost": s["lost"],
            "win_rate": s["win_rate"],
            "ci_95": s["ci_95"],
            "profit_factor": s["profit_factor"],
            "total_pnl_pct": s["total_pnl_pct"],
        }
    
    # Factor attribution (global)
    factor_wins = defaultdict(int)
    factor_total = defaultdict(int)
    for s in resolved:
        for f in s.get("key_factors", []):
            factor_total[f] += 1
            if s["status"] == "won":
                factor_wins[f] += 1
    factor_stats = {}
    for f in factor_total:
        ft = factor_total[f]
        fw = factor_wins[f]
        if ft >= 10:
            fwr = fw / ft
            factor_stats[f] = {
                "won": fw, "total": ft,
                "win_rate": round(fwr, 4),
                "edge_vs_baseline": round(fwr - overall_wr, 4),
            }
    
    return {
        "total_assets": len(results),
        "total_signals": len(all_signals),
        "total_resolved": total_resolved,
        "overall_win_rate": round(overall_wr, 4),
        "ci_95": [round(ci_low, 4), round(ci_high, 4)],
        "global_rsi_zones": global_rsi_zones,
        "asset_stats": dict(sorted(asset_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True)),
        "factor_attribution": dict(sorted(factor_stats.items(), key=lambda x: x[1]["edge_vs_baseline"], reverse=True)),
        "errors": errors,
        "per_asset_details": {k: {**v, "signals": v["signals"][:50]} for k, v in results.items()},  # Truncate for JSON size
    }
