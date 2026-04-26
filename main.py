from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

app = FastAPI(title="EGX Stock API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EGX_MAP = {
    "COMI": "COMI.CA", "HRHO": "HRHO.CA", "TMGH": "TMGH.CA",
    "SWDY": "SWDY.CA", "ORWE": "ORWE.CA", "JUFO": "JUFO.CA",
    "PHDC": "PHDC.CA", "MNHD": "MNHD.CA", "CLHO": "CLHO.CA",
    "AMOC": "AMOC.CA", "ABUK": "ABUK.CA", "EKHW": "EKHW.CA",
    "ESRS": "ESRS.CA", "SKPC": "SKPC.CA", "MCIT": "MCIT.CA",
    "EXPA": "EXPA.CA", "EFIC": "EFIC.CA", "ORAS": "ORAS.CA",
    "EGAL": "EGAL.CA", "ETEL": "ETEL.CA", "MFPC": "MFPC.CA",
    "HELI": "HELI.CA", "AIRC": "AIRC.CA", "GBCO": "GBCO.CA",
}

def to_ticker(symbol: str) -> str:
    s = symbol.upper().strip()
    if s.endswith(".CA"):
        return s
    return EGX_MAP.get(s, s + ".CA")

def safe(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    delta = pd.Series(closes).diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1]
    return round(float(v), 2) if not np.isnan(v) else None

def calc_macd(closes):
    s = pd.Series(closes)
    ema12 = s.ewm(span=12).mean()
    ema26 = s.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return {
        "macd": round(float(macd.iloc[-1]), 4),
        "signal": round(float(signal.iloc[-1]), 4),
        "histogram": round(float(macd.iloc[-1] - signal.iloc[-1]), 4),
    }

def calc_bb(closes, period=20):
    s = pd.Series(closes)
    ma = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return {
        "upper": round(float(upper.iloc[-1]), 2),
        "middle": round(float(ma.iloc[-1]), 2),
        "lower": round(float(lower.iloc[-1]), 2),
    }

@app.get("/")
def root():
    return {"status": "EGX Stock API is live ✅", "version": "1.0.0"}

@app.get("/stock/{symbol}")
def get_stock(symbol: str, period: str = "3mo"):
    ticker_sym = to_ticker(symbol)
    try:
        t = yf.Ticker(ticker_sym)
        hist = t.history(period=period)
        if hist.empty:
            raise HTTPException(404, f"No data for {symbol}. Check the ticker symbol.")

        info = t.info
        fi = t.fast_info

        closes = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()
        dates = [str(d.date()) for d in hist.index]

        cur = closes[-1]
        prev = closes[-2] if len(closes) > 1 else cur

        ma20 = round(float(pd.Series(closes).rolling(20).mean().iloc[-1]), 2) if len(closes) >= 20 else None
        ma50 = round(float(pd.Series(closes).rolling(50).mean().iloc[-1]), 2) if len(closes) >= 50 else None
        ma200 = round(float(pd.Series(closes).rolling(200).mean().iloc[-1]), 2) if len(closes) >= 200 else None

        week_chg = round((cur - closes[-6]) / closes[-6] * 100, 2) if len(closes) >= 6 else None
        month_chg = round((cur - closes[-22]) / closes[-22] * 100, 2) if len(closes) >= 22 else None
        q3_chg = round((cur - closes[-66]) / closes[-66] * 100, 2) if len(closes) >= 66 else None
        ytd_chg = round((cur - closes[0]) / closes[0] * 100, 2)

        rsi = calc_rsi(closes)
        macd = calc_macd(closes)
        bb = calc_bb(closes)

        # Support & Resistance (simple)
        high_20 = round(max(hist["High"].tail(20)), 2)
        low_20 = round(min(hist["Low"].tail(20)), 2)

        return {
            "symbol": symbol.upper(),
            "ticker": ticker_sym,
            "name": info.get("longName") or info.get("shortName") or symbol.upper(),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "currency": info.get("currency", "EGP"),
            "exchange": "EGX",
            "price": {
                "current": round(cur, 2),
                "previous_close": round(prev, 2),
                "open": round(float(hist["Open"].iloc[-1]), 2),
                "high": round(float(hist["High"].iloc[-1]), 2),
                "low": round(float(hist["Low"].iloc[-1]), 2),
                "day_change": round(cur - prev, 2),
                "day_change_pct": round((cur - prev) / prev * 100, 2),
                "week_change_pct": week_chg,
                "month_change_pct": month_chg,
                "three_month_change_pct": q3_chg,
                "ytd_change_pct": ytd_chg,
                "year_high": safe(fi.get("yearHigh")),
                "year_low": safe(fi.get("yearLow")),
                "market_cap": safe(fi.get("marketCap")),
                "volume": int(hist["Volume"].iloc[-1]),
                "avg_volume_10d": safe(fi.get("tenDayAverageVolume")),
            },
            "fundamentals": {
                "pe_ratio": safe(info.get("trailingPE")),
                "forward_pe": safe(info.get("forwardPE")),
                "book_value": safe(info.get("bookValue")),
                "price_to_book": safe(info.get("priceToBook")),
                "eps": safe(info.get("trailingEps")),
                "roe": safe(info.get("returnOnEquity")),
                "roa": safe(info.get("returnOnAssets")),
                "debt_to_equity": safe(info.get("debtToEquity")),
                "profit_margin": safe(info.get("profitMargins")),
                "revenue_growth": safe(info.get("revenueGrowth")),
                "earnings_growth": safe(info.get("earningsGrowth")),
                "dividend_yield": safe(info.get("dividendYield")),
                "beta": safe(info.get("beta")),
                "fifty_day_avg": safe(fi.get("fiftyDayAverage")),
                "two_hundred_day_avg": safe(fi.get("twoHundredDayAverage")),
                "shares_outstanding": safe(fi.get("shares")),
            },
            "technical": {
                "rsi_14": rsi,
                "macd": macd,
                "bollinger_bands": bb,
                "ma20": ma20,
                "ma50": ma50,
                "ma200": ma200,
                "resistance_20d": high_20,
                "support_20d": low_20,
                "price_vs_ma20": "above" if ma20 and cur > ma20 else "below",
                "price_vs_ma50": "above" if ma50 and cur > ma50 else "below",
                "price_vs_ma200": "above" if ma200 and cur > ma200 else "below",
            },
            "history": {
                "dates": dates,
                "closes": [round(c, 2) for c in closes],
                "opens": [round(float(o), 2) for o in hist["Open"].tolist()],
                "highs": [round(float(h), 2) for h in hist["High"].tolist()],
                "lows": [round(float(l), 2) for l in hist["Low"].tolist()],
                "volumes": [int(v) for v in volumes],
            },
            "last_updated": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/compare")
def compare(symbols: str):
    result = {}
    for sym in symbols.upper().split(","):
        sym = sym.strip()
        try:
            ticker_sym = to_ticker(sym)
            t = yf.Ticker(ticker_sym)
            hist = t.history(period="1mo")
            fi = t.fast_info
            if hist.empty:
                result[sym] = {"error": "Not found"}
                continue
            closes = hist["Close"].tolist()
            cur = closes[-1]
            prev = closes[-2] if len(closes) > 1 else cur
            result[sym] = {
                "name": t.info.get("longName", sym),
                "price": round(cur, 2),
                "change_pct": round((cur - prev) / prev * 100, 2),
                "month_change_pct": round((cur - closes[0]) / closes[0] * 100, 2),
                "market_cap": safe(fi.get("marketCap")),
                "pe": safe(t.info.get("trailingPE")),
                "year_high": safe(fi.get("yearHigh")),
                "year_low": safe(fi.get("yearLow")),
            }
        except Exception as e:
            result[sym] = {"error": str(e)}
    return result

@app.get("/market/egx30")
def egx30():
    try:
        t = yf.Ticker("^CASE30")
        fi = t.fast_info
        hist = t.history(period="3mo")
        closes = hist["Close"].tolist()
        dates = [str(d.date()) for d in hist.index]
        cur = closes[-1] if closes else 0
        prev = closes[-2] if len(closes) > 1 else cur
        month_chg = round((cur - closes[-22]) / closes[-22] * 100, 2) if len(closes) >= 22 else None
        return {
            "index": "EGX30",
            "current": round(cur, 2),
            "change": round(cur - prev, 2),
            "change_pct": round((cur - prev) / prev * 100, 2) if prev else 0,
            "month_change_pct": month_chg,
            "year_high": safe(fi.get("yearHigh")),
            "year_low": safe(fi.get("yearLow")),
            "history": {"dates": dates, "closes": [round(c, 2) for c in closes]},
        }
    except Exception as e:
        raise HTTPException(500, str(e))
