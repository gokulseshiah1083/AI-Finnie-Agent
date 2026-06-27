import logging
import time
from functools import lru_cache
from typing import Optional

import yfinance as yf
import pandas as pd

from src.core.config import CONFIG

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, dict]] = {}


def _is_fresh(timestamp: float) -> bool:
    ttl = CONFIG["market_data"]["cache_ttl_minutes"] * 60
    return (time.time() - timestamp) < ttl


def get_stock_info(ticker: str) -> Optional[dict]:
    ticker = ticker.upper().strip()
    if ticker in _cache:
        ts, data = _cache[ticker]
        if _is_fresh(ts):
            return data

    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="1mo")

        if hist.empty:
            return None

        current_price = float(hist["Close"].iloc[-1])
        month_ago_price = float(hist["Close"].iloc[0])
        monthly_change = ((current_price - month_ago_price) / month_ago_price) * 100

        year_hist = t.history(period="1y")
        if not year_hist.empty:
            year_ago_price = float(year_hist["Close"].iloc[0])
            yearly_change = ((current_price - year_ago_price) / year_ago_price) * 100
        else:
            yearly_change = None

        result = {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "current_price": round(current_price, 2),
            "monthly_change_pct": round(monthly_change, 2),
            "yearly_change_pct": round(yearly_change, 2) if yearly_change is not None else None,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }
        _cache[ticker] = (time.time(), result)
        return result
    except Exception as e:
        logger.error(f"Failed to fetch data for {ticker}: {e}")
        return None


def get_market_overview() -> dict:
    symbols = CONFIG["market_data"]["default_symbols"]
    overview = {}
    for sym in symbols:
        data = get_stock_info(sym)
        if data:
            overview[sym] = data
    return overview


def get_price_history(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(ticker.upper())
        hist = t.history(period=period)
        return hist if not hist.empty else None
    except Exception as e:
        logger.error(f"Failed to fetch history for {ticker}: {e}")
        return None
