import logging
import os
import time
from typing import Optional

import requests
import pandas as pd
import yfinance as yf

from src.core.config import CONFIG

logger = logging.getLogger(__name__)

_cache: dict[str, tuple[float, dict]] = {}
_history_cache: dict[str, tuple[float, pd.DataFrame]] = {}

BASE_URL = CONFIG["market_data"]["alpha_vantage_base_url"]
TIMEOUT = CONFIG["market_data"]["alpha_vantage_timeout"]


def _get_av_key() -> Optional[str]:
    return os.getenv("ALPHA_VANTAGE_API_KEY")


def _is_fresh(timestamp: float) -> bool:
    ttl = CONFIG["market_data"]["cache_ttl_minutes"] * 60
    return (time.time() - timestamp) < ttl


def _av_request(params: dict) -> Optional[dict]:
    key = _get_av_key()
    if not key:
        return None
    try:
        resp = requests.get(
            BASE_URL,
            params={**params, "apikey": key},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        # Alpha Vantage returns error messages inside JSON
        if "Error Message" in data or "Information" in data:
            msg = data.get("Error Message") or data.get("Information", "")
            logger.warning(f"Alpha Vantage: {msg}")
            return None
        return data
    except Exception as e:
        logger.warning(f"Alpha Vantage request failed: {e}")
        return None


# ── Alpha Vantage fetchers ────────────────────────────────────────────────────

def _av_get_stock_info(ticker: str) -> Optional[dict]:
    """Fetch quote + overview from Alpha Vantage."""
    quote_data = _av_request({"function": "GLOBAL_QUOTE", "symbol": ticker})
    overview_data = _av_request({"function": "OVERVIEW", "symbol": ticker})

    if not quote_data:
        return None

    quote = quote_data.get("Global Quote", {})
    if not quote or not quote.get("05. price"):
        return None

    overview = overview_data or {}

    current_price = float(quote.get("05. price", 0))
    prev_close = float(quote.get("08. previous close", 0) or 0)
    change_pct = float(quote.get("10. change percent", "0%").replace("%", "") or 0)

    # 52-week range from overview
    week_high = overview.get("52WeekHigh")
    week_low = overview.get("52WeekLow")

    # Monthly change from TIME_SERIES_DAILY
    monthly_change = _av_monthly_change(ticker, current_price)

    return {
        "ticker": ticker,
        "name": overview.get("Name", ticker),
        "current_price": round(current_price, 2),
        "daily_change_pct": round(change_pct, 2),
        "monthly_change_pct": monthly_change,
        "yearly_change_pct": None,  # would need 1yr daily series — expensive on free tier
        "market_cap": overview.get("MarketCapitalization"),
        "pe_ratio": _safe_float(overview.get("PERatio")),
        "dividend_yield": _safe_float(overview.get("DividendYield")),
        "52w_high": _safe_float(week_high),
        "52w_low": _safe_float(week_low),
        "sector": overview.get("Sector"),
        "industry": overview.get("Industry"),
        "source": "Alpha Vantage",
    }


def _av_monthly_change(ticker: str, current_price: float) -> Optional[float]:
    """Calculate ~1-month change using daily time series (compact = last 100 days)."""
    data = _av_request({
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": "compact",
    })
    if not data:
        return None
    series = data.get("Time Series (Daily)", {})
    if not series:
        return None
    dates = sorted(series.keys())
    if len(dates) < 21:
        return None
    # ~21 trading days ≈ 1 month ago
    month_ago_close = float(series[dates[-21]]["4. close"])
    return round(((current_price - month_ago_close) / month_ago_close) * 100, 2)


def _av_get_price_history(ticker: str, months: int = 6) -> Optional[pd.DataFrame]:
    """Fetch daily price history from Alpha Vantage."""
    outputsize = "full" if months > 3 else "compact"
    data = _av_request({
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": outputsize,
    })
    if not data:
        return None
    series = data.get("Time Series (Daily)", {})
    if not series:
        return None

    rows = [
        {"Date": pd.Timestamp(date), "Close": float(vals["4. close"])}
        for date, vals in series.items()
    ]
    df = pd.DataFrame(rows).sort_values("Date").set_index("Date")

    cutoff = pd.Timestamp.now() - pd.DateOffset(months=months)
    return df[df.index >= cutoff]


def _av_get_news(topics: str = "financial_markets") -> list[dict]:
    """Fetch financial news from Alpha Vantage NEWS_SENTIMENT endpoint."""
    data = _av_request({
        "function": "NEWS_SENTIMENT",
        "topics": topics,
        "limit": CONFIG["market_data"]["news_limit"],
        "sort": "LATEST",
    })
    if not data:
        return []

    items = []
    for article in data.get("feed", []):
        items.append({
            "title": article.get("title", ""),
            "publisher": article.get("source", ""),
            "summary": article.get("summary", ""),
            "url": article.get("url", ""),
            "sentiment": article.get("overall_sentiment_label", ""),
        })
    return items


# ── yfinance fallbacks ────────────────────────────────────────────────────────

def _yf_get_stock_info(ticker: str) -> Optional[dict]:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="1mo")
        if hist.empty:
            return None
        current_price = float(hist["Close"].iloc[-1])
        month_ago_price = float(hist["Close"].iloc[0])
        monthly_change = ((current_price - month_ago_price) / month_ago_price) * 100
        return {
            "ticker": ticker,
            "name": info.get("longName") or info.get("shortName", ticker),
            "current_price": round(current_price, 2),
            "daily_change_pct": None,
            "monthly_change_pct": round(monthly_change, 2),
            "yearly_change_pct": None,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": info.get("dividendYield"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "source": "yfinance (fallback)",
        }
    except Exception as e:
        logger.error(f"yfinance fallback failed for {ticker}: {e}")
        return None


def _yf_get_price_history(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    try:
        hist = yf.Ticker(ticker).history(period=period)
        if hist.empty:
            return None
        return hist[["Close"]]
    except Exception as e:
        logger.error(f"yfinance history fallback failed for {ticker}: {e}")
        return None


def _yf_get_news() -> list[dict]:
    all_news, seen = [], set()
    for sym in ["SPY", "QQQ", "^GSPC"]:
        try:
            for item in (yf.Ticker(sym).news or [])[:5]:
                title = item.get("title", "")
                if title and title not in seen:
                    seen.add(title)
                    all_news.append({
                        "title": title,
                        "publisher": item.get("publisher", ""),
                        "summary": item.get("summary", ""),
                        "url": item.get("link", ""),
                        "sentiment": "",
                    })
        except Exception:
            pass
    return all_news[:10]


# ── Public API ────────────────────────────────────────────────────────────────

def get_stock_info(ticker: str) -> Optional[dict]:
    ticker = ticker.upper().strip()
    if ticker in _cache:
        ts, data = _cache[ticker]
        if _is_fresh(ts):
            return data

    result = _av_get_stock_info(ticker)
    # Alpha Vantage makes three separate calls (quote, overview, time series) to
    # build one info dict. A rate limit or transient failure on just the time
    # series call still leaves a truthy dict with monthly_change_pct=None — that
    # used to short-circuit the "or" below and skip the yfinance fallback
    # entirely. Treat a missing monthly_change_pct as an incomplete AV result
    # and prefer a full yfinance result when one is available.
    if not result or result.get("monthly_change_pct") is None:
        fallback = _yf_get_stock_info(ticker)
        if fallback:
            result = fallback
    if result:
        _cache[ticker] = (time.time(), result)
    return result


def get_market_overview() -> dict:
    overview = {}
    for sym in CONFIG["market_data"]["default_symbols"]:
        data = get_stock_info(sym)
        if data:
            overview[sym] = data
    return overview


def get_price_history(ticker: str, period: str = "6mo") -> Optional[pd.DataFrame]:
    ticker = ticker.upper().strip()
    cache_key = f"{ticker}:{period}"
    if cache_key in _history_cache:
        ts, df = _history_cache[cache_key]
        if _is_fresh(ts):
            return df

    # Convert period string to months for AV
    months_map = {"1mo": 1, "3mo": 3, "6mo": 6, "1y": 12, "2y": 24}
    months = months_map.get(period, 6)

    df = _av_get_price_history(ticker, months) or _yf_get_price_history(ticker, period)
    if df is not None:
        _history_cache[cache_key] = (time.time(), df)
    return df


def get_financial_news(topics: str = "financial_markets") -> list[dict]:
    return _av_get_news(topics) or _yf_get_news()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_float(val) -> Optional[float]:
    try:
        f = float(val)
        return f if f > 0 else None
    except (TypeError, ValueError):
        return None
