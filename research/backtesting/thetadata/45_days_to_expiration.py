"""
Covered Call (Short Call) Backtesting System
=============================================
Strategy:
- Start on the first trading day of the year
- Execute every order at 10:00 AM EST
- Sell a call with ~25 DTE and delta ~0.2 at bid price
- Roll the call (buy back at ask + sell new) when:
    1. DTE <= 21, OR
    2. 50% profit earned
- Slippage of $0.01 per roll (applied to both buy and sell legs)

Uses ThetaData v3 REST API (local terminal on port 25510).
"""

import datetime
import logging
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
THETADATA_BASE = os.environ.get("THETADATA_BASE", "http://127.0.0.1:25510")
ROOT = "SPY"
YEAR = 2025  # backtest year
TARGET_DTE = 45
TARGET_DELTA = 0.20
ROLL_DTE_THRESHOLD = 21
PROFIT_TARGET_PCT = 0.50
SLIPPAGE = 0.01  # per leg
MS_10AM_EST = 10 * 60 * 60 * 1000  # 36_000_000 ms from midnight

OUTPUT_DIR = Path(__file__).parent / "output"
TRADE_LOG_FILE = OUTPUT_DIR / "trade_log.xlsx"
CHART_FILE = OUTPUT_DIR / "backtest_chart.png"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ThetaData REST client helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict | None = None, retries: int = 3) -> dict:
    """Call a ThetaData REST endpoint with basic retry logic."""
    url = f"{THETADATA_BASE}{endpoint}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data
        except (requests.RequestException, ValueError) as exc:
            log.warning("Request %s attempt %d failed: %s", url, attempt, exc)
            if attempt < retries:
                time.sleep(1 * attempt)
            else:
                raise


def get_expirations(root: str) -> list[datetime.date]:
    """Return sorted list of available expiration dates for *root*."""
    data = _get("/v2/list/expirations", {"root": root})
    raw = data["response"]
    dates = []
    for item in raw:
        if isinstance(item, dict):
            d = item.get("date", item.get("exp"))
        else:
            d = item
        dates.append(_parse_date(d))
    return sorted(dates)


def get_strikes(root: str, exp: datetime.date) -> list[float]:
    """Return available strikes (in dollars) for *root* / *exp*."""
    data = _get("/v2/list/strikes", {"root": root, "exp": _fmt_date(exp)})
    raw = data["response"]
    # Strikes are returned in millidollars (e.g., 480000 -> $480.00)
    return sorted(s / 1000.0 if s > 100_000 else s for s in raw)


def get_bulk_quotes_at_time(
    root: str,
    exp: datetime.date,
    date: datetime.date,
    ms_of_day: int = MS_10AM_EST,
) -> pd.DataFrame:
    """Fetch bid/ask for ALL strikes of *root*/*exp* on *date* at given time."""
    data = _get(
        "/v2/bulk_hist/option/quote",
        {
            "root": root,
            "exp": _fmt_date(exp),
            "start_date": _fmt_date(date),
            "end_date": _fmt_date(date),
            "ivl": 0,
        },
    )
    df = _to_dataframe(data)
    if df.empty:
        return df
    df = _filter_closest_time(df, ms_of_day)
    return df


def get_bulk_greeks_at_time(
    root: str,
    exp: datetime.date,
    date: datetime.date,
    ms_of_day: int = MS_10AM_EST,
) -> pd.DataFrame:
    """Fetch greeks for ALL strikes of *root*/*exp* on *date* at given time."""
    data = _get(
        "/v2/bulk_hist/option/greeks",
        {
            "root": root,
            "exp": _fmt_date(exp),
            "start_date": _fmt_date(date),
            "end_date": _fmt_date(date),
            "ivl": 0,
        },
    )
    df = _to_dataframe(data)
    if df.empty:
        return df
    df = _filter_closest_time(df, ms_of_day)
    return df


def get_option_quote(
    root: str,
    exp: datetime.date,
    strike: float,
    right: str,
    date: datetime.date,
    ms_of_day: int = MS_10AM_EST,
) -> dict:
    """Return bid/ask for a single option contract at the given time."""
    strike_raw = int(round(strike * 1000))
    data = _get(
        "/v2/hist/option/quote",
        {
            "root": root,
            "exp": _fmt_date(exp),
            "strike": strike_raw,
            "right": right,
            "start_date": _fmt_date(date),
            "end_date": _fmt_date(date),
            "ivl": 0,
        },
    )
    df = _to_dataframe(data)
    if df.empty:
        return {"bid": None, "ask": None}
    df = _filter_closest_time(df, ms_of_day)
    if df.empty:
        return {"bid": None, "ask": None}
    row = df.iloc[0]
    return {
        "bid": row.get("bid", row.get("bid_price")),
        "ask": row.get("ask", row.get("ask_price")),
    }


def get_option_greeks(
    root: str,
    exp: datetime.date,
    strike: float,
    right: str,
    date: datetime.date,
    ms_of_day: int = MS_10AM_EST,
) -> dict:
    """Return greeks for a single option contract at the given time."""
    strike_raw = int(round(strike * 1000))
    data = _get(
        "/v2/hist/option/greeks",
        {
            "root": root,
            "exp": _fmt_date(exp),
            "strike": strike_raw,
            "right": right,
            "start_date": _fmt_date(date),
            "end_date": _fmt_date(date),
            "ivl": 0,
        },
    )
    df = _to_dataframe(data)
    if df.empty:
        return {"delta": None}
    df = _filter_closest_time(df, ms_of_day)
    if df.empty:
        return {"delta": None}
    row = df.iloc[0]
    return {"delta": row.get("delta")}


def get_stock_eod(root: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Return daily OHLCV data for *root*."""
    data = _get(
        "/v2/hist/stock/eod",
        {
            "root": root,
            "start_date": _fmt_date(start),
            "end_date": _fmt_date(end),
        },
    )
    df = _to_dataframe(data)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _fmt_date(d: datetime.date) -> str:
    return d.strftime("%Y%m%d")


def _parse_date(val) -> datetime.date:
    s = str(val)
    return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _to_dataframe(data: dict) -> pd.DataFrame:
    """Convert a ThetaData JSON response to a DataFrame.

    ThetaData responses come in two common shapes:
      1. {"header": {"format": [col1, col2, ...]}, "response": [[v1, v2, ...], ...]}
      2. {"response": [{col1: v1, ...}, ...]}
    """
    header = data.get("header", {})
    columns = header.get("format", header.get("columns"))
    response = data.get("response", [])

    if not response:
        return pd.DataFrame()

    if columns and isinstance(response[0], list):
        return pd.DataFrame(response, columns=[c.lower() for c in columns])
    elif isinstance(response[0], dict):
        return pd.DataFrame(response)
    else:
        return pd.DataFrame(response, columns=["value"])


def _filter_closest_time(df: pd.DataFrame, target_ms: int) -> pd.DataFrame:
    """Keep only rows whose ms_of_day is closest to *target_ms*."""
    if "ms_of_day" not in df.columns:
        return df
    diff = (df["ms_of_day"] - target_ms).abs()
    min_diff = diff.min()
    return df.loc[diff == min_diff].copy()


# ---------------------------------------------------------------------------
# Strategy logic
# ---------------------------------------------------------------------------

class Position:
    """Tracks a single short call position."""

    def __init__(
        self,
        open_date: datetime.date,
        exp: datetime.date,
        strike: float,
        sell_price: float,
        delta: float,
    ):
        self.open_date = open_date
        self.exp = exp
        self.strike = strike
        self.sell_price = sell_price
        self.delta = delta

    def dte_as_of(self, date: datetime.date) -> int:
        return (self.exp - date).days

    def __repr__(self):
        return (
            f"Position(strike={self.strike}, exp={self.exp}, "
            f"sell_price={self.sell_price:.2f}, delta={self.delta:.3f})"
        )


def find_best_expiration(
    expirations: list[datetime.date], ref_date: datetime.date, target_dte: int
) -> datetime.date:
    """Pick the expiration closest to *ref_date + target_dte*."""
    target = ref_date + datetime.timedelta(days=target_dte)
    return min(expirations, key=lambda e: abs((e - target).days))


def select_strike_by_delta(
    root: str,
    exp: datetime.date,
    date: datetime.date,
    target_delta: float,
) -> tuple[float, float, float] | None:
    """Find the call strike whose delta is closest to *target_delta*.

    Returns (strike, bid, delta) or None if no data available.
    """
    greeks_df = get_bulk_greeks_at_time(root, exp, date)
    quotes_df = get_bulk_quotes_at_time(root, exp, date)

    if greeks_df.empty or quotes_df.empty:
        log.warning("No bulk data for %s exp=%s on %s", root, exp, date)
        return None

    # Filter to calls only
    for df in (greeks_df, quotes_df):
        if "right" in df.columns:
            df.drop(df[df["right"].astype(str).str.upper() != "C"].index, inplace=True)

    if greeks_df.empty:
        return None

    # Find strike with delta closest to target
    greeks_df = greeks_df.copy()
    greeks_df["delta_diff"] = (greeks_df["delta"].astype(float) - target_delta).abs()
    best = greeks_df.loc[greeks_df["delta_diff"].idxmin()]
    strike = float(best["strike"])

    # Normalise strike if in millidollars
    if strike > 100_000:
        strike = strike / 1000.0

    # Look up the bid from quotes
    quotes_df = quotes_df.copy()
    strike_col = quotes_df["strike"].astype(float)
    if strike_col.max() > 100_000:
        strike_col = strike_col / 1000.0
        quotes_df["strike"] = strike_col

    match = quotes_df.loc[(quotes_df["strike"] - strike).abs() < 0.01]
    if match.empty:
        q = get_option_quote(root, exp, strike, "C", date)
        bid = q["bid"]
    else:
        bid = float(match.iloc[0].get("bid", match.iloc[0].get("bid_price", 0)))

    delta = float(best["delta"])
    return strike, bid, delta


def run_backtest(year: int = YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the covered call backtest for the given *year*.

    Returns
    -------
    trade_log : DataFrame  -- every buy/sell with required columns
    stock_eod : DataFrame  -- daily SPY prices for charting
    """
    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(year, 12, 31)

    # Trading days
    trading_days = pd.bdate_range(start_date, end_date).date.tolist()
    log.info(
        "Backtest period: %s - %s  (%d trading days)",
        start_date, end_date, len(trading_days),
    )

    # Fetch all expirations once
    all_exps = get_expirations(ROOT)
    year_exps = [e for e in all_exps if e.year in (year, year + 1)]

    # Fetch stock EOD for charting
    stock_eod = get_stock_eod(ROOT, start_date, end_date)

    # ---- State ----
    position: Position | None = None
    account_change = 0.0  # cumulative P&L (per-share premium)
    trade_rows: list[dict] = []

    def log_trade(
        date: datetime.date,
        action: str,
        price: float,
        dte: int,
        change: float,
    ):
        nonlocal account_change
        account_change += change
        trade_rows.append(
            {
                "Date": date,
                "Position": action,
                "Right": "Call",
                "Price": round(price, 4),
                "DTE": dte,
                "Account Change ($)": round(change * 100, 2),  # per contract
                "Cumulative P&L ($)": round(account_change * 100, 2),
            },
        )
        log.info(
            "  %s  %-4s  price=%.4f  DTE=%d  chg=%.2f  cumPnL=%.2f",
            date, action, price, dte, change * 100, account_change * 100,
        )

    def open_new_position(day: datetime.date) -> Position | None:
        exp = find_best_expiration(year_exps, day, TARGET_DTE)
        result = select_strike_by_delta(ROOT, exp, day, TARGET_DELTA)
        if result is None:
            log.warning("Could not find suitable strike on %s", day)
            return None
        strike, bid, delta = result
        log.info(
            "Opening position on %s: strike=%.2f exp=%s bid=%.4f delta=%.3f",
            day, strike, exp, bid, delta,
        )
        return Position(open_date=day, exp=exp, strike=strike, sell_price=bid, delta=delta)

    # ---- Main loop ----
    for day in trading_days:
        if not year_exps:
            break

        # -- Check for roll / close on existing position --
        if position is not None:
            dte_today = position.dte_as_of(day)

            # If past expiration, close at 0
            if day >= position.exp:
                log.info("Position expired on %s", day)
                log_trade(day, "Buy", 0.0, 0, 0.0)
                position = None
            else:
                # Fetch current ask price
                q = get_option_quote(ROOT, position.exp, position.strike, "C", day)
                current_ask = q.get("ask")

                if current_ask is not None:
                    current_ask = float(current_ask)
                    profit_pct = (
                        (position.sell_price - current_ask) / position.sell_price
                        if position.sell_price > 0
                        else 0
                    )

                    should_roll = False
                    roll_reason = ""

                    if dte_today <= ROLL_DTE_THRESHOLD:
                        should_roll = True
                        roll_reason = f"DTE={dte_today}<={ROLL_DTE_THRESHOLD}"
                    elif profit_pct >= PROFIT_TARGET_PCT:
                        should_roll = True
                        roll_reason = f"profit={profit_pct:.0%}>={PROFIT_TARGET_PCT:.0%}"

                    if should_roll:
                        buy_price = current_ask + SLIPPAGE
                        log.info("Rolling on %s (%s): buy back @ %.4f", day, roll_reason, buy_price)
                        log_trade(day, "Buy", buy_price, dte_today, -buy_price)
                        position = None

                        # Open new position immediately
                        new_pos = open_new_position(day)
                        if new_pos is not None:
                            actual_sell = new_pos.sell_price - SLIPPAGE
                            new_pos.sell_price = actual_sell
                            position = new_pos
                            log_trade(day, "Sell", actual_sell, new_pos.dte_as_of(day), actual_sell)

        # -- Open initial position if we have none --
        if position is None and len(trade_rows) == 0:
            new_pos = open_new_position(day)
            if new_pos is not None:
                position = new_pos
                log_trade(day, "Sell", position.sell_price, position.dte_as_of(day), position.sell_price)

        # Also open if we lost a position (e.g. expired) and haven't re-entered
        if position is None and len(trade_rows) > 0 and trade_rows[-1]["Position"] == "Buy":
            new_pos = open_new_position(day)
            if new_pos is not None:
                actual_sell = new_pos.sell_price - SLIPPAGE
                new_pos.sell_price = actual_sell
                position = new_pos
                log_trade(day, "Sell", actual_sell, new_pos.dte_as_of(day), actual_sell)

    # -- Close remaining position at year end --
    if position is not None:
        last_day = trading_days[-1]
        q = get_option_quote(ROOT, position.exp, position.strike, "C", last_day)
        ask = float(q.get("ask", 0) or 0)
        buy_price = ask + SLIPPAGE
        log_trade(last_day, "Buy", buy_price, position.dte_as_of(last_day), -buy_price)
        position = None

    trade_log = pd.DataFrame(trade_rows)
    return trade_log, stock_eod


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(trade_log: pd.DataFrame) -> dict:
    """Compute annual return, max drawdown, etc."""
    if trade_log.empty:
        return {}

    cumulative = trade_log["Cumulative P&L ($)"].values

    peak = np.maximum.accumulate(cumulative)
    drawdown = cumulative - peak
    max_dd = drawdown.min()

    total_return = cumulative[-1]

    metrics = {
        "Total P&L (per contract, $)": round(total_return, 2),
        "Max Drawdown (per contract, $)": round(max_dd, 2),
        "Number of Sells": int((trade_log["Position"] == "Sell").sum()),
        "Number of Buys": int((trade_log["Position"] == "Buy").sum()),
        "Total Trades": len(trade_log),
    }
    return metrics


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------

def plot_backtest(trade_log: pd.DataFrame, stock_eod: pd.DataFrame):
    """Plot SPY price with trade markers and save to file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot SPY price
    if not stock_eod.empty:
        dates = stock_eod["date"] if "date" in stock_eod.columns else stock_eod.index
        close = stock_eod["close"] if "close" in stock_eod.columns else stock_eod.iloc[:, -2]
        ax.plot(dates, close, color="black", linewidth=1, label="SPY Close")

    # Overlay trade markers
    if not trade_log.empty and not stock_eod.empty:
        # Build date -> close price map
        if "date" in stock_eod.columns:
            price_map = dict(
                zip(
                    pd.to_datetime(stock_eod["date"]).dt.date,
                    stock_eod["close"],
                )
            )
        else:
            price_map = {}

        sells = trade_log[trade_log["Position"] == "Sell"]
        buys = trade_log[trade_log["Position"] == "Buy"]

        for _, row in sells.iterrows():
            d = row["Date"]
            y = price_map.get(d)
            if y is not None:
                ax.scatter(pd.Timestamp(d), y, marker="x", color="green", s=120, zorder=5)

        for _, row in buys.iterrows():
            d = row["Date"]
            y = price_map.get(d)
            if y is not None:
                ax.scatter(pd.Timestamp(d), y, marker="x", color="blue", s=120, zorder=5)

        # Legend proxies
        ax.scatter([], [], marker="x", color="green", s=120, label="Sell Call")
        ax.scatter([], [], marker="x", color="blue", s=120, label="Buy Call")

    ax.set_title(f"SPY Covered Call Backtest ({YEAR})", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("SPY Price ($)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(CHART_FILE), dpi=150)
    log.info("Chart saved to %s", CHART_FILE)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("Covered Call Backtest -- %s %d", ROOT, YEAR)
    log.info("=" * 60)

    trade_log, stock_eod = run_backtest(YEAR)

    if trade_log.empty:
        log.warning("No trades were executed. Check ThetaData terminal and data availability.")
        return

    # Save trade log to Excel
    trade_log.to_excel(str(TRADE_LOG_FILE), index=False, sheet_name="Trades")
    log.info("Trade log saved to %s", TRADE_LOG_FILE)

    # Performance metrics
    metrics = compute_metrics(trade_log)
    log.info("-" * 40)
    log.info("Performance Metrics")
    log.info("-" * 40)
    for k, v in metrics.items():
        log.info("  %-35s %s", k, v)

    # Print trade log table
    print("\n" + trade_log.to_string(index=False))

    # Chart
    plot_backtest(trade_log, stock_eod)


if __name__ == "__main__":
    main()

