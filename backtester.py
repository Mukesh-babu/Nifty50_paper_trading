"""Historical backtesting utilities for the NIFTY50 options strategies."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

from algo_trading_main import (
    TradingConfig,
    TechnicalIndicators,
    OptionsCalculator,
    Position,
    STRATEGY_REGISTRY,
)

IST = pytz.timezone("Asia/Kolkata")

class Backtester:
    """Offline simulator for strategy evaluation on historical data."""

    def __init__(
        self,
        symbol: Optional[str] = None,
        interval: Optional[str] = None,
        lookback_days: Optional[int] = None,
        strategies: Optional[List[str]] = None,
        auto_apply_best: bool = True,
    ) -> None:
        TradingConfig.reload()
        defaults = TradingConfig.get_backtest_defaults()
        self.symbol = symbol or defaults.get("symbol", "^NSEI")
        self.interval = interval or defaults.get("interval", "1m")
        self.lookback_days = lookback_days or defaults.get("lookback_days", 5)
        self.auto_apply_best = auto_apply_best
        self.strategy_names = strategies or TradingConfig.ACTIVE_STRATEGIES or list(STRATEGY_REGISTRY.keys())
        self.notes: List[str] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _interval_cap(self) -> int:
        if self.interval == "1m":
            max_days = int(TradingConfig.BACKTEST.get("max_lookback_1m", 7))
            return max(1, max_days)
        return self.lookback_days

    def _date_window(self) -> tuple[datetime, datetime]:
        cap = self._interval_cap()
        actual = min(self.lookback_days, cap)
        if self.lookback_days > cap:
            self.notes.append(
                f"Lookback truncated from {self.lookback_days}d to {cap}d for {self.interval} data limitations."
            )
        end_dt = datetime.now(IST)
        start_dt = end_dt - timedelta(days=actual)
        return start_dt, end_dt

    def _fetch_data(self) -> pd.DataFrame:
        start_dt, end_dt = self._date_window()
        fetch_end = end_dt + timedelta(days=1)
        data = yf.download(
            self.symbol,
            start=start_dt.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            interval=self.interval,
            progress=False,
            auto_adjust=False,
        )
        if data.empty:
            raise ValueError("No historical data returned for the requested window")

        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        if data.index.tz is None:
            data.index = data.index.tz_localize("UTC").tz_convert(IST)
        else:
            data.index = data.index.tz_convert(IST)
        data = data.sort_index()
        data = data.dropna()
        if data.empty:
            raise ValueError("Historical data contained only NaNs after cleaning")
        return data

    @staticmethod
    def _nearest_expiry_iso(current: datetime) -> str:
        days_ahead = (1 - current.weekday()) % 7  # Tuesday expiry (weekday=1)
        if days_ahead == 0:
            days_ahead = 7
        expiry_date = current.date() + timedelta(days=days_ahead)
        return expiry_date.isoformat()

    @staticmethod
    def _days_to_expiry(expiry_iso: str, now_dt: datetime) -> int:
        try:
            exp_date = datetime.strptime(expiry_iso, "%Y-%m-%d").date()
            delta = (exp_date - now_dt.date()).days
            return max(1, delta)
        except Exception:
            return 1

    def _calc_indicators(self, prices: List[float], current_close: float) -> Dict:
        if len(prices) < max(TradingConfig.RSI_PERIOD, TradingConfig.BB_PERIOD) + 1:
            return {
                "rsi": 50.0,
                "bb_upper": current_close,
                "bb_middle": current_close,
                "bb_lower": current_close,
                "volatility": 0.20,
                "vol_regime": "normal",
            }

        rsi = TechnicalIndicators.calculate_rsi(prices, TradingConfig.RSI_PERIOD)
        bb_u, bb_m, bb_l = TechnicalIndicators.calculate_bollinger_bands(
            prices, TradingConfig.BB_PERIOD, TradingConfig.BB_STD
        )
        vol = TechnicalIndicators.calculate_volatility(prices, TradingConfig.VOL_LOOKBACK)

        segments = []
        step = TradingConfig.VOL_LOOKBACK
        start = max(0, len(prices) - 3 * step)
        for i in range(start, len(prices), step):
            seg = TechnicalIndicators.calculate_volatility(prices[i:i + step], TradingConfig.VOL_LOOKBACK)
            if seg:
                segments.append(seg)
        vol_ma = np.mean(segments) if segments else vol
        if vol > vol_ma * TradingConfig.VOL_THRESHOLD:
            regime = "high"
        elif vol < vol_ma * 0.8:
            regime = "low"
        else:
            regime = "normal"

        return {
            "rsi": rsi,
            "bb_upper": bb_u,
            "bb_middle": bb_m,
            "bb_lower": bb_l,
            "volatility": vol,
            "vol_ma": vol_ma,
            "vol_regime": regime,
        }

    # ------------------------------------------------------------------
    def _run_strategy(self, strategy_name: str, data: pd.DataFrame) -> Dict:
        strat_cls = STRATEGY_REGISTRY.get(strategy_name)
        if not strat_cls:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = strat_cls()
        if hasattr(strategy, "refresh_from_config"):
            strategy.refresh_from_config()

        cash = TradingConfig.TOTAL_CAPITAL
        realized_pnl = 0.0
        open_positions: Dict[str, Position] = {}
        daily_trades: Dict[str, int] = {}
        last_trade_time: Optional[datetime] = None
        equity_curve: List[Dict] = []
        trades: List[Dict] = []
        pnl_series: List[float] = []
        close_history: List[float] = []
        peak_equity = cash
        max_drawdown = 0.0
        wins = 0

        for ts, row in data.iterrows():
            close_history.append(float(row["Close"]))
            market_data = {
                "timestamp": ts,
                "symbol": self.symbol,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row.get("Volume", 0) or 0),
            }
            tech = self._calc_indicators(close_history, market_data["close"])

            # Update open positions
            to_close: List[tuple[str, str, float]] = []
            for pid, pos in open_positions.items():
                dte = self._days_to_expiry(pos.expiry_date, ts)
                option_price = OptionsCalculator.get_realistic_premium(
                    market_data["close"],
                    pos.strike,
                    pos.option_type == "CE",
                    dte,
                    max(0.10, tech.get("volatility", 0.20)),
                )
                pos.update_price(option_price, market_data["close"])
                reason = pos.should_exit(market_data["close"], tech.get("vol_regime", "normal"))
                if reason:
                    to_close.append((pid, reason, option_price))

            for pid, reason, option_price in to_close:
                pos = open_positions.pop(pid)
                exit_time = ts.to_pydatetime()
                gross = (option_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots
                net = gross - TradingConfig.FEES_PER_ORDER
                realized_pnl += net
                cash += option_price * TradingConfig.LOT_SIZE * pos.qty_lots
                pnl_series.append(net)
                if net > 0:
                    wins += 1
                trade = {
                    "entry_timestamp": pos.entry_time.isoformat(),
                    "exit_timestamp": exit_time.isoformat(),
                    "symbol": pos.symbol,
                    "strategy": strategy_name,
                    "option_type": pos.option_type,
                    "strike_price": pos.strike,
                    "qty_lots": pos.qty_lots,
                    "entry_price": pos.entry_price,
                    "exit_price": option_price,
                    "pnl": net,
                    "reason_exit": reason,
                    "holding_minutes": (exit_time - pos.entry_time).total_seconds() / 60.0,
                    "entry_spot": pos.entry_spot,
                    "exit_spot": market_data["close"],
                    "max_profit": (pos.peak_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots,
                    "max_loss": (pos.max_loss_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots,
                }
                trades.append(trade)

            # Determine new entries
            day_key = ts.strftime("%Y-%m-%d")
            daily_trades.setdefault(day_key, 0)
            can_enter = (
                len(open_positions) < TradingConfig.MAX_OPEN_POSITIONS
                and daily_trades[day_key] < TradingConfig.MAX_TRADES_PER_DAY
            )
            if can_enter and last_trade_time is not None:
                if (ts - last_trade_time).total_seconds() < TradingConfig.COOLDOWN_MINUTES * 60:
                    can_enter = False

            if can_enter:
                if hasattr(strategy, "generate_signal"):
                    if strategy_name in ("Momentum Breakout", "Volatility Regime", "Adaptive Trend"):
                        signal = strategy.generate_signal(market_data, tech, close_history)
                    else:
                        signal = strategy.generate_signal(market_data, tech)
                else:
                    signal = None

                if signal and signal.get("confidence", 0) >= TradingConfig.ENTRY_CONFIDENCE_MIN:
                    is_call = signal["option_type"] == "CE"
                    base_strike = round(market_data["close"] / 50) * 50
                    adjustment = signal.get("strike_adjustment", 0)
                    strike = int(base_strike + adjustment if is_call else base_strike - adjustment)
                    expiry_iso = self._nearest_expiry_iso(ts.to_pydatetime())
                    dte = self._days_to_expiry(expiry_iso, ts.to_pydatetime())
                    entry_price = OptionsCalculator.get_realistic_premium(
                        market_data["close"],
                        strike,
                        is_call,
                        dte,
                        max(0.10, tech.get("volatility", 0.20)),
                    )
                    max_risk = cash * TradingConfig.RISK_PER_TRADE
                    risk_per_lot = max(1.0, entry_price * TradingConfig.LOT_SIZE * TradingConfig.BASE_SL_PCT)
                    lots = max(1, min(int(max_risk / risk_per_lot), TradingConfig.MAX_LOTS_CAP))
                    required_capital = entry_price * TradingConfig.LOT_SIZE * lots
                    if required_capital <= cash * 0.10 and lots > 0:
                        symbol = f"NIFTY{strike}{'CE' if is_call else 'PE'}"
                        pid = f"{symbol}_{ts.strftime('%Y%m%d_%H%M%S')}"
                        pos = Position(
                            symbol=symbol,
                            option_type=signal["option_type"],
                            strike=strike,
                            entry_price=entry_price,
                            qty_lots=lots,
                            entry_time=ts.to_pydatetime(),
                            strategy=strategy_name,
                            entry_spot=market_data["close"],
                            expiry_date=expiry_iso,
                        )
                        open_positions[pid] = pos
                        cash -= required_capital
                        daily_trades[day_key] += 1
                        last_trade_time = ts

            open_value = sum(
                pos.current_price * TradingConfig.LOT_SIZE * pos.qty_lots for pos in open_positions.values()
            )
            equity = cash + open_value
            peak_equity = max(peak_equity, equity)
            if peak_equity > 0:
                dd = (peak_equity - equity) / peak_equity
                max_drawdown = max(max_drawdown, dd)
            equity_curve.append({"timestamp": ts.isoformat(), "equity": equity})

        # Force close remaining positions at last seen price
        if open_positions:
            final_ts = data.index[-1].to_pydatetime()
            last_close = float(data.iloc[-1]["Close"])
            for pid, pos in list(open_positions.items()):
                option_price = OptionsCalculator.get_realistic_premium(
                    last_close,
                    pos.strike,
                    pos.option_type == "CE",
                    1,
                    0.20,
                )
                gross = (option_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots
                net = gross - TradingConfig.FEES_PER_ORDER
                realized_pnl += net
                cash += option_price * TradingConfig.LOT_SIZE * pos.qty_lots
                pnl_series.append(net)
                if net > 0:
                    wins += 1
                trades.append({
                    "entry_timestamp": pos.entry_time.isoformat(),
                    "exit_timestamp": final_ts.isoformat(),
                    "symbol": pos.symbol,
                    "strategy": strategy_name,
                    "option_type": pos.option_type,
                    "strike_price": pos.strike,
                    "qty_lots": pos.qty_lots,
                    "entry_price": pos.entry_price,
                    "exit_price": option_price,
                    "pnl": net,
                    "reason_exit": "FORCED_EXIT_END_OF_DATA",
                    "holding_minutes": (final_ts - pos.entry_time).total_seconds() / 60.0,
                    "entry_spot": pos.entry_spot,
                    "exit_spot": last_close,
                    "max_profit": (pos.peak_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots,
                    "max_loss": (pos.max_loss_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots,
                })
            open_positions.clear()
            equity_curve.append({"timestamp": final_ts.isoformat(), "equity": cash})

        total_trades = len(trades)
        total_pnl = realized_pnl
        avg_pnl = total_pnl / total_trades if total_trades else 0.0
        win_rate = (wins / total_trades) * 100 if total_trades else 0.0
        sharpe = 0.0
        if len(pnl_series) > 1 and np.std(pnl_series) != 0:
            sharpe = (np.mean(pnl_series) / np.std(pnl_series)) * np.sqrt(len(pnl_series))

        return {
            "name": strategy_name,
            "total_trades": total_trades,
            "winning_trades": wins,
            "losing_trades": total_trades - wins,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_trade_pnl": avg_pnl,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe": sharpe,
            "equity_curve": equity_curve,
            "trades": trades,
            "final_capital": cash,
        }

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        data = self._fetch_data()
        results = []
        best: Optional[Dict] = None
        for name in self.strategy_names:
            try:
                res = self._run_strategy(name, data)
                results.append(res)
                if not best or res["total_pnl"] > best["total_pnl"]:
                    best = res
            except Exception as exc:
                self.notes.append(f"Strategy {name} failed during backtest: {exc}")

        if best and self.auto_apply_best:
            TradingConfig.set_active_strategies([best["name"]])

        payload = {
            "symbol": self.symbol,
            "interval": self.interval,
            "lookback_days": self.lookback_days,
            "data_points": int(len(data)),
            "start": data.index[0].isoformat(),
            "end": data.index[-1].isoformat(),
            "strategies": results,
            "best_strategy": best,
            "notes": self.notes,
        }
        return payload


def run_backtest(symbol: Optional[str] = None, interval: Optional[str] = None, lookback_days: Optional[int] = None) -> Dict:
    """Convenience wrapper for scripts and CLI usage."""
    tester = Backtester(symbol=symbol, interval=interval, lookback_days=lookback_days)
    return tester.run()
