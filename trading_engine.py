# trading_engine.py
# Live Trading Engine (exports TradingEngine)
# Version: 1.1

import time
import threading
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import schedule
import yfinance as yf

# ---- Import platform building blocks ----
try:
    from algo_trading_main import (
        TradingConfig, DatabaseManager, TechnicalIndicators,
        OptionsCalculator, Position, STRATEGY_REGISTRY,
        export_trades_to_csv, logger
    )
except Exception as e:
    print("Error: Please ensure algo_trading_main.py is available:", e)
    raise

# ---- NSE client for real option LTPs (never from OI) ----
try:
    from nse_client import NseClient
except Exception as e:
    NseClient = None
    logger.warning("nse_client.py not available, will fallback to BS pricing only: %s", e)


IST = pytz.timezone("Asia/Kolkata")


def _ensure_aware_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Return a timezone-aware datetime in IST for comparisons."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return IST.localize(dt)
    return dt.astimezone(IST)


class MarketDataFeed:
    """Real-time market data feed for the underlying index using yfinance."""
    def __init__(self, symbol: str = "^NSEI"):
        self.symbol = symbol
        self.current_price: float = 0.0
        self.is_running: bool = False
        self.data_queue: "queue.Queue[Dict]" = queue.Queue()
        self.historical_prices: List[float] = []
        self.last_update: Optional[datetime] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Market data feed started")

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Market data feed stopped")

    def _loop(self):
        while self.is_running:
            try:
                if not self._is_market_open():
                    time.sleep(60)
                    continue

                ticker = yf.Ticker(self.symbol)
                data = ticker.history(period="1d", interval="1m")
                if not data.empty:
                    latest = data.iloc[-1]
                    market_data = {
                        "timestamp": datetime.now(IST),
                        "symbol": self.symbol,
                        "open": float(latest["Open"]),
                        "high": float(latest["High"]),
                        "low": float(latest["Low"]),
                        "close": float(latest["Close"]),
                        "volume": int(latest.get("Volume", 0)),
                    }
                    self.current_price = market_data["close"]
                    self.historical_prices.append(self.current_price)
                    if len(self.historical_prices) > 300:
                        self.historical_prices = self.historical_prices[-300:]

                    self.data_queue.put(market_data)
                    self.last_update = datetime.now(IST)
                    logger.debug("📊 %s @ ₹%.2f", self.symbol, self.current_price)
                time.sleep(30)

            except Exception as e:
                logger.error("Error fetching market data: %s", e)
                time.sleep(60)

    @staticmethod
    def _is_market_open() -> bool:
        now = datetime.now(IST)
        current_time = now.time()
        market_open = datetime.strptime(TradingConfig.MARKET_OPEN, "%H:%M").time()
        market_close = datetime.strptime(TradingConfig.MARKET_CLOSE, "%H:%M").time()
        is_weekday = now.weekday() < 5
        return is_weekday and (market_open <= current_time <= market_close)

    def get_latest(self) -> Optional[Dict]:
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def last_update_age(self) -> Optional[float]:
        last_update = _ensure_aware_ist(self.last_update)
        if not last_update:
            return None
        now = datetime.now(IST)
        try:
            return (now - last_update).total_seconds()
        except Exception as e:
            logger.debug("Market data age calculation failed: %s", e)
            return None

    def prices(self) -> List[float]:
        return self.historical_prices[-300:].copy()


class TradingEngine:
    """Main engine that orchestrates feed, signals, risk, and persistence."""
    def __init__(self):
        self.db = DatabaseManager()
        self.feed = MarketDataFeed()
        self.positions: Dict[str, Position] = {}
        TradingConfig.reload()
        self.strategies = self._instantiate_strategies()

        self.is_running = False
        self.current_capital = TradingConfig.TOTAL_CAPITAL
        self.daily_trade_count = 0
        self.last_trade_time: Optional[datetime] = None
        self.current_day = None

        # Performance
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_capital = self.current_capital

        # NSE client
        self.nse = NseClient("NIFTY") if NseClient else None

        self._trading_thread: Optional[threading.Thread] = None
        self._scheduler_thread: Optional[threading.Thread] = None

    # ---------- Lifecycle ----------
    def start(self):
        if self.is_running:
            return
        logger.info(
            f"Starting Trading Engine | Capital ₹{self.current_capital:,.2f} | "
            f"Risk/trade {TradingConfig.RISK_PER_TRADE * 100:.1f}%"
        )

        self.refresh_config()
        self.is_running = True
        self.feed.start()

        self._trading_thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._trading_thread.start()

        # schedule daily jobs
        schedule.clear()
        schedule.every().day.at("09:00").do(self._daily_reset)
        schedule.every().day.at("15:45").do(self._end_of_day_cleanup)

        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()

        logger.info("✅ Trading engine started")

    def stop(self):
        if not self.is_running:
            return
        logger.info("🛑 Stopping trading engine…")
        self.is_running = False
        try:
            self._close_all_positions("SYSTEM_SHUTDOWN")
        finally:
            self.feed.stop()
            csv_file = export_trades_to_csv(self.db)
            if csv_file:
                logger.info("📁 Trades exported: %s", csv_file)
        logger.info("✅ Trading engine stopped")

    def refresh_config(self):
        """Reload config values and refresh strategy parameters at runtime."""
        TradingConfig.reload()
        self.strategies = self._instantiate_strategies()
        for pos in self.positions.values():
            pos.fixed_risk_rupees = TradingConfig.FIXED_RISK_RUPEES
            pos.target_trigger_rupees = TradingConfig.TARGET_TRIGGER_RUPEES
        if not self.is_running:
            self.current_capital = TradingConfig.TOTAL_CAPITAL
            self.peak_capital = self.current_capital
        logger.info("🔄 Trading engine configuration refreshed")

    def _instantiate_strategies(self) -> List:
        strategies = []
        for name in TradingConfig.ACTIVE_STRATEGIES:
            strat_cls = STRATEGY_REGISTRY.get(name)
            if not strat_cls:
                logger.warning("Unknown strategy in config: %s", name)
                continue
            strat = strat_cls()
            if hasattr(strat, "refresh_from_config"):
                try:
                    strat.refresh_from_config()
                except Exception as exc:
                    logger.debug("Strategy refresh failed (%s): %s", name, exc)
            strategies.append(strat)

        if not strategies:
            # Fallback to ensure engine has at least one strategy active
            default_name, default_cls = next(iter(STRATEGY_REGISTRY.items()))
            logger.warning("No valid strategies active. Falling back to %s", default_name)
            strategies.append(default_cls())

        return strategies

    # ---------- Core loop ----------
    def _trading_loop(self):
        while self.is_running:
            try:
                md = self.feed.get_latest()
                if not md:
                    time.sleep(1)
                    continue

                prices = self.feed.prices()
                tech = self._calc_indicators(prices, md)

                self._update_positions(md, tech)
                self._check_exit_signals(md, tech)

                if self._can_enter():
                    self._check_entry_signals(md, tech, prices)

                self._update_metrics()
                time.sleep(3)

            except Exception as e:
                logger.error("Error in trading loop: %s", e)
                time.sleep(8)

    # ---------- Indicators ----------
    def _calc_indicators(self, prices: List[float], md: Dict) -> Dict:
        if len(prices) < max(TradingConfig.RSI_PERIOD, TradingConfig.BB_PERIOD) + 1:
            return {
                "rsi": 50.0,
                "bb_upper": md["close"],
                "bb_middle": md["close"],
                "bb_lower": md["close"],
                "volatility": 0.20,
                "vol_regime": "normal",
            }
        rsi = TechnicalIndicators.calculate_rsi(prices, TradingConfig.RSI_PERIOD)
        bb_u, bb_m, bb_l = TechnicalIndicators.calculate_bollinger_bands(
            prices, TradingConfig.BB_PERIOD, TradingConfig.BB_STD
        )
        vol = TechnicalIndicators.calculate_volatility(prices, TradingConfig.VOL_LOOKBACK)

        # rolling vol mean
        chunks = []
        step = TradingConfig.VOL_LOOKBACK
        start = max(0, len(prices) - 3 * step)
        for i in range(start, len(prices), step):
            ch = TechnicalIndicators.calculate_volatility(prices[i:i + step], TradingConfig.VOL_LOOKBACK)
            if ch:
                chunks.append(ch)
        vol_ma = np.mean(chunks) if chunks else vol

        if vol > vol_ma * TradingConfig.VOL_THRESHOLD:
            regime = "high"
        elif vol < vol_ma * 0.8:
            regime = "low"
        else:
            regime = "normal"

        return {
            "rsi": rsi, "bb_upper": bb_u, "bb_middle": bb_m, "bb_lower": bb_l,
            "volatility": vol, "vol_regime": regime, "vol_ma": vol_ma
        }

    # ---------- Positions ----------
    def _live_option_ltp(self, spot: float, strike: int, is_call: bool, days_to_expiry: int) -> float:
        """
        Try NSE lastPrice first (never OI). Fallback to BS if NSE unavailable.
        """
        try:
            if self.nse:
                expiry_iso = self._nearest_expiry_iso()
                if expiry_iso:
                    ltp = self.nse.get_option_ltp(strike=int(strike), is_call=is_call, expiry_iso=expiry_iso)
                    if ltp and ltp > 0:
                        return float(ltp)
        except Exception as e:
            logger.debug("NSE LTP fetch failed, falling back to BS: %s", e)

        vol = 0.20
        return float(OptionsCalculator.get_realistic_premium(
            spot, strike, is_call, max(1, days_to_expiry), vol
        ))

    def _update_positions(self, md: Dict, tech: Dict):
        spot = md["close"]
        to_close = []
        for pid, pos in list(self.positions.items()):
            dte = self._days_to_expiry_from_iso(pos.expiry_date) if getattr(pos, "expiry_date", "") else 1
            ltp = self._live_option_ltp(spot, int(pos.strike), pos.option_type == "CE", dte)
            pos.update_price(ltp, spot)

    def _check_exit_signals(self, md: Dict, tech: Dict):
        regime = tech.get("vol_regime", "normal")
        to_close: List[tuple] = []
        for pid, pos in self.positions.items():
            reason = pos.should_exit(md["close"], regime)
            if reason:
                to_close.append((pid, reason))
        for pid, reason in to_close:
            self._close_position(pid, reason, md)

    def _check_entry_signals(self, md: Dict, tech: Dict, prices: List[float]):
        for strat in self.strategies:
            try:
                if strat.__class__.__name__ in ("MomentumBreakoutStrategy", "VolatilityRegimeStrategy"):
                    sig = strat.generate_signal(md, tech, prices)
                else:
                    sig = strat.generate_signal(md, tech)
                if sig and sig.get("confidence", 0) >= TradingConfig.ENTRY_CONFIDENCE_MIN:
                    self._enter(sig, md, tech, strat.name)
                    break
            except Exception as e:
                logger.error("Signal error (%s): %s", strat.name, e)

    def _enter(self, signal: Dict, md: Dict, tech: Dict, strat_name: str):
        spot = md["close"]
        is_call = signal["option_type"] == "CE"
        base_strike = round(spot / 50) * 50  # NIFTY strikes 50-wide
        strike = int(base_strike + (signal.get("strike_adjustment", 0) if is_call
                                    else -signal.get("strike_adjustment", 0)))

        entry_time = datetime.now(IST)
        expiry_iso = self._nearest_expiry_iso() or (entry_time.date() + timedelta(days=self._days_to_next_tuesday())).isoformat()
        dte = self._days_to_expiry_from_iso(expiry_iso)

        # price from NSE (fallback to BS)
        entry_price = self._live_option_ltp(spot, strike, is_call, dte)

        # position sizing
        max_risk = self.current_capital * TradingConfig.RISK_PER_TRADE
        risk_per_lot = max(1.0, entry_price * TradingConfig.LOT_SIZE * TradingConfig.BASE_SL_PCT)

        lots = max(1, min(int(max_risk / risk_per_lot), 5))

        required_capital = entry_price * TradingConfig.LOT_SIZE * lots
        if required_capital > self.current_capital:
            logger.warning(
                "🚫 Entry rejected (insufficient cash) | %s needs ₹%.2f but only ₹%.2f available",
                strat_name, required_capital, self.current_capital
            )
            return

        capital_cap = self.current_capital * TradingConfig.PREMIUM_CAPITAL_FRACTION
        if required_capital > capital_cap:
            logger.warning(
                "🚫 Entry rejected (premium %.0f%% of cash > %.0f%% cap) | %s",
                (required_capital / max(self.current_capital, 1)) * 100,
                TradingConfig.PREMIUM_CAPITAL_FRACTION * 100,
                strat_name
            )
            return

        symbol = f"NIFTY{strike}{'CE' if is_call else 'PE'}"
        pid = f"{symbol}_{entry_time.strftime('%Y%m%d_%H%M%S')}"

        lots = max(1, min(int(max_risk / risk_per_lot), 5))

        premium_cap_pct = getattr(TradingConfig, "MAX_PREMIUM_ALLOCATION_PCT", 0.10)
        lot_cost = entry_price * TradingConfig.LOT_SIZE
        if lot_cost <= 0:
            logger.warning("🚫 Entry rejected (invalid lot cost): %s", strat_name)
            return

        max_lots_capital = int((self.current_capital * premium_cap_pct) / lot_cost)
        if max_lots_capital <= 0:
            logger.warning("🚫 Entry rejected (capital limit): %s", strat_name)
            return

        lots = min(lots, max_lots_capital)
        if lots <= 0:
            logger.warning("🚫 Entry rejected (zero lots after cap): %s", strat_name)
            return

        required_capital = lot_cost * lots

        symbol = f"NIFTY{strike}{'CE' if is_call else 'PE'}"
        pid = f"{symbol}_{entry_time.strftime('%Y%m%d_%H%M%S')}"


        pos = Position(
            symbol=symbol,
            option_type=signal["option_type"],
            strike=strike,
            entry_price=entry_price,
            qty_lots=lots,
            entry_time=entry_time,
            strategy=strat_name,
            entry_spot=spot,
            expiry_date=expiry_iso,
        )
        self.positions[pid] = pos

        self.daily_trade_count += 1
        self.last_trade_time = entry_time
        self.total_trades += 1
        self.current_capital -= required_capital

        logger.info("🟢 ENTRY %s @ ₹%.2f | Lots:%d | Spot ₹%.2f | %s | Exp:%s",
                    symbol, entry_price, lots, spot, signal.get("reason"), expiry_iso)

    def _close_position(self, pid: str, reason: str, md: Dict):
        if pid not in self.positions:
            return
        pos = self.positions[pid]
        exit_time = datetime.now(IST)
        exit_price = pos.current_price

        gross = (exit_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots
        net = gross - TradingConfig.FEES_PER_ORDER

        self.current_capital += (exit_price * TradingConfig.LOT_SIZE * pos.qty_lots)
        self.total_pnl += net
        if net > 0:
            self.winning_trades += 1

        # drawdown stats
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        dd = (self.peak_capital - self.current_capital) / max(self.peak_capital, 1)
        self.max_drawdown = max(self.max_drawdown, dd)

        # persist trade
        trade = {
            "entry_timestamp": pos.entry_time.isoformat(),
            "exit_timestamp": exit_time.isoformat(),
            "symbol": pos.symbol,
            "strategy": pos.strategy,
            "option_type": pos.option_type,
            "strike_price": pos.strike,
            "qty_lots": pos.qty_lots,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "pnl": net,
            "holding_minutes": pos.get_holding_minutes(),
            "entry_spot": pos.entry_spot,
            "exit_spot": md["close"],
            "volatility": 0.20,
            "vol_regime": "normal",
            "reason_entry": "STRATEGY_SIGNAL",
            "reason_exit": reason,
            "max_profit": (pos.peak_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots,
            "max_loss": (pos.max_loss_price - pos.entry_price) * TradingConfig.LOT_SIZE * pos.qty_lots,
        }
        try:
            self.db.insert_trade(trade)
        except Exception as e:
            logger.warning("DB insert failed: %s", e)

        del self.positions[pid]
        logger.info(f"🔴 EXIT: {pos.symbol} @ ₹{exit_price:.2f} | P&L: ₹{net:.2f} | "
           f"Reason: {reason} | Hold: {pos.get_holding_minutes():.0f}min | "
           f"Capital: ₹{self.current_capital:,.2f}")

    def _close_all_positions(self, reason: str):
        for pid in list(self.positions.keys()):
            md = {"close": self.feed.current_price}
            self._close_position(pid, reason, md)

    # ---------- Risk gates ----------
    def _can_enter(self) -> bool:
        if self.daily_trade_count >= TradingConfig.MAX_TRADES_PER_DAY:
            return False
        if len(self.positions) >= TradingConfig.MAX_OPEN_POSITIONS:
            return False
        now = datetime.now(IST)
        last_trade_time = _ensure_aware_ist(self.last_trade_time)
        if last_trade_time is not None:
            self.last_trade_time = last_trade_time
            if (now - last_trade_time).total_seconds() < TradingConfig.COOLDOWN_MINUTES * 60:
                return False
        if not self.feed._is_market_open():
            return False
        data_age = self.feed.last_update_age()
        if data_age is None:
            logger.warning("🚫 Entry blocked: awaiting live market data")
            return False
        if data_age > TradingConfig.DATA_STALENESS_SECONDS:
            logger.warning("🚫 Entry blocked: market data stale (%.1fs old)", data_age)
            return False
        if self.current_capital < TradingConfig.TOTAL_CAPITAL * 0.5:
            logger.warning("🚫 Trading halted: capital below 50%% of initial")
            return False
        return True

    # ---------- Scheduler ----------
    def _daily_reset(self):
        logger.info("🔄 Daily reset")
        self.daily_trade_count = 0
        self.current_day = datetime.now(IST).date()

    def _end_of_day_cleanup(self):
        logger.info("🌅 EOD cleanup")
        self._close_all_positions("END_OF_DAY")
        csv = export_trades_to_csv(self.db, f"daily_trades_{datetime.now(IST).strftime('%Y%m%d')}.csv")
        if csv:
            logger.info("📁 Daily trades exported: %s", csv)

    def _run_scheduler(self):
        while self.is_running:
            try:
                schedule.run_pending()
            except Exception as e:
                logger.debug("Scheduler error: %s", e)
            time.sleep(60)

    # ---------- Metrics / Status ----------
    def _update_metrics(self):
        # placeholder for real-time analytics if needed
        pass


    def get_status(self) -> Dict:
        try:
            unreal = sum(p.unrealized_pnl for p in self.positions.values())
            positions_data = []
            for p in self.positions.values():
                try:
                    positions_data.append(p.to_dict())
                except Exception as e:
                    logger.debug("Position to_dict failed: %s", e)

            last_update = _ensure_aware_ist(self.feed.last_update)
            data_age = self.feed.last_update_age()
            market_data_stale = data_age is None or data_age > TradingConfig.DATA_STALENESS_SECONDS

    def get_status(self) -> Dict:
        try:
            unreal = sum(p.unrealized_pnl for p in self.positions.values())
            positions_data = []
            for p in self.positions.values():
                try:
                    positions_data.append(p.to_dict())
                except Exception as e:
                    logger.debug("Position to_dict failed: %s", e)

            return {
                "is_running": self.is_running,
                "current_capital": self.current_capital,
                "total_pnl": self.total_pnl,

                "unrealized_pnl": unreal,
                "total_trades": self.total_trades,

                "unrealized_pnl": unreal,
                "total_trades": self.total_trades,

                "winning_trades": self.winning_trades,
                "win_rate": (self.winning_trades / max(self.total_trades, 1)) * 100,
                "daily_trade_count": self.daily_trade_count,
                "open_positions": len(self.positions),
                "max_drawdown": self.max_drawdown * 100,
                "current_price": self.feed.current_price,

                "last_update": last_update.isoformat() if last_update else None,
                "market_data_age_seconds": data_age,
                "market_data_stale": market_data_stale,
                "positions": positions_data,
                "strategy_stats": [s.get_statistics() for s in self.strategies],

                "last_update": self.feed.last_update.isoformat() if self.feed.last_update else None,
                "positions": positions_data,
                "strategy_stats": [s.get_statistics() for s in self.strategies],
                "active_strategies": TradingConfig.ACTIVE_STRATEGIES,
                "available_strategies": list(STRATEGY_REGISTRY.keys()),

            }
        except Exception as e:
            logger.error("get_status failed: %s", e)
            return {
                "is_running": self.is_running,
                "current_capital": self.current_capital,
                "total_pnl": 0.0,
                "total_trades": 0,
                "open_positions": 0,
                "error": str(e),
            }

    # ---------- Expiry helpers ----------
    def _nearest_expiry_iso(self) -> Optional[str]:
        """
        Prefer NSE-provided nearest non-past expiry.
        This naturally reflects Tuesday weeklies for 2025.
        """
        try:
            if self.nse:
                return self.nse.get_nearest_expiry_iso()
        except Exception as e:
            logger.debug("Nearest expiry fetch failed: %s", e)
        # fallback to next Tuesday
        return (datetime.now(IST).date() + timedelta(days=self._days_to_next_tuesday())).isoformat()

    @staticmethod
    def _days_to_expiry_from_iso(expiry_iso: str) -> int:
        try:
            exp_date = datetime.strptime(expiry_iso, "%Y-%m-%d").date()
            today = datetime.now(IST).date()
            d = (exp_date - today).days
            return max(1, d)
        except Exception:
            return 1

    @staticmethod
    def _days_to_next_tuesday() -> int:
        d = datetime.now(IST).date()
        days_ahead = (1 - d.weekday()) % 7  # Tue = 1
        return 7 if days_ahead == 0 else days_ahead


# no top-level run; importing this module is side-effect free
