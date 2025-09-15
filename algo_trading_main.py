# AI_ALGO_TRADING_PLATFORM - Main Trading Engine
# Advanced Algorithmic Trading System with Paper Trading & Live Dashboard
# Author: AI Trading Systems
# Version: 1.2

import os
import sys
import json
import time
import threading
import logging
import sqlite3
from copy import deepcopy
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
import pytz
from flask import Flask, render_template, jsonify, request
from scipy.stats import norm
import math
import random
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('algo_trading.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        handler.stream = sys.stdout

from config.manager import DEFAULT_CONFIG, ensure_config, load_config, update_config as _persist_config


# -------------------------------------------------
# Config
# -------------------------------------------------
class TradingConfig:
    """Centralized configuration management loaded from ``config/config.json``."""

    # Defaults (overwritten by reload)
    TOTAL_CAPITAL = DEFAULT_CONFIG["trading"]["total_capital"]
    RISK_PER_TRADE = DEFAULT_CONFIG["trading"]["risk_per_trade"]
    LOT_SIZE = DEFAULT_CONFIG["trading"]["lot_size"]
    FEES_PER_ORDER = DEFAULT_CONFIG["trading"]["fees_per_order"]

    BASE_SL_PCT = DEFAULT_CONFIG["trading"]["base_sl_pct"]
    BASE_TP_PCT = DEFAULT_CONFIG["trading"]["base_tp_pct"]
    PARTIAL_TP_PCT = DEFAULT_CONFIG["trading"]["partial_tp_pct"]
    TRAIL_PCT = DEFAULT_CONFIG["trading"]["trail_pct"]

    ENTRY_CONFIDENCE_MIN = DEFAULT_CONFIG["trading"]["entry_confidence_min"]
    MAX_LOTS_CAP = DEFAULT_CONFIG["trading"]["max_lots_cap"]
    MAX_PREMIUM_ALLOCATION_PCT = DEFAULT_CONFIG["trading"].get("max_premium_allocation_pct", 0.10)
    FIXED_RISK_RUPEES = DEFAULT_CONFIG["trading"]["fixed_risk_rupees"]
    TARGET_TRIGGER_RUPEES = DEFAULT_CONFIG["trading"]["target_trigger_rupees"]
    TRAIL_AFTER_TRIGGER = DEFAULT_CONFIG["trading"]["trail_after_trigger"]

    MAX_TRADES_PER_DAY = DEFAULT_CONFIG["trading"]["max_trades_per_day"]
    MAX_OPEN_POSITIONS = DEFAULT_CONFIG["trading"]["max_open_positions"]
    COOLDOWN_MINUTES = DEFAULT_CONFIG["trading"]["cooldown_minutes"]

    MARKET_OPEN = DEFAULT_CONFIG["trading"]["market_open"]
    MARKET_CLOSE = DEFAULT_CONFIG["trading"]["market_close"]
    EOD_EXIT = DEFAULT_CONFIG["trading"]["eod_exit"]

    RSI_PERIOD = DEFAULT_CONFIG["indicators"]["rsi_period"]
    RSI_OVERSOLD = DEFAULT_CONFIG["indicators"]["rsi_oversold"]
    RSI_OVERBOUGHT = DEFAULT_CONFIG["indicators"]["rsi_overbought"]
    BB_PERIOD = DEFAULT_CONFIG["indicators"]["bb_period"]
    BB_STD = DEFAULT_CONFIG["indicators"]["bb_std"]
    VOL_LOOKBACK = DEFAULT_CONFIG["indicators"]["vol_lookback"]
    VOL_THRESHOLD = DEFAULT_CONFIG["indicators"]["vol_threshold"]

    MOMENTUM_LOOKBACK = DEFAULT_CONFIG["strategy_params"]["Momentum Breakout"]["lookback_period"]
    MOMENTUM_RANGE_THRESHOLD = DEFAULT_CONFIG["strategy_params"]["Momentum Breakout"]["range_threshold"]
    MOMENTUM_STRIKE_ADJUSTMENT = DEFAULT_CONFIG["strategy_params"]["Momentum Breakout"]["strike_adjustment"]

    VOLATILITY_TRIGGER = DEFAULT_CONFIG["strategy_params"]["Volatility Regime"]["vol_threshold"]
    VOL_REGIME_RSI_BUY = DEFAULT_CONFIG["strategy_params"]["Volatility Regime"]["rsi_buy"]
    VOL_REGIME_RSI_SELL = DEFAULT_CONFIG["strategy_params"]["Volatility Regime"]["rsi_sell"]
    VOL_REGIME_STRIKE_ADJUSTMENT = DEFAULT_CONFIG["strategy_params"]["Volatility Regime"]["strike_adjustment"]

    ADAPTIVE_SHORT_EMA = DEFAULT_CONFIG["strategy_params"]["Adaptive Trend"]["short_ema"]
    ADAPTIVE_LONG_EMA = DEFAULT_CONFIG["strategy_params"]["Adaptive Trend"]["long_ema"]
    ADAPTIVE_RSI_CONFIRM = DEFAULT_CONFIG["strategy_params"]["Adaptive Trend"]["trend_confirmation_rsi"]
    ADAPTIVE_STRIKE_ADJUSTMENT = DEFAULT_CONFIG["strategy_params"]["Adaptive Trend"]["strike_adjustment"]
    ADAPTIVE_VOL_FLOOR = DEFAULT_CONFIG["strategy_params"]["Adaptive Trend"]["volatility_floor"]

    ACTIVE_STRATEGIES: List[str] = list(DEFAULT_CONFIG["trading"]["active_strategies"])
    STRATEGY_PARAMETERS: Dict[str, Dict] = deepcopy(DEFAULT_CONFIG["strategy_params"])
    BACKTEST = deepcopy(DEFAULT_CONFIG["backtest"])

    _raw_config: Dict[str, Dict] = {}
    _lock = threading.RLock()

    @classmethod
    def reload(cls) -> Dict[str, Dict]:
        """Reload configuration from disk and populate class-level attributes."""

        with cls._lock:
            ensure_config()
            config = load_config()
            trading = config.get("trading", {})
            indicators = config.get("indicators", {})
            strategy_params = config.get("strategy_params", {})
            backtest = config.get("backtest", {})

            cls.TOTAL_CAPITAL = float(trading.get("total_capital", cls.TOTAL_CAPITAL))
            cls.RISK_PER_TRADE = float(trading.get("risk_per_trade", cls.RISK_PER_TRADE))
            cls.LOT_SIZE = int(trading.get("lot_size", cls.LOT_SIZE))
            cls.FEES_PER_ORDER = float(trading.get("fees_per_order", cls.FEES_PER_ORDER))
            cls.BASE_SL_PCT = float(trading.get("base_sl_pct", cls.BASE_SL_PCT))
            cls.BASE_TP_PCT = float(trading.get("base_tp_pct", cls.BASE_TP_PCT))
            cls.PARTIAL_TP_PCT = float(trading.get("partial_tp_pct", cls.PARTIAL_TP_PCT))
            cls.TRAIL_PCT = float(trading.get("trail_pct", cls.TRAIL_PCT))
            cls.ENTRY_CONFIDENCE_MIN = int(trading.get("entry_confidence_min", cls.ENTRY_CONFIDENCE_MIN))
            cls.MAX_LOTS_CAP = int(trading.get("max_lots_cap", cls.MAX_LOTS_CAP))
            cls.MAX_PREMIUM_ALLOCATION_PCT = float(
                trading.get("max_premium_allocation_pct", cls.MAX_PREMIUM_ALLOCATION_PCT)
            )
            cls.FIXED_RISK_RUPEES = float(trading.get("fixed_risk_rupees", cls.FIXED_RISK_RUPEES))
            cls.TARGET_TRIGGER_RUPEES = float(trading.get("target_trigger_rupees", cls.TARGET_TRIGGER_RUPEES))
            cls.TRAIL_AFTER_TRIGGER = bool(trading.get("trail_after_trigger", cls.TRAIL_AFTER_TRIGGER))
            cls.MAX_TRADES_PER_DAY = int(trading.get("max_trades_per_day", cls.MAX_TRADES_PER_DAY))
            cls.MAX_OPEN_POSITIONS = int(trading.get("max_open_positions", cls.MAX_OPEN_POSITIONS))
            cls.COOLDOWN_MINUTES = int(trading.get("cooldown_minutes", cls.COOLDOWN_MINUTES))
            cls.MARKET_OPEN = str(trading.get("market_open", cls.MARKET_OPEN))
            cls.MARKET_CLOSE = str(trading.get("market_close", cls.MARKET_CLOSE))
            cls.EOD_EXIT = str(trading.get("eod_exit", cls.EOD_EXIT))

            cls.RSI_PERIOD = int(indicators.get("rsi_period", cls.RSI_PERIOD))
            cls.RSI_OVERSOLD = float(strategy_params.get("Mean Reversion", {}).get(
                "rsi_oversold", indicators.get("rsi_oversold", cls.RSI_OVERSOLD)
            ))
            cls.RSI_OVERBOUGHT = float(strategy_params.get("Mean Reversion", {}).get(
                "rsi_overbought", indicators.get("rsi_overbought", cls.RSI_OVERBOUGHT)
            ))
            cls.BB_PERIOD = int(indicators.get("bb_period", cls.BB_PERIOD))
            cls.BB_STD = float(indicators.get("bb_std", cls.BB_STD))
            cls.VOL_LOOKBACK = int(indicators.get("vol_lookback", cls.VOL_LOOKBACK))
            cls.VOL_THRESHOLD = float(indicators.get("vol_threshold", cls.VOL_THRESHOLD))

            momentum = strategy_params.get("Momentum Breakout", {})
            cls.MOMENTUM_LOOKBACK = int(momentum.get("lookback_period", cls.MOMENTUM_LOOKBACK))
            cls.MOMENTUM_RANGE_THRESHOLD = float(momentum.get("range_threshold", cls.MOMENTUM_RANGE_THRESHOLD))
            cls.MOMENTUM_STRIKE_ADJUSTMENT = int(momentum.get("strike_adjustment", cls.MOMENTUM_STRIKE_ADJUSTMENT))

            vol_params = strategy_params.get("Volatility Regime", {})
            cls.VOLATILITY_TRIGGER = float(vol_params.get("vol_threshold", cls.VOLATILITY_TRIGGER))
            cls.VOL_REGIME_RSI_BUY = float(vol_params.get("rsi_buy", cls.VOL_REGIME_RSI_BUY))
            cls.VOL_REGIME_RSI_SELL = float(vol_params.get("rsi_sell", cls.VOL_REGIME_RSI_SELL))
            cls.VOL_REGIME_STRIKE_ADJUSTMENT = int(vol_params.get("strike_adjustment", cls.VOL_REGIME_STRIKE_ADJUSTMENT))

            adaptive = strategy_params.get("Adaptive Trend", {})
            cls.ADAPTIVE_SHORT_EMA = int(adaptive.get("short_ema", cls.ADAPTIVE_SHORT_EMA))
            cls.ADAPTIVE_LONG_EMA = int(adaptive.get("long_ema", cls.ADAPTIVE_LONG_EMA))
            cls.ADAPTIVE_RSI_CONFIRM = float(adaptive.get("trend_confirmation_rsi", cls.ADAPTIVE_RSI_CONFIRM))
            cls.ADAPTIVE_STRIKE_ADJUSTMENT = int(adaptive.get("strike_adjustment", cls.ADAPTIVE_STRIKE_ADJUSTMENT))
            cls.ADAPTIVE_VOL_FLOOR = float(adaptive.get("volatility_floor", cls.ADAPTIVE_VOL_FLOOR))

            cls.ACTIVE_STRATEGIES = list(trading.get("active_strategies", cls.ACTIVE_STRATEGIES))
            cls.STRATEGY_PARAMETERS = deepcopy(strategy_params)
            cls.BACKTEST = deepcopy(backtest) if backtest else deepcopy(DEFAULT_CONFIG["backtest"])

            cls._raw_config = deepcopy(config)
            return deepcopy(config)

    @classmethod
    def to_dict(cls) -> Dict[str, Dict]:
        with cls._lock:
            if not cls._raw_config:
                cls.reload()
            return deepcopy(cls._raw_config)

    @classmethod
    def to_dashboard(cls) -> Dict[str, Dict]:
        """Return a sanitized dictionary representation for API responses."""
        cls.reload()
        return {
            "trading": {
                "total_capital": cls.TOTAL_CAPITAL,
                "risk_per_trade": cls.RISK_PER_TRADE,
                "lot_size": cls.LOT_SIZE,
                "fees_per_order": cls.FEES_PER_ORDER,
                "base_sl_pct": cls.BASE_SL_PCT,
                "base_tp_pct": cls.BASE_TP_PCT,
                "partial_tp_pct": cls.PARTIAL_TP_PCT,
                "trail_pct": cls.TRAIL_PCT,
                "entry_confidence_min": cls.ENTRY_CONFIDENCE_MIN,
                "max_lots_cap": cls.MAX_LOTS_CAP,
                "max_premium_allocation_pct": cls.MAX_PREMIUM_ALLOCATION_PCT,
                "fixed_risk_rupees": cls.FIXED_RISK_RUPEES,
                "target_trigger_rupees": cls.TARGET_TRIGGER_RUPEES,
                "trail_after_trigger": cls.TRAIL_AFTER_TRIGGER,
                "max_trades_per_day": cls.MAX_TRADES_PER_DAY,
                "max_open_positions": cls.MAX_OPEN_POSITIONS,
                "cooldown_minutes": cls.COOLDOWN_MINUTES,
                "market_open": cls.MARKET_OPEN,
                "market_close": cls.MARKET_CLOSE,
                "eod_exit": cls.EOD_EXIT,
                "active_strategies": cls.ACTIVE_STRATEGIES,
            },
            "indicators": {
                "rsi_period": cls.RSI_PERIOD,
                "rsi_oversold": cls.RSI_OVERSOLD,
                "rsi_overbought": cls.RSI_OVERBOUGHT,
                "bb_period": cls.BB_PERIOD,
                "bb_std": cls.BB_STD,
                "vol_lookback": cls.VOL_LOOKBACK,
                "vol_threshold": cls.VOL_THRESHOLD,
            },
            "strategy_params": deepcopy(cls.STRATEGY_PARAMETERS),
            "backtest": deepcopy(cls.BACKTEST),
        }

    @classmethod
    def update_config(cls, patch: Dict[str, Dict]) -> Dict[str, Dict]:
        with cls._lock:
            merged = _persist_config(patch)
            cls._raw_config = deepcopy(merged)
            cls.reload()
            return cls.to_dashboard()

    @classmethod
    def set_active_strategies(cls, strategies: List[str]) -> Dict[str, Dict]:
        return cls.update_config({"trading": {"active_strategies": strategies}})

    @classmethod
    def update_strategy_params(cls, name: str, params: Dict) -> Dict[str, Dict]:
        return cls.update_config({"strategy_params": {name: params}})

    @classmethod
    def get_strategy_params(cls, name: str) -> Dict:
        cls.reload()
        return deepcopy(cls.STRATEGY_PARAMETERS.get(name, {}))

    @classmethod
    def get_backtest_defaults(cls) -> Dict[str, Dict]:
        cls.reload()
        return deepcopy(cls.BACKTEST)


# Load configuration at import time so class attributes reflect file contents
TradingConfig.reload()

# -------------------------------------------------
# Database
# -------------------------------------------------
class DatabaseManager:
    """Database operations for storing trades and market data"""

    def __init__(self, db_path="algo_trading.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_timestamp TEXT,
                exit_timestamp TEXT,
                symbol TEXT,
                strategy TEXT,
                option_type TEXT,
                strike_price REAL,
                qty_lots INTEGER,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                holding_minutes REAL,
                entry_spot REAL,
                exit_spot REAL,
                volatility REAL,
                vol_regime TEXT,
                reason_entry TEXT,
                reason_exit TEXT,
                max_profit REAL,
                max_loss REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                symbol TEXT,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                close_price REAL,
                volume INTEGER,
                rsi REAL,
                bb_upper REAL,
                bb_middle REAL,
                bb_lower REAL,
                volatility REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Strategy performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                total_pnl REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                avg_trade_duration REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # System status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                component TEXT,
                status TEXT,
                last_update TEXT,
                message TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def insert_trade(self, trade_data: Dict):
        """Insert trade record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (
                entry_timestamp, exit_timestamp, symbol, strategy, option_type,
                strike_price, qty_lots, entry_price, exit_price, pnl,
                holding_minutes, entry_spot, exit_spot, volatility, vol_regime,
                reason_entry, reason_exit, max_profit, max_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade_data.get('entry_timestamp'),
            trade_data.get('exit_timestamp'),
            trade_data.get('symbol'),
            trade_data.get('strategy'),
            trade_data.get('option_type'),
            trade_data.get('strike_price'),
            trade_data.get('qty_lots'),
            trade_data.get('entry_price'),
            trade_data.get('exit_price'),
            trade_data.get('pnl'),
            trade_data.get('holding_minutes'),
            trade_data.get('entry_spot'),
            trade_data.get('exit_spot'),
            trade_data.get('volatility'),
            trade_data.get('vol_regime'),
            trade_data.get('reason_entry'),
            trade_data.get('reason_exit'),
            trade_data.get('max_profit'),
            trade_data.get('max_loss')
        ))

        conn.commit()
        trade_id = cursor.lastrowid
        conn.close()
        return trade_id

    def get_recent_trades(self, limit=100):
        """Get recent trades"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            "SELECT * FROM trades ORDER BY entry_timestamp DESC LIMIT ?",
            conn, params=[limit]
        )
        conn.close()
        return df

    def get_strategy_performance(self):
        """Get strategy performance metrics"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(
            """
            SELECT strategy,
                   COUNT(*) as total_trades,
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                   SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                   SUM(pnl) as total_pnl,
                   AVG(pnl) as avg_pnl,
                   AVG(holding_minutes) as avg_duration,
                   ROUND(100.0 * SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) as win_rate
            FROM trades 
            GROUP BY strategy
            """, conn
        )
        conn.close()
        return df

# -------------------------------------------------
# Indicators
# -------------------------------------------------
class TechnicalIndicators:
    """Technical analysis indicators"""

    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50.0

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def calculate_bollinger_bands(prices: List[float], period: int = 20, std_mult: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            current_price = prices[-1]
            return float(current_price), float(current_price), float(current_price)

        sma = float(np.mean(prices[-period:]))
        std = float(np.std(prices[-period:]))

        upper = sma + (std_mult * std)
        lower = sma - (std_mult * std)

        return float(upper), float(sma), float(lower)

    @staticmethod
    def calculate_volatility(prices: List[float], period: int = 20) -> float:
        """Calculate realized volatility"""
        if len(prices) < 2:
            return 0.20

        returns = np.diff(np.log(prices))
        if len(returns) < period:
            vol = float(np.std(returns) * np.sqrt(252 * 375))  # Annualized intraday
        else:
            vol = float(np.std(returns[-period:]) * np.sqrt(252 * 375))

        return max(0.05, min(1.0, vol))

# -------------------------------------------------
# Options pricing
# -------------------------------------------------
class OptionsCalculator:
    """Options pricing and Greeks calculations"""

    @staticmethod
    def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool = True) -> float:
        """Black-Scholes option pricing"""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return max(S - K, 0.0) if is_call else max(K - S, 0.0)

        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if is_call:
            return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
        else:
            return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

    @staticmethod
    def get_realistic_premium(spot: float, strike: float, is_call: bool, days_to_expiry: int, vol: float = 0.20) -> float:
        """Get realistic option premium based on market patterns"""
        if days_to_expiry <= 0:
            return max(0, spot - strike) if is_call else max(0, strike - spot)

        # Time to expiry in years
        T = days_to_expiry / 365.0

        # Risk-free rate (approximate Indian bond yields)
        r = 0.06

        # Base Black-Scholes price
        bs_price = OptionsCalculator.black_scholes_price(spot, strike, T, r, vol, is_call)

        # Market adjustments for Indian options
        moneyness = spot / strike if strike > 0 else 1.0

        # Add bid-ask spread and market impact
        spread = max(0.05, bs_price * 0.02)  # 2% spread

        # Volatility skew adjustment
        if is_call:
            if moneyness > 1.02:      # ITM
                skew_adj = 1.1
            elif moneyness > 0.98:    # ATM
                skew_adj = 1.0
            else:                     # OTM
                skew_adj = 0.9
        else:
            if moneyness < 0.98:      # ITM
                skew_adj = 1.1
            elif moneyness < 1.02:    # ATM
                skew_adj = 1.0
            else:                     # OTM
                skew_adj = 0.9

        final_price = float(bs_price * skew_adj + spread)

        # Add some randomness for realism (simulation only)
        noise = 1 + np.random.uniform(-0.05, 0.05)  # ±5% noise
        final_price *= noise

        return float(max(0.05, final_price))  # Minimum premium of ₹0.05

# -------------------------------------------------
# Strategy base & strategies
# -------------------------------------------------
class TradingStrategy:
    """Base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.signals_generated = 0
        self.trades_executed = 0

    def generate_signal(self, market_data: Dict, technical_data: Dict) -> Optional[Dict]:
        """Generate trading signal - to be implemented by subclasses"""
        raise NotImplementedError

    def get_statistics(self) -> Dict:
        """Get strategy statistics"""
        return {
            'name': self.name,
            'signals_generated': self.signals_generated,
            'trades_executed': self.trades_executed,
            'signal_to_trade_ratio': self.trades_executed / max(self.signals_generated, 1)
        }

class MeanReversionStrategy(TradingStrategy):
    """Mean reversion trading strategy"""

    def __init__(self):
        super().__init__("Mean Reversion")
        self.rsi_oversold = TradingConfig.RSI_OVERSOLD
        self.rsi_overbought = TradingConfig.RSI_OVERBOUGHT
        self.strike_adjustment = 0
        self.refresh_from_config()

    def refresh_from_config(self):
        params = TradingConfig.get_strategy_params(self.name)
        self.rsi_oversold = float(params.get("rsi_oversold", TradingConfig.RSI_OVERSOLD))
        self.rsi_overbought = float(params.get("rsi_overbought", TradingConfig.RSI_OVERBOUGHT))
        self.strike_adjustment = int(params.get("strike_adjustment", 0))

    def generate_signal(self, market_data: Dict, technical_data: Dict) -> Optional[Dict]:
        """Generate mean reversion signals"""
        current_price = market_data['close']
        rsi = technical_data.get('rsi', 50)
        bb_upper = technical_data.get('bb_upper', current_price)
        bb_lower = technical_data.get('bb_lower', current_price)
        vol_regime = technical_data.get('vol_regime', 'normal')

        self.signals_generated += 1

        # Trade in normal or high volatility
        if vol_regime not in ('normal', 'high'):
            return None

        signal = None

        # Oversold condition - Buy Call
        if rsi <= self.rsi_oversold and current_price <= bb_lower:
            signal = {
                'type': 'BUY',
                'option_type': 'CE',
                'reason': 'MEAN_REVERSION_OVERSOLD',
                'confidence': min(100, (self.rsi_oversold - rsi) * 2),
                'strike_adjustment': self.strike_adjustment
            }

        # Overbought condition - Buy Put
        elif rsi >= self.rsi_overbought and current_price >= bb_upper:
            signal = {
                'type': 'BUY',
                'option_type': 'PE',
                'reason': 'MEAN_REVERSION_OVERBOUGHT',
                'confidence': min(100, (rsi - self.rsi_overbought) * 2),
                'strike_adjustment': self.strike_adjustment
            }

        if signal:
            self.trades_executed += 1

        return signal

class MomentumBreakoutStrategy(TradingStrategy):
    """Momentum breakout trading strategy"""

    def __init__(self):
        super().__init__("Momentum Breakout")
        self.lookback_period = TradingConfig.MOMENTUM_LOOKBACK
        self.range_threshold = TradingConfig.MOMENTUM_RANGE_THRESHOLD
        self.strike_adjustment = TradingConfig.MOMENTUM_STRIKE_ADJUSTMENT
        self.refresh_from_config()

    def refresh_from_config(self):
        params = TradingConfig.get_strategy_params(self.name)
        self.lookback_period = int(params.get("lookback_period", TradingConfig.MOMENTUM_LOOKBACK))
        self.range_threshold = float(params.get("range_threshold", TradingConfig.MOMENTUM_RANGE_THRESHOLD))
        self.strike_adjustment = int(params.get("strike_adjustment", TradingConfig.MOMENTUM_STRIKE_ADJUSTMENT))

    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate momentum breakout signals"""
        if len(historical_prices) < self.lookback_period:
            return None

        current_price = market_data['close']
        vol_regime = technical_data.get('vol_regime', 'normal')
        volume = market_data.get('volume', 0)

        self.signals_generated += 1

        # Trade in high OR normal volatility (more trades)
        if vol_regime not in ('high', 'normal'):
            return None

        recent_high = max(historical_prices[-self.lookback_period:])
        recent_low = min(historical_prices[-self.lookback_period:])
        range_pct = (recent_high - recent_low) / max(recent_low, 1e-9)

        # Need decent range (loosened to 0.6%)
        if range_pct < self.range_threshold:
            return None

        signal = None

        # Upward breakout (buffer loosened to ~0.15%)
        if current_price > recent_high * 1.0015:
            signal = {
                'type': 'BUY',
                'option_type': 'CE',
                'reason': 'MOMENTUM_BREAKOUT_UP',
                'confidence': min(100, range_pct * 1000),  # Scale with range
                'strike_adjustment': self.strike_adjustment
            }

        # Downward breakout (buffer loosened)
        elif current_price < recent_low * 0.9985:
            signal = {
                'type': 'BUY',
                'option_type': 'PE',
                'reason': 'MOMENTUM_BREAKOUT_DOWN',
                'confidence': min(100, range_pct * 1000),
                'strike_adjustment': self.strike_adjustment
            }

        if signal:
            self.trades_executed += 1

        return signal

class VolatilityRegimeStrategy(TradingStrategy):
    """Volatility regime based trading strategy"""

    def __init__(self):
        super().__init__("Volatility Regime")
        self.vol_threshold = TradingConfig.VOLATILITY_TRIGGER
        self.rsi_buy = TradingConfig.VOL_REGIME_RSI_BUY
        self.rsi_sell = TradingConfig.VOL_REGIME_RSI_SELL
        self.strike_adjustment = TradingConfig.VOL_REGIME_STRIKE_ADJUSTMENT
        self.refresh_from_config()

    def refresh_from_config(self):
        params = TradingConfig.get_strategy_params(self.name)
        self.vol_threshold = float(params.get("vol_threshold", TradingConfig.VOLATILITY_TRIGGER))
        self.rsi_buy = float(params.get("rsi_buy", TradingConfig.VOL_REGIME_RSI_BUY))
        self.rsi_sell = float(params.get("rsi_sell", TradingConfig.VOL_REGIME_RSI_SELL))
        self.strike_adjustment = int(params.get("strike_adjustment", TradingConfig.VOL_REGIME_STRIKE_ADJUSTMENT))

    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate signals based on volatility regime changes"""
        if len(historical_prices) < 20:
            return None

        current_vol = technical_data.get('volatility', 0.20)
        vol_regime = technical_data.get('vol_regime', 'normal')
        current_price = market_data['close']
        rsi = technical_data.get('rsi', 50)

        self.signals_generated += 1

        signal = None

        # Volatility expansion strategy (loosened barrier)
        if vol_regime in ('high', 'normal') and current_vol > self.vol_threshold:
            if rsi < self.rsi_buy:  # Oversold in high/normal vol
                signal = {
                    'type': 'BUY',
                    'option_type': 'CE',
                    'reason': 'HIGH_VOL_OVERSOLD',
                    'confidence': 75,
                    'strike_adjustment': self.strike_adjustment
                }
            elif rsi > self.rsi_sell:  # Overbought in high/normal vol
                signal = {
                    'type': 'BUY',
                    'option_type': 'PE',
                    'reason': 'HIGH_VOL_OVERBOUGHT',
                    'confidence': 75,
                    'strike_adjustment': self.strike_adjustment
                }

        # Volatility contraction strategy (unchanged but available)
        elif vol_regime == 'low' and current_vol < 0.15:
            recent_prices = historical_prices[-5:]
            price_stability = np.std(recent_prices) / max(np.mean(recent_prices), 1e-9)

            if price_stability < 0.005:  # Very stable prices
                if rsi > 55:  # Slight upward bias
                    signal = {
                        'type': 'BUY',
                        'option_type': 'CE',
                        'reason': 'LOW_VOL_BREAKOUT_PREP_UP',
                        'confidence': 60,
                        'strike_adjustment': 100
                    }
                elif rsi < 45:  # Slight downward bias
                    signal = {
                        'type': 'BUY',
                        'option_type': 'PE',
                        'reason': 'LOW_VOL_BREAKOUT_PREP_DOWN',
                        'confidence': 60,
                        'strike_adjustment': 100
                    }

        if signal:
            self.trades_executed += 1

        return signal

class AdaptiveTrendStrategy(TradingStrategy):
    """Adaptive trend-following options strategy for NIFTY options."""

    def __init__(self):
        super().__init__("Adaptive Trend")
        self.short_ema = TradingConfig.ADAPTIVE_SHORT_EMA
        self.long_ema = TradingConfig.ADAPTIVE_LONG_EMA
        self.rsi_confirm = TradingConfig.ADAPTIVE_RSI_CONFIRM
        self.strike_adjustment = TradingConfig.ADAPTIVE_STRIKE_ADJUSTMENT
        self.vol_floor = TradingConfig.ADAPTIVE_VOL_FLOOR
        self.refresh_from_config()

    def refresh_from_config(self):
        params = TradingConfig.get_strategy_params(self.name)
        self.short_ema = int(params.get("short_ema", TradingConfig.ADAPTIVE_SHORT_EMA))
        self.long_ema = int(params.get("long_ema", TradingConfig.ADAPTIVE_LONG_EMA))
        self.rsi_confirm = float(params.get("trend_confirmation_rsi", TradingConfig.ADAPTIVE_RSI_CONFIRM))
        self.strike_adjustment = int(params.get("strike_adjustment", TradingConfig.ADAPTIVE_STRIKE_ADJUSTMENT))
        self.vol_floor = float(params.get("volatility_floor", TradingConfig.ADAPTIVE_VOL_FLOOR))

    @staticmethod
    def _ema(values: List[float], period: int) -> float:
        series = pd.Series(values)
        return float(series.ewm(span=period, adjust=False).mean().iloc[-1])

    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate signals using dual-EMA trend confirmation with RSI filter."""
        lookback = max(self.long_ema * 2, 50)
        if len(historical_prices) < lookback:
            return None

        prices_window = historical_prices[-lookback:]
        short_ema = self._ema(prices_window, self.short_ema)
        long_ema = self._ema(prices_window, self.long_ema)

        rsi = technical_data.get('rsi', 50)
        vol = technical_data.get('volatility', 0.20)

        if vol < self.vol_floor:
            return None

        self.signals_generated += 1
        signal = None
        trend_diff = long_ema if long_ema else 1e-9
        slope = (short_ema - long_ema) / trend_diff

        if short_ema > long_ema and rsi >= self.rsi_confirm:
            confidence = min(100, abs(slope) * 25000)
            if confidence >= TradingConfig.ENTRY_CONFIDENCE_MIN:
                signal = {
                    'type': 'BUY',
                    'option_type': 'CE',
                    'reason': 'ADAPTIVE_TREND_UP',
                    'confidence': confidence,
                    'strike_adjustment': self.strike_adjustment
                }
        elif short_ema < long_ema and rsi <= (100 - self.rsi_confirm):
            confidence = min(100, abs(slope) * 25000)
            if confidence >= TradingConfig.ENTRY_CONFIDENCE_MIN:
                signal = {
                    'type': 'BUY',
                    'option_type': 'PE',
                    'reason': 'ADAPTIVE_TREND_DOWN',
                    'confidence': confidence,
                    'strike_adjustment': self.strike_adjustment
                }

        if signal:
            self.trades_executed += 1

        return signal


STRATEGY_REGISTRY: Dict[str, type] = {
    "Mean Reversion": MeanReversionStrategy,
    "Momentum Breakout": MomentumBreakoutStrategy,
    "Volatility Regime": VolatilityRegimeStrategy,
    "Adaptive Trend": AdaptiveTrendStrategy,
}

# -------------------------------------------------
# Position / P&L container
# -------------------------------------------------
class Position:
    """Represents an open trading position"""

    def __init__(self, symbol: str, option_type: str, strike: float, entry_price: float,
                 qty_lots: int, entry_time: datetime, strategy: str, entry_spot: float,
                 expiry_date: str = ""):
        self.symbol = symbol
        self.option_type = option_type  # 'CE' or 'PE'
        self.strike = strike
        self.entry_price = entry_price
        self.qty_lots = qty_lots
        self.entry_time = entry_time
        self.strategy = strategy
        self.entry_spot = entry_spot
        self.expiry_date = expiry_date  # ISO 'YYYY-MM-DD'

        self.current_price = entry_price
        self.peak_price = entry_price
        self.max_loss_price = entry_price
        self.partial_profit_taken = False

        self.trailing_stop_active = False
        self.trail_high_price = entry_price

        self.unrealized_pnl = 0.0
        self.total_fees = TradingConfig.FEES_PER_ORDER  # Entry fee

        # rupee-based risk model parameters
        self.fixed_risk_rupees = TradingConfig.FIXED_RISK_RUPEES
        self.target_trigger_rupees = TradingConfig.TARGET_TRIGGER_RUPEES

    def update_price(self, new_price: float, current_spot: float):
        """Update current option price and calculate metrics"""
        self.current_price = new_price

        # Update peak and trough prices
        if new_price > self.peak_price:
            self.peak_price = new_price
        if new_price < self.max_loss_price:
            self.max_loss_price = new_price

        # Calculate unrealized P&L
        self.unrealized_pnl = (new_price - self.entry_price) * TradingConfig.LOT_SIZE * self.qty_lots - self.total_fees

        # Update trailing anchor
        if self.trailing_stop_active and new_price > self.trail_high_price:
            self.trail_high_price = new_price

    def get_return_pct(self) -> float:
        """Get current return percentage"""
        if self.entry_price <= 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price

    def get_holding_minutes(self) -> float:
        now = datetime.now(pytz.timezone('Asia/Kolkata'))
        if self.entry_time.tzinfo is None:
            entry_time = pytz.timezone('Asia/Kolkata').localize(self.entry_time)
        else:
            entry_time = self.entry_time
        return (now - entry_time).total_seconds() / 60.0

    def should_exit(self, current_spot: float, vol_regime: str = 'normal') -> Optional[str]:
        """
        Rupee-based risk management:
        - Hard stop: -₹500 (default via TradingConfig.FIXED_RISK_RUPEES)
        - When +₹500 reached: turn on trailing stop (breakeven floor + TRAIL_PCT)
        """
        # P&L in rupees (net of entry fee; exit fee applied on close by engine)
        pnl_rupees = (self.current_price - self.entry_price) * TradingConfig.LOT_SIZE * self.qty_lots - self.total_fees

        # 1) Fixed-rupee stop loss
        if pnl_rupees <= -self.fixed_risk_rupees:
            return "STOP_LOSS_FIXED"

        # 2) Trigger trailing after target profit
        if TradingConfig.TRAIL_AFTER_TRIGGER and not self.trailing_stop_active and pnl_rupees >= self.target_trigger_rupees:
            self.trailing_stop_active = True
            self.trail_high_price = max(self.trail_high_price, self.current_price)
            return None  # Start trailing; do not exit immediately

        # 3) Manage trailing stop once active
        if self.trailing_stop_active:
            self.trail_high_price = max(self.trail_high_price, self.current_price)
            # Breakeven floor + price trail
            trail_stop_price = max(self.entry_price, self.trail_high_price * (1 - TradingConfig.TRAIL_PCT))
            if self.current_price <= trail_stop_price:
                return "TRAILING_STOP"

        # 4) End-of-day flat
        current_time = datetime.now().time()
        eod_time = datetime.strptime(TradingConfig.EOD_EXIT, "%H:%M").time()
        if current_time >= eod_time:
            return "END_OF_DAY"

        return None

    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'option_type': self.option_type,
            'strike': self.strike,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'qty_lots': self.qty_lots,
            'entry_time': self.entry_time.isoformat() if hasattr(self.entry_time, 'isoformat') else str(self.entry_time),
            'strategy': self.strategy,
            'entry_spot': self.entry_spot,
            'unrealized_pnl': self.unrealized_pnl,
            'return_pct': self.get_return_pct(),
            'holding_minutes': self.get_holding_minutes(),
            'expiry_date': self.expiry_date
        }

# -------------------------------------------------
# CSV export
# -------------------------------------------------
def export_trades_to_csv(db_manager: DatabaseManager, filename: str = None) -> Optional[str]:
    """Export all trades to CSV file"""
    if filename is None:
        filename = f"algo_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    trades_df = db_manager.get_recent_trades(limit=10000)  # Up to 10k recent trades

    if trades_df.empty:
        logger.info("No trades to export")
        return None

    # Add calculated columns for better analysis
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_timestamp']).dt.date
    trades_df['entry_time'] = pd.to_datetime(trades_df['entry_timestamp']).dt.time
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_timestamp']).dt.date
    trades_df['exit_time'] = pd.to_datetime(trades_df['exit_timestamp']).dt.time
    trades_df['total_premium_paid'] = trades_df['entry_price'] * trades_df['qty_lots'] * TradingConfig.LOT_SIZE
    trades_df['total_premium_received'] = trades_df['exit_price'] * trades_df['qty_lots'] * TradingConfig.LOT_SIZE
    trades_df['roi_percentage'] = (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price'] * 100

    try:
        trades_df.to_csv(filename, index=False)
        logger.info(f"✅ Trades exported to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"❌ Error exporting trades: {e}")
        return None


def run_enhanced_backtest(symbol: Optional[str] = None,
                          interval: Optional[str] = None,
                          lookback_days: Optional[int] = None,
                          strategies: Optional[List[str]] = None) -> Dict:
    """Execute the advanced backtester and persist the best strategy selection."""
    from backtester import Backtester  # Local import to avoid circular dependency

    tester = Backtester(
        symbol=symbol,
        interval=interval,
        lookback_days=lookback_days,
        strategies=strategies,
        auto_apply_best=True,
    )
    result = tester.run()
    best = result.get("best_strategy") or {}
    if best:
        logger.info(
            "Backtest complete. Best strategy %s with total P&L ₹%.2f",
            best.get("name"),
            best.get("total_pnl", 0.0),
        )
    else:
        logger.warning("Backtest completed but no trades were generated.")
    return result

# -------------------------------------------------
# Entrypoint hint
# -------------------------------------------------
if __name__ == "__main__":
    print("AI Algo Trading Platform - Main Engine")
    print("This is the core trading engine module. Use trading_engine.py to run the live system.")
