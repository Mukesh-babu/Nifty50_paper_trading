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

# -------------------------------------------------
# Config
# -------------------------------------------------
class TradingConfig:
    """Centralized configuration management"""
    # Capital Management
    TOTAL_CAPITAL = 10000          # ₹10K starting capital for paper trading
    RISK_PER_TRADE = 0.02           # 2% risk per trade (still used for sanity checks)
    LOT_SIZE = 50                   # NIFTY lot size = 50
    FEES_PER_ORDER = 64             # Estimated fees per order
    PREMIUM_CAPITAL_FRACTION = 0.6  # Allow up to 60% of available cash per position

    # Risk Management (percent-based helpers; still used for sizing heuristics)
    BASE_SL_PCT = 0.03              # 3% initial stop loss assumption (for sizing heuristics)
    BASE_TP_PCT = 0.06              # (not used by rupee exit, but kept for analytics)
    PARTIAL_TP_PCT = 0.04           # (kept for analytics)
    TRAIL_PCT = 0.02                # 2% trailing stop (applies to option price when trailing)

    # New: Execution intensity & fixed-rupee risk model
    ENTRY_CONFIDENCE_MIN = 40       # Lower threshold → more entries
    MAX_LOTS_CAP = 10               # Allow up to 10 lots when risk allows

    FIXED_RISK_RUPEES = 500         # Hard stop per trade in ₹
    TARGET_TRIGGER_RUPEES = 500     # When reached, flip to trailing mode
    TRAIL_AFTER_TRIGGER = True      # Enable trailing after trigger

    # Trading Limits
    MAX_TRADES_PER_DAY = 5
    MAX_OPEN_POSITIONS = 3
    COOLDOWN_MINUTES = 30
    DATA_STALENESS_SECONDS = 120    # Require market data updates within the last 2 minutes

    # Market Hours (IST)
    MARKET_OPEN = "09:15"
    MARKET_CLOSE = "15:30"
    EOD_EXIT = "15:15"

    # Technical Indicators (slightly looser to increase signals)
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30               # was 25
    RSI_OVERBOUGHT = 70             # was 75
    BB_PERIOD = 20
    BB_STD = 2.0
    VOL_LOOKBACK = 20
    VOL_THRESHOLD = 1.2

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
                'strike_adjustment': 0  # ATM strike
            }

        # Overbought condition - Buy Put
        elif rsi >= self.rsi_overbought and current_price >= bb_upper:
            signal = {
                'type': 'BUY',
                'option_type': 'PE',
                'reason': 'MEAN_REVERSION_OVERBOUGHT',
                'confidence': min(100, (rsi - self.rsi_overbought) * 2),
                'strike_adjustment': 0  # ATM strike
            }

        if signal:
            self.trades_executed += 1

        return signal

class MomentumBreakoutStrategy(TradingStrategy):
    """Momentum breakout trading strategy"""

    def __init__(self):
        super().__init__("Momentum Breakout")
        self.lookback_period = 10

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
        if range_pct < 0.006:
            return None

        signal = None

        # Upward breakout (buffer loosened to ~0.15%)
        if current_price > recent_high * 1.0015:
            signal = {
                'type': 'BUY',
                'option_type': 'CE',
                'reason': 'MOMENTUM_BREAKOUT_UP',
                'confidence': min(100, range_pct * 1000),  # Scale with range
                'strike_adjustment': 50  # Slightly OTM
            }

        # Downward breakout (buffer loosened)
        elif current_price < recent_low * 0.9985:
            signal = {
                'type': 'BUY',
                'option_type': 'PE',
                'reason': 'MOMENTUM_BREAKOUT_DOWN',
                'confidence': min(100, range_pct * 1000),
                'strike_adjustment': 50  # Slightly OTM
            }

        if signal:
            self.trades_executed += 1

        return signal

class VolatilityRegimeStrategy(TradingStrategy):
    """Volatility regime based trading strategy"""

    def __init__(self):
        super().__init__("Volatility Regime")

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
        if vol_regime in ('high', 'normal') and current_vol > 0.25:
            if rsi < 40:  # Oversold in high/normal vol
                signal = {
                    'type': 'BUY',
                    'option_type': 'CE',
                    'reason': 'HIGH_VOL_OVERSOLD',
                    'confidence': 75,
                    'strike_adjustment': 25
                }
            elif rsi > 60:  # Overbought in high/normal vol
                signal = {
                    'type': 'BUY',
                    'option_type': 'PE',
                    'reason': 'HIGH_VOL_OVERBOUGHT',
                    'confidence': 75,
                    'strike_adjustment': 25
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

# -------------------------------------------------
# Entrypoint hint
# -------------------------------------------------
if __name__ == "__main__":
    print("AI Algo Trading Platform - Main Engine")
    print("This is the core trading engine module. Use trading_engine.py to run the live system.")
