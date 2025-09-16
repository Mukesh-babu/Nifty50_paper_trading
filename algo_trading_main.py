# AI_ALGO_TRADING_PLATFORM - Main Trading Engine
# Advanced Algorithmic Trading System with Paper Trading & Live Dashboard
# Author: AI Trading Systems
# Version: 1.2

import sys
import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from scipy.stats import norm

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
# Backtesting utilities
# -------------------------------------------------
def run_enhanced_backtest(symbol: str = "^NSEI", days: int = 5, interval: str = "5m",
                          export: bool = True) -> Dict[str, float]:
    """Run a lightweight moving-average backtest for diagnostics.

    The launcher expects this helper to exist.  Earlier revisions removed it,
    leading to ``ImportError`` at runtime.  This implementation keeps the
    runtime self-contained: it tries to download historical data via
    :mod:`yfinance` and, if none is available (e.g. offline environments),
    generates a synthetic intraday price series so that the analytics layer can
    still be exercised.

    Returns a dictionary with headline metrics so callers (CLI/UI) can inspect
    performance or surface the result to users.
    """

    logger.info("📈 Starting backtest for %s (%s, %dd)", symbol, interval, days)

    def _interval_to_minutes(value: str) -> int:
        try:
            if value.endswith("m"):
                return max(1, int(value[:-1]))
            if value.endswith("h"):
                return max(1, int(value[:-1]) * 60)
        except (ValueError, AttributeError):
            pass
        return 5

    minutes = _interval_to_minutes(interval)
    points_per_day = max(1, int(390 / minutes))

    history = pd.DataFrame()
    try:
        history = yf.download(
            symbol,
            period=f"{max(days, 1)}d",
            interval=interval,
            progress=False,
            threads=False,
        )
    except Exception as err:
        logger.warning("Historical download failed for %s: %s", symbol, err)

    if history is None:
        history = pd.DataFrame()

    if isinstance(history, pd.DataFrame) and isinstance(history.columns, pd.MultiIndex):
        try:
            history = history.xs(symbol, axis=1, level=-1)
        except Exception:
            history = history.droplevel(-1, axis=1)

    if isinstance(history, pd.DataFrame):
        history.columns = [str(col) for col in history.columns]

    if "Close" in history.columns:
        history = history.dropna(subset=["Close"]).copy()
    else:
        history = pd.DataFrame()

    if history.empty:
        logger.warning("No market data available; generating synthetic price series for smoke test")
        total_points = max(points_per_day * max(days, 1), 60)
        end_ts = datetime.now(pytz.timezone('Asia/Kolkata'))
        index = pd.date_range(
            end=end_ts,
            periods=total_points,
            freq=f"{minutes}T",
            tz=end_ts.tzinfo,
        )
        rng = np.random.default_rng(seed=42)
        base_price = 19500.0
        step_returns = rng.normal(loc=0.0001, scale=0.004, size=total_points)
        price_series = pd.Series(step_returns, index=index).add(1).cumprod() * base_price
        history = pd.DataFrame({
            "Open": price_series.shift(1).fillna(price_series.iloc[0]),
            "High": price_series * (1 + rng.uniform(0.0005, 0.0015, total_points)),
            "Low": price_series * (1 - rng.uniform(0.0005, 0.0015, total_points)),
            "Close": price_series,
            "Volume": rng.integers(120_000, 320_000, total_points),
        }, index=index)
    else:
        history.index = pd.to_datetime(history.index)
        tzinfo = getattr(history.index, "tz", None)
        if tzinfo is not None:
            try:
                history.index = history.index.tz_convert('Asia/Kolkata')
            except Exception:
                history.index = history.index.tz_localize('Asia/Kolkata', nonexistent='shift_forward', ambiguous='NaT')
            history.index = history.index.tz_localize(None)

    if history.empty:
        logger.error("Backtest aborted: unable to prepare price data")
        return {
            "symbol": symbol,
            "interval": interval,
            "total_trades": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "avg_trade_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe_ratio": 0.0,
            "annualized_return_pct": 0.0,
            "export_path": None,
        }

    history.sort_index(inplace=True)
    history["returns"] = history["Close"].pct_change().fillna(0.0)
    history["fast_ma"] = history["Close"].rolling(12).mean()
    history["slow_ma"] = history["Close"].rolling(36).mean()
    history["signal"] = np.where(history["fast_ma"] > history["slow_ma"], 1, -1)
    history.loc[history["slow_ma"].isna(), "signal"] = 0
    history["position"] = history["signal"].shift(1).fillna(0).astype(int)
    history["strategy_return"] = history["position"] * history["returns"]

    equity_curve = (1 + history["strategy_return"]).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    strategy_std = history["strategy_return"].std()
    sharpe = 0.0
    if strategy_std > 0:
        sharpe = float((history["strategy_return"].mean() / strategy_std) * np.sqrt(points_per_day * 252))

    total_growth = float(equity_curve.iloc[-1]) if not equity_curve.empty else 1.0
    total_days = max(1, (history.index[-1] - history.index[0]).days + 1)
    annualized_return = float(total_growth ** (252 / total_days) - 1) if total_days > 0 else 0.0

    trades: List[Dict[str, object]] = []
    current_pos = 0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None

    for ts, row in history.iterrows():
        desired_pos = int(row["position"])
        if current_pos == 0 and desired_pos != 0:
            current_pos = desired_pos
            entry_price = float(row["Close"])
            entry_time = ts
            continue

        if current_pos != 0 and desired_pos != current_pos:
            exit_price = float(row["Close"])
            exit_time = ts
            holding = 0.0
            if entry_time is not None:
                holding = max(0.0, (exit_time - entry_time).total_seconds() / 60.0)
            pct_return = ((exit_price - entry_price) / entry_price) * current_pos if entry_price else 0.0
            trades.append({
                "entry_time": pd.Timestamp(entry_time).isoformat() if entry_time else pd.Timestamp(exit_time).isoformat(),
                "exit_time": pd.Timestamp(exit_time).isoformat(),
                "direction": "LONG" if current_pos == 1 else "SHORT",
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pct_return": pct_return,
                "holding_minutes": holding,
            })

            if desired_pos != 0:
                current_pos = desired_pos
                entry_price = float(row["Close"])
                entry_time = ts
            else:
                current_pos = 0
                entry_price = 0.0
                entry_time = None

    if current_pos != 0 and entry_time is not None:
        exit_price = float(history.iloc[-1]["Close"])
        exit_time = history.index[-1]
        holding = max(0.0, (exit_time - entry_time).total_seconds() / 60.0)
        pct_return = ((exit_price - entry_price) / entry_price) * current_pos if entry_price else 0.0
        trades.append({
            "entry_time": pd.Timestamp(entry_time).isoformat(),
            "exit_time": pd.Timestamp(exit_time).isoformat(),
            "direction": "LONG" if current_pos == 1 else "SHORT",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pct_return": pct_return,
            "holding_minutes": holding,
        })

    trades_df = pd.DataFrame(trades)
    capital = float(TradingConfig.TOTAL_CAPITAL)
    if not trades_df.empty:
        trades_df["pnl_rupees"] = trades_df["pct_return"] * capital

    total_trades = int(len(trades_df))
    winning_trades = int((trades_df["pct_return"] > 0).sum()) if total_trades else 0
    total_return_pct = float(trades_df["pct_return"].sum() * 100) if total_trades else 0.0
    avg_trade_pct = float(trades_df["pct_return"].mean() * 100) if total_trades else 0.0
    win_rate = float((winning_trades / total_trades) * 100) if total_trades else 0.0

    export_path = None
    if export and not trades_df.empty:
        exports_dir = Path("exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        export_file = exports_dir / f"backtest_{symbol.replace('^', '').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(export_file, index=False)
        export_path = str(export_file)
        logger.info("📁 Backtest trades exported to %s", export_file)

    summary = {
        "symbol": symbol,
        "interval": interval,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 2),
        "total_return_pct": round(total_return_pct, 2),
        "avg_trade_pct": round(avg_trade_pct, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "sharpe_ratio": round(sharpe, 2),
        "annualized_return_pct": round(annualized_return * 100, 2),
        "export_path": export_path,
    }

    logger.info("✅ Backtest summary: %s", summary)
    return summary

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
