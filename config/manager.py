"""Utilities for loading and persisting trading configuration."""
from __future__ import annotations

import json
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "trading": {
        "total_capital": 100000,
        "risk_per_trade": 0.02,
        "lot_size": 50,
        "fees_per_order": 64,
        "base_sl_pct": 0.03,
        "base_tp_pct": 0.06,
        "partial_tp_pct": 0.04,
        "trail_pct": 0.02,
        "entry_confidence_min": 40,
        "max_lots_cap": 10,
        "max_premium_allocation_pct": 0.10,
        "fixed_risk_rupees": 500,
        "target_trigger_rupees": 500,
        "trail_after_trigger": True,
        "max_trades_per_day": 5,
        "max_open_positions": 3,
        "cooldown_minutes": 30,
        "market_open": "09:15",
        "market_close": "15:30",
        "eod_exit": "15:15",
        "active_strategies": [
            "Mean Reversion",
            "Momentum Breakout",
            "Volatility Regime",
        ],
    },
    "indicators": {
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "bb_period": 20,
        "bb_std": 2.0,
        "vol_lookback": 20,
        "vol_threshold": 1.2,
    },
    "strategy_params": {
        "Mean Reversion": {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "strike_adjustment": 0,
        },
        "Momentum Breakout": {
            "lookback_period": 10,
            "range_threshold": 0.006,
            "strike_adjustment": 50,
        },
        "Volatility Regime": {
            "vol_threshold": 0.25,
            "rsi_buy": 40,
            "rsi_sell": 60,
            "strike_adjustment": 50,
        },
        "Adaptive Trend": {
            "short_ema": 21,
            "long_ema": 55,
            "trend_confirmation_rsi": 55,
            "strike_adjustment": 50,
            "volatility_floor": 0.18,
        },
    },
    "backtest": {
        "symbol": "^NSEI",
        "interval": "1m",
        "lookback_days": 5,
        "max_lookback_1m": 7,
    },
}

_lock = threading.RLock()
_cache: Dict[str, Any] | None = None

def ensure_config() -> None:
    """Ensure the configuration file exists on disk."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)

def load_config(force: bool = False) -> Dict[str, Any]:
    """Load configuration from disk with optional caching."""
    global _cache
    with _lock:
        if _cache is not None and not force:
            return deepcopy(_cache)
        ensure_config()
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        _cache = data
        return deepcopy(data)

def save_config(config: Dict[str, Any]) -> None:
    """Persist configuration to disk and refresh cache."""
    global _cache
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    _cache = deepcopy(config)

def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return the merged copy."""
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result

def update_config(partial_updates: Dict[str, Any]) -> Dict[str, Any]:
    """Apply ``partial_updates`` to the stored configuration and persist it."""
    current = load_config()
    merged = deep_update(current, partial_updates)
    save_config(merged)
    return merged

def reset_to_defaults() -> Dict[str, Any]:
    """Overwrite the configuration with factory defaults."""
    save_config(DEFAULT_CONFIG)
    return deepcopy(DEFAULT_CONFIG)
