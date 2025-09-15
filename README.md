# AI Algorithmic Trading Platform

## Overview
Advanced algorithmic trading platform with paper trading capabilities for the Indian options market.

## Features
- ✅ Real-time market data integration
- ✅ Multiple trading strategies (9+ strategies)
- ✅ Advanced risk management
- ✅ Web-based dashboard with real-time updates
- ✅ Trade logging and CSV export
- ✅ Performance analytics
- ✅ Paper trading (no real money at risk)
- ✅ Configurable parameters from dashboard
- ✅ Built-in historical backtesting with best-strategy auto selection

## Quick Start

### 1. Installation
Run the setup script:
```bash
python setup.py
```

### 2. Start Trading System
```bash
# Windows
start_trading.bat

# Linux/Mac
./start_trading.sh

# Or manually
python run.py both
```

### 3. Access Dashboard
Open your browser and go to: http://localhost:5000

## Strategies Available

1. **Mean Reversion** - Trades oversold/overbought conditions
2. **Momentum Breakout** - Follows price breakouts with volume
3. **Volatility Regime** - Adapts to volatility conditions
4. **Pair Trading** - Statistical arbitrage based on price ratios
5. **News Based** - Reacts to sudden price movements
6. **Scalping** - High-frequency quick profit strategy  
7. **Machine Learning** - AI-based pattern recognition
8. **Options Flow** - Based on options activity analysis
9. **Arbitrage** - Statistical arbitrage opportunities

## Configuration

Edit `config/config.json` (or use the dashboard Configuration tab) to customize:
- Capital amount (propagated to engine, dashboard, and backtester)
- Risk parameters & per-trade limits
- Active strategy roster and strategy-specific tuning
- Market timing and trailing/target behaviour

## Files Structure

```
├── algo_trading_main.py      # Core trading engine
├── trading_engine.py         # Live trading system
├── dashboard_app.py          # Web dashboard
├── advanced_strategies.py    # Additional strategies
├── run.py                   # Main launcher
├── config/
│   └── config.json          # Configuration
├── templates/
│   └── dashboard.html       # Dashboard template
├── data/                    # Market data storage
├── logs/                    # System logs
└── exports/                 # Trade exports
```

## Risk Management

- **Paper Trading Only**: No real money at risk
- **Position Sizing**: Automatic based on capital and risk
- **Stop Losses**: Dynamic based on volatility
- **Daily Limits**: Maximum trades per day
- **Cooldown Periods**: Prevents overtrading

## Backtesting

- Use the dashboard **Backtesting** tab to run historical simulations.
- Supports intraday 1-minute data (up to 7 days via Yahoo Finance limits) and higher intervals.
- Automatically applies the most profitable strategy from the run to the live engine configuration (optional toggle).
- Detailed equity curves and trade breakdowns are displayed directly in the UI.

## Dashboard Features

- Real-time P&L tracking
- Open positions monitoring
- Strategy performance comparison
- Trade history with filtering
- Performance charts and analytics
- CSV export functionality
- System control (start/stop)

## Development

To add new strategies:
1. Extend the `TradingStrategy` class
2. Implement `generate_signal()` method
3. Add to strategy factory in `advanced_strategies.py`

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review configuration in `config/config.json`
3. Ensure all dependencies are installed

## Disclaimer

This software is for educational and paper trading purposes only. 
Not financial advice. Always do your own research before real trading.

## License

This project is for educational purposes. Modify and use as needed.
