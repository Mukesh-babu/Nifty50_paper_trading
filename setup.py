# SETUP.PY - Installation and Setup Script
# Setup script for AI Algo Trading Platform
# Version: 1.0

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    """Print setup header"""
    print("="*80)
    print("🤖 AI ALGORITHMIC TRADING PLATFORM SETUP")
    print("="*80)
    print("Setting up your advanced algo trading environment...")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("📋 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    requirements = [
        "yfinance>=0.2.18",
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scipy>=1.9.0",
        "flask>=2.3.0",
        "flask-socketio>=5.3.0",
        "plotly>=5.15.0",
        "schedule>=1.2.0",
        "pytz>=2023.3",
        "python-socketio>=5.8.0",
        "python-engineio>=4.7.0",
        "requests>=2.31.0"
    ]
    
    failed_packages = []
    
    for package in requirements:
        try:
            print(f"   Installing {package}...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                                  capture_output=True, text=True, check=True)
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {package}")
            print(f"      Error: {e.stderr}")
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Warning: {len(failed_packages)} packages failed to install:")
        for pkg in failed_packages:
            print(f"   - {pkg}")
        print("\n   Please install these manually using: pip install <package_name>")
        return False
    
    print("\n✅ All packages installed successfully!")
    return True

def create_directory_structure():
    """Create necessary directories"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data",
        "logs",
        "exports",
        "templates",
        "static/css",
        "static/js",
        "backups",
        "config"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"   📂 Created: {directory}")
    
    print("✅ Directory structure created successfully!")

def create_config_files():
    """Create configuration files"""
    print("\n⚙️  Creating configuration files...")
    
    # Create requirements.txt
    requirements_content = """# AI Algo Trading Platform Requirements
# Core packages for algorithmic trading system

# Data and Analysis
yfinance>=0.2.18
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0

# Web Framework and Real-time Communication
flask>=2.3.0
flask-socketio>=5.3.0
python-socketio>=5.8.0
python-engineio>=4.7.0

# Visualization and Charts
plotly>=5.15.0

# Scheduling and Time Management
schedule>=1.2.0
pytz>=2023.3

# HTTP Requests
requests>=2.31.0

# Database (SQLite is built-in)
# Additional packages for production:
# psycopg2-binary>=2.9.0  # PostgreSQL
# pymongo>=4.4.0          # MongoDB
"""
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content)
    
    # Create config.json
    config_content = """{
    "trading": {
        "total_capital": 100000,
        "risk_per_trade": 0.02,
        "lot_size": 75,
        "fees_per_order": 64,
        "max_trades_per_day": 5,
        "max_open_positions": 3,
        "cooldown_minutes": 30
    },
    "market": {
        "symbol": "^NSEI",
        "market_open": "09:15",
        "market_close": "15:30",
        "eod_exit": "15:15"
    },
    "strategies": {
        "enabled": [
            "Mean Reversion",
            "Momentum Breakout",
            "Volatility Regime"
        ],
        "risk_levels": {
            "conservative": ["Mean Reversion", "Pair Trading", "Arbitrage"],
            "moderate": ["Volatility Regime", "Options Flow"],
            "aggressive": ["Momentum Breakout", "News Based", "Scalping", "Machine Learning"]
        }
    },
    "dashboard": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": false,
        "auto_export": true
    },
    "database": {
        "type": "sqlite",
        "filename": "algo_trading.db"
    }
}"""
    
    with open("config/config.json", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    # Create run script
    run_script_content = """#!/usr/bin/env python3
# RUN.PY - Main application launcher
# Run this script to start the complete trading system

import sys
import os
import threading
import time
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def start_trading_engine():
    \"\"\"Start the trading engine\"\"\"
    try:
        from trading_engine import TradingEngine
        print("🚀 Starting Trading Engine...")
        
        engine = TradingEngine()
        engine.start()
        
        # Keep running
        while engine.is_running:
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("🛑 Trading engine stopped by user")
        if 'engine' in locals():
            engine.stop()
    except Exception as e:
        print(f"❌ Error in trading engine: {e}")

def start_dashboard():
    \"\"\"Start the web dashboard\"\"\"
    try:
        from dashboard_app import app, socketio
        print("🌐 Starting Web Dashboard on http://localhost:5000")
        
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"❌ Error in dashboard: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🤖 AI ALGORITHMIC TRADING PLATFORM")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "engine":
            start_trading_engine()
        elif sys.argv[1] == "dashboard":
            start_dashboard()
        elif sys.argv[1] == "both":
            # Start both in separate threads
            engine_thread = threading.Thread(target=start_trading_engine)
            dashboard_thread = threading.Thread(target=start_dashboard)
            
            engine_thread.daemon = True
            dashboard_thread.daemon = True
            
            engine_thread.start()
            time.sleep(2)  # Give engine time to start
            dashboard_thread.start()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("🛑 Shutting down system...")
        else:
            print("Usage: python run.py [engine|dashboard|both]")
    else:
        print("Select mode:")
        print("1. Trading Engine Only")
        print("2. Dashboard Only") 
        print("3. Both (Recommended)")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            start_trading_engine()
        elif choice == "2":
            start_dashboard()
        elif choice == "3":
            # Start both
            engine_thread = threading.Thread(target=start_trading_engine)
            dashboard_thread = threading.Thread(target=start_dashboard)
            
            engine_thread.daemon = True
            dashboard_thread.daemon = True
            
            engine_thread.start()
            time.sleep(2)
            dashboard_thread.start()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("🛑 Shutting down system...")
        else:
            print("Invalid choice. Exiting.")
"""
    
    with open("run.py", "w", encoding="utf-8") as f:
        f.write(run_script_content)
    
    # Create HTML template
    html_template = open("dashboard_template.html", "r").read() if os.path.exists("dashboard_template.html") else ""
    
    with open("templates/dashboard.html", "w", encoding="utf-8") as f:
        f.write(html_template if html_template else "<!-- Dashboard template will be created -->")
    
    print("✅ Configuration files created successfully!")

def create_startup_scripts():
    """Create platform-specific startup scripts"""
    print("\n🚀 Creating startup scripts...")
    
    # Windows batch file
    windows_script = """@echo off
title AI Algo Trading Platform
echo ================================
echo AI ALGORITHMIC TRADING PLATFORM
echo ================================
echo.
echo Starting the trading system...
echo.
python run.py both
pause
"""
    
    with open("start_trading.bat", "w", encoding="utf-8") as f:
        f.write(windows_script)
    
    # Linux/Mac shell script
    unix_script = """#!/bin/bash
clear
echo "================================"
echo "AI ALGORITHMIC TRADING PLATFORM"
echo "================================"
echo ""
echo "Starting the trading system..."
echo ""
python3 run.py both
"""
    
    with open("start_trading.sh", "w", encoding="utf-8") as f:
        f.write(unix_script)
    
    # Make shell script executable on Unix systems
    if platform.system() != "Windows":
        try:
            os.chmod("start_trading.sh", 0o755)
        except:
            pass
    
    print("✅ Startup scripts created!")

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    test_modules = [
        ("yfinance", "yfinance"),
        ("pandas", "pd"),
        ("numpy", "np"),
        ("scipy", "scipy"),
        ("flask", "Flask"),
        ("plotly", "plotly"),
        ("schedule", "schedule"),
        ("pytz", "pytz")
    ]
    
    failed_imports = []
    
    for module_name, import_name in test_modules:
        try:
            if import_name == "Flask":
                exec(f"from flask import {import_name}")
            else:
                exec(f"import {import_name}")
            print(f"   ✅ {module_name}")
        except ImportError as e:
            print(f"   ❌ {module_name} - {e}")
            failed_imports.append(module_name)
    
    if failed_imports:
        print(f"\n⚠️  Warning: {len(failed_imports)} modules failed to import")
        return False
    
    print("\n✅ All modules imported successfully!")
    return True

def create_sample_data():
    """Create sample configuration and data files"""
    print("\n📋 Creating sample data files...")
    
    # Create sample trading log
    log_content = """# AI Algo Trading Platform - Sample Log
# This file will contain trading logs when the system runs

[INFO] System initialized
[INFO] Market data feed started
[INFO] Trading strategies loaded
[INFO] Dashboard ready at http://localhost:5000
"""
    
    with open("logs/sample.log", "w",encoding="utf-8") as f:
        f.write(log_content)
    
    # Create README file
    readme_content = """# AI Algorithmic Trading Platform

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

Edit `config/config.json` to customize:
- Capital amount
- Risk parameters
- Strategy selection
- Market timing

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
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ Sample data files created!")

def setup_complete():
    """Display setup completion message"""
    print("\n" + "="*80)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*80)
    print()
    print("Your AI Algorithmic Trading Platform is ready!")
    print()
    print("📋 Next Steps:")
    print("   1. Review configuration in config/config.json")
    print("   2. Start the system:")
    print("      • Windows: Double-click start_trading.bat")
    print("      • Linux/Mac: ./start_trading.sh")
    print("      • Manual: python run.py both")
    print("   3. Open dashboard: http://localhost:5000")
    print()
    print("📚 Documentation: README.md")
    print("📊 Dashboard: http://localhost:5000")
    print("📁 Trade exports: exports/ directory")
    print("📝 Logs: logs/ directory")
    print()
    print("⚠️  Important: This is a PAPER TRADING system.")
    print("   No real money is at risk. Perfect for learning!")
    print()
    print("🚀 Happy Trading!")
    print("="*80)

def main():
    """Main setup function"""
    try:
        print_header()
        
        # Check Python version
        if not check_python_version():
            return False
        
        # Install requirements
        if not install_requirements():
            print("\n⚠️  Some packages failed to install. You may need to install them manually.")
            print("   Continue anyway? (y/n): ", end="")
            if input().lower() != 'y':
                return False
        
        # Create directory structure
        create_directory_structure()
        
        # Create configuration files
        create_config_files()
        
        # Create startup scripts
        create_startup_scripts()
        
        # Test imports
        if not test_imports():
            print("\n⚠️  Some modules failed to import. Please check your installation.")
            print("   Continue anyway? (y/n): ", end="")
            if input().lower() != 'y':
                return False
        
        # Create sample data
        create_sample_data()
        
        # Setup complete
        setup_complete()
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n🛑 Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)