#!/usr/bin/env python3
# MAIN_LAUNCHER.PY - Complete AI Algo Trading Platform Launcher
# Comprehensive launcher for the AI Algorithmic Trading Platform
# Version: 1.0

import os
import sys
import json
import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import io
import sys
# Configure UTF-8 encoding for output streams (Windows compatibility)
try:
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except (AttributeError, OSError):
    # Fallback for environments where reconfiguration isn't needed/possible
    pass

# Ensure we can import our modules
sys.path.append(str(Path(__file__).parent))

BASIC_DIRS = ["data", "logs", "exports", "config"]

def print_banner():
    """Print application banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║     AI ALGORITHMIC TRADING PLATFORM                                       ║
║                                                                              ║
║    Advanced Paper Trading System for Indian Options Market                  ║
║    ✅ Real-time Data • ✅ 9+ Strategies • ✅ Risk Management                ║
║    ✅ Web Dashboard • ✅ Performance Analytics • ✅ CSV Export               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)

def check_dependencies(verbose: bool = True) -> bool:
    """Check if all required dependencies are available"""
    if verbose:
        print("🔍 Checking dependencies...")
    required_modules = [
        ('yfinance', 'Market data'),
        ('pandas', 'Data analysis'),
        ('numpy', 'Numerical computing'),
        ('flask', 'Web framework'),
        ('plotly', 'Charts and visualization'),
        ('schedule', 'Task scheduling'),
        ('sqlite3', 'Database (built-in)')
    ]
    missing_modules = []
    for module, description in required_modules:
        try:
            if module == 'sqlite3':
                import sqlite3  # noqa: F401
            else:
                __import__(module)
            if verbose:
                print(f"   ✅ {module:<12} - {description}")
        except ImportError:
            if verbose:
                print(f"   ❌ {module:<12} - {description} (MISSING)")
            missing_modules.append(module)
    if missing_modules and verbose:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_modules)}")
        print("   Run: pip install " + ' '.join(missing_modules))
    return len(missing_modules) == 0

def load_config(config_path: Path | None = None):
    """Load configuration from config file"""
    if config_path is None:
        config_path = Path("config/config.json")
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        print("   Using default configuration")
        return get_default_config()


def write_config(config: dict, config_path: Path | None = None):
    """Write configuration to file (creates config directory if needed)"""
    if config_path is None:
        config_path = Path("config/config.json")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Config written to: {config_path}")
    except Exception as e:
        print(f"❌ Failed to write config: {e}")

def get_default_config():
    """Get default configuration"""
    return {
        "trading": {
            "total_capital": 100000,
            "risk_per_trade": 0.02,
            "max_trades_per_day": 5,
            "market": "NSE",
            "symbol_universe": ["NIFTY", "BANKNIFTY"],
            "paper_trading": True
        },
        "dashboard": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": False
        },
        "storage": {
            "db_path": "algo_trading.db",
            "export_dir": "exports"
        },
        "logging": {
            "level": "INFO",
            "log_dir": "logs"
        }
    }

def ensure_dirs():
    """Ensure required directories exist"""
    for d in BASIC_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)

def start_trading_engine():
    """Start the trading engine"""
    try:
        print("Starting Trading Engine...")
        from trading_engine import TradingEngine  # user-implemented
        engine = TradingEngine()
        engine.start()
        print("✅ Trading Engine started successfully")
        print("💰 Paper trading active - no real money at risk")
        print("📊 Monitor performance via web dashboard")
        # Keep the engine running
        try:
            while getattr(engine, "is_running", True):
                time.sleep(60)  # Check every minute
                # Print status every 5 minutes
                now = datetime.now()
                if now.minute % 5 == 0:
                    try:
                        status = engine.get_status()
                        print(
                            f"🕒 {now:%Y-%m-%d %H:%M} | "
                            f"💼 Capital: ₹{status.get('current_capital', 0):,.0f} | "
                            f"P&L: ₹{status.get('total_pnl', 0):,.0f} | "
                            f"Trades: {status.get('total_trades', 0)} | "
                            f"Positions: {status.get('open_positions', 0)}"
                        )
                    except Exception as e:
                        print(f"⚠️  Status check failed: {e}")
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested by user")
            try:
                engine.stop()
            except Exception:
                pass
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure trading_engine.py (with TradingEngine) is available")
    except Exception as e:
        print(f"❌ Error starting trading engine: {e}")
        import traceback
        traceback.print_exc()

def start_dashboard(config):
    """Start the web dashboard"""
    try:
        print("🌐 Starting Web Dashboard...")
        from dashboard_app import app, socketio, start_dashboard_updates  # user-implemented
        host = config.get('dashboard', {}).get('host', '0.0.0.0')
        port = config.get('dashboard', {}).get('port', 5000)
        debug = config.get('dashboard', {}).get('debug', False)
        print(f"📱 Dashboard URL: http://localhost:{port}")
        print("🎛️  Dashboard features:")
        print("   • Real-time performance monitoring")
        print("   • Strategy comparison and analysis")
        print("   • Trade history and export")
        print("   • System control (start/stop)")
        # Start background updates
        try:
            start_dashboard_updates()
        except TypeError:
            # Allow dashboards that don't require a call signature
            start_dashboard_updates  # noqa: B018 (noop ref)
        # Run the dashboard (blocking)
        socketio.run(app, host=host, port=port, debug=debug)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure dashboard_app.py is available (Flask-SocketIO app)")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        import traceback
        traceback.print_exc()

def run_backtest():
    """Run historical backtest"""
    try:
        print("📈 Running Historical Backtest...")
        from algo_trading_main import run_enhanced_backtest  # user-implemented
        print("🔄 This will run the backtest on historical data")
        print("📊 Results will be saved to CSV file")
        run_enhanced_backtest()
        print("✅ Backtest finished")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure algo_trading_main.py (run_enhanced_backtest) is available")
    except Exception as e:
        print(f"❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()

def run_analysis():
    """Run comprehensive strategy analysis"""
    try:
        print("📊 Running Strategy Analysis...")
        from strategy_analyzer import run_comprehensive_analysis  # user-implemented
        run_comprehensive_analysis()
        print("✅ Analysis finished")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Please ensure strategy_analyzer.py (run_comprehensive_analysis) is available")
    except Exception as e:
        print(f"❌ Error running analysis: {e}")
        import traceback
        traceback.print_exc()

def setup_system():
    """Setup the system"""
    try:
        print("⚙️  Running System Setup...")
        ensure_dirs()
        default_cfg = get_default_config()
        cfg_path = Path("config/config.json")
        if not cfg_path.exists():
            write_config(default_cfg, cfg_path)
        # Optional: run user-provided setup if exists
        try:
            from setup import main as setup_main  # user-implemented optional
            setup_main()
        except ImportError:
            print("ℹ️  setup.py not found — base directories/config created.")
        print("✅ System setup complete.")
    except Exception as e:
        print(f"❌ Error in setup: {e}")
        import traceback
        traceback.print_exc()

def start_both_services(config):
    """Start both trading engine and dashboard"""
    print("Starting Complete Trading System...")
    # Don't use daemon=True to avoid signal handler issues
    engine_thread = threading.Thread(target=start_trading_engine)
    engine_thread.start()
    print("Waiting for trading engine to initialize...")
    time.sleep(3)
    start_dashboard(config)

def show_system_status():
    """Show current system status"""
    print("\n📋 SYSTEM STATUS")
    print("-"*40)
    # Check DB
    db_path = Path("algo_trading.db")
    if db_path.exists():
        print(f"✅ Database: {db_path} ({db_path.stat().st_size} bytes)")
        try:
            from algo_trading_main import DatabaseManager  # user-implemented
            db = DatabaseManager()
            recent_trades = db.get_recent_trades(limit=10)
            print(f"📊 Recent trades: {len(recent_trades)}")
        except Exception as e:
            print(f"⚠️  Database access issue: {e}")
    else:
        print("❌ Database: Not found (no trades yet)")
    # Check dirs
    for directory in BASIC_DIRS:
        path = Path(directory)
        if path.exists():
            files = list(path.glob('*'))
            print(f"✅ {directory}/: {len(files)} file(s)")
        else:
            print(f"❌ {directory}/: Not found")
    # Check config
    config_path = Path("config/config.json")
    if config_path.exists():
        print(f"✅ Configuration: Available")
    else:
        print(f"❌ Configuration: Not found")
    print("\n💡 Tip: Run option 6 (System Setup) if any components are missing")

def show_help():
    """Show help information"""
    help_text = """
🆘 AI ALGO TRADING PLATFORM - HELP

OVERVIEW:
This is a paper trading system for the Indian options market. It uses real market data
but doesn't execute actual trades — perfect for learning and strategy testing.

 QUICK START:
1) Create a virtual environment and install dependencies:
   python -m venv .venv && . .venv/bin/activate  (Windows: .venv\\Scripts\\activate)
   pip install -r requirements.txt
   (or run the launcher to see what's missing)

2) Initialize folders & default config:
   python MAIN_LAUNCHER.py --mode setup

3) Start everything (engine + dashboard):
   python MAIN_LAUNCHER.py --mode both

4) Open the dashboard URL shown in the console.

🧭 MODES:
  --mode engine     Start the paper trading engine only
  --mode dashboard  Start the web dashboard only
  --mode both       Start engine and dashboard together
  --mode backtest   Run historical backtest (writes CSV/plots)
  --mode analysis   Run strategy analysis
  --mode setup      Create required folders & default config
  --mode status     Print current system status
  --mode menu       Interactive menu (TUI)

⚙️ FILES EXPECTED (user-implemented):
  trading_engine.py       -> class TradingEngine { start(), stop(), get_status(), is_running }
  dashboard_app.py        -> Flask-SocketIO app: (app, socketio, start_dashboard_updates)
  algo_trading_main.py    -> functions: run_enhanced_backtest(), class DatabaseManager
  strategy_analyzer.py    -> function: run_comprehensive_analysis()
  setup.py (optional)     -> function: main()

🪪 LOGS & DATA:
  - logs/        runtime logs
  - data/        cached or downloaded datasets
  - exports/     CSV/plot exports
  - config/      configuration files (config.json)
  - algo_trading.db  SQLite database (created on first trade)

🧰 TROUBLESHOOTING:
  • Missing module -> pip install <name> (see dependency check output)
  • Dashboard import error -> verify dashboard_app.py exposes app/socketio
  • Engine import error -> verify TradingEngine class exists and can start
  • Backtest stuck -> confirm network/data source and error messages
  • Windows port in use -> change dashboard.port in config/config.json

📣 NOTE:
This launcher never auto-runs backtests from the main entry point. You must choose a
specific mode or use the interactive menu.
"""
    print(help_text)

def show_interactive_menu():
    """Show interactive menu"""
    while True:
        print("\n" + "="*60)
        print("🤖 AI ALGO TRADING PLATFORM - MAIN MENU")
        print("="*60)
        print("1. 🚀 Start Trading Engine (Paper Trading)")
        print("2. 🌐 Start Web Dashboard")
        print("3. 🔥 Start Both (Recommended)")
        print("4. 📈 Run Historical Backtest")
        print("5. 📊 Run Strategy Analysis")
        print("6. ⚙️  System Setup")
        print("7. 📋 System Status")
        print("8. 🆘 Help")
        print("9. ❌ Exit")
        print("-"*60)
        choice = input("Enter your choice (1-9): ").strip()
        if choice == '1':
            start_trading_engine()
        elif choice == '2':
            config = load_config()
            start_dashboard(config)
        elif choice == '3':
            config = load_config()
            start_both_services(config)
        elif choice == '4':
            run_backtest()
        elif choice == '5':
            run_analysis()
        elif choice == '6':
            setup_system()
        elif choice == '7':
            show_system_status()
        elif choice == '8':
            show_help()
        elif choice == '9':
            print("👋 Thank you for using AI Algo Trading Platform!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def parse_args():
    parser = argparse.ArgumentParser(
        prog="MAIN_LAUNCHER.py",
        description="AI Algorithmic Trading Platform Launcher (Paper Trading for Indian Options)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["engine", "dashboard", "both", "backtest", "analysis", "setup", "status", "menu"],
        default="menu",
        help="Which component(s) to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.json",
        help="Path to config JSON"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Hide ASCII banner on start"
    )
    parser.add_argument(
        "--skip-dep-check",
        action="store_true",
        help="Skip the dependency check"
    )
    return parser.parse_args()

def main():
    # Parsing command-line arguments
    args = parse_args()
    
    if not args.no_banner:
        print_banner()
    
    ensure_dirs()

    # Check dependencies
    if not args.skip_dep_check:
        deps_ok = check_dependencies(verbose=True)
        if not deps_ok and args.mode in {"engine", "dashboard", "both", "backtest", "analysis"}:
            print("❌ Cannot continue: required dependencies missing. Use --mode setup or install deps.")
            sys.exit(2)

    # Load the configuration
    cfg_path = Path(args.config) if args.config else Path("config/config.json")
    config = load_config(cfg_path)  # This will handle config loading and defaulting

    try:
        # Execute based on selected mode
        if args.mode == "engine":
            start_trading_engine()
        elif args.mode == "dashboard":
            start_dashboard(config)
        elif args.mode == "both":
            start_both_services(config)
        elif args.mode == "backtest":
            run_backtest()
        elif args.mode == "analysis":
            run_analysis()
        elif args.mode == "setup":
            setup_system()
        elif args.mode == "status":
            show_system_status()
        elif args.mode == "menu":
            show_interactive_menu()
        else:
            show_help()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting gracefully.")
    except Exception as e:
        print(f"❌ Unhandled error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
