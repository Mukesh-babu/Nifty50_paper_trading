#!/usr/bin/env python3
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
    """Start the trading engine"""
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
    """Start the web dashboard"""
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
