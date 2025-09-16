# DASHBOARD_APP.PY - Real-time Trading Dashboard
# Web-based dashboard for monitoring algo trading system
# Version: 1.1

from flask import Flask, render_template, jsonify, request, send_file
from flask_socketio import SocketIO, emit
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import plotly.graph_objs as go
import plotly.utils
import time
import sys
import threading

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent

# --- Imports & logger fallback ---
import logging
logger = None
try:
    from trading_engine import TradingEngine  # single import, used everywhere
    from algo_trading_main import DatabaseManager, export_trades_to_csv, logger as ext_logger
    logger = ext_logger

except ImportError as e:
    print("Error: Please ensure algo_trading_main.py is available:", e)
    # Fallback logger
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    logger = logging.getLogger("dashboard_fallback")

    # we still attempt to run with a best-effort DB manager
    try:
        from algo_trading_main import DatabaseManager, export_trades_to_csv
    except Exception as e2:
        logger.error("Critical: Could not import DatabaseManager/export_trades_to_csv: %s", e2)
        sys.exit(1)

# --- Flask / SocketIO ---
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config['SECRET_KEY'] = 'algo_trading_secret_key_2024'

# Force threading backend to avoid eventlet/gevent mismatches unless you explicitly install/configure them.
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# --- Globals ---
trading_engine = None
_engine_lock = threading.RLock()
db_manager = DatabaseManager()

# ============= Helpers =============

def _df_safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True) if df is not None and not df.empty else pd.DataFrame()

def _iso(dt) -> str:
    try:
        if pd.isna(dt):
            return None
        if isinstance(dt, str):
            return dt
        if isinstance(dt, (pd.Timestamp, datetime)):
            return dt.isoformat()
        return str(dt)
    except Exception:
        return None

def _get_live_status_from_engine():
    with _engine_lock:
        if trading_engine and getattr(trading_engine, "is_running", False):
            try:
                return trading_engine.get_status()
            except Exception as e:
                logger.warning("get_status() failed; falling back to DB: %s", e)
    return None

def _get_status_from_db():
    """
    OPTIONAL: If you persist status snapshots in your DB, fetch the latest here.
    If you don't, consider deriving from recent trades P&L as a weaker fallback.
    """
    status = {
        'is_running': False,
        'current_capital': 100000,
        'total_pnl': 0,
        'total_trades': 0,
        'winning_trades': 0,
        'win_rate': 0.0,
        'open_positions': 0,
        'current_price': 0
    }
    try:
        trades_df = db_manager.get_recent_trades(limit=1000)
        if trades_df is not None and not trades_df.empty:
            tdf = _df_safe_copy(trades_df)
            total_trades = len(tdf)
            total_pnl = float(tdf["pnl"].sum())
            wins = int((tdf["pnl"] > 0).sum())
            status.update({
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'winning_trades': wins,
                'win_rate': (wins / total_trades) * 100 if total_trades else 0.0,
            })
    except Exception as e:
        logger.debug("DB status fallback failed: %s", e)
    return status

# ============= Charts =============

def create_performance_chart(trades_df: pd.DataFrame):
    """Create performance chart using Plotly (cumulative P&L vs trade # and vs time)."""
    tdf = _df_safe_copy(trades_df)
    if tdf.empty or "pnl" not in tdf.columns:
        return json.dumps({})

    # Prefer a chronological sort (entry or exit timestamp if present)
    time_col = "exit_timestamp" if "exit_timestamp" in tdf.columns else "entry_timestamp"
    if time_col in tdf.columns:
        tdf[time_col] = pd.to_datetime(tdf[time_col], errors="coerce")
        tdf = tdf.sort_values(time_col)
    else:
        tdf = tdf.reset_index(drop=True)

    tdf["cumulative_pnl"] = tdf["pnl"].cumsum()
    tdf["trade_num"] = range(1, len(tdf) + 1)
    tdf["time_x"] = tdf[time_col] if time_col in tdf.columns else tdf["trade_num"]

    trace_trade_num = go.Scatter(
        x=tdf["trade_num"],
        y=tdf["cumulative_pnl"],
        mode='lines+markers',
        name='Cumulative P&L (by trade #)',
        line=dict(width=3),
        marker=dict(size=6)
    )

    trace_time = go.Scatter(
        x=tdf["time_x"],
        y=tdf["cumulative_pnl"],
        mode='lines',
        name='Cumulative P&L (by time)',
        line=dict(width=2, dash="dot")
    )

    layout = go.Layout(
        title='Trading Performance',
        xaxis=dict(title='Trade # / Time'),
        yaxis=dict(title='Cumulative P&L (₹)'),
        hovermode='x unified',
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig = go.Figure(data=[trace_trade_num, trace_time], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_strategy_performance_chart():
    """Create strategy performance comparison chart."""
    strategy_df = _df_safe_copy(db_manager.get_strategy_performance())
    if strategy_df.empty or "strategy" not in strategy_df.columns:
        return json.dumps({})

    strategy_df["total_pnl"] = strategy_df.get("total_pnl", 0.0)

    trace1 = go.Bar(
        x=strategy_df['strategy'],
        y=strategy_df['total_pnl'],
        name='Total P&L'
    )

    layout = go.Layout(
        title='Strategy Performance Comparison',
        xaxis=dict(title='Strategy'),
        yaxis=dict(title='Total P&L (₹)'),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig = go.Figure(data=[trace1], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_hourly_performance_chart():
    """Create hourly performance analysis chart."""
    trades_df = _df_safe_copy(db_manager.get_recent_trades(limit=1000))
    if trades_df.empty:
        return json.dumps({})

    if "entry_timestamp" in trades_df.columns:
        trades_df["entry_timestamp"] = pd.to_datetime(trades_df["entry_timestamp"], errors="coerce")
        trades_df["entry_hour"] = trades_df["entry_timestamp"].dt.hour
    else:
        trades_df["entry_hour"] = 0

    hourly_pnl = trades_df.groupby('entry_hour', as_index=False)['pnl'].sum()

    trace1 = go.Bar(
        x=hourly_pnl['entry_hour'],
        y=hourly_pnl['pnl'],
        name='Hourly P&L'
    )

    layout = go.Layout(
        title='Hourly Trading Performance',
        xaxis=dict(title='Hour of Day', tickmode='linear', dtick=1),
        yaxis=dict(title='Total P&L (₹)'),
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )

    fig = go.Figure(data=[trace1], layout=layout)
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ============= Routes =============

@app.route('/')
def dashboard():
    """Main dashboard page"""
    template_path = BASE_DIR / "templates" / "dashboard.html"
    if not template_path.exists():
        template_path.parent.mkdir(parents=True, exist_ok=True)
        with template_path.open('w', encoding='utf-8') as f:
            f.write(get_dashboard_html())
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.warning("Template render failed, falling back to inline HTML: %s", e)
        return get_dashboard_html()

def get_dashboard_html():
    """Return basic dashboard HTML if template is missing"""
    return '''<!DOCTYPE html>
<html><head><title>AI Trading Dashboard</title>
<style>body{background:#1a1a1a;color:#fff;font-family:Arial;margin:24px}</style>
</head><body>
<h1>AI Trading Dashboard</h1>
<div id="status">Loading...</div>
<script>
setInterval(function(){
    fetch('/api/status').then(r=>r.json()).then(data=>{
        document.getElementById('status').innerHTML =
          'Running: ' + (data.is_running ? 'Yes' : 'No') +
          ' | Capital: ' + (data.current_capital || 0) +
          ' | PnL: ' + (data.total_pnl || 0) +
          ' | Trades: ' + (data.total_trades || 0);
    });
}, 5000);
</script>
</body></html>'''

@app.route('/api/status')
def get_status():
    """Get current system status"""
    # 1) Try engine (in-process)
    status = _get_live_status_from_engine()
    # 2) Fallback to DB if engine not in-process
    if status is None:
        status = _get_status_from_db()
        status['error'] = 'Trading engine not connected to this process'
    return jsonify(status)

@app.route('/api/trades')
def get_trades():
    """Get recent trades"""
    try:
        limit = request.args.get('limit', 100, type=int)
        trades_df = _df_safe_copy(db_manager.get_recent_trades(limit=limit))
        if trades_df.empty:
            return jsonify({'trades': []})

        # Ensure JSON-serializable timestamps
        for col in ("entry_timestamp", "exit_timestamp"):
            if col in trades_df.columns:
                trades_df[col] = pd.to_datetime(trades_df[col], errors="coerce").map(_iso)

        trades_list = trades_df.to_dict('records')
        return jsonify({'trades': trades_list})

    except Exception as e:
        logger.error("Error getting trades: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/performance')
def get_performance():
    """Get performance metrics and charts"""
    try:
        trades_df = _df_safe_copy(db_manager.get_recent_trades(limit=1000))

        # Charts
        perf_chart = create_performance_chart(trades_df)
        strategy_chart = create_strategy_performance_chart()
        hourly_chart = create_hourly_performance_chart()

        # Metrics
        if trades_df.empty:
            metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_trade_pnl': 0.0,
                'max_win': 0.0,
                'max_loss': 0.0
            }
        else:
            total_trades = len(trades_df)
            winning_trades = int((trades_df['pnl'] > 0).sum())
            total_pnl = float(trades_df['pnl'].sum())
            metrics = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': (winning_trades / total_trades) * 100 if total_trades else 0.0,
                'total_pnl': total_pnl,
                'avg_trade_pnl': total_pnl / total_trades if total_trades else 0.0,
                'max_win': float(trades_df['pnl'].max()),
                'max_loss': float(trades_df['pnl'].min())
            }

        return jsonify({
            'metrics': metrics,
            'performance_chart': perf_chart,
            'strategy_chart': strategy_chart,
            'hourly_chart': hourly_chart
        })

    except Exception as e:
        logger.error("Error getting performance data: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/strategies')
def get_strategies():
    """Get strategy performance breakdown"""
    try:
        strategy_df = _df_safe_copy(db_manager.get_strategy_performance())
        if strategy_df.empty:
            return jsonify({'strategies': []})

        strategies_list = strategy_df.to_dict('records')
        return jsonify({'strategies': strategies_list})

    except Exception as e:
        logger.error("Error getting strategy data: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/export')
def export_trades():
    """Export trades to CSV"""
    try:
        filename = export_trades_to_csv(db_manager)
        if filename and os.path.exists(filename):
            return send_file(filename, as_attachment=True, download_name=os.path.basename(filename))
        else:
            return jsonify({'error': 'No trades to export or file creation failed'}), 404

    except Exception as e:
        logger.error("Error exporting trades: %s", e)
        return jsonify({'error': str(e)}), 500

@app.route('/api/control/<action>')
def control_system(action):
    """Control trading system (start/stop)"""
    global trading_engine

    try:
        if action == 'start':
            # Lazy-import here to avoid "cannot import name TradingEngine" at module import time
            if trading_engine is None:
                try:
                    from trading_engine import TradingEngine as _TradingEngine
                except Exception as ie:
                    # Give a clear error message back to the UI
                    return jsonify({'status': 'error', 'message': f'Import error for TradingEngine: {ie}'}), 500

                trading_engine = _TradingEngine()
                trading_engine.start()
                return jsonify({'status': 'success', 'message': 'Trading engine started'})

            elif not trading_engine.is_running:
                trading_engine.start()
                return jsonify({'status': 'success', 'message': 'Trading engine restarted'})
            else:
                return jsonify({'status': 'info', 'message': 'Trading engine already running'})

        elif action == 'stop':
            if trading_engine and trading_engine.is_running:
                trading_engine.stop()
                return jsonify({'status': 'success', 'message': 'Trading engine stopped'})
            else:
                return jsonify({'status': 'info', 'message': 'Trading engine not running'})

        else:
            return jsonify({'status': 'error', 'message': f'Unknown action: {action}'}), 400

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Control error: {str(e)}'}), 500

# ============= Socket.IO =============

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected to trading dashboard'})
    logger.info("Client connected to dashboard")
    # Push an immediate status & metrics snapshot on connect
    try:
        status = _get_live_status_from_engine() or _get_status_from_db()
        socketio.emit('status_update', status)
        # Optionally push metrics too
        trades_df = _df_safe_copy(db_manager.get_recent_trades(limit=1000))
        metrics_payload = {
            'metrics': {
                'total_trades': int(len(trades_df)) if not trades_df.empty else 0,
                'total_pnl': float(trades_df['pnl'].sum()) if not trades_df.empty else 0.0
            }
        }
        socketio.emit('metrics_update', metrics_payload)
    except Exception as e:
        logger.debug("Initial push on connect failed: %s", e)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("Client disconnected from dashboard")

def dashboard_update_task():
    """SocketIO-friendly background task to send real-time updates."""
    while True:
        try:
            status = _get_live_status_from_engine() or _get_status_from_db()
            socketio.emit('status_update', status)
            socketio.sleep(5)  # non-blocking sleep in SocketIO context
        except Exception as e:
            logger.warning("Dashboard update error: %s", e)
            socketio.sleep(10)

def ensure_background_task():
    """Start the SocketIO background task if not running."""
    # Using an attribute to keep a single task
    if not hasattr(ensure_background_task, "_started"):
        socketio.start_background_task(target=dashboard_update_task)
        ensure_background_task._started = True
        logger.info("Dashboard update background task started")

def start_dashboard_updates() -> bool:
    """Public helper to start Socket.IO background updates.

    Flask launchers import this symbol (see ``main_launcher.py``) to ensure
    the dashboard emits live status messages.  Older versions of this module
    did not expose the helper which caused ``ImportError`` at runtime.  The
    function is idempotent thanks to :func:`ensure_background_task` keeping an
    internal flag, so calling it repeatedly is safe.
    """

    ensure_background_task()
    return True

# ============= Main =============

if __name__ == '__main__':
    start_dashboard_updates()
    logger.info("🌐 Starting Trading Dashboard on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, use_reloader=False)
