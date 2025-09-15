# STRATEGY_ANALYZER.PY - Advanced Strategy Analysis and Comparison
# Comprehensive tool for analyzing and comparing trading strategies
# Version: 1.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from algo_trading_main import DatabaseManager, TradingConfig, export_trades_to_csv
    from advanced_strategies import StrategyFactory
except ImportError:
    print("Warning: Core modules not available. Some features may not work.")

class StrategyAnalyzer:
    """Comprehensive strategy analysis and comparison tool"""
    
    def __init__(self, db_path: str = "algo_trading.db"):
        self.db_manager = DatabaseManager(db_path)
        self.analysis_results = {}
        
    def analyze_all_strategies(self, days_back: int = 30) -> Dict:
        """Analyze performance of all strategies"""
        print("📊 Analyzing strategy performance...")
        
        # Get recent trades
        trades_df = self.db_manager.get_recent_trades(limit=10000)
        
        if trades_df.empty:
            print("❌ No trade data available for analysis")
            return {}
        
        # Filter by date range if specified
        if days_back > 0:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            trades_df['entry_date'] = pd.to_datetime(trades_df['entry_timestamp'])
            trades_df = trades_df[trades_df['entry_date'] >= cutoff_date]
        
        # Analyze each strategy
        strategy_analysis = {}
        strategies = trades_df['strategy'].unique()
        
        for strategy in strategies:
            strategy_trades = trades_df[trades_df['strategy'] == strategy]
            strategy_analysis[strategy] = self._analyze_strategy_performance(strategy_trades)
        
        # Overall portfolio analysis
        portfolio_analysis = self._analyze_portfolio_performance(trades_df)
        
        self.analysis_results = {
            'strategies': strategy_analysis,
            'portfolio': portfolio_analysis,
            'comparison': self._compare_strategies(strategy_analysis),
            'risk_metrics': self._calculate_risk_metrics(trades_df),
            'recommendations': self._generate_recommendations(strategy_analysis)
        }
        
        return self.analysis_results
    
    def _analyze_strategy_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze performance metrics for a single strategy"""
        if trades_df.empty:
            return self._empty_analysis()
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] <= 0])
        
        total_pnl = trades_df['pnl'].sum()
        avg_pnl = trades_df['pnl'].mean()
        
        win_rate = (winning_trades / total_trades) * 100
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
        
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Calculate consecutive wins/losses
        trades_df_sorted = trades_df.sort_values('entry_timestamp')
        wins_losses = (trades_df_sorted['pnl'] > 0).astype(int)
        
        max_consecutive_wins = self._max_consecutive(wins_losses, 1)
        max_consecutive_losses = self._max_consecutive(wins_losses, 0)
        
        # Calculate drawdown
        cumulative_pnl = trades_df_sorted['pnl'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        
        # Average holding time
        avg_holding_time = trades_df['holding_minutes'].mean()
        
        # Volatility of returns
        returns_volatility = trades_df['pnl'].std()
        
        # Sharpe ratio (simplified)
        if returns_volatility > 0:
            sharpe_ratio = avg_pnl / returns_volatility
        else:
            sharpe_ratio = 0
        
        # Recovery factor
        recovery_factor = total_pnl / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = total_pnl / abs(max_drawdown) if max_drawdown != 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'max_drawdown': max_drawdown,
            'avg_holding_time': avg_holding_time,
            'returns_volatility': returns_volatility,
            'sharpe_ratio': sharpe_ratio,
            'recovery_factor': recovery_factor,
            'calmar_ratio': calmar_ratio,
            'expectancy': (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)
        }
    
    def _analyze_portfolio_performance(self, trades_df: pd.DataFrame) -> Dict:
        """Analyze overall portfolio performance"""
        if trades_df.empty:
            return self._empty_analysis()
        
        # Daily P&L analysis
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_timestamp']).dt.date
        daily_pnl = trades_df.groupby('entry_date')['pnl'].sum()
        
        profitable_days = len(daily_pnl[daily_pnl > 0])
        total_days = len(daily_pnl)
        
        best_day = daily_pnl.max()
        worst_day = daily_pnl.min()
        
        avg_daily_pnl = daily_pnl.mean()
        daily_volatility = daily_pnl.std()
        
        # Calculate portfolio metrics
        portfolio_analysis = self._analyze_strategy_performance(trades_df)
        portfolio_analysis.update({
            'trading_days': total_days,
            'profitable_days': profitable_days,
            'daily_win_rate': (profitable_days / total_days) * 100,
            'best_day': best_day,
            'worst_day': worst_day,
            'avg_daily_pnl': avg_daily_pnl,
            'daily_volatility': daily_volatility,
            'daily_sharpe': avg_daily_pnl / daily_volatility if daily_volatility > 0 else 0
        })
        
        return portfolio_analysis
    
    def _compare_strategies(self, strategy_analysis: Dict) -> Dict:
        """Compare strategies and rank them"""
        if not strategy_analysis:
            return {}
        
        comparison_metrics = ['total_pnl', 'win_rate', 'profit_factor', 'sharpe_ratio', 'recovery_factor']
        comparison_data = []
        
        for strategy, metrics in strategy_analysis.items():
            row = {'strategy': strategy}
            for metric in comparison_metrics:
                row[metric] = metrics.get(metric, 0)
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Rank strategies
        rankings = {}
        for metric in comparison_metrics:
            if metric in ['max_drawdown']:  # Lower is better
                rankings[f'{metric}_ranking'] = comparison_df[metric].rank(ascending=True)
            else:  # Higher is better
                rankings[f'{metric}_ranking'] = comparison_df[metric].rank(ascending=False)
        
        # Overall score (weighted average of rankings)
        weights = {
            'total_pnl_ranking': 0.25,
            'win_rate_ranking': 0.20,
            'profit_factor_ranking': 0.20,
            'sharpe_ratio_ranking': 0.20,
            'recovery_factor_ranking': 0.15
        }
        
        overall_score = sum(rankings[metric] * weight for metric, weight in weights.items())
        comparison_df['overall_ranking'] = overall_score.rank(ascending=True)
        
        return {
            'comparison_table': comparison_df.to_dict('records'),
            'best_strategy': comparison_df.loc[comparison_df['overall_ranking'].idxmin(), 'strategy'],
            'worst_strategy': comparison_df.loc[comparison_df['overall_ranking'].idxmax(), 'strategy'],
            'rankings': rankings
        }
    
    def _calculate_risk_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """Calculate advanced risk metrics"""
        if trades_df.empty:
            return {}
        
        returns = trades_df['pnl'].values
        
        # Value at Risk (VaR)
        var_95 = np.percentile(returns, 5)  # 95% VaR
        var_99 = np.percentile(returns, 1)  # 99% VaR
        
        # Conditional Value at Risk (CVaR)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
        
        # Maximum Adverse Excursion
        max_adverse_excursion = trades_df['max_loss'].mean() if 'max_loss' in trades_df.columns else 0
        
        # Maximum Favorable Excursion  
        max_favorable_excursion = trades_df['max_profit'].mean() if 'max_profit' in trades_df.columns else 0
        
        # Skewness and Kurtosis
        skewness = pd.Series(returns).skew()
        kurtosis = pd.Series(returns).kurtosis()
        
        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'cvar_95': cvar_95,
            'cvar_99': cvar_99,
            'max_adverse_excursion': max_adverse_excursion,
            'max_favorable_excursion': max_favorable_excursion,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'sortino_ratio': sortino_ratio,
            'downside_deviation': downside_deviation
        }
    
    def _generate_recommendations(self, strategy_analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        if not strategy_analysis:
            recommendations.append("❌ No strategy data available for analysis")
            return recommendations
        
        # Find best and worst performing strategies
        strategies_by_pnl = sorted(strategy_analysis.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        
        if len(strategies_by_pnl) > 0:
            best_strategy = strategies_by_pnl[0]
            worst_strategy = strategies_by_pnl[-1]
            
            recommendations.append(f"🏆 Best performing strategy: {best_strategy[0]} (₹{best_strategy[1]['total_pnl']:.2f} P&L)")
            
            if len(strategies_by_pnl) > 1:
                recommendations.append(f"⚠️  Worst performing strategy: {worst_strategy[0]} (₹{worst_strategy[1]['total_pnl']:.2f} P&L)")
        
        # Analyze win rates
        high_win_rate_strategies = [name for name, metrics in strategy_analysis.items() if metrics['win_rate'] > 60]
        if high_win_rate_strategies:
            recommendations.append(f"✅ High win rate strategies (>60%): {', '.join(high_win_rate_strategies)}")
        
        # Identify risky strategies
        high_drawdown_strategies = [name for name, metrics in strategy_analysis.items() if metrics['max_drawdown'] < -1000]
        if high_drawdown_strategies:
            recommendations.append(f"🚨 High drawdown strategies (>₹1000): {', '.join(high_drawdown_strategies)}")
        
        # Capital allocation recommendations
        profitable_strategies = [name for name, metrics in strategy_analysis.items() if metrics['total_pnl'] > 0]
        if profitable_strategies:
            recommendations.append(f"💰 Focus capital on profitable strategies: {', '.join(profitable_strategies)}")
        
        # Risk management recommendations
        avg_holding_times = [metrics['avg_holding_time'] for metrics in strategy_analysis.values()]
        if avg_holding_times:
            avg_holding = np.mean(avg_holding_times)
            if avg_holding > 120:  # More than 2 hours
                recommendations.append("⏰ Consider reducing holding times - average is >2 hours")
            elif avg_holding < 30:  # Less than 30 minutes
                recommendations.append("⏰ Very short holding times - consider if fees are eating profits")
        
        return recommendations
    
    def _max_consecutive(self, series: pd.Series, value: int) -> int:
        """Calculate maximum consecutive occurrences of a value"""
        max_consecutive = 0
        current_consecutive = 0
        
        for val in series:
            if val == value:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_win': 0,
            'max_loss': 0,
            'profit_factor': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'max_drawdown': 0,
            'avg_holding_time': 0,
            'returns_volatility': 0,
            'sharpe_ratio': 0,
            'recovery_factor': 0,
            'calmar_ratio': 0,
            'expectancy': 0
        }
    
    def generate_detailed_report(self, save_to_file: bool = True) -> str:
        """Generate comprehensive analysis report"""
        if not self.analysis_results:
            self.analyze_all_strategies()
        
        report = []
        report.append("=" * 100)
        report.append("🤖 AI ALGORITHMIC TRADING PLATFORM - STRATEGY ANALYSIS REPORT")
        report.append("=" * 100)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Portfolio Summary
        portfolio = self.analysis_results.get('portfolio', {})
        report.append("📊 PORTFOLIO OVERVIEW")
        report.append("-" * 50)
        report.append(f"Total Trades: {portfolio.get('total_trades', 0)}")
        report.append(f"Total P&L: ₹{portfolio.get('total_pnl', 0):,.2f}")
        report.append(f"Win Rate: {portfolio.get('win_rate', 0):.1f}%")
        report.append(f"Trading Days: {portfolio.get('trading_days', 0)}")
        report.append(f"Average Daily P&L: ₹{portfolio.get('avg_daily_pnl', 0):,.2f}")
        report.append(f"Max Drawdown: ₹{portfolio.get('max_drawdown', 0):,.2f}")
        report.append(f"Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.2f}")
        report.append("")
        
        # Strategy Comparison
        comparison = self.analysis_results.get('comparison', {})
        if comparison and 'comparison_table' in comparison:
            report.append("🏆 STRATEGY RANKINGS")
            report.append("-" * 50)
            comparison_table = comparison['comparison_table']
            
            # Sort by overall ranking
            sorted_strategies = sorted(comparison_table, key=lambda x: x.get('overall_ranking', 999))
            
            report.append(f"{'Rank':<5} {'Strategy':<20} {'Total P&L':<12} {'Win Rate':<10} {'Profit Factor':<15}")
            report.append("-" * 70)
            
            for i, strategy in enumerate(sorted_strategies, 1):
                report.append(f"{i:<5} {strategy['strategy']:<20} ₹{strategy['total_pnl']:<11,.2f} {strategy['win_rate']:<9.1f}% {strategy['profit_factor']:<15.2f}")
            
            report.append("")
        
        # Individual Strategy Analysis
        strategies = self.analysis_results.get('strategies', {})
        if strategies:
            report.append("📈 DETAILED STRATEGY ANALYSIS")
            report.append("-" * 50)
            
            for strategy_name, metrics in strategies.items():
                report.append(f"\n🎯 {strategy_name.upper()}")
                report.append("-" * 30)
                report.append(f"Trades: {metrics['total_trades']} | Win Rate: {metrics['win_rate']:.1f}% | Total P&L: ₹{metrics['total_pnl']:,.2f}")
                report.append(f"Average Win: ₹{metrics['avg_win']:,.2f} | Average Loss: ₹{metrics['avg_loss']:,.2f}")
                report.append(f"Max Win: ₹{metrics['max_win']:,.2f} | Max Loss: ₹{metrics['max_loss']:,.2f}")
                report.append(f"Profit Factor: {metrics['profit_factor']:.2f} | Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
                report.append(f"Max Drawdown: ₹{metrics['max_drawdown']:,.2f} | Recovery Factor: {metrics['recovery_factor']:.2f}")
                report.append(f"Avg Holding Time: {metrics['avg_holding_time']:.1f} minutes")
                
                # Performance rating
                if metrics['total_pnl'] > 0 and metrics['win_rate'] > 50:
                    rating = "🟢 EXCELLENT"
                elif metrics['total_pnl'] > 0:
                    rating = "🟡 GOOD"
                elif metrics['total_pnl'] > -500:
                    rating = "🟠 FAIR"
                else:
                    rating = "🔴 POOR"
                
                report.append(f"Performance Rating: {rating}")
        
        # Risk Analysis
        risk_metrics = self.analysis_results.get('risk_metrics', {})
        if risk_metrics:
            report.append("\n  RISK ANALYSIS")
            report.append("-" * 50)
            report.append(f"Value at Risk (95%): ₹{risk_metrics.get('var_95', 0):,.2f}")
            report.append(f"Value at Risk (99%): ₹{risk_metrics.get('var_99', 0):,.2f}")
            report.append(f"Conditional VaR (95%): ₹{risk_metrics.get('cvar_95', 0):,.2f}")
            report.append(f"Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.2f}")
            report.append(f"Downside Deviation: {risk_metrics.get('downside_deviation', 0):,.2f}")
            report.append(f"Return Skewness: {risk_metrics.get('skewness', 0):.2f}")
            report.append(f"Return Kurtosis: {risk_metrics.get('kurtosis', 0):.2f}")
        
        # Recommendations
        recommendations = self.analysis_results.get('recommendations', [])
        if recommendations:
            report.append("\n💡 RECOMMENDATIONS")
            report.append("-" * 50)
            for rec in recommendations:
                report.append(f"• {rec}")
        
        # Capital Allocation Suggestions
        if strategies:
            report.append("\n💰 SUGGESTED CAPITAL ALLOCATION")
            report.append("-" * 50)
            
            # Calculate allocation based on Sharpe ratio and total PnL
            total_score = 0
            strategy_scores = {}
            
            for name, metrics in strategies.items():
                if metrics['total_pnl'] > 0 and metrics['sharpe_ratio'] > 0:
                    score = metrics['sharpe_ratio'] * (1 + metrics['total_pnl'] / 1000)  # Weight by PnL
                    strategy_scores[name] = max(0, score)
                    total_score += strategy_scores[name]
            
            if total_score > 0:
                for name, score in sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True):
                    allocation_pct = (score / total_score) * 100
                    if allocation_pct >= 5:  # Only show significant allocations
                        report.append(f"{name}: {allocation_pct:.1f}% of capital")
            else:
                report.append("⚠️  No profitable strategies with positive Sharpe ratio found.")
                report.append("Consider revising strategy parameters or market conditions.")
        
        # Trading Calendar Analysis
        if portfolio.get('trading_days', 0) > 0:
            report.append(f"\n📅 TRADING CALENDAR")
            report.append("-" * 50)
            report.append(f"Profitable Days: {portfolio.get('profitable_days', 0)} ({portfolio.get('daily_win_rate', 0):.1f}%)")
            report.append(f"Best Trading Day: ₹{portfolio.get('best_day', 0):,.2f}")
            report.append(f"Worst Trading Day: ₹{portfolio.get('worst_day', 0):,.2f}")
            report.append(f"Daily Volatility: ₹{portfolio.get('daily_volatility', 0):,.2f}")
            report.append(f"Daily Sharpe Ratio: {portfolio.get('daily_sharpe', 0):.2f}")
        
        report.append("\n" + "=" * 100)
        report.append("📋 ANALYSIS COMPLETE")
        report.append("=" * 100)
        report.append("\n⚠️  Disclaimer: This analysis is based on historical paper trading data.")
        report.append("Past performance does not guarantee future results.")
        report.append("Always conduct your own research before making trading decisions.")
        
        report_text = "\n".join(report)
        
        if save_to_file:
            filename = f"strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            try:
                with open(f"exports/{filename}", "w", encoding="utf-8") as f:
                    f.write(report_text)
                print(f"📁 Analysis report saved to: exports/{filename}")
            except Exception as e:
                print(f"❌ Error saving report: {e}")
        
        return report_text
    
    def export_analysis_to_csv(self) -> str:
        """Export analysis results to CSV files"""
        if not self.analysis_results:
            self.analyze_all_strategies()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Export strategy comparison
            comparison = self.analysis_results.get('comparison', {})
            if comparison and 'comparison_table' in comparison:
                comparison_df = pd.DataFrame(comparison['comparison_table'])
                comparison_file = f"exports/strategy_comparison_{timestamp}.csv"
                comparison_df.to_csv(comparison_file, index=False)
                print(f"📊 Strategy comparison exported to: {comparison_file}")
            
            # Export detailed metrics
            strategies = self.analysis_results.get('strategies', {})
            if strategies:
                detailed_data = []
                for strategy_name, metrics in strategies.items():
                    row = {'strategy': strategy_name}
                    row.update(metrics)
                    detailed_data.append(row)
                
                detailed_df = pd.DataFrame(detailed_data)
                detailed_file = f"exports/strategy_detailed_metrics_{timestamp}.csv"
                detailed_df.to_csv(detailed_file, index=False)
                print(f"📈 Detailed metrics exported to: {detailed_file}")
            
            # Export portfolio summary
            portfolio = self.analysis_results.get('portfolio', {})
            if portfolio:
                portfolio_df = pd.DataFrame([portfolio])
                portfolio_file = f"exports/portfolio_summary_{timestamp}.csv"
                portfolio_df.to_csv(portfolio_file, index=False)
                print(f"💼 Portfolio summary exported to: {portfolio_file}")
            
            return f"Analysis exported successfully at {timestamp}"
            
        except Exception as e:
            error_msg = f"❌ Error exporting analysis: {e}"
            print(error_msg)
            return error_msg
    
    def create_performance_visualization(self, save_plots: bool = True):
        """Create performance visualization charts"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('dark_background')
            
            if not self.analysis_results:
                self.analyze_all_strategies()
            
            strategies = self.analysis_results.get('strategies', {})
            if not strategies:
                print("❌ No strategy data available for visualization")
                return
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('AI Algo Trading - Strategy Performance Analysis', fontsize=16, color='white')
            
            strategy_names = list(strategies.keys())
            
            # 1. Total P&L Comparison
            pnl_values = [metrics['total_pnl'] for metrics in strategies.values()]
            colors = ['green' if pnl > 0 else 'red' for pnl in pnl_values]
            
            axes[0,0].bar(strategy_names, pnl_values, color=colors, alpha=0.7)
            axes[0,0].set_title('Total P&L by Strategy', color='white')
            axes[0,0].set_ylabel('P&L (₹)', color='white')
            axes[0,0].tick_params(axis='x', rotation=45, colors='white')
            axes[0,0].tick_params(axis='y', colors='white')
            axes[0,0].grid(True, alpha=0.3)
            
            # 2. Win Rate Comparison
            win_rates = [metrics['win_rate'] for metrics in strategies.values()]
            axes[0,1].bar(strategy_names, win_rates, color='cyan', alpha=0.7)
            axes[0,1].set_title('Win Rate by Strategy', color='white')
            axes[0,1].set_ylabel('Win Rate (%)', color='white')
            axes[0,1].tick_params(axis='x', rotation=45, colors='white')
            axes[0,1].tick_params(axis='y', colors='white')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].set_ylim(0, 100)
            
            # 3. Risk-Return Scatter
            sharpe_ratios = [metrics['sharpe_ratio'] for metrics in strategies.values()]
            axes[1,0].scatter(sharpe_ratios, pnl_values, c=colors, s=100, alpha=0.7)
            axes[1,0].set_title('Risk-Return Profile', color='white')
            axes[1,0].set_xlabel('Sharpe Ratio', color='white')
            axes[1,0].set_ylabel('Total P&L (₹)', color='white')
            axes[1,0].tick_params(colors='white')
            axes[1,0].grid(True, alpha=0.3)
            
            # Add strategy labels
            for i, name in enumerate(strategy_names):
                axes[1,0].annotate(name[:10], (sharpe_ratios[i], pnl_values[i]), 
                                 xytext=(5, 5), textcoords='offset points', 
                                 fontsize=8, color='white')
            
            # 4. Drawdown Analysis
            drawdowns = [abs(metrics['max_drawdown']) for metrics in strategies.values()]
            axes[1,1].bar(strategy_names, drawdowns, color='orange', alpha=0.7)
            axes[1,1].set_title('Maximum Drawdown by Strategy', color='white')
            axes[1,1].set_ylabel('Max Drawdown (₹)', color='white')
            axes[1,1].tick_params(axis='x', rotation=45, colors='white')
            axes[1,1].tick_params(axis='y', colors='white')
            axes[1,1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_plots:
                filename = f"exports/strategy_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black')
                print(f"📊 Performance visualization saved to: {filename}")
            
            plt.show()
            
        except ImportError:
            print("❌ Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"❌ Error creating visualization: {e}")

def run_comprehensive_analysis():
    """Run comprehensive strategy analysis"""
    print("🚀 Starting Comprehensive Strategy Analysis...")
    
    analyzer = StrategyAnalyzer()
    
    try:
        # Run analysis
        results = analyzer.analyze_all_strategies(days_back=30)
        
        if not results:
            print("❌ No data available for analysis")
            return
        
        # Generate report
        print("\n📋 Generating detailed report...")
        report = analyzer.generate_detailed_report(save_to_file=True)
        
        # Print summary to console
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        
        portfolio = results.get('portfolio', {})
        print(f"📈 Total Trades: {portfolio.get('total_trades', 0)}")
        print(f"💰 Total P&L: ₹{portfolio.get('total_pnl', 0):,.2f}")
        print(f"🎯 Win Rate: {portfolio.get('win_rate', 0):.1f}%")
        print(f"📉 Max Drawdown: ₹{portfolio.get('max_drawdown', 0):,.2f}")
        
        comparison = results.get('comparison', {})
        if comparison.get('best_strategy'):
            print(f"🏆 Best Strategy: {comparison['best_strategy']}")
        
        # Export to CSV
        print("\n📤 Exporting analysis data...")
        analyzer.export_analysis_to_csv()
        
        # Create visualizations
        print("\n📊 Creating performance visualizations...")
        analyzer.create_performance_visualization()
        
        # Show recommendations
        recommendations = results.get('recommendations', [])
        if recommendations:
            print("\n💡 KEY RECOMMENDATIONS:")
            for rec in recommendations[:5]:  # Show top 5
                print(f"   • {rec}")
        
        print("\n✅ Comprehensive analysis completed!")
        
    except Exception as e:
        print(f"❌ Error in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_comprehensive_analysis()