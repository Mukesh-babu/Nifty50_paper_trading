# ADVANCED_STRATEGIES.PY - Additional Trading Strategies
# Advanced algorithmic trading strategies for Indian options market
# Version: 1.0

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import math
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from algo_trading_main import TradingStrategy, TradingConfig, TechnicalIndicators
except ImportError:
    print("Warning: Could not import base classes. Ensure algo_trading_main.py is available.")

class PairTradingStrategy(TradingStrategy):
    """Pairs trading strategy using statistical arbitrage"""
    
    def __init__(self):
        super().__init__("Pair Trading")
        self.lookback_period = 30
        self.z_threshold = 2.0
        
    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate pairs trading signals based on mean reversion of price ratios"""
        if len(historical_prices) < self.lookback_period:
            return None
            
        current_price = market_data['close']
        vol_regime = technical_data.get('vol_regime', 'normal')
        
        self.signals_generated += 1
        
        # Calculate z-score of current price vs historical mean
        price_series = np.array(historical_prices[-self.lookback_period:])
        price_mean = np.mean(price_series)
        price_std = np.std(price_series)
        
        if price_std == 0:
            return None
            
        z_score = (current_price - price_mean) / price_std
        
        signal = None
        
        # Generate signals based on z-score extremes
        if z_score > self.z_threshold:  # Price too high, expect reversion down
            signal = {
                'type': 'BUY',
                'option_type': 'PE',
                'reason': 'PAIR_TRADE_OVERVALUED',
                'confidence': min(100, abs(z_score) * 30),
                'strike_adjustment': 25  # Slightly OTM
            }
        elif z_score < -self.z_threshold:  # Price too low, expect reversion up
            signal = {
                'type': 'BUY',
                'option_type': 'CE',
                'reason': 'PAIR_TRADE_UNDERVALUED',
                'confidence': min(100, abs(z_score) * 30),
                'strike_adjustment': 25
            }
            
        # Only trade in favorable volatility conditions
        if signal and vol_regime == 'low':
            return None
            
        if signal:
            self.trades_executed += 1
            
        return signal

class NewsBasedStrategy(TradingStrategy):
    """Strategy that reacts to market events and news (simulated)"""
    
    def __init__(self):
        super().__init__("News Based")
        self.event_impact_threshold = 0.005  # 0.5% price movement
        self.lookback_minutes = 15
        
    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate signals based on sudden price movements (news events)"""
        if len(historical_prices) < 10:
            return None
            
        current_price = market_data['close']
        vol_regime = technical_data.get('vol_regime', 'normal')
        
        self.signals_generated += 1
        
        # Detect sudden price movements (simulated news impact)
        recent_prices = historical_prices[-self.lookback_minutes:]
        if len(recent_prices) < 5:
            return None
            
        price_change = (current_price - recent_prices[0]) / recent_prices[0]
        volatility = np.std(np.diff(recent_prices)) / np.mean(recent_prices)
        
        signal = None
        
        # React to significant price movements
        if abs(price_change) > self.event_impact_threshold and volatility > 0.01:
            if price_change > 0:  # Sudden upward movement - expect continuation
                signal = {
                    'type': 'BUY',
                    'option_type': 'CE',
                    'reason': 'NEWS_MOMENTUM_UP',
                    'confidence': min(100, abs(price_change) * 1000),
                    'strike_adjustment': 75  # More OTM for higher leverage
                }
            else:  # Sudden downward movement - expect continuation
                signal = {
                    'type': 'BUY',
                    'option_type': 'PE',
                    'reason': 'NEWS_MOMENTUM_DOWN',
                    'confidence': min(100, abs(price_change) * 1000),
                    'strike_adjustment': 75
                }
                
        # Only trade significant events
        if signal and signal['confidence'] < 60:
            return None
            
        if signal:
            self.trades_executed += 1
            
        return signal

class ScalpingStrategy(TradingStrategy):
    """High-frequency scalping strategy for quick profits"""
    
    def __init__(self):
        super().__init__("Scalping")
        self.min_volume_threshold = 100000  # Minimum volume for liquidity
        self.spread_threshold = 0.002  # Maximum 0.2% spread
        self.target_profit = 0.015  # 1.5% target profit
        
    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate scalping signals based on short-term price action"""
        if len(historical_prices) < 5:
            return None
            
        current_price = market_data['close']
        volume = market_data.get('volume', 0)
        vol_regime = technical_data.get('vol_regime', 'normal')
        
        self.signals_generated += 1
        
        # Only scalp in high-volume, high-volatility conditions
        if volume < self.min_volume_threshold or vol_regime != 'high':
            return None
            
        # Look for short-term momentum
        recent_prices = historical_prices[-3:]
        if len(recent_prices) < 3:
            return None
            
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        price_acceleration = recent_prices[-1] - 2*recent_prices[-2] + recent_prices[-3]
        
        signal = None
        
        # Scalp on momentum with acceleration
        if momentum > 0.001 and price_acceleration > 0:  # Upward momentum accelerating
            signal = {
                'type': 'BUY',
                'option_type': 'CE',
                'reason': 'SCALP_MOMENTUM_UP',
                'confidence': 80,
                'strike_adjustment': 0  # ATM for quick moves
            }
        elif momentum < -0.001 and price_acceleration < 0:  # Downward momentum accelerating
            signal = {
                'type': 'BUY',
                'option_type': 'PE',
                'reason': 'SCALP_MOMENTUM_DOWN',
                'confidence': 80,
                'strike_adjustment': 0
            }
            
        if signal:
            self.trades_executed += 1
            
        return signal

class MachineLearningStrategy(TradingStrategy):
    """AI/ML-based strategy using statistical patterns"""
    
    def __init__(self):
        super().__init__("Machine Learning")
        self.feature_window = 20
        self.prediction_threshold = 0.7
        
    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate ML-based signals using feature engineering"""
        if len(historical_prices) < self.feature_window:
            return None
            
        current_price = market_data['close']
        vol_regime = technical_data.get('vol_regime', 'normal')
        
        self.signals_generated += 1
        
        # Feature engineering
        features = self._extract_features(historical_prices, technical_data)
        
        # Simple ML prediction (replace with actual ML model in production)
        prediction = self._simple_ml_prediction(features)
        
        signal = None
        
        if prediction['confidence'] > self.prediction_threshold:
            if prediction['direction'] == 'up':
                signal = {
                    'type': 'BUY',
                    'option_type': 'CE',
                    'reason': 'ML_PREDICTION_UP',
                    'confidence': prediction['confidence'] * 100,
                    'strike_adjustment': 50
                }
            elif prediction['direction'] == 'down':
                signal = {
                    'type': 'BUY',
                    'option_type': 'PE',
                    'reason': 'ML_PREDICTION_DOWN',
                    'confidence': prediction['confidence'] * 100,
                    'strike_adjustment': 50
                }
                
        if signal:
            self.trades_executed += 1
            
        return signal
    
    def _extract_features(self, prices: List[float], technical_data: Dict) -> Dict:
        """Extract features for ML prediction"""
        prices_array = np.array(prices[-self.feature_window:])
        
        features = {
            'price_momentum_5': (prices_array[-1] - prices_array[-5]) / prices_array[-5] if len(prices_array) >= 5 else 0,
            'price_momentum_10': (prices_array[-1] - prices_array[-10]) / prices_array[-10] if len(prices_array) >= 10 else 0,
            'volatility': technical_data.get('volatility', 0.2),
            'rsi': technical_data.get('rsi', 50),
            'bb_position': self._calculate_bb_position(prices_array[-1], technical_data),
            'volume_ratio': 1.0,  # Placeholder
            'time_of_day': datetime.now().hour / 24.0,
            'price_std': np.std(prices_array) / np.mean(prices_array),
            'price_skewness': stats.skew(prices_array) if len(prices_array) >= 3 else 0,
            'price_kurtosis': stats.kurtosis(prices_array) if len(prices_array) >= 4 else 0
        }
        
        return features
    
    def _calculate_bb_position(self, current_price: float, technical_data: Dict) -> float:
        """Calculate position within Bollinger Bands"""
        bb_upper = technical_data.get('bb_upper', current_price)
        bb_lower = technical_data.get('bb_lower', current_price)
        
        if bb_upper == bb_lower:
            return 0.5
            
        return (current_price - bb_lower) / (bb_upper - bb_lower)
    
    def _simple_ml_prediction(self, features: Dict) -> Dict:
        """Simple ML prediction logic (replace with actual model)"""
        # This is a simplified rule-based system mimicking ML output
        score = 0
        
        # Momentum features
        if features['price_momentum_5'] > 0.002:
            score += 0.3
        elif features['price_momentum_5'] < -0.002:
            score -= 0.3
            
        # RSI features
        if features['rsi'] < 30:
            score += 0.2
        elif features['rsi'] > 70:
            score -= 0.2
            
        # Bollinger Band position
        bb_pos = features['bb_position']
        if bb_pos < 0.2:
            score += 0.25
        elif bb_pos > 0.8:
            score -= 0.25
            
        # Volatility regime
        if features['volatility'] > 0.25:
            score *= 1.2  # Amplify signals in high vol
        elif features['volatility'] < 0.15:
            score *= 0.8  # Dampen signals in low vol
            
        # Convert to prediction
        confidence = min(1.0, abs(score))
        direction = 'up' if score > 0 else 'down' if score < 0 else 'neutral'
        
        return {
            'direction': direction,
            'confidence': confidence,
            'score': score
        }

class OptionsFlowStrategy(TradingStrategy):
    """Strategy based on options flow analysis (simulated)"""
    
    def __init__(self):
        super().__init__("Options Flow")
        self.flow_threshold = 0.6  # 60% call/put ratio threshold
        
    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate signals based on simulated options flow data"""
        if len(historical_prices) < 10:
            return None
            
        current_price = market_data['close']
        vol_regime = technical_data.get('vol_regime', 'normal')
        
        self.signals_generated += 1
        
        # Simulate options flow (in real implementation, get actual data)
        call_put_ratio = self._simulate_options_flow(historical_prices, technical_data)
        
        signal = None
        
        # Generate signals based on extreme options flow
        if call_put_ratio > (1 + self.flow_threshold):  # Heavy call buying
            signal = {
                'type': 'BUY',
                'option_type': 'CE',
                'reason': 'HEAVY_CALL_FLOW',
                'confidence': min(100, (call_put_ratio - 1) * 100),
                'strike_adjustment': 25
            }
        elif call_put_ratio < (1 - self.flow_threshold):  # Heavy put buying
            signal = {
                'type': 'BUY',
                'option_type': 'PE',
                'reason': 'HEAVY_PUT_FLOW',
                'confidence': min(100, (1 - call_put_ratio) * 100),
                'strike_adjustment': 25
            }
            
        if signal:
            self.trades_executed += 1
            
        return signal
    
    def _simulate_options_flow(self, prices: List[float], technical_data: Dict) -> float:
        """Simulate options flow data (replace with actual data in production)"""
        # This simulates options flow based on price momentum and volatility
        recent_change = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        volatility = technical_data.get('volatility', 0.2)
        
        # Simulate call/put ratio based on market conditions
        base_ratio = 1.0  # Neutral
        momentum_impact = recent_change * 10  # Convert to ratio impact
        vol_impact = (volatility - 0.2) * 2  # Volatility impact
        
        # Add some randomness to simulate real market conditions
        noise = np.random.normal(0, 0.1)
        
        call_put_ratio = base_ratio + momentum_impact + vol_impact + noise
        return max(0.1, min(3.0, call_put_ratio))  # Cap between 0.1 and 3.0

class ArbitrageStrategy(TradingStrategy):
    """Statistical arbitrage strategy"""
    
    def __init__(self):
        super().__init__("Arbitrage")
        self.price_deviation_threshold = 0.003  # 0.3% deviation
        
    def generate_signal(self, market_data: Dict, technical_data: Dict, historical_prices: List[float]) -> Optional[Dict]:
        """Generate arbitrage signals based on price deviations"""
        if len(historical_prices) < 20:
            return None
            
        current_price = market_data['close']
        vol_regime = technical_data.get('vol_regime', 'normal')
        
        self.signals_generated += 1
        
        # Calculate theoretical fair value
        fair_value = self._calculate_fair_value(historical_prices, technical_data)
        deviation = (current_price - fair_value) / fair_value
        
        signal = None
        
        # Generate signals based on price deviation from fair value
        if abs(deviation) > self.price_deviation_threshold:
            if deviation > 0:  # Overvalued
                signal = {
                    'type': 'BUY',
                    'option_type': 'PE',
                    'reason': 'ARBITRAGE_OVERVALUED',
                    'confidence': min(100, abs(deviation) * 2000),
                    'strike_adjustment': 0  # ATM for arbitrage
                }
            else:  # Undervalued
                signal = {
                    'type': 'BUY',
                    'option_type': 'CE',
                    'reason': 'ARBITRAGE_UNDERVALUED',
                    'confidence': min(100, abs(deviation) * 2000),
                    'strike_adjustment': 0
                }
                
        if signal:
            self.trades_executed += 1
            
        return signal
    
    def _calculate_fair_value(self, prices: List[float], technical_data: Dict) -> float:
        """Calculate theoretical fair value"""
        # Simple fair value calculation (can be enhanced with more sophisticated models)
        sma_20 = np.mean(prices[-20:])
        volatility_adj = technical_data.get('volatility', 0.2) - 0.2
        
        # Adjust SMA based on volatility regime
        fair_value = sma_20 * (1 + volatility_adj * 0.1)
        
        return fair_value

# Strategy factory for easy strategy management
class StrategyFactory:
    """Factory class to create and manage trading strategies"""
    
    @staticmethod
    def create_all_strategies() -> List[TradingStrategy]:
        """Create all available strategies"""
        from algo_trading_main import MeanReversionStrategy, MomentumBreakoutStrategy, VolatilityRegimeStrategy
        
        strategies = [
            MeanReversionStrategy(),
            MomentumBreakoutStrategy(),
            VolatilityRegimeStrategy(),
            PairTradingStrategy(),
            NewsBasedStrategy(),
            ScalpingStrategy(),
            MachineLearningStrategy(),
            OptionsFlowStrategy(),
            ArbitrageStrategy()
        ]
        
        return strategies
    
    @staticmethod
    def create_conservative_strategies() -> List[TradingStrategy]:
        """Create conservative strategies with lower risk"""
        from algo_trading_main import MeanReversionStrategy, VolatilityRegimeStrategy
        
        return [
            MeanReversionStrategy(),
            VolatilityRegimeStrategy(),
            PairTradingStrategy(),
            ArbitrageStrategy()
        ]
    
    @staticmethod
    def create_aggressive_strategies() -> List[TradingStrategy]:
        """Create aggressive strategies with higher risk/reward"""
        from algo_trading_main import MomentumBreakoutStrategy
        
        return [
            MomentumBreakoutStrategy(),
            NewsBasedStrategy(),
            ScalpingStrategy(),
            MachineLearningStrategy(),
            OptionsFlowStrategy()
        ]
    
    @staticmethod
    def get_strategy_by_name(name: str) -> Optional[TradingStrategy]:
        """Get specific strategy by name"""
        strategy_map = {
            'mean_reversion': MeanReversionStrategy,
            'momentum_breakout': MomentumBreakoutStrategy,
            'volatility_regime': VolatilityRegimeStrategy,
            'pair_trading': PairTradingStrategy,
            'news_based': NewsBasedStrategy,
            'scalping': ScalpingStrategy,
            'machine_learning': MachineLearningStrategy,
            'options_flow': OptionsFlowStrategy,
            'arbitrage': ArbitrageStrategy
        }
        
        strategy_class = strategy_map.get(name.lower())
        if strategy_class:
            return strategy_class()
        return None

if __name__ == "__main__":
    print("Advanced Trading Strategies Module")
    print("Available strategies:")
    
    strategies = StrategyFactory.create_all_strategies()
    for i, strategy in enumerate(strategies, 1):
        print(f"{i}. {strategy.name}")
    
    print(f"\nTotal strategies available: {len(strategies)}")