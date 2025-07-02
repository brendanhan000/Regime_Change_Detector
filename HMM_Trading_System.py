import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Import your existing model
from Back_testing_v2 import ImprovedMarketRegimeDetector


class HMMTradingSystem:
    """
    Trading system based on HMM Market Regime Detection
    Provides clear entry/exit signals with risk management
    """

    def __init__(self,
                 confidence_threshold: float = 0.65,
                 lookback_window: int = 252,
                 min_holding_period: int = 5,
                 max_position_size: float = 1.0,
                 stop_loss_pct: float = 0.05,
                 take_profit_pct: float = 0.15):
        """
        Initialize trading system parameters

        Args:
            confidence_threshold: Minimum confidence to take positions
            lookback_window: Days of data for model training
            min_holding_period: Minimum days to hold position
            max_position_size: Maximum position size (1.0 = 100%)
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        self.confidence_threshold = confidence_threshold
        self.lookback_window = lookback_window
        self.min_holding_period = min_holding_period
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # Trading state
        self.current_position = 0.0  # -1 to 1 (short to long)
        self.entry_price = None
        self.entry_date = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.days_in_position = 0

        # Model
        self.detector = None
        self.last_retrain_date = None

        # Trading log
        self.trading_log = []
        self.signals_log = []

    def calculate_position_size(self, regime: str, confidence: float,
                                current_price: float, volatility: float) -> float:
        """
        Calculate position size based on regime, confidence, and risk management
        """
        # Base position sizes by regime
        base_sizes = {
            'Bull': 0.8,  # 80% long in bull market
            'Sideways': 0.2,  # 20% long in sideways market
            'Bear': -0.3  # 30% short in bear market
        }

        base_size = base_sizes.get(regime, 0.0)

        # Adjust by confidence
        confidence_adjusted = base_size * confidence

        # Adjust by volatility (reduce size in high volatility)
        volatility_factor = min(1.0, 0.2 / max(volatility, 0.1))
        vol_adjusted = confidence_adjusted * volatility_factor

        # Apply maximum position size limit
        final_size = np.clip(vol_adjusted, -self.max_position_size, self.max_position_size)

        return final_size

    def should_retrain_model(self, current_date: datetime) -> bool:
        """Check if model should be retrained"""
        if self.last_retrain_date is None:
            return True

        days_since_retrain = (current_date - self.last_retrain_date).days
        return days_since_retrain >= 21  # Retrain monthly

    def get_trading_signal(self, prices: pd.Series, current_date: datetime) -> Dict:
        """
        Generate trading signal for current date

        Returns:
            Dict with signal information
        """
        try:
            # Get data up to current date
            historical_data = prices.loc[:current_date]

            if len(historical_data) < self.lookback_window:
                return {
                    'signal': 'HOLD',
                    'regime': 'Unknown',
                    'confidence': 0.0,
                    'target_position': 0.0,
                    'reason': 'Insufficient data'
                }

            # Retrain model if needed
            if self.should_retrain_model(current_date):
                train_data = historical_data.tail(self.lookback_window)
                self.detector = ImprovedMarketRegimeDetector(n_regimes=3, random_state=42)
                self.detector.fit(train_data)
                self.last_retrain_date = current_date
                print(f"Model retrained on {current_date.strftime('%Y-%m-%d')}")

            # Get current regime prediction
            regime_pred, confidence_pred = self.detector.predict_regime(historical_data)

            current_regime = regime_pred.iloc[-1]
            current_confidence = confidence_pred.iloc[-1]
            current_price = prices.loc[current_date]

            # Calculate volatility
            returns = historical_data.pct_change().dropna()
            volatility = returns.tail(20).std() * np.sqrt(252)

            # Calculate target position
            target_position = self.calculate_position_size(
                current_regime, current_confidence, current_price, volatility
            )

            # Generate signal
            signal = self._generate_signal(current_regime, current_confidence,
                                           target_position, current_price)

            return {
                'signal': signal,
                'regime': current_regime,
                'confidence': current_confidence,
                'target_position': target_position,
                'current_price': current_price,
                'volatility': volatility,
                'reason': self._get_signal_reason(signal, current_regime, current_confidence)
            }

        except Exception as e:
            return {
                'signal': 'HOLD',
                'regime': 'Error',
                'confidence': 0.0,
                'target_position': 0.0,
                'reason': f'Error: {str(e)}'
            }

    def _generate_signal(self, regime: str, confidence: float,
                         target_position: float, current_price: float) -> str:
        """Generate trading signal based on regime and position"""

        # Check stop loss and take profit first
        if self.current_position != 0 and self.entry_price is not None:
            if self._should_stop_loss(current_price):
                return 'STOP_LOSS'
            if self._should_take_profit(current_price):
                return 'TAKE_PROFIT'

        # Check minimum holding period
        if self.current_position != 0 and self.days_in_position < self.min_holding_period:
            return 'HOLD'

        # Check confidence threshold
        if confidence < self.confidence_threshold:
            if self.current_position != 0:
                return 'CLOSE'  # Close position if confidence too low
            else:
                return 'HOLD'  # Don't enter if confidence too low

        # Determine signal based on target position
        position_diff = target_position - self.current_position

        if abs(position_diff) < 0.1:  # Small difference
            return 'HOLD'
        elif position_diff > 0.2:  # Significant increase needed
            return 'BUY'
        elif position_diff < -0.2:  # Significant decrease needed
            return 'SELL'
        else:
            return 'HOLD'

    def _should_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss should be triggered"""
        if self.stop_loss_price is None:
            return False

        if self.current_position > 0:  # Long position
            return current_price <= self.stop_loss_price
        elif self.current_position < 0:  # Short position
            return current_price >= self.stop_loss_price

        return False

    def _should_take_profit(self, current_price: float) -> bool:
        """Check if take profit should be triggered"""
        if self.take_profit_price is None:
            return False

        if self.current_position > 0:  # Long position
            return current_price >= self.take_profit_price
        elif self.current_position < 0:  # Short position
            return current_price <= self.take_profit_price

        return False

    def _get_signal_reason(self, signal: str, regime: str, confidence: float) -> str:
        """Get explanation for signal"""
        reasons = {
            'BUY': f'{regime} regime detected with {confidence:.1%} confidence',
            'SELL': f'{regime} regime detected with {confidence:.1%} confidence',
            'HOLD': f'Holding current position',
            'CLOSE': f'Closing position due to low confidence ({confidence:.1%})',
            'STOP_LOSS': f'Stop loss triggered',
            'TAKE_PROFIT': f'Take profit triggered'
        }
        return reasons.get(signal, 'Unknown reason')

    def execute_signal(self, signal_info: Dict, current_date: datetime) -> Dict:
        """
        Execute trading signal and update position

        Returns:
            Dict with execution information
        """
        signal = signal_info['signal']
        current_price = signal_info['current_price']
        target_position = signal_info['target_position']

        execution_info = {
            'date': current_date,
            'signal': signal,
            'price': current_price,
            'old_position': self.current_position,
            'new_position': self.current_position,
            'trade_size': 0.0,
            'trade_type': 'NONE'
        }

        if signal in ['BUY', 'SELL']:
            # Calculate trade size
            trade_size = target_position - self.current_position

            # Execute trade
            self.current_position = target_position
            self.entry_price = current_price
            self.entry_date = current_date
            self.days_in_position = 0

            # Set stop loss and take profit
            if self.current_position > 0:  # Long position
                self.stop_loss_price = current_price * (1 - self.stop_loss_pct)
                self.take_profit_price = current_price * (1 + self.take_profit_pct)
            elif self.current_position < 0:  # Short position
                self.stop_loss_price = current_price * (1 + self.stop_loss_pct)
                self.take_profit_price = current_price * (1 - self.take_profit_pct)
            else:
                self.stop_loss_price = None
                self.take_profit_price = None

            execution_info.update({
                'new_position': self.current_position,
                'trade_size': trade_size,
                'trade_type': 'LONG' if trade_size > 0 else 'SHORT'
            })

        elif signal in ['CLOSE', 'STOP_LOSS', 'TAKE_PROFIT']:
            # Close position
            trade_size = -self.current_position
            self.current_position = 0.0
            self.entry_price = None
            self.entry_date = None
            self.stop_loss_price = None
            self.take_profit_price = None
            self.days_in_position = 0

            execution_info.update({
                'new_position': 0.0,
                'trade_size': trade_size,
                'trade_type': 'CLOSE'
            })

        # Update days in position
        if self.current_position != 0:
            self.days_in_position += 1

        # Log the trade
        if execution_info['trade_size'] != 0:
            self.trading_log.append(execution_info.copy())

        return execution_info

    def get_current_status(self) -> Dict:
        """Get current trading system status"""
        return {
            'current_position': self.current_position,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'days_in_position': self.days_in_position,
            'total_trades': len(self.trading_log)
        }

    def backtest_strategy(self, ticker: str, start_date: str, end_date: str) -> Dict:
        """
        Backtest the trading strategy
        """
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date)
        prices = data['Close'].squeeze()

        # Initialize tracking
        self.trading_log = []
        self.signals_log = []
        daily_returns = []

        # Run through each trading day
        print(f"Backtesting {ticker} from {start_date} to {end_date}")

        for current_date in prices.index[self.lookback_window:]:
            # Get signal
            signal_info = self.get_trading_signal(prices, current_date)

            # Execute signal
            execution_info = self.execute_signal(signal_info, current_date)

            # Calculate daily return
            if len(daily_returns) > 0:
                daily_return = (prices.loc[current_date] / prices.loc[prev_date] - 1) * self.current_position
            else:
                daily_return = 0.0

            daily_returns.append(daily_return)

            # Log signal
            self.signals_log.append({
                'date': current_date,
                'price': prices.loc[current_date],
                'regime': signal_info['regime'],
                'confidence': signal_info['confidence'],
                'signal': signal_info['signal'],
                'position': self.current_position,
                'daily_return': daily_return
            })

            prev_date = current_date

        # Calculate performance metrics
        returns_series = pd.Series(daily_returns, index=prices.index[self.lookback_window:])

        total_return = (1 + returns_series).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_series)) - 1
        volatility = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Max drawdown
        cumulative = (1 + returns_series).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = [t for t in self.trading_log if t['trade_size'] *
                          (prices.loc[t['date']] - self.entry_price) > 0]
        win_rate = len(winning_trades) / len(self.trading_log) if self.trading_log else 0

        return {
            'returns': returns_series,
            'signals': pd.DataFrame(self.signals_log),
            'trades': pd.DataFrame(self.trading_log),
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(self.trading_log)
        }


# Example usage
def main():
    """Example of how to use the trading system"""

    # Initialize trading system
    trading_system = HMMTradingSystem(
        confidence_threshold=0.65,
        lookback_window=252,
        min_holding_period=5,
        max_position_size=1.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.15
    )

    # Backtest
    results = trading_system.backtest_strategy(
        ticker='SPY',
        start_date='2020-01-01',
        end_date='2025-07-01'
    )

    print("\n=== TRADING SYSTEM BACKTEST RESULTS ===")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Annualized Return: {results['annualized_return']:.2%}")
    print(f"Volatility: {results['volatility']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")

    # Show recent signals
    print("\n=== RECENT SIGNALS ===")
    recent_signals = results['signals'].tail(10)
    for _, row in recent_signals.iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d')}: {row['signal']} - "
              f"{row['regime']} ({row['confidence']:.1%}) - Position: {row['position']:.1f}")


if __name__ == "__main__":
    main()