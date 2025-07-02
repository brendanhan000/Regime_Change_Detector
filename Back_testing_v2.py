import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from typing import Dict, Tuple, List
from scipy.stats import norm

warnings.filterwarnings('ignore')

# Import the MarketRegimeDetector from the main file
from hmmlearn import hmm


class ImprovedMarketRegimeDetector:
    """Improved Market Regime Detector with better feature engineering and model stability"""

    def __init__(self, n_regimes=3, random_state=42):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        self.features = None
        self.returns = None
        self.scaler = StandardScaler()
        self.random_state = random_state
        self.feature_names = []

    def prepare_features(self, prices):
        """Enhanced feature engineering with more robust indicators"""
        if not isinstance(prices, pd.Series):
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]
                if hasattr(prices, 'squeeze'):
                    prices = prices.squeeze()
            else:
                prices = pd.Series(prices.flatten() if hasattr(prices, 'flatten') else prices)

        returns = prices.pct_change().dropna()
        self.returns = returns

        # Enhanced feature set
        features_dict = {}

        # 1. Multiple timeframe returns
        features_dict['returns_1d'] = returns
        features_dict['returns_5d'] = prices.pct_change(5)
        features_dict['returns_10d'] = prices.pct_change(10)
        features_dict['returns_20d'] = prices.pct_change(20)

        # 2. Rolling statistics with multiple windows
        for window in [5, 10, 20]:
            features_dict[f'roll_mean_{window}d'] = returns.rolling(window=window).mean()
            features_dict[f'roll_std_{window}d'] = returns.rolling(window=window).std()
            features_dict[f'roll_skew_{window}d'] = returns.rolling(window=window).skew()
            # Fixed: Use kurt() instead of kurtosis()
            features_dict[f'roll_kurt_{window}d'] = returns.rolling(window=window).kurt()

        # 3. Volatility measures
        features_dict['realized_vol_10d'] = returns.rolling(window=10).std() * np.sqrt(252)
        features_dict['realized_vol_20d'] = returns.rolling(window=20).std() * np.sqrt(252)
        features_dict['vol_ratio'] = (returns.rolling(window=5).std() /
                                      returns.rolling(window=20).std().replace(0, np.nan))

        # 4. Moving average indicators
        for window in [10, 20, 50]:
            ma = prices.rolling(window=window).mean()
            features_dict[f'price_to_ma_{window}d'] = (prices - ma) / ma
            features_dict[f'ma_slope_{window}d'] = ma.diff(5) / ma.shift(5)

        # 5. Technical indicators
        # RSI
        delta = returns
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        features_dict['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_window = 20
        bb_ma = prices.rolling(window=bb_window).mean()
        bb_std = prices.rolling(window=bb_window).std()
        features_dict['bb_position'] = (prices - bb_ma) / (2 * bb_std + 1e-10)
        features_dict['bb_width'] = bb_std / (bb_ma + 1e-10)

        # 6. Momentum indicators
        features_dict['momentum_10_5'] = (prices.pct_change(10) /
                                          (prices.pct_change(5).replace(0, np.nan) + 1e-10))
        features_dict['momentum_20_10'] = (prices.pct_change(20) /
                                           (prices.pct_change(10).replace(0, np.nan) + 1e-10))

        # 7. Drawdown measures
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        features_dict['drawdown'] = (cumulative - running_max) / running_max
        features_dict['drawdown_duration'] = (cumulative < running_max).astype(int).groupby(
            (cumulative >= running_max).cumsum()).cumsum()

        # 8. VIX-like indicator (using rolling volatility)
        features_dict['vix_proxy'] = returns.rolling(window=20).std() * np.sqrt(252) * 100

        # 9. Additional regime-discriminating features
        # Price momentum
        features_dict['price_momentum_5'] = (prices / prices.shift(5) - 1)
        features_dict['price_momentum_10'] = (prices / prices.shift(10) - 1)
        features_dict['price_momentum_20'] = (prices / prices.shift(20) - 1)

        # Volatility regimes
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        features_dict['vol_regime'] = vol_20 / (vol_60 + 1e-10)

        # Trend strength
        features_dict['trend_strength'] = abs(returns.rolling(20).mean()) / (returns.rolling(20).std() + 1e-10)

        # Create DataFrame and find common index
        features_df = pd.DataFrame(features_dict)

        # Use a more conservative approach to handle NaN values
        # Only keep data where we have at least 70% of features
        valid_threshold = 0.7 * len(features_df.columns)
        features_df = features_df.dropna(thresh=valid_threshold)

        # Fill remaining NaN values with forward fill then backward fill
        features_df = features_df.fillna(method='ffill').fillna(method='bfill')

        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()

        if len(features_df) == 0:
            raise ValueError("No valid features after preprocessing")

        # Normalize features
        features_normalized = pd.DataFrame(
            self.scaler.fit_transform(features_df),
            index=features_df.index,
            columns=features_df.columns
        )

        # Add small noise to prevent numerical issues
        np.random.seed(self.random_state)
        noise = np.random.normal(0, 1e-6, features_normalized.shape)
        features_normalized += noise

        self.features = features_normalized
        self.feature_names = list(features_normalized.columns)

        print(f"Prepared {len(features_normalized)} samples with {len(self.feature_names)} features")
        return features_normalized.values

    def fit(self, prices):
        """Fit the HMM model with improved initialization and stability"""
        X = self.prepare_features(prices)

        if len(X) < 50:  # Need minimum data for stable fitting
            raise ValueError(f"Insufficient data for model fitting: {len(X)} samples")

        print(f"Fitting HMM model with {len(X)} samples and {X.shape[1]} features")

        # Try multiple random initializations and pick the best
        best_model = None
        best_score = -np.inf
        successful_fits = 0

        for seed in range(10):  # Try more different initializations
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type="diag",  # Use diagonal covariance for stability
                    n_iter=100,  # Reduced iterations for stability
                    tol=1e-4,  # Looser tolerance
                    random_state=seed + self.random_state,
                    init_params="stmc"
                )

                # Set minimum covariance to prevent numerical issues
                model.min_covar = 1e-6

                # Fit the model
                model.fit(X)

                # Calculate score (log-likelihood)
                score = model.score(X)
                successful_fits += 1

                print(f"  Seed {seed}: Score = {score:.2f}")

                if score > best_score and not np.isnan(score) and not np.isinf(score):
                    best_score = score
                    best_model = model

            except Exception as e:
                print(f"  Seed {seed}: Failed - {str(e)[:50]}...")
                continue

        if best_model is None or successful_fits == 0:
            raise ValueError(f"Failed to fit any model successfully out of 10 attempts")

        print(f"Best model score: {best_score:.2f} (from {successful_fits} successful fits)")
        self.model = best_model

        # Assign regime names based on characteristics
        states = self.model.predict(X)
        self._assign_regime_names(states)

        return self

    def _assign_regime_names(self, states):
        """Assign meaningful names to regimes based on their characteristics"""
        regime_stats = {}

        for i in range(self.n_regimes):
            regime_mask = states == i
            if np.sum(regime_mask) > 0:
                regime_returns = self.returns.iloc[-len(states):][regime_mask]
                regime_stats[i] = {
                    'mean_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'count': np.sum(regime_mask)
                }
            else:
                # Handle empty regimes
                regime_stats[i] = {
                    'mean_return': 0,
                    'volatility': 0,
                    'count': 0
                }

        # Sort regimes by mean return
        sorted_regimes = sorted(regime_stats.items(), key=lambda x: x[1]['mean_return'])

        # Assign names based on return characteristics
        self.regime_names = {}
        for idx, (regime_id, stats) in enumerate(sorted_regimes):
            if idx == 0:
                self.regime_names[regime_id] = 'Bear'
            elif idx == len(sorted_regimes) - 1:
                self.regime_names[regime_id] = 'Bull'
            else:
                self.regime_names[regime_id] = 'Sideways'

        print(f"Regime assignment: {self.regime_names}")
        for regime_id, stats in regime_stats.items():
            regime_name = self.regime_names[regime_id]
            print(f"  {regime_name}: {stats['count']} samples, "
                  f"mean return: {stats['mean_return']:.4f}, "
                  f"volatility: {stats['volatility']:.4f}")

    def predict_regime(self, prices):
        """Predict regime with confidence scores"""
        if self.model is None:
            raise ValueError("Model must be fitted first")

        X = self.prepare_features(prices)

        # Get state probabilities
        log_probs = self.model.predict_proba(X)
        states = self.model.predict(X)

        # Create regime series with confidence
        regime_series = pd.Series(
            [self.regime_names[state] for state in states],
            index=self.features.index,
            name='Regime'
        )

        # Calculate confidence as the maximum probability
        confidence = np.max(log_probs, axis=1)
        confidence_series = pd.Series(
            confidence,
            index=self.features.index,
            name='Confidence'
        )

        return regime_series, confidence_series


class ImprovedHMMBacktester:
    """Enhanced backtesting system with better regime labeling and evaluation"""

    def __init__(self, lookback_window: int = 252, rebalance_freq: int = 21,
                 min_regime_duration: int = 5):
        self.lookback_window = lookback_window
        self.rebalance_freq = rebalance_freq
        self.min_regime_duration = min_regime_duration
        self.results = {}

    def create_improved_true_regimes(self, prices: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Create more accurate 'true' regime labels using multiple indicators
        """
        # Calculate various market indicators
        indicators = {}

        # 1. Rolling returns (multiple timeframes)
        indicators['ret_5d'] = returns.rolling(5).mean() * 252
        indicators['ret_20d'] = returns.rolling(20).mean() * 252
        indicators['ret_60d'] = returns.rolling(60).mean() * 252

        # 2. Volatility measures
        indicators['vol_20d'] = returns.rolling(20).std() * np.sqrt(252)
        indicators['vol_60d'] = returns.rolling(60).std() * np.sqrt(252)

        # 3. Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        indicators['drawdown'] = (cumulative - running_max) / running_max

        # 4. Moving average relationships
        ma_20 = prices.rolling(20).mean()
        ma_60 = prices.rolling(60).mean()
        indicators['price_vs_ma20'] = (prices - ma_20) / ma_20
        indicators['price_vs_ma60'] = (prices - ma_60) / ma_60
        indicators['ma_trend'] = (ma_20 - ma_60) / ma_60

        # 5. VIX-like measure
        indicators['vix_proxy'] = returns.rolling(20).std() * np.sqrt(252)

        # Create DataFrame
        df = pd.DataFrame(indicators)
        df = df.dropna()

        # Initialize regime labels
        true_regimes = pd.Series(index=prices.index, dtype=str, name='true_regime')
        true_regimes[:] = 'Sideways'  # Default to sideways

        # Improved regime classification with more nuanced thresholds
        for idx in df.index:
            row = df.loc[idx]

            # Bear market conditions (multiple must be true)
            bear_conditions = [
                row['drawdown'] < -0.08,  # 8% drawdown
                row['ret_20d'] < -0.10,  # Negative 20-day returns
                row['price_vs_ma20'] < -0.02,  # Below 20-day MA
                row['vol_20d'] > 0.25,  # High volatility
                row['ma_trend'] < -0.01  # Downward trend
            ]

            # Bull market conditions
            bull_conditions = [
                row['drawdown'] > -0.03,  # Small drawdown
                row['ret_20d'] > 0.15,  # Strong positive 20-day returns
                row['price_vs_ma20'] > 0.02,  # Above 20-day MA
                row['ma_trend'] > 0.01,  # Upward trend
                row['vol_20d'] < 0.25  # Moderate volatility
            ]

            # Sideways conditions (when neither bull nor bear conditions are met strongly)
            sideways_conditions = [
                abs(row['drawdown']) < 0.08,  # Small drawdown
                abs(row['ret_20d']) < 0.15,  # Moderate returns
                abs(row['price_vs_ma20']) < 0.05,  # Near moving average
                row['vol_20d'] < 0.30  # Not extremely high volatility
            ]

            # Classification logic with more balanced approach
            bear_score = sum(bear_conditions)
            bull_score = sum(bull_conditions)
            sideways_score = sum(sideways_conditions)

            if bear_score >= 3 and bear_score > bull_score:
                true_regimes.loc[idx] = 'Bear'
            elif bull_score >= 3 and bull_score > bear_score:
                true_regimes.loc[idx] = 'Bull'
            else:
                true_regimes.loc[idx] = 'Sideways'

        # Apply minimum duration filter to reduce noise
        true_regimes = self._apply_min_duration_filter(true_regimes)

        # Fill any remaining NaN values
        true_regimes = true_regimes.fillna(method='ffill').fillna(method='bfill')
        true_regimes = true_regimes.fillna('Sideways')

        # Print regime distribution for debugging
        regime_dist = true_regimes.value_counts(normalize=True)
        print(f"True regime distribution: {regime_dist.to_dict()}")

        return true_regimes

    def _apply_min_duration_filter(self, regimes: pd.Series) -> pd.Series:
        """Apply minimum duration filter to reduce regime switching noise"""
        if len(regimes) == 0:
            return regimes

        filtered_regimes = regimes.copy()

        # Find regime changes
        regime_changes = regimes != regimes.shift(1)
        regime_starts = regimes.index[regime_changes]

        for i in range(len(regime_starts) - 1):
            start_idx = regime_starts[i]
            end_idx = regime_starts[i + 1]

            # If regime duration is too short, change it to the previous regime
            duration_days = (end_idx - start_idx).days if hasattr(end_idx - start_idx, 'days') else 1
            if duration_days < self.min_regime_duration:
                if i > 0:
                    prev_regime = filtered_regimes.loc[regime_starts[i - 1]]
                    filtered_regimes.loc[start_idx:end_idx] = prev_regime

        return filtered_regimes

    def rolling_prediction(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, List[float]]:
        """
        Enhanced rolling prediction with confidence tracking
        """
        returns = prices.pct_change().dropna()
        true_regimes = self.create_improved_true_regimes(prices, returns)

        predicted_regimes = pd.Series(index=prices.index, dtype=str, name='predicted_regime')
        confidence_scores = pd.Series(index=prices.index, dtype=float, name='confidence')
        accuracies = []

        # Start predictions after we have enough data
        start_idx = self.lookback_window

        print(f"Starting rolling predictions from index {start_idx}")

        for i in range(start_idx, len(prices), self.rebalance_freq):
            # Get training data
            train_end = i
            train_start = max(0, train_end - self.lookback_window)
            train_prices = prices.iloc[train_start:train_end]

            # Get prediction period
            pred_end = min(len(prices), i + self.rebalance_freq)
            pred_prices = prices.iloc[train_start:pred_end]

            try:
                print(f"Processing period {i // self.rebalance_freq + 1}: "
                      f"train [{train_start}:{train_end}], predict [{i}:{pred_end}]")

                # Train model
                detector = ImprovedMarketRegimeDetector(n_regimes=3, random_state=42)
                detector.fit(train_prices)

                # Make predictions
                regime_pred, confidence_pred = detector.predict_regime(pred_prices)

                # Store predictions for the new period only
                new_pred_start = len(regime_pred) - (pred_end - i)
                new_predictions = regime_pred.iloc[new_pred_start:]
                new_confidence = confidence_pred.iloc[new_pred_start:]

                predicted_regimes.loc[new_predictions.index] = new_predictions.values
                confidence_scores.loc[new_confidence.index] = new_confidence.values

                # Calculate accuracy for this period
                if len(new_predictions) > 0:
                    true_period = true_regimes.loc[new_predictions.index]
                    valid_mask = (true_period.notna()) & (new_predictions.notna())

                    if valid_mask.sum() > 0:
                        accuracy = accuracy_score(
                            true_period[valid_mask],
                            new_predictions[valid_mask]
                        )
                        accuracies.append(accuracy)

                        # Print progress
                        regime_dist = new_predictions.value_counts()
                        print(f"  Period accuracy: {accuracy:.3f}, "
                              f"predicted regimes: {regime_dist.to_dict()}")

            except Exception as e:
                print(f"Error at index {i}: {e}")
                # Fill with 'Sideways' as default
                pred_indices = prices.index[i:pred_end]
                predicted_regimes.loc[pred_indices] = 'Sideways'
                confidence_scores.loc[pred_indices] = 0.33
                accuracies.append(0.33)  # Random guess accuracy for 3 classes

        # Clean up series
        predicted_regimes = predicted_regimes.dropna()
        confidence_scores = confidence_scores.dropna()

        # Print final distribution
        if len(predicted_regimes) > 0:
            final_dist = predicted_regimes.value_counts(normalize=True)
            print(f"Final predicted regime distribution: {final_dist.to_dict()}")

        return predicted_regimes, true_regimes, confidence_scores, accuracies

    def calculate_regime_based_returns(self, prices: pd.Series, predicted_regimes: pd.Series,
                                       confidence_scores: pd.Series = None) -> pd.DataFrame:
        """
        Calculate returns for different regime-based strategies including confidence-weighted
        """
        returns = prices.pct_change().dropna()

        # Align all series
        common_index = returns.index.intersection(predicted_regimes.index)
        returns_aligned = returns.loc[common_index]
        regimes_aligned = predicted_regimes.loc[common_index]

        if confidence_scores is not None:
            confidence_aligned = confidence_scores.loc[common_index]
        else:
            confidence_aligned = pd.Series(1.0, index=common_index)

        # Strategy returns
        strategy_returns = pd.DataFrame(index=common_index)

        # Buy and Hold
        strategy_returns['Buy_Hold'] = returns_aligned

        # Basic regime strategy
        regime_weights = {'Bull': 1.0, 'Sideways': 0.5, 'Bear': 0.0}
        strategy_returns['Regime_Strategy'] = returns_aligned * regimes_aligned.map(regime_weights)

        # Confidence-weighted strategy
        base_weights = regimes_aligned.map(regime_weights)
        confidence_weights = base_weights * confidence_aligned + (1 - confidence_aligned) * 0.5
        strategy_returns['Confidence_Weighted'] = returns_aligned * confidence_weights

        # Tactical strategy with confidence adjustment
        tactical_weights = {'Bull': 1.2, 'Sideways': 0.5, 'Bear': -0.2}
        base_tactical = regimes_aligned.map(tactical_weights)
        confidence_tactical = base_tactical * confidence_aligned + (1 - confidence_aligned) * 0.5
        strategy_returns['Confidence_Tactical'] = returns_aligned * confidence_tactical

        return strategy_returns

    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {
                'Total Return': 0, 'Annualized Return': 0, 'Volatility': 0,
                'Sharpe Ratio': 0, 'Max Drawdown': 0, 'Win Rate': 0,
                'Calmar Ratio': 0, 'Sortino Ratio': 0
            }

        total_return = (1 + returns_clean).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns_clean)) - 1 if len(returns_clean) > 0 else 0
        volatility = returns_clean.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

        # Drawdown calculation
        cumulative = (1 + returns_clean).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Additional metrics
        win_rate = (returns_clean > 0).mean()
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns_clean[returns_clean < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0

        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Calmar Ratio': calmar_ratio,
            'Sortino Ratio': sortino_ratio
        }

    def run_backtest(self, ticker: str = 'SPY', years: int = 2) -> Dict:
        """Run complete enhanced backtest"""
        print(f"Running enhanced backtest for {ticker} over {years} years...")

        # Download data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365 + 200)  # Extra days for warmup

        data = yf.download(ticker, start=start_date, end=end_date)
        prices = data['Close'].squeeze()

        print(f"Downloaded {len(prices)} data points from {prices.index[0]} to {prices.index[-1]}")

        # Run rolling predictions
        predicted_regimes, true_regimes, confidence_scores, accuracies = self.rolling_prediction(prices)

        # Calculate overall accuracy
        common_index = predicted_regimes.index.intersection(true_regimes.index)
        if len(common_index) > 0:
            overall_accuracy = accuracy_score(
                true_regimes.loc[common_index],
                predicted_regimes.loc[common_index]
            )
        else:
            overall_accuracy = 0.0

        # Calculate high-confidence accuracy
        high_conf_threshold = 0.7
        if len(confidence_scores) > 0:
            high_conf_mask = confidence_scores > high_conf_threshold
            high_conf_common = common_index.intersection(confidence_scores[high_conf_mask].index)
            if len(high_conf_common) > 0:
                high_conf_accuracy = accuracy_score(
                    true_regimes.loc[high_conf_common],
                    predicted_regimes.loc[high_conf_common]
                )
            else:
                high_conf_accuracy = 0.0
        else:
            high_conf_accuracy = 0.0

        # Calculate strategy returns
        strategy_returns = self.calculate_regime_based_returns(prices, predicted_regimes, confidence_scores)

        # Calculate performance metrics
        performance_metrics = {}
        for strategy in strategy_returns.columns:
            performance_metrics[strategy] = self.calculate_performance_metrics(
                strategy_returns[strategy].dropna()
            )

        # Get regime distribution
        regime_distribution = predicted_regimes.value_counts(normalize=True) if len(
            predicted_regimes) > 0 else pd.Series()

        # Store results
        results = {
            'prices': prices,
            'predicted_regimes': predicted_regimes,
            'true_regimes': true_regimes,
            'confidence_scores': confidence_scores,
            'strategy_returns': strategy_returns,
            'performance_metrics': performance_metrics,
            'overall_accuracy': overall_accuracy,
            'high_conf_accuracy': high_conf_accuracy,
            'rolling_accuracies': accuracies,
            'regime_distribution': regime_distribution
        }

        self.results = results
        return results

    def plot_results(self):
        """Create comprehensive plots of backtest results"""
        if not self.results:
            print("No results to plot. Run backtest first.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(24, 16))

        # Plot 1: Price and Regime Evolution
        ax1 = axes[0, 0]
        prices = self.results['prices']
        regimes = self.results['predicted_regimes']
        confidence = self.results['confidence_scores']

        # Plot price
        ax1.plot(prices.index, prices.values, 'k-', linewidth=1, alpha=0.7, label='Price')

        # Color code regimes with confidence
        colors = {'Bear': 'red', 'Sideways': 'orange', 'Bull': 'green'}
        for regime in colors.keys():
            mask = regimes == regime
            if mask.any():
                regime_prices = prices.loc[regimes.index[mask]]
                regime_conf = confidence.loc[regimes.index[mask]] if len(confidence) > 0 else pd.Series(1.0, index=
                regimes.index[mask])
                ax1.scatter(regimes.index[mask], regime_prices,
                            c=colors[regime], alpha=regime_conf * 0.8, s=20, label=f'{regime} Regime')

        ax1.set_title('Price Evolution with Predicted Regimes\n(Transparency = Confidence)')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Strategy Performance
        ax2 = axes[0, 1]
        strategy_returns = self.results['strategy_returns']
        cumulative_returns = (1 + strategy_returns).cumprod()

        for strategy in cumulative_returns.columns:
            ax2.plot(cumulative_returns.index, cumulative_returns[strategy],
                     linewidth=2, label=strategy)

        ax2.set_title('Cumulative Strategy Returns')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Performance Metrics Comparison
        ax3 = axes[0, 2]
        metrics_data = self.results['performance_metrics']
        strategies = list(metrics_data.keys())
        metrics_to_plot = ['Annualized Return', 'Sharpe Ratio', 'Max Drawdown']

        x = np.arange(len(strategies))
        width = 0.25

        for i, metric in enumerate(metrics_to_plot):
            values = [metrics_data[strategy][metric] for strategy in strategies]
            ax3.bar(x + i * width, values, width, label=metric)

        ax3.set_title('Performance Metrics Comparison')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Value')
        ax3.set_xticks(x + width)
        ax3.set_xticklabels(strategies, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Regime Distribution
        ax4 = axes[1, 0]
        regime_dist = self.results['regime_distribution']
        if len(regime_dist) > 0:
            colors_dist = [colors.get(regime, 'gray') for regime in regime_dist.index]
            ax4.pie(regime_dist.values, labels=regime_dist.index, autopct='%1.1f%%',
                    colors=colors_dist)
        ax4.set_title('Predicted Regime Distribution')

        # Plot 5: Rolling Accuracy
        ax5 = axes[1, 1]
        rolling_acc = self.results['rolling_accuracies']
        if len(rolling_acc) > 0:
            ax5.plot(range(len(rolling_acc)), rolling_acc, 'b-', linewidth=2)
            ax5.axhline(y=1 / 3, color='r', linestyle='--', alpha=0.7, label='Random Guess')
            ax5.axhline(y=np.mean(rolling_acc), color='g', linestyle='--', alpha=0.7,
                        label=f'Mean Accuracy: {np.mean(rolling_acc):.3f}')
        ax5.set_title('Rolling Prediction Accuracy')
        ax5.set_xlabel('Rebalancing Period')
        ax5.set_ylabel('Accuracy')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Confidence Distribution
        ax6 = axes[1, 2]
        if len(confidence) > 0:
            ax6.hist(confidence.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax6.axvline(x=confidence.mean(), color='red', linestyle='--',
                        label=f'Mean: {confidence.mean():.3f}')
        ax6.set_title('Confidence Score Distribution')
        ax6.set_xlabel('Confidence Score')
        ax6.set_ylabel('Frequency')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print summary statistics
        self.print_summary()

    def print_summary(self):
        """Print detailed summary of backtest results"""
        if not self.results:
            print("No results to summarize. Run backtest first.")
            return

        print("\n" + "=" * 80)
        print("ENHANCED HMM MARKET REGIME BACKTEST SUMMARY")
        print("=" * 80)

        # Overall Performance
        print(f"\n📊 OVERALL ACCURACY METRICS:")
        print(f"Overall Accuracy: {self.results['overall_accuracy']:.1%}")
        print(f"High Confidence Accuracy: {self.results['high_conf_accuracy']:.1%}")
        print(f"Rolling Periods: {len(self.results['rolling_accuracies'])}")
        if len(self.results['rolling_accuracies']) > 0:
            print(f"Mean Rolling Accuracy: {np.mean(self.results['rolling_accuracies']):.1%}")
            print(f"Std Rolling Accuracy: {np.std(self.results['rolling_accuracies']):.1%}")

        # Regime Distribution
        print(f"\n🎯 REGIME DISTRIBUTION:")
        regime_dist = self.results['regime_distribution']
        for regime, pct in regime_dist.items():
            print(f"{regime}: {pct:.1%}")

        # Strategy Performance
        print(f"\n💰 STRATEGY PERFORMANCE:")
        performance = self.results['performance_metrics']

        print(
            f"{'Strategy':<20} {'Total Ret':<10} {'Ann. Ret':<10} {'Volatility':<12} {'Sharpe':<8} {'Max DD':<10}")
        print("-" * 80)

        for strategy, metrics in performance.items():
            print(f"{strategy:<20} {metrics['Total Return']:>9.1%} {metrics['Annualized Return']:>9.1%} "
                  f"{metrics['Volatility']:>11.1%} {metrics['Sharpe Ratio']:>7.2f} {metrics['Max Drawdown']:>9.1%}")

        # Risk-Adjusted Performance
        print(f"\n📈 RISK-ADJUSTED METRICS:")
        print(f"{'Strategy':<20} {'Calmar':<8} {'Sortino':<8} {'Win Rate':<10}")
        print("-" * 50)

        for strategy, metrics in performance.items():
            print(f"{strategy:<20} {metrics['Calmar Ratio']:>7.2f} {metrics['Sortino Ratio']:>7.2f} "
                  f"{metrics['Win Rate']:>9.1%}")

        # Confidence Statistics
        confidence = self.results['confidence_scores']
        if len(confidence) > 0:
            print(f"\n🔍 CONFIDENCE STATISTICS:")
            print(f"Mean Confidence: {confidence.mean():.3f}")
            print(f"Std Confidence: {confidence.std():.3f}")
            print(f"Min Confidence: {confidence.min():.3f}")
            print(f"Max Confidence: {confidence.max():.3f}")
            print(f"High Confidence (>0.7) %: {(confidence > 0.7).mean():.1%}")

        print("\n" + "=" * 80)

def main():
    """Main function to run the enhanced backtest"""
    # Initialize backtester
    backtester = ImprovedHMMBacktester(
        lookback_window=252,  # 1 year training window
        rebalance_freq=21,  # Monthly rebalancing
        min_regime_duration=5  # Minimum 5 days for regime
    )

    # Run backtest
    try:
        results = backtester.run_backtest(ticker='SPY', years=3)

        # Plot results
        backtester.plot_results()

        # Additional analysis: Test different parameters
        print("\n" + "=" * 60)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("=" * 60)

        # Test different rebalancing frequencies
        rebalance_freqs = [10, 21, 42]  # 2 weeks, 1 month, 2 months

        for freq in rebalance_freqs:
            print(f"\nTesting rebalancing frequency: {freq} days")
            test_backtester = ImprovedHMMBacktester(
                lookback_window=252,
                rebalance_freq=freq,
                min_regime_duration=5
            )

            try:
                test_results = test_backtester.run_backtest(ticker='SPY', years=2)

                # Quick summary
                perf = test_results['performance_metrics']['Confidence_Weighted']
                print(f"  Confidence Weighted Strategy:")
                print(f"    Annual Return: {perf['Annualized Return']:.1%}")
                print(f"    Sharpe Ratio: {perf['Sharpe Ratio']:.2f}")
                print(f"    Max Drawdown: {perf['Max Drawdown']:.1%}")
                print(f"    Overall Accuracy: {test_results['overall_accuracy']:.1%}")

            except Exception as e:
                print(f"  Error: {e}")

        # Test different assets
        assets = ['QQQ', 'IWM', 'EFA']  # Tech, Small Cap, International

        print(f"\n" + "=" * 40)
        print("CROSS-ASSET ANALYSIS")
        print("=" * 40)

        for asset in assets:
            print(f"\nTesting asset: {asset}")
            asset_backtester = ImprovedHMMBacktester(
                lookback_window=252,
                rebalance_freq=21,
                min_regime_duration=5
            )

            try:
                asset_results = asset_backtester.run_backtest(ticker=asset, years=2)

                # Quick summary
                perf = asset_results['performance_metrics']['Confidence_Weighted']
                print(f"  Confidence Weighted Strategy:")
                print(f"    Annual Return: {perf['Annualized Return']:.1%}")
                print(f"    Sharpe Ratio: {perf['Sharpe Ratio']:.2f}")
                print(f"    Max Drawdown: {perf['Max Drawdown']:.1%}")
                print(f"    Overall Accuracy: {asset_results['overall_accuracy']:.1%}")

            except Exception as e:
                print(f"  Error: {e}")

    except Exception as e:
        print(f"Error running main backtest: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()