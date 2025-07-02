import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
import yfinance as yf
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class MarketRegimeDetector:
    def __init__(self, n_regimes=3):
        """
        Initialize the Market Regime Detector

        Parameters:
        n_regimes (int): Number of market regimes (default: 3 for bear/bull/sideways)
        """
        self.n_regimes = n_regimes
        self.model = None
        self.regime_names = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        self.features = None
        self.returns = None

    def prepare_features(self, prices):
        """
        Prepare features for HMM model

        Parameters:
        prices (pd.Series): Price data

        Returns:
        np.array: Feature matrix
        """
        # Ensure prices is a pandas Series with proper index
        if not isinstance(prices, pd.Series):
            if isinstance(prices, pd.DataFrame):
                # If it's a DataFrame, take the first column
                prices = prices.iloc[:, 0]
                if hasattr(prices, 'squeeze'):
                    prices = prices.squeeze()
            else:
                # If it's a numpy array, convert to Series
                prices = pd.Series(prices.flatten() if hasattr(prices, 'flatten') else prices)

        # Calculate returns
        returns = prices.pct_change().dropna()
        self.returns = returns

        # Calculate rolling statistics with shorter windows for stability
        window_short = 10
        window_long = 20

        rolling_mean = returns.rolling(window=window_short).mean()
        rolling_std = returns.rolling(window=window_short).std()

        # Calculate momentum indicators
        momentum_5 = prices.pct_change(5)
        momentum_10 = prices.pct_change(10)

        # Calculate volatility (annualized)
        volatility = returns.rolling(window=window_short).std() * np.sqrt(252)

        # Price relative to moving averages
        ma_short = prices.rolling(window=window_short).mean()
        ma_long = prices.rolling(window=window_long).mean()
        price_to_ma_short = (prices - ma_short) / ma_short
        price_to_ma_long = (prices - ma_long) / ma_long

        # RSI-like indicator
        delta = returns
        gain = delta.where(delta > 0, 0).rolling(window=window_short).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window_short).mean()
        rs = gain / (loss + 1e-10)  # Add small value to avoid division by zero
        rsi = 100 - (100 / (1 + rs))

        # Align all series to the same index (start from where all features are available)
        series_list = [returns, rolling_mean, rolling_std, momentum_5, momentum_10,
                       volatility, price_to_ma_short, price_to_ma_long, rsi]

        # Find common index
        common_index = returns.index
        for series in series_list:
            if hasattr(series, 'dropna'):
                common_index = common_index.intersection(series.dropna().index)

        # Take only the most recent data to avoid too much history affecting current regime
        if len(common_index) > 500:
            common_index = common_index[-500:]  # Keep last 500 data points

        # Combine features using the common index
        features_df = pd.DataFrame({
            'returns': returns.loc[common_index],
            'rolling_mean': rolling_mean.loc[common_index],
            'rolling_std': rolling_std.loc[common_index],
            'momentum_5': momentum_5.loc[common_index],
            'momentum_10': momentum_10.loc[common_index],
            'volatility': volatility.loc[common_index],
            'price_to_ma_short': price_to_ma_short.loc[common_index],
            'price_to_ma_long': price_to_ma_long.loc[common_index],
            'rsi': rsi.loc[common_index]
        }, index=common_index)

        # Drop any remaining NaN values
        features_df = features_df.dropna()

        # Handle any infinite or very large values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()

        # Standardize features
        features_normalized = (features_df - features_df.mean()) / features_df.std()

        # Replace any remaining NaN values with 0
        features_normalized = features_normalized.fillna(0)

        # Add small amount of noise to avoid perfect correlations
        np.random.seed(42)
        noise = np.random.normal(0, 1e-6, features_normalized.shape)
        features_normalized += noise

        self.features = features_normalized
        return features_normalized.values

    def fit(self, prices):
        """
        Fit the HMM model to price data

        Parameters:
        prices (pd.Series): Price data with datetime index
        """
        # Prepare features
        X = self.prepare_features(prices)

        # Initialize and fit Gaussian HMM with more robust settings
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",  # Use diagonal covariance to avoid singularity
            n_iter=100,
            tol=1e-4,
            random_state=42,
            init_params="stmc"  # Initialize all parameters
        )

        # Set minimum covariance to ensure numerical stability
        self.model.min_covar = 1e-6

        # Fit the model
        self.model.fit(X)

        # Predict states
        states = self.model.predict(X)

        # Assign regime names based on mean returns
        regime_returns = {}
        for i in range(self.n_regimes):
            regime_mask = states == i
            if np.sum(regime_mask) > 0:
                regime_returns[i] = np.mean(self.returns.iloc[-len(states):][regime_mask])

        # Sort regimes by mean returns (bear < sideways < bull)
        sorted_regimes = sorted(regime_returns.items(), key=lambda x: x[1])

        # Update regime names mapping
        self.regime_names = {}
        regime_labels = ['Bear', 'Sideways', 'Bull']
        for idx, (regime_id, _) in enumerate(sorted_regimes):
            self.regime_names[regime_id] = regime_labels[idx]

        return self

    def predict_regime(self, prices):
        """
        Predict market regimes for given prices

        Parameters:
        prices (pd.Series): Price data

        Returns:
        pd.Series: Predicted regimes
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        X = self.prepare_features(prices)
        states = self.model.predict(X)

        # Map states to regime names
        regime_series = pd.Series(
            [self.regime_names[state] for state in states],
            index=self.features.index,
            name='Regime'
        )

        return regime_series

    def get_regime_probabilities(self, prices):
        """
        Get probabilities for each regime

        Parameters:
        prices (pd.Series): Price data

        Returns:
        pd.DataFrame: Probabilities for each regime
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")

        X = self.prepare_features(prices)
        probs = self.model.predict_proba(X)

        # Create DataFrame with regime probabilities
        prob_df = pd.DataFrame(
            probs,
            index=self.features.index,
            columns=[self.regime_names[i] for i in range(self.n_regimes)]
        )

        return prob_df

    def plot_regimes(self, prices, regimes, save_path=None):
        """
        Plot price data with regime overlay

        Parameters:
        prices (pd.Series): Price data
        regimes (pd.Series): Predicted regimes
        save_path (str): Path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

        # Plot prices
        ax1.plot(prices.index, prices.values, 'k-', linewidth=1, alpha=0.7)
        ax1.set_ylabel('Price')
        ax1.set_title('Stock Price and Market Regimes')
        ax1.grid(True, alpha=0.3)

        # Color code regimes
        colors = {'Bear': 'red', 'Sideways': 'orange', 'Bull': 'green'}
        for regime in colors.keys():
            mask = regimes == regime
            if mask.any():
                ax1.scatter(regimes.index[mask], prices.loc[regimes.index[mask]],
                            c=colors[regime], alpha=0.6, s=20, label=regime)

        ax1.legend()

        # Plot regime timeline
        regime_numeric = regimes.map({'Bear': 0, 'Sideways': 1, 'Bull': 2})
        ax2.plot(regime_numeric.index, regime_numeric.values, 'o-', markersize=3)
        ax2.set_ylabel('Regime')
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['Bear', 'Sideways', 'Bull'])
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def get_regime_statistics(self, regimes):
        """
        Get statistics for each regime

        Parameters:
        regimes (pd.Series): Predicted regimes

        Returns:
        pd.DataFrame: Regime statistics
        """
        stats = {}

        for regime in ['Bear', 'Sideways', 'Bull']:
            mask = regimes == regime
            if mask.any():
                regime_returns = self.returns.loc[regimes.index[mask]]
                stats[regime] = {
                    'Count': mask.sum(),
                    'Percentage': (mask.sum() / len(regimes)) * 100,
                    'Mean_Return': regime_returns.mean() * 252,  # Annualized
                    'Volatility': regime_returns.std() * np.sqrt(252),  # Annualized
                    'Sharpe_Ratio': (regime_returns.mean() / regime_returns.std()) * np.sqrt(
                        252) if regime_returns.std() > 0 else 0
                }

        return pd.DataFrame(stats).T


# Example usage and demonstration
def main():
    # Download sample data (S&P 500)
    print("Downloading S&P 500 data...")
    ticker = "^GSPC"
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)  # 3 years of data (reduced for stability)

    data = yf.download(ticker, start=start_date, end=end_date)
    # Extract the Close price as a Series
    prices = data['Close'].squeeze()  # Use squeeze() to ensure it's a Series

    print(f"Data downloaded: {len(prices)} data points from {prices.index[0]} to {prices.index[-1]}")
    print(f"Price data type: {type(prices)}")
    print(f"Price data shape: {prices.shape}")

    # Initialize and fit the model
    detector = MarketRegimeDetector(n_regimes=3)

    try:
        detector.fit(prices)
        print("Model fitted successfully!")

        # Predict regimes
        regimes = detector.predict_regime(prices)

        # Get regime probabilities
        probabilities = detector.get_regime_probabilities(prices)

        # Get regime statistics
        stats = detector.get_regime_statistics(regimes)

        print("\n=== Market Regime Statistics ===")
        print(stats.round(4))

        # Display recent regimes
        print(f"\n=== Recent Regime Predictions ===")
        print(regimes.tail(10))

        # Display current regime probabilities
        print(f"\n=== Current Regime Probabilities ===")
        print(probabilities.iloc[-1].round(4))

        # Plot results
        detector.plot_regimes(prices, regimes)

        # Plot regime probabilities
        plt.figure(figsize=(15, 8))
        for regime in probabilities.columns:
            plt.plot(probabilities.index, probabilities[regime],
                     label=f'{regime} Probability', linewidth=2)

        plt.title('Market Regime Probabilities Over Time')
        plt.xlabel('Date')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        return detector, regimes, probabilities, stats

    except Exception as e:
        print(f"Error during model fitting: {e}")
        print("Trying with simpler model configuration...")

        # Fallback to even simpler model
        detector_simple = MarketRegimeDetector(n_regimes=2)  # Just bull/bear
        detector_simple.fit(prices)

        regimes = detector_simple.predict_regime(prices)
        probabilities = detector_simple.get_regime_probabilities(prices)
        stats = detector_simple.get_regime_statistics(regimes)

        print("\n=== Simplified Model Results ===")
        print(stats.round(4))

        return detector_simple, regimes, probabilities, stats


if __name__ == "__main__":
    # Run the example
    detector, regimes, probabilities, stats = main()

    # Additional analysis
    print("\n=== Model Parameters ===")
    print(f"Number of regimes: {detector.n_regimes}")
    print(f"Transition matrix:")
    print(detector.model.transmat_.round(3))

    print(f"\nMean returns by regime:")
    for i in range(detector.n_regimes):
        mean_return = detector.model.means_[i][0]  # First feature is returns
        print(f"{detector.regime_names[i]}: {mean_return:.6f}")