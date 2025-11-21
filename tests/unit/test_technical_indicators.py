"""
Unit tests for technical indicators calculations.
Testing: SMA, EMA, RSI, MACD, Bollinger Bands, Stochastic Oscillator
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.mark.unit
class TestSimpleMovingAverage:
    """Tests for Simple Moving Average (SMA) calculation."""

    def test_sma_calculation_basic(self):
        """Test basic SMA calculation with known values."""
        # Given: A series of prices
        prices = pd.Series([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
        window = 5

        # When: SMA is calculated
        from technical_indicators import calculate_sma
        result = calculate_sma(prices, window)

        # Then: First 4 values should be NaN, then moving averages
        assert pd.isna(result.iloc[0:4]).all()
        assert result.iloc[4] == 14.0  # (10+12+14+16+18)/5
        assert result.iloc[5] == 16.0  # (12+14+16+18+20)/5
        assert result.iloc[9] == 24.0  # (20+22+24+26+28)/5

    def test_sma_with_insufficient_data(self):
        """Test SMA when data length is less than window size."""
        prices = pd.Series([10, 12, 14])
        window = 5

        from technical_indicators import calculate_sma
        result = calculate_sma(prices, window)

        # All values should be NaN
        assert pd.isna(result).all()

    def test_sma_with_different_windows(self):
        """Test SMA with multiple window sizes (20, 50, 200 days)."""
        prices = pd.Series(range(1, 201))  # 200 data points

        from technical_indicators import calculate_sma

        sma_20 = calculate_sma(prices, 20)
        sma_50 = calculate_sma(prices, 50)
        sma_200 = calculate_sma(prices, 200)

        assert not pd.isna(sma_20.iloc[-1])
        assert not pd.isna(sma_50.iloc[-1])
        assert not pd.isna(sma_200.iloc[-1])

        # Later SMAs should have higher values for increasing series
        assert sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]

    def test_sma_with_empty_series(self):
        """Test SMA with empty price series."""
        prices = pd.Series([])

        from technical_indicators import calculate_sma
        result = calculate_sma(prices, 5)

        assert len(result) == 0

    def test_sma_with_nan_values(self):
        """Test SMA handling of NaN values in data."""
        prices = pd.Series([10, 12, np.nan, 16, 18, 20])

        from technical_indicators import calculate_sma
        result = calculate_sma(prices, 3)

        # Should handle NaN appropriately
        assert isinstance(result, pd.Series)


@pytest.mark.unit
class TestExponentialMovingAverage:
    """Tests for Exponential Moving Average (EMA) calculation."""

    def test_ema_calculation_basic(self):
        """Test basic EMA calculation."""
        prices = pd.Series([22, 24, 25, 23, 26, 28, 26, 29, 27, 28])
        window = 5

        from technical_indicators import calculate_ema
        result = calculate_ema(prices, window)

        # EMA should exist for all values after initial
        assert not pd.isna(result.iloc[-1])
        assert len(result) == len(prices)

    def test_ema_more_responsive_than_sma(self):
        """Test that EMA reacts faster to price changes than SMA."""
        # Create a price series with a sudden jump
        prices = pd.Series([20] * 10 + [30] * 10)

        from technical_indicators import calculate_sma, calculate_ema

        sma = calculate_sma(prices, 10)
        ema = calculate_ema(prices, 10)

        # After the jump, EMA should be closer to new price than SMA
        assert ema.iloc[15] > sma.iloc[15]

    def test_ema_common_periods(self):
        """Test EMA with common periods (12, 26 for MACD)."""
        prices = pd.Series(range(1, 51))

        from technical_indicators import calculate_ema

        ema_12 = calculate_ema(prices, 12)
        ema_26 = calculate_ema(prices, 26)

        assert not pd.isna(ema_12.iloc[-1])
        assert not pd.isna(ema_26.iloc[-1])


@pytest.mark.unit
class TestRelativeStrengthIndex:
    """Tests for Relative Strength Index (RSI) calculation."""

    def test_rsi_range_bounds(self):
        """Test that RSI values are always between 0 and 100."""
        # Create volatile price data
        np.random.seed(42)
        prices = pd.Series(np.random.uniform(50, 150, 100))

        from technical_indicators import calculate_rsi
        rsi = calculate_rsi(prices, period=14)

        # Remove NaN values
        rsi_values = rsi.dropna()

        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()

    def test_rsi_overbought_condition(self):
        """Test RSI in overbought condition (rising prices)."""
        # Steadily increasing prices
        prices = pd.Series(range(50, 100))

        from technical_indicators import calculate_rsi
        rsi = calculate_rsi(prices, period=14)

        # RSI should be high (>70 typically overbought)
        assert rsi.iloc[-1] > 70

    def test_rsi_oversold_condition(self):
        """Test RSI in oversold condition (falling prices)."""
        # Steadily decreasing prices
        prices = pd.Series(range(100, 50, -1))

        from technical_indicators import calculate_rsi
        rsi = calculate_rsi(prices, period=14)

        # RSI should be low (<30 typically oversold)
        assert rsi.iloc[-1] < 30

    def test_rsi_neutral_condition(self):
        """Test RSI with sideways/neutral price action."""
        # Prices oscillating around same level
        prices = pd.Series([50, 52, 49, 51, 50, 52, 49, 51] * 5)

        from technical_indicators import calculate_rsi
        rsi = calculate_rsi(prices, period=14)

        # RSI should be near 50 (neutral)
        assert 40 < rsi.iloc[-1] < 60

    def test_rsi_period_14(self):
        """Test standard RSI with 14-period."""
        prices = pd.Series(range(1, 51))

        from technical_indicators import calculate_rsi
        rsi = calculate_rsi(prices, period=14)

        # First 14 values should be NaN
        assert pd.isna(rsi.iloc[:14]).all()
        assert not pd.isna(rsi.iloc[14])


@pytest.mark.unit
class TestMACD:
    """Tests for Moving Average Convergence Divergence (MACD) calculation."""

    def test_macd_basic_calculation(self):
        """Test MACD line, signal line, and histogram calculation."""
        prices = pd.Series(range(1, 101))

        from technical_indicators import calculate_macd
        macd_line, signal_line, histogram = calculate_macd(prices)

        # All three components should be returned
        assert isinstance(macd_line, pd.Series)
        assert isinstance(signal_line, pd.Series)
        assert isinstance(histogram, pd.Series)

        # Lengths should match
        assert len(macd_line) == len(signal_line) == len(histogram) == len(prices)

    def test_macd_histogram_calculation(self):
        """Test that histogram = MACD line - signal line."""
        prices = pd.Series(range(50, 150))

        from technical_indicators import calculate_macd
        macd_line, signal_line, histogram = calculate_macd(prices)

        # Check histogram calculation (ignoring NaN values)
        valid_indices = ~pd.isna(histogram)
        calculated_histogram = macd_line[valid_indices] - signal_line[valid_indices]

        pd.testing.assert_series_equal(
            histogram[valid_indices],
            calculated_histogram,
            check_names=False
        )

    def test_macd_bullish_crossover(self):
        """Test MACD bullish crossover (MACD crosses above signal)."""
        # Create price data that goes from down to up trend
        prices = pd.Series(list(range(100, 50, -1)) + list(range(50, 100)))

        from technical_indicators import calculate_macd
        macd_line, signal_line, histogram = calculate_macd(prices)

        # Find crossover points where MACD crosses above signal
        crossovers = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))

        # Should have at least one bullish crossover
        assert crossovers.any()

    def test_macd_bearish_crossover(self):
        """Test MACD bearish crossover (MACD crosses below signal)."""
        # Create price data that goes from up to down trend
        prices = pd.Series(list(range(50, 100)) + list(range(100, 50, -1)))

        from technical_indicators import calculate_macd
        macd_line, signal_line, histogram = calculate_macd(prices)

        # Find crossover points where MACD crosses below signal
        crossovers = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

        # Should have at least one bearish crossover
        assert crossovers.any()

    def test_macd_custom_parameters(self):
        """Test MACD with custom fast, slow, and signal periods."""
        prices = pd.Series(range(1, 101))

        from technical_indicators import calculate_macd

        # Custom parameters
        macd_line, signal_line, histogram = calculate_macd(
            prices,
            fast_period=8,
            slow_period=21,
            signal_period=5
        )

        assert not pd.isna(macd_line.iloc[-1])
        assert not pd.isna(signal_line.iloc[-1])


@pytest.mark.unit
class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_bands_basic(self):
        """Test basic Bollinger Bands calculation."""
        prices = pd.Series(range(50, 150))

        from technical_indicators import calculate_bollinger_bands
        upper, middle, lower = calculate_bollinger_bands(prices, period=20, std_dev=2)

        # All three bands should be returned
        assert isinstance(upper, pd.Series)
        assert isinstance(middle, pd.Series)
        assert isinstance(lower, pd.Series)

        # Lengths should match
        assert len(upper) == len(middle) == len(lower) == len(prices)

    def test_bollinger_bands_ordering(self):
        """Test that upper > middle > lower always."""
        np.random.seed(42)
        prices = pd.Series(np.random.uniform(50, 150, 100))

        from technical_indicators import calculate_bollinger_bands
        upper, middle, lower = calculate_bollinger_bands(prices)

        # Check ordering (ignoring NaN values)
        valid_indices = ~pd.isna(upper)
        assert (upper[valid_indices] >= middle[valid_indices]).all()
        assert (middle[valid_indices] >= lower[valid_indices]).all()

    def test_bollinger_bands_middle_is_sma(self):
        """Test that middle band equals SMA."""
        prices = pd.Series(range(50, 150))
        period = 20

        from technical_indicators import calculate_bollinger_bands, calculate_sma
        upper, middle, lower = calculate_bollinger_bands(prices, period=period)
        sma = calculate_sma(prices, window=period)

        # Middle band should equal SMA
        pd.testing.assert_series_equal(middle, sma, check_names=False)

    def test_bollinger_bands_volatility_expansion(self):
        """Test that bands widen during high volatility."""
        # Low volatility period
        low_vol_prices = pd.Series([100] * 30)
        # High volatility period
        high_vol_prices = pd.Series([100, 110, 90, 105, 95, 115, 85] * 5)

        from technical_indicators import calculate_bollinger_bands

        upper_low, middle_low, lower_low = calculate_bollinger_bands(low_vol_prices)
        upper_high, middle_high, lower_high = calculate_bollinger_bands(high_vol_prices)

        # Band width for low volatility
        low_width = upper_low.iloc[-1] - lower_low.iloc[-1]
        # Band width for high volatility
        high_width = upper_high.iloc[-1] - lower_high.iloc[-1]

        # High volatility should have wider bands
        assert high_width > low_width

    def test_bollinger_bands_different_std_dev(self):
        """Test Bollinger Bands with different standard deviations."""
        prices = pd.Series(range(50, 150))

        from technical_indicators import calculate_bollinger_bands

        upper_2std, middle_2std, lower_2std = calculate_bollinger_bands(prices, std_dev=2)
        upper_3std, middle_3std, lower_3std = calculate_bollinger_bands(prices, std_dev=3)

        # 3 std dev bands should be wider than 2 std dev bands
        assert upper_3std.iloc[-1] > upper_2std.iloc[-1]
        assert lower_3std.iloc[-1] < lower_2std.iloc[-1]


@pytest.mark.unit
class TestStochasticOscillator:
    """Tests for Stochastic Oscillator calculation."""

    def test_stochastic_range_bounds(self):
        """Test that Stochastic values are between 0 and 100."""
        # Create OHLC data
        np.random.seed(42)
        high = pd.Series(np.random.uniform(100, 150, 50))
        low = pd.Series(np.random.uniform(50, 100, 50))
        close = pd.Series(np.random.uniform(60, 140, 50))

        from technical_indicators import calculate_stochastic
        k_values, d_values = calculate_stochastic(high, low, close)

        # Remove NaN values
        k_valid = k_values.dropna()
        d_valid = d_values.dropna()

        assert (k_valid >= 0).all() and (k_valid <= 100).all()
        assert (d_valid >= 0).all() and (d_valid <= 100).all()

    def test_stochastic_overbought(self):
        """Test Stochastic in overbought condition."""
        # Price consistently at high end of range
        high = pd.Series([110] * 30)
        low = pd.Series([50] * 30)
        close = pd.Series([108] * 30)

        from technical_indicators import calculate_stochastic
        k_values, d_values = calculate_stochastic(high, low, close)

        # Should be near 100 (overbought)
        assert k_values.iloc[-1] > 80

    def test_stochastic_oversold(self):
        """Test Stochastic in oversold condition."""
        # Price consistently at low end of range
        high = pd.Series([110] * 30)
        low = pd.Series([50] * 30)
        close = pd.Series([52] * 30)

        from technical_indicators import calculate_stochastic
        k_values, d_values = calculate_stochastic(high, low, close)

        # Should be near 0 (oversold)
        assert k_values.iloc[-1] < 20


@pytest.mark.unit
class TestAverageTrueRange:
    """Tests for Average True Range (ATR) - volatility indicator."""

    def test_atr_calculation(self):
        """Test basic ATR calculation."""
        np.random.seed(42)
        high = pd.Series(np.random.uniform(100, 150, 50))
        low = pd.Series(np.random.uniform(50, 100, 50))
        close = pd.Series(np.random.uniform(60, 140, 50))

        from technical_indicators import calculate_atr
        atr = calculate_atr(high, low, close, period=14)

        # ATR should always be positive
        atr_valid = atr.dropna()
        assert (atr_valid > 0).all()

    def test_atr_higher_with_volatility(self):
        """Test that ATR increases with volatility."""
        # Low volatility
        high_low = pd.Series([101] * 30)
        low_low = pd.Series([99] * 30)
        close_low = pd.Series([100] * 30)

        # High volatility
        high_high = pd.Series([120] * 30)
        low_high = pd.Series([80] * 30)
        close_high = pd.Series([100] * 30)

        from technical_indicators import calculate_atr

        atr_low = calculate_atr(high_low, low_low, close_low)
        atr_high = calculate_atr(high_high, low_high, close_high)

        # High volatility should have higher ATR
        assert atr_high.iloc[-1] > atr_low.iloc[-1]


@pytest.mark.unit
class TestOnBalanceVolume:
    """Tests for On-Balance Volume (OBV) calculation."""

    def test_obv_calculation(self):
        """Test basic OBV calculation."""
        close = pd.Series([100, 102, 101, 105, 103, 107])
        volume = pd.Series([1000, 1500, 1200, 1800, 1300, 2000])

        from technical_indicators import calculate_obv
        obv = calculate_obv(close, volume)

        # OBV should have same length as input
        assert len(obv) == len(close)

    def test_obv_increases_on_up_days(self):
        """Test OBV increases when price goes up."""
        close = pd.Series([100, 105, 110])
        volume = pd.Series([1000, 1000, 1000])

        from technical_indicators import calculate_obv
        obv = calculate_obv(close, volume)

        # OBV should be increasing
        assert obv.iloc[1] > obv.iloc[0]
        assert obv.iloc[2] > obv.iloc[1]

    def test_obv_decreases_on_down_days(self):
        """Test OBV decreases when price goes down."""
        close = pd.Series([100, 95, 90])
        volume = pd.Series([1000, 1000, 1000])

        from technical_indicators import calculate_obv
        obv = calculate_obv(close, volume)

        # OBV should be decreasing
        assert obv.iloc[1] < obv.iloc[0]
        assert obv.iloc[2] < obv.iloc[1]


@pytest.mark.unit
class TestVolumeWeightedAveragePrice:
    """Tests for Volume-Weighted Average Price (VWAP)."""

    def test_vwap_calculation(self):
        """Test basic VWAP calculation."""
        high = pd.Series([105, 110, 108])
        low = pd.Series([95, 100, 98])
        close = pd.Series([100, 105, 103])
        volume = pd.Series([1000, 1500, 1200])

        from technical_indicators import calculate_vwap
        vwap = calculate_vwap(high, low, close, volume)

        # VWAP should exist
        assert not pd.isna(vwap.iloc[-1])
        assert len(vwap) == len(close)

    def test_vwap_within_price_range(self):
        """Test that VWAP is within high/low range."""
        np.random.seed(42)
        high = pd.Series(np.random.uniform(100, 150, 50))
        low = pd.Series(np.random.uniform(50, 100, 50))
        close = pd.Series(np.random.uniform(60, 140, 50))
        volume = pd.Series(np.random.uniform(1000, 5000, 50))

        from technical_indicators import calculate_vwap
        vwap = calculate_vwap(high, low, close, volume)

        # VWAP should generally be within the overall price range
        assert vwap.iloc[-1] >= low.min()
        assert vwap.iloc[-1] <= high.max()
