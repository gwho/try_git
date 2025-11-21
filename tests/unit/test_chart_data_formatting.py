"""
Unit tests for chart data formatting and validation.
Testing: Chart.js format, data validation, color schemes, date formatting
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.mark.unit
class TestChartJSFormatter:
    """Tests for formatting data for Chart.js library."""

    def test_format_line_chart_single_stock(self):
        """Test formatting single stock data for line chart."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = pd.Series(range(100, 110))

        from chart_formatter import format_line_chart

        chart_data = format_line_chart('AAPL', dates, prices)

        # Should have Chart.js structure
        assert 'label' in chart_data
        assert 'data' in chart_data
        assert 'borderColor' in chart_data
        assert 'backgroundColor' in chart_data
        assert 'fill' in chart_data

        assert chart_data['label'] == 'AAPL'
        assert len(chart_data['data']) == 10

    def test_format_multi_stock_comparison(self):
        """Test formatting multiple stocks for comparison chart."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        stocks = {
            'AAPL': pd.Series(range(100, 110)),
            'GOOGL': pd.Series(range(200, 210)),
            'MSFT': pd.Series(range(150, 160))
        }

        from chart_formatter import format_multi_stock_chart

        chart_data = format_multi_stock_chart(dates, stocks)

        # Should have labels and datasets
        assert 'labels' in chart_data
        assert 'datasets' in chart_data

        # Labels should be formatted dates
        assert len(chart_data['labels']) == 10

        # Should have dataset for each stock
        assert len(chart_data['datasets']) == 3

        # Each dataset should have unique color
        colors = [ds['borderColor'] for ds in chart_data['datasets']]
        assert len(set(colors)) == 3  # All different

    def test_format_candlestick_chart(self):
        """Test formatting OHLC data for candlestick chart."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        ohlc_data = pd.DataFrame({
            'Open': [100, 105, 103, 108, 110],
            'High': [107, 110, 109, 112, 115],
            'Low': [98, 103, 101, 106, 108],
            'Close': [105, 103, 108, 110, 112]
        })

        from chart_formatter import format_candlestick_chart

        chart_data = format_candlestick_chart('AAPL', dates, ohlc_data)

        # Each data point should have o, h, l, c
        assert len(chart_data['data']) == 5

        for point in chart_data['data']:
            assert 'o' in point
            assert 'h' in point
            assert 'l' in point
            assert 'c' in point

            # High should be >= Low
            assert point['h'] >= point['l']

    def test_format_with_indicators_overlay(self):
        """Test formatting price data with indicator overlays."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        prices = pd.Series(range(100, 130))
        sma_20 = pd.Series([np.nan] * 19 + list(range(110, 121)))

        from chart_formatter import format_with_indicators

        chart_data = format_with_indicators(
            'AAPL',
            dates,
            prices,
            indicators={'SMA(20)': sma_20}
        )

        # Should have dataset for price and each indicator
        assert len(chart_data['datasets']) == 2

        # Price dataset
        assert chart_data['datasets'][0]['label'] == 'AAPL'

        # SMA dataset
        assert 'SMA(20)' in chart_data['datasets'][1]['label']

    def test_format_bollinger_bands(self):
        """Test formatting Bollinger Bands for chart."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        upper = pd.Series(range(110, 140))
        middle = pd.Series(range(100, 130))
        lower = pd.Series(range(90, 120))

        from chart_formatter import format_bollinger_bands

        chart_data = format_bollinger_bands('AAPL', dates, upper, middle, lower)

        # Should have 3 datasets: upper, middle, lower
        assert len(chart_data['datasets']) >= 3

        # Upper and lower should have fill area
        for ds in chart_data['datasets']:
            if 'Upper' in ds['label'] or 'Lower' in ds['label']:
                assert 'fill' in ds


@pytest.mark.unit
class TestDataValidation:
    """Tests for validating chart data before sending to frontend."""

    def test_validate_dates_format(self):
        """Test validation of date format."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')

        from chart_validator import validate_dates

        # Should pass validation
        is_valid, error = validate_dates(dates)
        assert is_valid is True
        assert error is None

    def test_validate_dates_with_gaps(self):
        """Test validation of dates with gaps (non-trading days)."""
        dates = pd.DatetimeIndex([
            '2024-01-01', '2024-01-02', '2024-01-05', '2024-01-08'
        ])

        from chart_validator import validate_dates

        # Should still be valid (gaps are OK)
        is_valid, error = validate_dates(dates)
        assert is_valid is True

    def test_validate_dates_not_chronological(self):
        """Test validation fails for non-chronological dates."""
        dates = pd.DatetimeIndex([
            '2024-01-05', '2024-01-03', '2024-01-04'
        ])

        from chart_validator import validate_dates

        # Should fail validation
        is_valid, error = validate_dates(dates)
        assert is_valid is False
        assert 'chronological' in error.lower()

    def test_validate_price_data(self):
        """Test validation of price data."""
        prices = pd.Series([100.5, 102.3, 101.8, 105.2])

        from chart_validator import validate_price_data

        is_valid, error = validate_price_data(prices)
        assert is_valid is True

    def test_validate_price_data_with_nan(self):
        """Test validation handles NaN values appropriately."""
        prices = pd.Series([100, np.nan, 102, 103])

        from chart_validator import validate_price_data

        is_valid, error = validate_price_data(prices, allow_nan=False)
        # Should fail if NaN not allowed
        assert is_valid is False

    def test_validate_price_data_negative(self):
        """Test validation fails for negative prices."""
        prices = pd.Series([100, 102, -5, 103])

        from chart_validator import validate_price_data

        is_valid, error = validate_price_data(prices)
        assert is_valid is False
        assert 'negative' in error.lower()

    def test_validate_data_length_match(self):
        """Test validation of matching lengths for dates and prices."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        prices = pd.Series(range(100, 105))  # Only 5 values

        from chart_validator import validate_data_lengths

        is_valid, error = validate_data_lengths(dates, prices)
        assert is_valid is False
        assert 'length' in error.lower()

    def test_validate_sufficient_data_points(self):
        """Test validation of minimum data points."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        prices = pd.Series([100, 101, 102])

        from chart_validator import validate_sufficient_data

        # Need at least 5 data points
        is_valid, error = validate_sufficient_data(dates, prices, min_points=5)
        assert is_valid is False


@pytest.mark.unit
class TestDateFormatting:
    """Tests for date formatting for chart display."""

    def test_format_dates_for_display(self):
        """Test formatting dates for chart labels."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')

        from chart_formatter import format_dates_for_display

        formatted = format_dates_for_display(dates)

        # Should be list of formatted strings
        assert isinstance(formatted, list)
        assert len(formatted) == 5

        # Check format (e.g., "Jan 01" or "2024-01-01")
        for date_str in formatted:
            assert isinstance(date_str, str)
            assert len(date_str) > 0

    def test_format_dates_monthly(self):
        """Test formatting dates for monthly display."""
        dates = pd.date_range('2024-01-01', periods=12, freq='MS')

        from chart_formatter import format_dates_for_display

        formatted = format_dates_for_display(dates, format='monthly')

        # Should show month names
        assert 'Jan' in formatted[0] or 'January' in formatted[0]

    def test_format_dates_yearly(self):
        """Test formatting dates for yearly display."""
        dates = pd.date_range('2020-01-01', periods=5, freq='YS')

        from chart_formatter import format_dates_for_display

        formatted = format_dates_for_display(dates, format='yearly')

        # Should show years
        assert '2020' in formatted[0]

    def test_auto_format_dates_by_range(self):
        """Test automatic date formatting based on date range."""
        from chart_formatter import auto_format_dates

        # Short range: daily format
        dates_daily = pd.date_range('2024-01-01', periods=30, freq='D')
        formatted_daily = auto_format_dates(dates_daily)
        assert len(formatted_daily) == 30

        # Long range: might use monthly/yearly
        dates_yearly = pd.date_range('2020-01-01', periods=365, freq='D')
        formatted_yearly = auto_format_dates(dates_yearly)
        assert len(formatted_yearly) == 365


@pytest.mark.unit
class TestColorSchemes:
    """Tests for color assignment and schemes."""

    def test_assign_unique_colors(self):
        """Test assigning unique colors to multiple stocks."""
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']

        from chart_formatter import assign_colors

        colors = assign_colors(symbols)

        # Should have color for each symbol
        assert len(colors) == 5

        # All colors should be different
        assert len(set(colors.values())) == 5

        # Colors should be valid hex or rgb
        for color in colors.values():
            assert isinstance(color, str)
            assert len(color) > 0

    def test_consistent_color_assignment(self):
        """Test that same symbols get same colors."""
        from chart_formatter import assign_colors

        colors1 = assign_colors(['AAPL', 'GOOGL'])
        colors2 = assign_colors(['AAPL', 'GOOGL'])

        # AAPL should get same color both times
        assert colors1['AAPL'] == colors2['AAPL']
        assert colors1['GOOGL'] == colors2['GOOGL']

    def test_color_scheme_selection(self):
        """Test different color scheme options."""
        symbols = ['AAPL', 'GOOGL', 'MSFT']

        from chart_formatter import assign_colors

        colors_default = assign_colors(symbols, scheme='default')
        colors_vibrant = assign_colors(symbols, scheme='vibrant')
        colors_pastel = assign_colors(symbols, scheme='pastel')

        # Should return different colors for different schemes
        assert colors_default['AAPL'] != colors_vibrant['AAPL']

    def test_indicator_color_differentiation(self):
        """Test that indicators get distinct colors from prices."""
        from chart_formatter import get_indicator_color

        price_color = '#3B82F6'  # Blue
        sma_color = get_indicator_color('sma', base_color=price_color)
        ema_color = get_indicator_color('ema', base_color=price_color)

        # Indicator colors should be different from price and each other
        assert sma_color != price_color
        assert ema_color != price_color
        assert sma_color != ema_color


@pytest.mark.unit
class TestNormalization:
    """Tests for price normalization for comparison."""

    def test_normalize_to_percentage(self):
        """Test normalizing prices to percentage change."""
        prices = pd.Series([100, 105, 110, 108, 112])

        from chart_formatter import normalize_to_percentage

        normalized = normalize_to_percentage(prices)

        # Should start at 0%
        assert normalized.iloc[0] == 0

        # Last value: (112-100)/100 * 100 = 12%
        assert abs(normalized.iloc[-1] - 12) < 0.1

    def test_normalize_to_base_100(self):
        """Test normalizing prices to base 100."""
        prices = pd.Series([50, 55, 60, 58, 62])

        from chart_formatter import normalize_to_base_100

        normalized = normalize_to_base_100(prices)

        # Should start at 100
        assert normalized.iloc[0] == 100

        # Last value: (62/50) * 100 = 124
        assert abs(normalized.iloc[-1] - 124) < 0.1

    def test_normalize_multiple_stocks(self):
        """Test normalizing multiple stocks together."""
        stocks = {
            'AAPL': pd.Series([100, 110, 120]),
            'GOOGL': pd.Series([2000, 2100, 2200]),
            'MSFT': pd.Series([300, 315, 330])
        }

        from chart_formatter import normalize_multiple_stocks

        normalized = normalize_multiple_stocks(stocks)

        # All should start at 100
        for symbol in stocks:
            assert normalized[symbol].iloc[0] == 100

    def test_denormalize_prices(self):
        """Test converting normalized prices back to actual."""
        original_prices = pd.Series([100, 110, 120])

        from chart_formatter import normalize_to_base_100, denormalize_prices

        normalized = normalize_to_base_100(original_prices)
        denormalized = denormalize_prices(normalized, original_prices.iloc[0])

        # Should match original
        pd.testing.assert_series_equal(denormalized, original_prices, check_names=False)


@pytest.mark.unit
class TestChartOptions:
    """Tests for Chart.js options generation."""

    def test_generate_line_chart_options(self):
        """Test generating options for line chart."""
        from chart_formatter import generate_chart_options

        options = generate_chart_options(chart_type='line')

        # Should have required Chart.js options
        assert 'responsive' in options
        assert 'plugins' in options
        assert 'scales' in options

    def test_generate_options_with_indicators(self):
        """Test chart options when indicators are present."""
        from chart_formatter import generate_chart_options

        options = generate_chart_options(
            chart_type='line',
            has_indicators=True
        )

        # Should configure legend to show/hide indicators
        assert 'plugins' in options
        assert 'legend' in options['plugins']

    def test_generate_options_with_volume(self):
        """Test chart options with volume subplot."""
        from chart_formatter import generate_chart_options

        options = generate_chart_options(
            chart_type='line',
            include_volume=True
        )

        # Should have multiple y-axes
        assert 'scales' in options
        # Check for multiple axes configuration

    def test_tooltip_configuration(self):
        """Test tooltip formatting in chart options."""
        from chart_formatter import generate_chart_options

        options = generate_chart_options(chart_type='line')

        # Should have tooltip configuration
        assert 'plugins' in options
        assert 'tooltip' in options['plugins']

        # Tooltip should format numbers appropriately
        tooltip = options['plugins']['tooltip']
        assert 'callbacks' in tooltip or 'mode' in tooltip


@pytest.mark.unit
class TestVolumeFormatting:
    """Tests for volume data formatting."""

    def test_format_volume_data(self):
        """Test formatting volume data for display."""
        volume = pd.Series([1000000, 1500000, 1200000, 1800000])

        from chart_formatter import format_volume_data

        formatted = format_volume_data(volume)

        # Should have same length
        assert len(formatted) == 4

        # Should be formatted (not raw numbers)
        # e.g., "1.0M" instead of 1000000

    def test_volume_as_bar_chart(self):
        """Test formatting volume as bar chart dataset."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        volume = pd.Series([1000000, 1500000, 1200000, 1800000, 1600000])

        from chart_formatter import format_volume_bars

        chart_data = format_volume_bars('AAPL', dates, volume)

        # Should have bar chart properties
        assert chart_data['type'] == 'bar'
        assert 'yAxisID' in chart_data  # Separate axis for volume

    def test_volume_color_by_price_change(self):
        """Test volume bars colored by price change."""
        prices = pd.Series([100, 105, 103, 108, 107])
        volume = pd.Series([1000000, 1500000, 1200000, 1800000, 1600000])

        from chart_formatter import format_volume_with_price_color

        chart_data = format_volume_with_price_color(prices, volume)

        # Should have colors array (green for up, red for down)
        assert 'backgroundColor' in chart_data

        colors = chart_data['backgroundColor']
        # Second bar should be green (price up)
        # Third bar should be red (price down)
        assert isinstance(colors, list)


@pytest.mark.unit
class TestChartDataSerialization:
    """Tests for serializing chart data to JSON."""

    def test_serialize_chart_data_to_json(self):
        """Test that chart data can be serialized to JSON."""
        import json
        from chart_formatter import format_multi_stock_chart

        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        stocks = {
            'AAPL': pd.Series([100, 105, 110, 108, 112])
        }

        chart_data = format_multi_stock_chart(dates, stocks)

        # Should be JSON serializable
        json_str = json.dumps(chart_data)
        assert isinstance(json_str, str)

        # Should be able to deserialize
        deserialized = json.loads(json_str)
        assert 'labels' in deserialized
        assert 'datasets' in deserialized

    def test_handle_nan_in_json(self):
        """Test that NaN values are handled in JSON serialization."""
        import json
        from chart_formatter import format_line_chart

        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        prices = pd.Series([100, np.nan, 110, 108, 112])

        chart_data = format_line_chart('AAPL', dates, prices)

        # Should be JSON serializable (NaN converted to null)
        json_str = json.dumps(chart_data, default=str)
        assert isinstance(json_str, str)

    def test_handle_datetime_in_json(self):
        """Test that datetime objects are properly serialized."""
        import json
        from chart_formatter import format_multi_stock_chart

        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        stocks = {'AAPL': pd.Series([100, 105, 110])}

        chart_data = format_multi_stock_chart(dates, stocks)

        # Dates should be converted to strings
        for label in chart_data['labels']:
            assert isinstance(label, str)

        # Should be JSON serializable
        json_str = json.dumps(chart_data)
        assert isinstance(json_str, str)


@pytest.mark.unit
class TestResponsiveFormatting:
    """Tests for responsive chart formatting."""

    def test_format_for_mobile(self):
        """Test chart formatting optimized for mobile."""
        from chart_formatter import format_for_device

        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        stocks = {'AAPL': pd.Series(range(100, 130))}

        chart_data = format_for_device(dates, stocks, device='mobile')

        # Mobile might have fewer data points or simplified view
        assert 'labels' in chart_data
        assert 'datasets' in chart_data

    def test_format_for_desktop(self):
        """Test chart formatting for desktop."""
        from chart_formatter import format_for_device

        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        stocks = {'AAPL': pd.Series(range(100, 200))}

        chart_data = format_for_device(dates, stocks, device='desktop')

        # Desktop can handle more data points
        assert len(chart_data['labels']) == 100

    def test_adaptive_data_points(self):
        """Test adaptive sampling based on screen size."""
        from chart_formatter import adaptive_sampling

        dates = pd.date_range('2024-01-01', periods=500, freq='D')
        prices = pd.Series(range(100, 600))

        # Sample down to 100 points
        sampled_dates, sampled_prices = adaptive_sampling(dates, prices, max_points=100)

        assert len(sampled_dates) <= 100
        assert len(sampled_prices) <= 100
