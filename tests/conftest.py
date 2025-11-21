"""
Pytest configuration and shared fixtures for stock comparison tests.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


@pytest.fixture
def app():
    """Create Flask app for testing."""
    from app import app as flask_app
    flask_app.config['TESTING'] = True
    flask_app.config['DEBUG'] = False
    return flask_app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_dates():
    """Generate sample date range for testing."""
    return pd.date_range('2024-01-01', periods=180, freq='D')


@pytest.fixture
def sample_stock_data(sample_dates):
    """Generate sample stock price data."""
    np.random.seed(42)

    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(sample_dates))
    prices = base_price * (1 + returns).cumprod()

    data = pd.DataFrame({
        'Date': sample_dates,
        'Open': prices * np.random.uniform(0.98, 1.02, len(sample_dates)),
        'High': prices * np.random.uniform(1.01, 1.05, len(sample_dates)),
        'Low': prices * np.random.uniform(0.95, 0.99, len(sample_dates)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, len(sample_dates))
    })

    return data


@pytest.fixture
def sample_multiple_stocks(sample_dates):
    """Generate data for multiple stocks."""
    np.random.seed(42)

    stocks = {}
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    base_prices = [150, 2800, 380, 3400, 240]

    for symbol, base_price in zip(symbols, base_prices):
        returns = np.random.normal(0.001, 0.02, len(sample_dates))
        prices = base_price * (1 + returns).cumprod()

        stocks[symbol] = pd.DataFrame({
            'Date': sample_dates,
            'Open': prices * np.random.uniform(0.98, 1.02, len(sample_dates)),
            'High': prices * np.random.uniform(1.01, 1.05, len(sample_dates)),
            'Low': prices * np.random.uniform(0.95, 0.99, len(sample_dates)),
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(sample_dates))
        })

    return stocks


@pytest.fixture
def mock_yfinance_data(sample_multiple_stocks):
    """Mock yfinance API responses."""
    def mock_ticker(symbol):
        """Create mock Ticker object."""
        ticker = MagicMock()

        if symbol in sample_multiple_stocks:
            stock_data = sample_multiple_stocks[symbol]

            # Mock history method
            ticker.history.return_value = stock_data

            # Mock info property
            ticker.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Inc.',
                'currentPrice': float(stock_data['Close'].iloc[-1]),
                'previousClose': float(stock_data['Close'].iloc[-2]),
                'open': float(stock_data['Open'].iloc[-1]),
                'dayHigh': float(stock_data['High'].iloc[-1]),
                'dayLow': float(stock_data['Low'].iloc[-1]),
                'volume': int(stock_data['Volume'].iloc[-1]),
                'marketCap': np.random.randint(1000000000, 3000000000000),
                'trailingPE': np.random.uniform(15, 35),
                'trailingEps': np.random.uniform(3, 15),
                'dividendYield': np.random.uniform(0, 0.03),
                'beta': np.random.uniform(0.8, 1.5),
                'fiftyTwoWeekHigh': float(stock_data['High'].max()),
                'fiftyTwoWeekLow': float(stock_data['Low'].min())
            }
        else:
            # Invalid symbol
            ticker.history.return_value = pd.DataFrame()
            ticker.info = {}

        return ticker

    with patch('yfinance.Ticker', side_effect=mock_ticker):
        yield


@pytest.fixture
def mock_yfinance_failure():
    """Mock yfinance API failure."""
    def mock_ticker_failure(symbol):
        """Create mock Ticker that raises exception."""
        ticker = MagicMock()
        ticker.history.side_effect = Exception("API Error")
        ticker.info = {}
        return ticker

    with patch('yfinance.Ticker', side_effect=mock_ticker_failure):
        yield


@pytest.fixture
def sample_price_series():
    """Generate simple price series for testing."""
    return pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109])


@pytest.fixture
def sample_ohlc_data():
    """Generate OHLC data for testing."""
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    np.random.seed(42)

    close_prices = 100 * (1 + np.random.normal(0.001, 0.02, 30)).cumprod()

    return pd.DataFrame({
        'Date': dates,
        'Open': close_prices * np.random.uniform(0.98, 1.02, 30),
        'High': close_prices * np.random.uniform(1.01, 1.05, 30),
        'Low': close_prices * np.random.uniform(0.95, 0.99, 30),
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, 30)
    })


@pytest.fixture
def sample_returns():
    """Generate sample returns data."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 100))


@pytest.fixture
def trending_up_prices():
    """Generate steadily increasing prices."""
    return pd.Series(np.linspace(100, 150, 50))


@pytest.fixture
def trending_down_prices():
    """Generate steadily decreasing prices."""
    return pd.Series(np.linspace(150, 100, 50))


@pytest.fixture
def volatile_prices():
    """Generate highly volatile prices."""
    np.random.seed(42)
    base = 100
    volatility = 0.05
    returns = np.random.normal(0, volatility, 100)
    return pd.Series(base * (1 + returns).cumprod())


@pytest.fixture
def stable_prices():
    """Generate stable, low-volatility prices."""
    np.random.seed(42)
    base = 100
    volatility = 0.005
    returns = np.random.normal(0, volatility, 100)
    return pd.Series(base * (1 + returns).cumprod())


@pytest.fixture
def mock_stock_info():
    """Mock stock info data."""
    return {
        'symbol': 'AAPL',
        'shortName': 'Apple Inc.',
        'currentPrice': 150.25,
        'previousClose': 148.50,
        'open': 149.00,
        'dayHigh': 151.00,
        'dayLow': 148.75,
        'volume': 75000000,
        'marketCap': 2500000000000,
        'trailingPE': 28.5,
        'trailingEps': 5.27,
        'dividendYield': 0.0055,
        'beta': 1.25,
        'fiftyTwoWeekHigh': 180.00,
        'fiftyTwoWeekLow': 120.00
    }


@pytest.fixture
def sample_correlation_data():
    """Generate correlated stock returns for testing."""
    np.random.seed(42)

    # Generate correlated returns
    mean = [0.001, 0.001, 0.001]
    cov = [
        [0.0004, 0.0002, 0.0001],
        [0.0002, 0.0004, 0.00015],
        [0.0001, 0.00015, 0.0004]
    ]

    returns = np.random.multivariate_normal(mean, cov, 100)

    return {
        'AAPL': pd.Series(returns[:, 0]),
        'GOOGL': pd.Series(returns[:, 1]),
        'MSFT': pd.Series(returns[:, 2])
    }


@pytest.fixture
def sample_technical_indicators():
    """Generate sample technical indicator data."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    prices = pd.Series(100 * (1 + np.random.normal(0.001, 0.02, 50)).cumprod())

    # Simple moving averages
    sma_20 = prices.rolling(window=20).mean()
    sma_50 = prices.rolling(window=50).mean()

    # EMA
    ema_12 = prices.ewm(span=12).mean()
    ema_26 = prices.ewm(span=26).mean()

    # Bollinger Bands
    sma = prices.rolling(window=20).mean()
    std = prices.rolling(window=20).std()
    bb_upper = sma + (std * 2)
    bb_lower = sma - (std * 2)

    return {
        'dates': dates,
        'prices': prices,
        'sma_20': sma_20,
        'sma_50': sma_50,
        'ema_12': ema_12,
        'ema_26': ema_26,
        'bb_upper': bb_upper,
        'bb_middle': sma,
        'bb_lower': bb_lower
    }


@pytest.fixture
def empty_dataframe():
    """Return empty DataFrame for testing edge cases."""
    return pd.DataFrame()


@pytest.fixture
def single_row_dataframe():
    """Return DataFrame with single row."""
    return pd.DataFrame({
        'Date': [datetime.now()],
        'Close': [100.0],
        'Volume': [1000000]
    })


class MockYFinanceTicker:
    """Mock class for yfinance Ticker."""

    def __init__(self, symbol, data=None, should_fail=False):
        self.symbol = symbol
        self.data = data
        self.should_fail = should_fail

    def history(self, start=None, end=None, period=None):
        """Mock history method."""
        if self.should_fail:
            raise Exception("API Error")

        if self.data is not None:
            # Filter by date range if provided
            if start and end:
                mask = (self.data['Date'] >= start) & (self.data['Date'] <= end)
                return self.data[mask]
            return self.data

        return pd.DataFrame()

    @property
    def info(self):
        """Mock info property."""
        if self.should_fail or self.data is None:
            return {}

        return {
            'symbol': self.symbol,
            'shortName': f'{self.symbol} Inc.',
            'currentPrice': float(self.data['Close'].iloc[-1]),
            'marketCap': 1000000000000
        }


@pytest.fixture
def mock_ticker_factory(sample_multiple_stocks):
    """Factory fixture for creating mock tickers."""
    def create_ticker(symbol, should_fail=False):
        data = sample_multiple_stocks.get(symbol)
        return MockYFinanceTicker(symbol, data, should_fail)

    return create_ticker


# Markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )


# Helper functions for tests
@pytest.fixture
def assert_valid_chart_data():
    """Fixture providing chart data validation helper."""
    def validator(chart_data):
        """Validate chart data structure."""
        assert 'labels' in chart_data
        assert 'datasets' in chart_data
        assert isinstance(chart_data['labels'], list)
        assert isinstance(chart_data['datasets'], list)

        for dataset in chart_data['datasets']:
            assert 'label' in dataset
            assert 'data' in dataset
            assert isinstance(dataset['data'], list)

        return True

    return validator


@pytest.fixture
def assert_valid_api_response():
    """Fixture providing API response validation helper."""
    def validator(response, expected_keys):
        """Validate API response structure."""
        import json

        assert response.status_code == 200
        data = json.loads(response.data)

        for key in expected_keys:
            assert key in data

        return data

    return validator


@pytest.fixture
def create_mock_comparison_request():
    """Factory for creating comparison request payloads."""
    def factory(symbols=None, period='6mo', indicators=None, normalize=False):
        """Create comparison request payload."""
        payload = {
            'symbols': symbols or ['AAPL', 'GOOGL'],
            'period': period
        }

        if indicators:
            payload['indicators'] = indicators

        if normalize:
            payload['normalize'] = True

        return payload

    return factory


# Setup and teardown helpers
@pytest.fixture(autouse=True)
def reset_caches():
    """Reset any caches between tests."""
    # This would clear any caching mechanisms
    # Implementation depends on actual caching strategy
    yield
    # Cleanup after test


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_csv_data(temp_data_dir):
    """Create sample CSV file for testing."""
    csv_path = temp_data_dir / "AAPL_test.csv"

    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Close': range(100, 130),
        'Volume': [1000000] * 30
    })

    data.to_csv(csv_path, index=False)
    return csv_path


# Performance testing helpers
@pytest.fixture
def measure_time():
    """Fixture for measuring execution time."""
    import time

    times = {}

    def measure(name):
        """Context manager for timing code blocks."""
        class Timer:
            def __enter__(self):
                self.start = time.time()
                return self

            def __exit__(self, *args):
                self.end = time.time()
                times[name] = self.end - self.start

        return Timer()

    measure.times = times
    return measure
