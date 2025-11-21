"""
Sample stock data for testing purposes.
Provides realistic stock data without requiring external API calls.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_stock_data(symbol, start_date, end_date, base_price=100, volatility=0.02):
    """
    Generate realistic stock data for testing.

    Args:
        symbol: Stock symbol
        start_date: Start date
        end_date: End date
        base_price: Starting price
        volatility: Price volatility (standard deviation of returns)

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    n_days = len(dates)

    np.random.seed(hash(symbol) % 2**32)  # Consistent data for same symbol

    # Generate returns
    returns = np.random.normal(0.0005, volatility, n_days)
    prices = base_price * (1 + returns).cumprod()

    # Generate OHLC from close prices
    open_prices = prices * np.random.uniform(0.99, 1.01, n_days)
    high_prices = prices * np.random.uniform(1.005, 1.03, n_days)
    low_prices = prices * np.random.uniform(0.97, 0.995, n_days)

    # Ensure high is highest and low is lowest
    high_prices = np.maximum(high_prices, np.maximum(open_prices, prices))
    low_prices = np.minimum(low_prices, np.minimum(open_prices, prices))

    # Generate volume
    avg_volume = 50000000 if base_price > 200 else 20000000
    volume = np.random.normal(avg_volume, avg_volume * 0.3, n_days)
    volume = np.abs(volume).astype(int)

    data = pd.DataFrame({
        'Date': dates,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': prices,
        'Volume': volume
    })

    return data


# Pre-generated data for common stocks
AAPL_DATA = generate_stock_data('AAPL', '2023-01-01', '2024-12-31', base_price=150, volatility=0.02)
GOOGL_DATA = generate_stock_data('GOOGL', '2023-01-01', '2024-12-31', base_price=140, volatility=0.025)
MSFT_DATA = generate_stock_data('MSFT', '2023-01-01', '2024-12-31', base_price=380, volatility=0.018)
AMZN_DATA = generate_stock_data('AMZN', '2023-01-01', '2024-12-31', base_price=170, volatility=0.03)
TSLA_DATA = generate_stock_data('TSLA', '2023-01-01', '2024-12-31', base_price=240, volatility=0.04)

# Market data (S&P 500 proxy)
SPY_DATA = generate_stock_data('SPY', '2023-01-01', '2024-12-31', base_price=450, volatility=0.015)


def get_sample_data(symbol, period='6mo'):
    """
    Get sample data for a stock symbol.

    Args:
        symbol: Stock symbol
        period: Time period ('1mo', '3mo', '6mo', '1y', '2y', '5y')

    Returns:
        DataFrame with stock data
    """
    data_map = {
        'AAPL': AAPL_DATA,
        'GOOGL': GOOGL_DATA,
        'MSFT': MSFT_DATA,
        'AMZN': AMZN_DATA,
        'TSLA': TSLA_DATA,
        'SPY': SPY_DATA
    }

    if symbol not in data_map:
        return pd.DataFrame()  # Empty for unknown symbols

    data = data_map[symbol].copy()

    # Filter by period
    period_days = {
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365,
        '2y': 730,
        '5y': 1825
    }

    if period in period_days:
        days = period_days[period]
        data = data.tail(days)

    return data


def get_sample_info(symbol):
    """
    Get sample info data for a stock.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with stock info
    """
    info_map = {
        'AAPL': {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'longName': 'Apple Inc.',
            'currentPrice': 150.25,
            'previousClose': 148.50,
            'open': 149.00,
            'dayHigh': 151.00,
            'dayLow': 148.75,
            'volume': 75000000,
            'marketCap': 2500000000000,
            'trailingPE': 28.5,
            'forwardPE': 25.3,
            'trailingEps': 5.27,
            'forwardEps': 5.94,
            'dividendYield': 0.0055,
            'beta': 1.25,
            'fiftyTwoWeekHigh': 180.00,
            'fiftyTwoWeekLow': 120.00,
            'fiftyDayAverage': 155.00,
            'twoHundredDayAverage': 145.00,
            'currency': 'USD',
            'exchange': 'NMS'
        },
        'GOOGL': {
            'symbol': 'GOOGL',
            'shortName': 'Alphabet Inc.',
            'longName': 'Alphabet Inc.',
            'currentPrice': 140.50,
            'previousClose': 138.75,
            'open': 139.00,
            'dayHigh': 141.50,
            'dayLow': 138.50,
            'volume': 25000000,
            'marketCap': 1800000000000,
            'trailingPE': 26.8,
            'forwardPE': 23.5,
            'trailingEps': 5.24,
            'forwardEps': 5.98,
            'dividendYield': 0.0,
            'beta': 1.05,
            'fiftyTwoWeekHigh': 150.00,
            'fiftyTwoWeekLow': 110.00,
            'fiftyDayAverage': 138.00,
            'twoHundredDayAverage': 130.00,
            'currency': 'USD',
            'exchange': 'NMS'
        },
        'MSFT': {
            'symbol': 'MSFT',
            'shortName': 'Microsoft Corporation',
            'longName': 'Microsoft Corporation',
            'currentPrice': 380.25,
            'previousClose': 378.50,
            'open': 379.00,
            'dayHigh': 382.00,
            'dayLow': 377.75,
            'volume': 22000000,
            'marketCap': 2800000000000,
            'trailingPE': 32.5,
            'forwardPE': 28.3,
            'trailingEps': 11.70,
            'forwardEps': 13.44,
            'dividendYield': 0.0075,
            'beta': 0.92,
            'fiftyTwoWeekHigh': 400.00,
            'fiftyTwoWeekLow': 320.00,
            'fiftyDayAverage': 385.00,
            'twoHundredDayAverage': 360.00,
            'currency': 'USD',
            'exchange': 'NMS'
        },
        'AMZN': {
            'symbol': 'AMZN',
            'shortName': 'Amazon.com Inc.',
            'longName': 'Amazon.com, Inc.',
            'currentPrice': 170.50,
            'previousClose': 168.25,
            'open': 169.00,
            'dayHigh': 172.00,
            'dayLow': 168.00,
            'volume': 45000000,
            'marketCap': 1750000000000,
            'trailingPE': 65.5,
            'forwardPE': 45.2,
            'trailingEps': 2.60,
            'forwardEps': 3.77,
            'dividendYield': 0.0,
            'beta': 1.15,
            'fiftyTwoWeekHigh': 180.00,
            'fiftyTwoWeekLow': 135.00,
            'fiftyDayAverage': 168.00,
            'twoHundredDayAverage': 155.00,
            'currency': 'USD',
            'exchange': 'NMS'
        },
        'TSLA': {
            'symbol': 'TSLA',
            'shortName': 'Tesla Inc.',
            'longName': 'Tesla, Inc.',
            'currentPrice': 240.75,
            'previousClose': 235.50,
            'open': 238.00,
            'dayHigh': 245.00,
            'dayLow': 236.50,
            'volume': 120000000,
            'marketCap': 750000000000,
            'trailingPE': 75.2,
            'forwardPE': 55.8,
            'trailingEps': 3.20,
            'forwardEps': 4.31,
            'dividendYield': 0.0,
            'beta': 2.05,
            'fiftyTwoWeekHigh': 300.00,
            'fiftyTwoWeekLow': 180.00,
            'fiftyDayAverage': 235.00,
            'twoHundredDayAverage': 220.00,
            'currency': 'USD',
            'exchange': 'NMS'
        },
        'SPY': {
            'symbol': 'SPY',
            'shortName': 'SPDR S&P 500 ETF Trust',
            'longName': 'SPDR S&P 500 ETF Trust',
            'currentPrice': 450.25,
            'previousClose': 448.75,
            'open': 449.00,
            'dayHigh': 451.50,
            'dayLow': 448.50,
            'volume': 75000000,
            'marketCap': 0,
            'trailingPE': 0,
            'forwardPE': 0,
            'trailingEps': 0,
            'forwardEps': 0,
            'dividendYield': 0.013,
            'beta': 1.0,
            'fiftyTwoWeekHigh': 470.00,
            'fiftyTwoWeekLow': 400.00,
            'fiftyDayAverage': 455.00,
            'twoHundredDayAverage': 435.00,
            'currency': 'USD',
            'exchange': 'PCX'
        }
    }

    return info_map.get(symbol, {})


def get_correlated_stocks(symbols, correlation_matrix, period='6mo'):
    """
    Generate correlated stock data.

    Args:
        symbols: List of stock symbols
        correlation_matrix: Desired correlation matrix (numpy array)
        period: Time period

    Returns:
        Dictionary of DataFrames with correlated stock data
    """
    period_days = {
        '1mo': 30,
        '3mo': 90,
        '6mo': 180,
        '1y': 365
    }

    n_days = period_days.get(period, 180)
    n_stocks = len(symbols)

    # Generate correlated returns
    mean = [0.0005] * n_stocks
    cov = correlation_matrix * 0.0004  # Scale to realistic volatility

    np.random.seed(42)
    returns = np.random.multivariate_normal(mean, cov, n_days)

    # Generate prices from returns
    base_prices = [150, 140, 380, 170, 240][:n_stocks]
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')

    stocks = {}
    for i, symbol in enumerate(symbols):
        prices = base_prices[i] * (1 + returns[:, i]).cumprod()

        stocks[symbol] = pd.DataFrame({
            'Date': dates,
            'Open': prices * np.random.uniform(0.99, 1.01, n_days),
            'High': prices * np.random.uniform(1.005, 1.03, n_days),
            'Low': prices * np.random.uniform(0.97, 0.995, n_days),
            'Close': prices,
            'Volume': np.random.randint(10000000, 100000000, n_days)
        })

    return stocks


# Test scenarios
TEST_SCENARIOS = {
    'bull_market': {
        'description': 'Strong upward trend',
        'returns': lambda n: np.random.normal(0.002, 0.015, n)
    },
    'bear_market': {
        'description': 'Downward trend',
        'returns': lambda n: np.random.normal(-0.002, 0.02, n)
    },
    'high_volatility': {
        'description': 'High volatility period',
        'returns': lambda n: np.random.normal(0, 0.05, n)
    },
    'low_volatility': {
        'description': 'Low volatility period',
        'returns': lambda n: np.random.normal(0.0005, 0.005, n)
    },
    'crash': {
        'description': 'Market crash scenario',
        'returns': lambda n: np.concatenate([
            np.random.normal(0.001, 0.015, n//2),
            np.random.normal(-0.05, 0.03, n//2)
        ])
    }
}


def generate_scenario_data(scenario, symbol='TEST', days=180, base_price=100):
    """
    Generate stock data for a specific test scenario.

    Args:
        scenario: Scenario name from TEST_SCENARIOS
        symbol: Stock symbol
        days: Number of days
        base_price: Starting price

    Returns:
        DataFrame with stock data
    """
    if scenario not in TEST_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    returns = TEST_SCENARIOS[scenario]['returns'](days)
    prices = base_price * (1 + returns).cumprod()

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * np.random.uniform(0.99, 1.01, days),
        'High': prices * np.random.uniform(1.005, 1.03, days),
        'Low': prices * np.random.uniform(0.97, 0.995, days),
        'Close': prices,
        'Volume': np.random.randint(10000000, 100000000, days)
    })

    return data
