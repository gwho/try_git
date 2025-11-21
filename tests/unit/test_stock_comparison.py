"""
Unit tests for stock comparison logic.
Testing: Multi-stock comparison, relative performance, correlation, normalization
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.mark.unit
class TestStockComparison:
    """Tests for comparing multiple stocks."""

    def test_compare_two_stocks_basic(self):
        """Test basic comparison of two stocks."""
        # Mock stock data for two stocks
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        stock1_data = pd.DataFrame({
            'Date': dates,
            'Close': range(100, 130),
            'Volume': [1000000] * 30
        })
        stock2_data = pd.DataFrame({
            'Date': dates,
            'Close': range(200, 230),
            'Volume': [2000000] * 30
        })

        from stock_comparison import compare_stocks

        result = compare_stocks(['AAPL', 'GOOGL'], [stock1_data, stock2_data])

        # Should return comparison data for both stocks
        assert 'AAPL' in result
        assert 'GOOGL' in result
        assert len(result) == 2

    def test_compare_multiple_stocks(self):
        """Test comparison of more than 2 stocks (3-5 stocks)."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        stock_data = []

        for i in range(5):
            data = pd.DataFrame({
                'Date': dates,
                'Close': range(100 + i*10, 130 + i*10),
                'Volume': [1000000 * (i+1)] * 30
            })
            stock_data.append(data)

        from stock_comparison import compare_stocks

        result = compare_stocks(stocks, stock_data)

        assert len(result) == 5
        for symbol in stocks:
            assert symbol in result

    def test_compare_stocks_with_different_date_ranges(self):
        """Test comparison when stocks have different date ranges."""
        # Stock 1: 30 days
        dates1 = pd.date_range('2024-01-01', periods=30, freq='D')
        stock1_data = pd.DataFrame({
            'Date': dates1,
            'Close': range(100, 130)
        })

        # Stock 2: 25 days (shorter)
        dates2 = pd.date_range('2024-01-01', periods=25, freq='D')
        stock2_data = pd.DataFrame({
            'Date': dates2,
            'Close': range(200, 225)
        })

        from stock_comparison import compare_stocks

        # Should align dates and only include overlapping periods
        result = compare_stocks(['AAPL', 'GOOGL'], [stock1_data, stock2_data])

        assert result is not None
        # Both stocks should have data for the overlapping period
        assert len(result['AAPL']['dates']) == len(result['GOOGL']['dates'])

    def test_normalize_prices_for_comparison(self):
        """Test normalization of stock prices to percentage change from start."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        stock1_prices = pd.Series([100, 102, 104, 103, 105, 107, 106, 108, 110, 109])
        stock2_prices = pd.Series([50, 51, 52, 51.5, 52.5, 53.5, 53, 54, 55, 54.5])

        from stock_comparison import normalize_prices

        norm1, norm2 = normalize_prices([stock1_prices, stock2_prices])

        # Both should start at 100 (or 0 for percentage change)
        assert norm1.iloc[0] == 100
        assert norm2.iloc[0] == 100

        # Final values should reflect percentage change from start
        expected_final_1 = (109 / 100) * 100  # 109%
        expected_final_2 = (54.5 / 50) * 100  # 109%

        assert abs(norm1.iloc[-1] - expected_final_1) < 0.01
        assert abs(norm2.iloc[-1] - expected_final_2) < 0.01


@pytest.mark.unit
class TestRelativePerformance:
    """Tests for calculating relative performance metrics."""

    def test_calculate_percentage_change(self):
        """Test percentage change calculation over period."""
        stock_data = pd.DataFrame({
            'Close': [100, 110, 105, 115, 120]
        })

        from stock_comparison import calculate_percentage_change

        pct_change = calculate_percentage_change(stock_data)

        # From 100 to 120 is 20% gain
        assert pct_change == 20.0

    def test_calculate_returns(self):
        """Test daily returns calculation."""
        prices = pd.Series([100, 102, 101, 105, 103])

        from stock_comparison import calculate_daily_returns

        returns = calculate_daily_returns(prices)

        # First value should be NaN (no previous price)
        assert pd.isna(returns.iloc[0])

        # Second value: (102-100)/100 = 0.02 or 2%
        assert abs(returns.iloc[1] - 0.02) < 0.0001

        # Third value: (101-102)/102 ≈ -0.0098
        assert returns.iloc[2] < 0

    def test_calculate_cumulative_returns(self):
        """Test cumulative returns calculation."""
        daily_returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])

        from stock_comparison import calculate_cumulative_returns

        cum_returns = calculate_cumulative_returns(daily_returns)

        # Should be monotonic calculation: (1+r1)*(1+r2)*...
        assert len(cum_returns) == len(daily_returns)
        assert cum_returns.iloc[-1] != 0

    def test_calculate_volatility(self):
        """Test volatility (standard deviation of returns)."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, 0.02])

        from stock_comparison import calculate_volatility

        volatility = calculate_volatility(returns)

        # Volatility should be positive
        assert volatility > 0

        # More volatile returns should have higher volatility
        stable_returns = pd.Series([0.001] * 8)
        stable_volatility = calculate_volatility(stable_returns)

        assert volatility > stable_volatility

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation (risk-adjusted returns)."""
        returns = pd.Series([0.01, 0.02, 0.015, 0.018, 0.022, 0.019, 0.021, 0.017])
        risk_free_rate = 0.02  # 2% annual

        from stock_comparison import calculate_sharpe_ratio

        sharpe = calculate_sharpe_ratio(returns, risk_free_rate)

        # Sharpe ratio can be positive or negative
        assert isinstance(sharpe, float)

    def test_compare_performance_metrics(self):
        """Test comparison of performance metrics between stocks."""
        stock1_returns = pd.Series([0.01, 0.02, -0.01, 0.03])
        stock2_returns = pd.Series([0.02, 0.03, -0.02, 0.04])

        from stock_comparison import compare_performance_metrics

        metrics = compare_performance_metrics([stock1_returns, stock2_returns])

        # Should include metrics for both stocks
        assert 'stock_0' in metrics
        assert 'stock_1' in metrics

        # Each should have standard metrics
        for stock in metrics.values():
            assert 'total_return' in stock
            assert 'volatility' in stock
            assert 'sharpe_ratio' in stock


@pytest.mark.unit
class TestCorrelation:
    """Tests for correlation analysis between stocks."""

    def test_calculate_correlation_two_stocks(self):
        """Test correlation calculation between two stocks."""
        stock1_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        stock2_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])  # Identical

        from stock_comparison import calculate_correlation

        correlation = calculate_correlation(stock1_returns, stock2_returns)

        # Identical series should have correlation of 1.0
        assert abs(correlation - 1.0) < 0.001

    def test_correlation_matrix_multiple_stocks(self):
        """Test correlation matrix for multiple stocks."""
        returns_data = {
            'AAPL': pd.Series([0.01, 0.02, -0.01, 0.03, 0.01]),
            'GOOGL': pd.Series([0.015, 0.018, -0.012, 0.028, 0.009]),
            'MSFT': pd.Series([0.012, 0.022, -0.008, 0.031, 0.011])
        }

        from stock_comparison import calculate_correlation_matrix

        corr_matrix = calculate_correlation_matrix(returns_data)

        # Should be a square matrix
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        assert corr_matrix.shape[0] == 3

        # Diagonal should be 1.0 (correlation with self)
        for i in range(len(corr_matrix)):
            assert abs(corr_matrix.iloc[i, i] - 1.0) < 0.001

    def test_negative_correlation(self):
        """Test detection of negative correlation."""
        stock1_returns = pd.Series([0.01, 0.02, 0.03, 0.04])
        stock2_returns = pd.Series([-0.01, -0.02, -0.03, -0.04])  # Inverse

        from stock_comparison import calculate_correlation

        correlation = calculate_correlation(stock1_returns, stock2_returns)

        # Should be strongly negative
        assert correlation < -0.9

    def test_uncorrelated_stocks(self):
        """Test detection of uncorrelated stocks."""
        np.random.seed(42)
        stock1_returns = pd.Series(np.random.randn(50))
        stock2_returns = pd.Series(np.random.randn(50))

        from stock_comparison import calculate_correlation

        correlation = calculate_correlation(stock1_returns, stock2_returns)

        # Should be close to 0 (uncorrelated random data)
        assert abs(correlation) < 0.5


@pytest.mark.unit
class TestBeta:
    """Tests for Beta calculation (stock volatility relative to market)."""

    def test_calculate_beta_basic(self):
        """Test basic beta calculation."""
        stock_returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.02])
        market_returns = pd.Series([0.01, 0.015, -0.005, 0.02, 0.015])

        from stock_comparison import calculate_beta

        beta = calculate_beta(stock_returns, market_returns)

        # Beta should be a float
        assert isinstance(beta, float)

    def test_beta_greater_than_one(self):
        """Test stock with beta > 1 (more volatile than market)."""
        market_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
        # Stock moves 1.5x the market
        stock_returns = market_returns * 1.5

        from stock_comparison import calculate_beta

        beta = calculate_beta(stock_returns, market_returns)

        # Beta should be approximately 1.5
        assert 1.3 < beta < 1.7

    def test_beta_less_than_one(self):
        """Test stock with beta < 1 (less volatile than market)."""
        market_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
        # Stock moves 0.5x the market
        stock_returns = market_returns * 0.5

        from stock_comparison import calculate_beta

        beta = calculate_beta(stock_returns, market_returns)

        # Beta should be approximately 0.5
        assert 0.3 < beta < 0.7

    def test_negative_beta(self):
        """Test stock with negative beta (inverse to market)."""
        market_returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
        # Stock moves opposite to market
        stock_returns = -market_returns

        from stock_comparison import calculate_beta

        beta = calculate_beta(stock_returns, market_returns)

        # Beta should be negative
        assert beta < 0


@pytest.mark.unit
class TestDrawdown:
    """Tests for drawdown calculation (peak to trough decline)."""

    def test_calculate_maximum_drawdown(self):
        """Test maximum drawdown calculation."""
        prices = pd.Series([100, 110, 105, 115, 95, 105, 100])

        from stock_comparison import calculate_max_drawdown

        max_dd = calculate_max_drawdown(prices)

        # From peak of 115 to trough of 95: (95-115)/115 ≈ -17.4%
        assert max_dd < 0
        assert abs(max_dd - (-0.174)) < 0.01

    def test_no_drawdown_increasing_prices(self):
        """Test drawdown with consistently increasing prices."""
        prices = pd.Series([100, 105, 110, 115, 120])

        from stock_comparison import calculate_max_drawdown

        max_dd = calculate_max_drawdown(prices)

        # Should be 0 or very small (no drawdown)
        assert max_dd <= 0.01

    def test_drawdown_series(self):
        """Test calculation of drawdown series over time."""
        prices = pd.Series([100, 110, 105, 115, 95, 105, 120])

        from stock_comparison import calculate_drawdown_series

        drawdowns = calculate_drawdown_series(prices)

        # Should have same length as prices
        assert len(drawdowns) == len(prices)

        # All drawdowns should be <= 0
        assert (drawdowns <= 0).all()

        # First value should be 0 (at initial peak)
        assert drawdowns.iloc[0] == 0


@pytest.mark.unit
class TestPortfolioMetrics:
    """Tests for portfolio-level metrics when comparing stocks."""

    def test_calculate_portfolio_value(self):
        """Test portfolio value calculation with weights."""
        stock_prices = {
            'AAPL': pd.Series([100, 105, 110]),
            'GOOGL': pd.Series([200, 210, 205])
        }
        weights = {'AAPL': 0.6, 'GOOGL': 0.4}

        from stock_comparison import calculate_portfolio_value

        portfolio_value = calculate_portfolio_value(stock_prices, weights)

        # Should return normalized portfolio value series
        assert len(portfolio_value) == 3
        assert portfolio_value.iloc[0] == 100  # Normalized to 100

    def test_equal_weight_portfolio(self):
        """Test equal-weighted portfolio."""
        stock_prices = {
            'AAPL': pd.Series([100, 110, 120]),
            'GOOGL': pd.Series([200, 210, 220]),
            'MSFT': pd.Series([150, 155, 160])
        }

        from stock_comparison import create_equal_weight_portfolio

        portfolio = create_equal_weight_portfolio(stock_prices)

        # Each stock should have 1/3 weight
        assert len(portfolio['weights']) == 3
        for weight in portfolio['weights'].values():
            assert abs(weight - 1/3) < 0.001

    def test_market_cap_weighted_portfolio(self):
        """Test market-cap weighted portfolio."""
        stock_prices = {
            'AAPL': pd.Series([100, 110, 120]),
            'GOOGL': pd.Series([200, 210, 220])
        }
        market_caps = {
            'AAPL': 3000000000000,  # 3T
            'GOOGL': 2000000000000   # 2T
        }

        from stock_comparison import create_market_cap_weighted_portfolio

        portfolio = create_market_cap_weighted_portfolio(stock_prices, market_caps)

        # AAPL should have 60% weight, GOOGL 40%
        assert abs(portfolio['weights']['AAPL'] - 0.6) < 0.001
        assert abs(portfolio['weights']['GOOGL'] - 0.4) < 0.001


@pytest.mark.unit
class TestRankingAndSorting:
    """Tests for ranking stocks by various metrics."""

    def test_rank_by_returns(self):
        """Test ranking stocks by total returns."""
        stock_data = {
            'AAPL': {'total_return': 0.15},
            'GOOGL': {'total_return': 0.22},
            'MSFT': {'total_return': 0.18}
        }

        from stock_comparison import rank_stocks_by_metric

        ranked = rank_stocks_by_metric(stock_data, 'total_return', ascending=False)

        # GOOGL should be first (highest return)
        assert ranked[0][0] == 'GOOGL'
        assert ranked[1][0] == 'MSFT'
        assert ranked[2][0] == 'AAPL'

    def test_rank_by_volatility(self):
        """Test ranking stocks by volatility (lower is better)."""
        stock_data = {
            'AAPL': {'volatility': 0.25},
            'GOOGL': {'volatility': 0.35},
            'MSFT': {'volatility': 0.20}
        }

        from stock_comparison import rank_stocks_by_metric

        ranked = rank_stocks_by_metric(stock_data, 'volatility', ascending=True)

        # MSFT should be first (lowest volatility)
        assert ranked[0][0] == 'MSFT'
        assert ranked[1][0] == 'AAPL'
        assert ranked[2][0] == 'GOOGL'

    def test_rank_by_sharpe_ratio(self):
        """Test ranking stocks by Sharpe ratio."""
        stock_data = {
            'AAPL': {'sharpe_ratio': 1.5},
            'GOOGL': {'sharpe_ratio': 1.8},
            'MSFT': {'sharpe_ratio': 1.2}
        }

        from stock_comparison import rank_stocks_by_metric

        ranked = rank_stocks_by_metric(stock_data, 'sharpe_ratio', ascending=False)

        # GOOGL should be first (highest Sharpe ratio)
        assert ranked[0][0] == 'GOOGL'


@pytest.mark.unit
class TestDataAlignment:
    """Tests for aligning stock data with different date ranges."""

    def test_align_stock_data_same_dates(self):
        """Test aligning stocks with identical date ranges."""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        stock1 = pd.DataFrame({'Date': dates, 'Close': range(100, 130)})
        stock2 = pd.DataFrame({'Date': dates, 'Close': range(200, 230)})

        from stock_comparison import align_stock_data

        aligned = align_stock_data([stock1, stock2])

        # Should have same dates
        assert len(aligned[0]) == len(aligned[1]) == 30

    def test_align_stock_data_different_dates(self):
        """Test aligning stocks with different date ranges."""
        dates1 = pd.date_range('2024-01-01', periods=30, freq='D')
        dates2 = pd.date_range('2024-01-10', periods=25, freq='D')

        stock1 = pd.DataFrame({'Date': dates1, 'Close': range(100, 130)})
        stock2 = pd.DataFrame({'Date': dates2, 'Close': range(200, 225)})

        from stock_comparison import align_stock_data

        aligned = align_stock_data([stock1, stock2])

        # Should only include overlapping dates
        assert len(aligned[0]) == len(aligned[1])
        assert len(aligned[0]) < 30  # Less than full range

    def test_align_with_missing_dates(self):
        """Test aligning stocks with gaps in data (non-trading days)."""
        # Stock 1 has some missing dates
        dates1 = pd.DatetimeIndex(['2024-01-01', '2024-01-02', '2024-01-04', '2024-01-05'])
        # Stock 2 has continuous dates
        dates2 = pd.date_range('2024-01-01', periods=5, freq='D')

        stock1 = pd.DataFrame({'Date': dates1, 'Close': [100, 102, 104, 106]})
        stock2 = pd.DataFrame({'Date': dates2, 'Close': [200, 202, 204, 206, 208]})

        from stock_comparison import align_stock_data

        aligned = align_stock_data([stock1, stock2], method='inner')

        # Should only include dates present in both
        assert len(aligned[0]) == len(aligned[1]) == 4
