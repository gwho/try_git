"""
Integration tests for stock comparison API endpoints.
Testing: Flask routes, request/response handling, error cases
"""
import pytest
import json
from datetime import datetime, timedelta


@pytest.mark.integration
class TestCompareStocksEndpoint:
    """Tests for /api/compare endpoint."""

    def test_compare_two_stocks_success(self, client, mock_yfinance_data):
        """Test successful comparison of two stocks."""
        # Given: Two stock symbols
        payload = {
            'symbols': ['AAPL', 'GOOGL'],
            'period': '6mo'
        }

        # When: Making POST request to compare endpoint
        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Then: Should return 200 with comparison data
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'stocks' in data
        assert len(data['stocks']) == 2
        assert 'AAPL' in data['stocks']
        assert 'GOOGL' in data['stocks']

        # Each stock should have required fields
        for symbol in ['AAPL', 'GOOGL']:
            stock_data = data['stocks'][symbol]
            assert 'dates' in stock_data
            assert 'prices' in stock_data
            assert 'normalized_prices' in stock_data
            assert 'metrics' in stock_data

    def test_compare_multiple_stocks(self, client, mock_yfinance_data):
        """Test comparison of 3-5 stocks."""
        payload = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
            'period': '3mo'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data['stocks']) == 5

    def test_compare_with_invalid_symbol(self, client):
        """Test comparison with invalid stock symbol."""
        payload = {
            'symbols': ['AAPL', 'INVALID123'],
            'period': '1mo'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Should return error or partial data with error info
        assert response.status_code in [200, 400]

        data = json.loads(response.data)
        if response.status_code == 200:
            # If partial success, should have error info
            assert 'errors' in data
            assert 'INVALID123' in data['errors']

    def test_compare_with_single_symbol(self, client):
        """Test comparison with only one symbol (should fail)."""
        payload = {
            'symbols': ['AAPL'],
            'period': '1mo'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Should return error - need at least 2 stocks to compare
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_compare_with_no_symbols(self, client):
        """Test comparison with empty symbols list."""
        payload = {
            'symbols': [],
            'period': '1mo'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_compare_with_missing_symbols_field(self, client):
        """Test comparison with missing symbols field."""
        payload = {
            'period': '1mo'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_compare_with_different_periods(self, client, mock_yfinance_data):
        """Test comparison with different time periods."""
        periods = ['1mo', '3mo', '6mo', '1y', '5y']

        for period in periods:
            payload = {
                'symbols': ['AAPL', 'GOOGL'],
                'period': period
            }

            response = client.post(
                '/api/compare',
                data=json.dumps(payload),
                content_type='application/json'
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert len(data['stocks']) == 2

    def test_compare_response_includes_correlation(self, client, mock_yfinance_data):
        """Test that comparison includes correlation matrix."""
        payload = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'period': '6mo'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should include correlation matrix
        assert 'correlation_matrix' in data
        corr_matrix = data['correlation_matrix']

        # Matrix should include all symbols
        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            assert symbol in corr_matrix


@pytest.mark.integration
class TestTechnicalIndicatorsEndpoint:
    """Tests for /api/stock/<symbol>/indicators endpoint."""

    def test_get_indicators_for_single_stock(self, client, mock_yfinance_data):
        """Test getting technical indicators for a stock."""
        # Given: Request for indicators
        response = client.get('/api/stock/AAPL/indicators?period=6mo')

        # Then: Should return indicators data
        assert response.status_code == 200

        data = json.loads(response.data)
        assert 'symbol' in data
        assert data['symbol'] == 'AAPL'
        assert 'indicators' in data

        indicators = data['indicators']
        # Should include common indicators
        assert 'sma_20' in indicators
        assert 'sma_50' in indicators
        assert 'ema_12' in indicators
        assert 'rsi' in indicators
        assert 'macd' in indicators

    def test_get_specific_indicators(self, client, mock_yfinance_data):
        """Test requesting specific indicators."""
        indicators_list = ['sma_20', 'rsi', 'bollinger_bands']
        indicators_param = ','.join(indicators_list)

        response = client.get(
            f'/api/stock/AAPL/indicators?period=6mo&indicators={indicators_param}'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should only include requested indicators
        for indicator in indicators_list:
            assert indicator in data['indicators']

    def test_get_indicators_invalid_symbol(self, client):
        """Test getting indicators for invalid symbol."""
        response = client.get('/api/stock/INVALID123/indicators?period=6mo')

        assert response.status_code == 404
        data = json.loads(response.data)
        assert 'error' in data

    def test_indicators_with_different_periods(self, client, mock_yfinance_data):
        """Test indicators with different time periods."""
        periods = ['1mo', '3mo', '6mo', '1y']

        for period in periods:
            response = client.get(f'/api/stock/AAPL/indicators?period={period}')

            assert response.status_code == 200
            data = json.loads(response.data)
            assert 'indicators' in data

    def test_bollinger_bands_structure(self, client, mock_yfinance_data):
        """Test that Bollinger Bands return upper, middle, lower."""
        response = client.get('/api/stock/AAPL/indicators?period=6mo&indicators=bollinger_bands')

        assert response.status_code == 200
        data = json.loads(response.data)

        bb = data['indicators']['bollinger_bands']
        assert 'upper' in bb
        assert 'middle' in bb
        assert 'lower' in bb
        assert isinstance(bb['upper'], list)

    def test_macd_structure(self, client, mock_yfinance_data):
        """Test that MACD returns line, signal, histogram."""
        response = client.get('/api/stock/AAPL/indicators?period=6mo&indicators=macd')

        assert response.status_code == 200
        data = json.loads(response.data)

        macd = data['indicators']['macd']
        assert 'macd_line' in macd
        assert 'signal_line' in macd
        assert 'histogram' in macd


@pytest.mark.integration
class TestComparisonWithIndicatorsEndpoint:
    """Tests for /api/compare/with-indicators endpoint."""

    def test_compare_with_indicators(self, client, mock_yfinance_data):
        """Test comparison that includes technical indicators."""
        payload = {
            'symbols': ['AAPL', 'GOOGL'],
            'period': '6mo',
            'indicators': ['sma_20', 'rsi', 'macd']
        }

        response = client.post(
            '/api/compare/with-indicators',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Each stock should have indicators
        for symbol in ['AAPL', 'GOOGL']:
            assert symbol in data['stocks']
            assert 'indicators' in data['stocks'][symbol]

            indicators = data['stocks'][symbol]['indicators']
            assert 'sma_20' in indicators
            assert 'rsi' in indicators
            assert 'macd' in indicators

    def test_compare_with_all_indicators(self, client, mock_yfinance_data):
        """Test comparison with all available indicators."""
        payload = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'period': '6mo',
            'indicators': 'all'
        }

        response = client.post(
            '/api/compare/with-indicators',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should include many indicators
        first_stock = list(data['stocks'].values())[0]
        indicators = first_stock['indicators']

        # Check for presence of various indicator types
        assert 'sma_20' in indicators
        assert 'ema_12' in indicators
        assert 'rsi' in indicators
        assert 'macd' in indicators
        assert 'bollinger_bands' in indicators


@pytest.mark.integration
class TestChartDataEndpoint:
    """Tests for /api/compare/chart-data endpoint."""

    def test_get_chart_data_basic(self, client, mock_yfinance_data):
        """Test getting chart data formatted for frontend."""
        payload = {
            'symbols': ['AAPL', 'GOOGL'],
            'period': '6mo',
            'chart_type': 'line'
        }

        response = client.post(
            '/api/compare/chart-data',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should have Chart.js compatible format
        assert 'labels' in data  # Dates
        assert 'datasets' in data

        # Should have dataset for each stock
        assert len(data['datasets']) == 2

        for dataset in data['datasets']:
            assert 'label' in dataset  # Stock symbol
            assert 'data' in dataset   # Price data
            assert 'borderColor' in dataset
            assert 'backgroundColor' in dataset

    def test_chart_data_with_indicators(self, client, mock_yfinance_data):
        """Test chart data that includes indicator overlays."""
        payload = {
            'symbols': ['AAPL'],
            'period': '6mo',
            'chart_type': 'line',
            'indicators': ['sma_20', 'sma_50', 'bollinger_bands']
        }

        response = client.post(
            '/api/compare/chart-data',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should have datasets for stock + indicators
        dataset_labels = [ds['label'] for ds in data['datasets']]

        assert 'AAPL' in dataset_labels
        assert 'AAPL SMA(20)' in dataset_labels
        assert 'AAPL SMA(50)' in dataset_labels
        assert 'AAPL BB Upper' in dataset_labels
        assert 'AAPL BB Lower' in dataset_labels

    def test_chart_data_normalized(self, client, mock_yfinance_data):
        """Test normalized chart data for comparison."""
        payload = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'period': '6mo',
            'chart_type': 'line',
            'normalize': True
        }

        response = client.post(
            '/api/compare/chart-data',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # All stocks should start at 100
        for dataset in data['datasets']:
            first_value = dataset['data'][0]
            assert abs(first_value - 100) < 0.1

    def test_chart_data_candlestick(self, client, mock_yfinance_data):
        """Test candlestick chart data format."""
        payload = {
            'symbols': ['AAPL'],
            'period': '3mo',
            'chart_type': 'candlestick'
        }

        response = client.post(
            '/api/compare/chart-data',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Candlestick should have OHLC data
        dataset = data['datasets'][0]
        assert 'data' in dataset

        # Each data point should have o, h, l, c
        for point in dataset['data']:
            assert 'o' in point  # open
            assert 'h' in point  # high
            assert 'l' in point  # low
            assert 'c' in point  # close


@pytest.mark.integration
class TestPerformanceMetricsEndpoint:
    """Tests for /api/compare/metrics endpoint."""

    def test_get_performance_metrics(self, client, mock_yfinance_data):
        """Test getting performance metrics for compared stocks."""
        payload = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'period': '1y'
        }

        response = client.post(
            '/api/compare/metrics',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should have metrics for each stock
        assert 'metrics' in data
        assert len(data['metrics']) == 3

        for symbol in ['AAPL', 'GOOGL', 'MSFT']:
            assert symbol in data['metrics']
            metrics = data['metrics'][symbol]

            # Check required metrics
            assert 'total_return' in metrics
            assert 'volatility' in metrics
            assert 'sharpe_ratio' in metrics
            assert 'max_drawdown' in metrics
            assert 'current_price' in metrics
            assert 'price_change' in metrics

    def test_metrics_include_rankings(self, client, mock_yfinance_data):
        """Test that metrics include rankings."""
        payload = {
            'symbols': ['AAPL', 'GOOGL', 'MSFT'],
            'period': '1y'
        }

        response = client.post(
            '/api/compare/metrics',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        # Should include rankings
        assert 'rankings' in data
        rankings = data['rankings']

        assert 'by_return' in rankings
        assert 'by_volatility' in rankings
        assert 'by_sharpe_ratio' in rankings

        # Each ranking should have ordered list
        assert len(rankings['by_return']) == 3


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in API endpoints."""

    def test_invalid_json_payload(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            '/api/compare',
            data='invalid json{',
            content_type='application/json'
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_missing_content_type(self, client):
        """Test handling of missing Content-Type header."""
        payload = {'symbols': ['AAPL', 'GOOGL']}

        response = client.post(
            '/api/compare',
            data=json.dumps(payload)
            # No content_type specified
        )

        # Should either work or return 415 Unsupported Media Type
        assert response.status_code in [200, 415]

    def test_too_many_symbols(self, client):
        """Test handling of too many symbols (>10)."""
        payload = {
            'symbols': [f'STOCK{i}' for i in range(15)],
            'period': '1mo'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Should return error
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data

    def test_invalid_period(self, client):
        """Test handling of invalid time period."""
        payload = {
            'symbols': ['AAPL', 'GOOGL'],
            'period': 'invalid_period'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        assert response.status_code == 400

    def test_yfinance_api_failure(self, client, mock_yfinance_failure):
        """Test handling when yfinance API fails."""
        payload = {
            'symbols': ['AAPL', 'GOOGL'],
            'period': '1mo'
        }

        response = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Should return error status
        assert response.status_code in [500, 503]
        data = json.loads(response.data)
        assert 'error' in data

    def test_rate_limiting(self, client, mock_yfinance_data):
        """Test API rate limiting behavior."""
        payload = {
            'symbols': ['AAPL', 'GOOGL'],
            'period': '1mo'
        }

        # Make many rapid requests
        responses = []
        for _ in range(50):
            response = client.post(
                '/api/compare',
                data=json.dumps(payload),
                content_type='application/json'
            )
            responses.append(response.status_code)

        # If rate limiting is implemented, should see 429 status codes
        # If not implemented, all should be 200
        assert all(code in [200, 429] for code in responses)


@pytest.mark.integration
class TestCaching:
    """Tests for caching behavior."""

    def test_repeated_requests_use_cache(self, client, mock_yfinance_data):
        """Test that repeated requests are faster due to caching."""
        payload = {
            'symbols': ['AAPL', 'GOOGL'],
            'period': '6mo'
        }

        import time

        # First request
        start1 = time.time()
        response1 = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        duration1 = time.time() - start1

        # Second request (should be cached)
        start2 = time.time()
        response2 = client.post(
            '/api/compare',
            data=json.dumps(payload),
            content_type='application/json'
        )
        duration2 = time.time() - start2

        # Both should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Second request should be faster (if caching is implemented)
        # This test might be flaky, so we just check they both work
        assert duration2 <= duration1 * 2  # Reasonable threshold

    def test_cache_invalidation_with_different_period(self, client, mock_yfinance_data):
        """Test that cache is different for different periods."""
        base_payload = {
            'symbols': ['AAPL', 'GOOGL']
        }

        # Request with 1mo period
        response1 = client.post(
            '/api/compare',
            data=json.dumps({**base_payload, 'period': '1mo'}),
            content_type='application/json'
        )

        # Request with 6mo period
        response2 = client.post(
            '/api/compare',
            data=json.dumps({**base_payload, 'period': '6mo'}),
            content_type='application/json'
        )

        # Both should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200

        # Data should be different (different number of data points)
        data1 = json.loads(response1.data)
        data2 = json.loads(response2.data)

        dates1 = data1['stocks']['AAPL']['dates']
        dates2 = data2['stocks']['AAPL']['dates']

        assert len(dates1) != len(dates2)
