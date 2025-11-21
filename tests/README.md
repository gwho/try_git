# Stock Comparison Tests

Comprehensive test suite for the stock comparison feature with technical charts, following Test-Driven Development (TDD) principles.

## Test Structure

```
tests/
├── conftest.py                          # Pytest configuration and fixtures
├── unit/                                # Unit tests
│   ├── test_technical_indicators.py     # Technical indicator calculations
│   ├── test_stock_comparison.py         # Stock comparison logic
│   └── test_chart_data_formatting.py    # Chart data formatting
├── integration/                         # Integration tests
│   └── test_comparison_api.py           # API endpoint tests
└── fixtures/                            # Test data and helpers
    └── sample_stock_data.py             # Mock stock data generators
```

## Running Tests

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Run tests in parallel (faster)
pytest -n auto
```

### Run with Coverage Report

```bash
pytest --cov=. --cov-report=html
# Open htmlcov/index.html to view coverage report
```

### Run Specific Test File

```bash
pytest tests/unit/test_technical_indicators.py
pytest tests/integration/test_comparison_api.py
```

### Run Specific Test

```bash
pytest tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic
```

## Test Coverage

### Unit Tests

#### Technical Indicators (`test_technical_indicators.py`)
- **Simple Moving Average (SMA)**
  - Basic calculation with known values
  - Different window sizes (20, 50, 200 days)
  - Insufficient data handling
  - NaN value handling

- **Exponential Moving Average (EMA)**
  - Basic calculation
  - Responsiveness compared to SMA
  - Common periods (12, 26 for MACD)

- **Relative Strength Index (RSI)**
  - Range bounds (0-100)
  - Overbought conditions (>70)
  - Oversold conditions (<30)
  - Neutral conditions (~50)

- **MACD (Moving Average Convergence Divergence)**
  - Line, signal, and histogram calculation
  - Bullish crossovers
  - Bearish crossovers
  - Custom parameters

- **Bollinger Bands**
  - Upper, middle, lower band calculation
  - Band ordering validation
  - Volatility expansion/contraction
  - Different standard deviations (2, 3)

- **Stochastic Oscillator**
  - Range bounds (0-100)
  - Overbought/oversold conditions

- **Average True Range (ATR)**
  - Volatility measurement
  - Positive values validation

- **On-Balance Volume (OBV)**
  - Volume-price relationship
  - Accumulation/distribution detection

- **Volume-Weighted Average Price (VWAP)**
  - Intraday price average
  - Within price range validation

#### Stock Comparison (`test_stock_comparison.py`)
- **Basic Comparison**
  - Two stock comparison
  - Multiple stock comparison (3-5 stocks)
  - Different date ranges alignment

- **Price Normalization**
  - Percentage change from start
  - Base 100 normalization

- **Performance Metrics**
  - Percentage change calculation
  - Daily returns
  - Cumulative returns
  - Volatility (standard deviation)
  - Sharpe ratio (risk-adjusted returns)

- **Correlation Analysis**
  - Two-stock correlation
  - Correlation matrix (multiple stocks)
  - Negative correlation detection
  - Uncorrelated stocks

- **Beta Calculation**
  - Beta > 1 (more volatile than market)
  - Beta < 1 (less volatile than market)
  - Negative beta (inverse to market)

- **Drawdown Analysis**
  - Maximum drawdown calculation
  - Drawdown series over time
  - No drawdown scenarios

- **Portfolio Metrics**
  - Portfolio value calculation
  - Equal-weight portfolios
  - Market-cap weighted portfolios

- **Ranking and Sorting**
  - Rank by returns
  - Rank by volatility
  - Rank by Sharpe ratio

- **Data Alignment**
  - Same date ranges
  - Different date ranges
  - Missing dates handling

#### Chart Data Formatting (`test_chart_data_formatting.py`)
- **Chart.js Formatting**
  - Line chart single stock
  - Multi-stock comparison
  - Candlestick (OHLC) charts
  - Indicator overlays
  - Bollinger Bands display

- **Data Validation**
  - Date format validation
  - Date gaps handling
  - Chronological order
  - Price data validation
  - NaN value handling
  - Negative price detection
  - Data length matching
  - Minimum data points

- **Date Formatting**
  - Display formatting
  - Monthly format
  - Yearly format
  - Auto-format by range

- **Color Schemes**
  - Unique color assignment
  - Consistent colors
  - Different schemes (default, vibrant, pastel)
  - Indicator color differentiation

- **Normalization**
  - Percentage normalization
  - Base 100 normalization
  - Multiple stock normalization
  - Denormalization

- **Chart Options**
  - Line chart options
  - Options with indicators
  - Options with volume subplot
  - Tooltip configuration

- **Volume Formatting**
  - Volume data formatting
  - Bar chart representation
  - Color by price change

- **Serialization**
  - JSON serialization
  - NaN handling in JSON
  - Datetime handling in JSON

- **Responsive Formatting**
  - Mobile optimization
  - Desktop optimization
  - Adaptive data point sampling

### Integration Tests

#### Comparison API (`test_comparison_api.py`)
- **Compare Stocks Endpoint** (`/api/compare`)
  - Successful two-stock comparison
  - Multiple stock comparison
  - Invalid symbol handling
  - Single symbol error
  - Empty symbols error
  - Missing fields handling
  - Different time periods
  - Correlation matrix inclusion

- **Technical Indicators Endpoint** (`/api/stock/<symbol>/indicators`)
  - Single stock indicators
  - Specific indicators request
  - Invalid symbol handling
  - Different time periods
  - Bollinger Bands structure
  - MACD structure

- **Comparison with Indicators** (`/api/compare/with-indicators`)
  - Multiple stocks with indicators
  - All indicators request

- **Chart Data Endpoint** (`/api/compare/chart-data`)
  - Basic chart data
  - Chart with indicator overlays
  - Normalized comparison
  - Candlestick format

- **Performance Metrics Endpoint** (`/api/compare/metrics`)
  - Performance metrics retrieval
  - Rankings inclusion

- **Error Handling**
  - Invalid JSON payload
  - Missing Content-Type
  - Too many symbols (>10)
  - Invalid period
  - yfinance API failure
  - Rate limiting

- **Caching**
  - Repeated request caching
  - Cache invalidation

## Test Fixtures

The `conftest.py` file provides numerous fixtures:

- **Flask App Fixtures**: `app`, `client`
- **Sample Data**: `sample_dates`, `sample_stock_data`, `sample_multiple_stocks`
- **Mock Data**: `mock_yfinance_data`, `mock_yfinance_failure`
- **Price Series**: `sample_price_series`, `trending_up_prices`, `trending_down_prices`, `volatile_prices`, `stable_prices`
- **Technical Data**: `sample_technical_indicators`, `sample_ohlc_data`
- **Helpers**: `assert_valid_chart_data`, `assert_valid_api_response`, `create_mock_comparison_request`

## Writing New Tests

### Example Unit Test

```python
import pytest

@pytest.mark.unit
class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic functionality."""
        # Arrange
        input_data = [1, 2, 3, 4, 5]

        # Act
        result = my_function(input_data)

        # Assert
        assert result == expected_output
```

### Example Integration Test

```python
import pytest
import json

@pytest.mark.integration
class TestNewEndpoint:
    def test_endpoint_success(self, client, mock_yfinance_data):
        """Test successful API call."""
        # Arrange
        payload = {'symbol': 'AAPL'}

        # Act
        response = client.post(
            '/api/new-endpoint',
            data=json.dumps(payload),
            content_type='application/json'
        )

        # Assert
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'expected_field' in data
```

## TDD Workflow

1. **Write the test first** - Define what you want the code to do
2. **Run the test** - It should fail (Red)
3. **Write minimal code** - Make the test pass (Green)
4. **Refactor** - Improve the code while keeping tests green
5. **Repeat** - For each new feature or bug fix

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest --cov=. --cov-report=xml
```

## Test Markers

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Tests with external dependencies
- `@pytest.mark.slow` - Long-running tests

Run specific markers:
```bash
pytest -m "unit and not slow"
```

## Debugging Tests

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Stop on first failure
pytest -x

# Drop into debugger on failure
pytest --pdb

# Run last failed tests
pytest --lf
```

## Test Data

Sample data is generated deterministically using seeded random number generators, ensuring:
- Reproducible test results
- Realistic stock price behavior
- No external API dependencies
- Fast test execution

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure all tests pass
3. Maintain >80% code coverage
4. Follow existing test patterns
5. Document complex test scenarios

## Notes

- Tests use mocked yfinance data to avoid external API calls
- All dates are timezone-naive for consistency
- Price data includes realistic OHLCV values
- Tests cover both success and failure scenarios
- Edge cases are thoroughly tested
