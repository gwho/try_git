# Testing Guide - Stock Comparison with Technical Charts

This document provides a quick start guide for using the TDD test suite for the stock comparison feature.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run All Tests

```bash
pytest
```

### 3. View Results

Tests will output results showing:
- ✅ Passed tests
- ❌ Failed tests (these guide implementation)
- Test coverage statistics

## Test-Driven Development (TDD) Workflow

### The Red-Green-Refactor Cycle

```
1. RED:    Write a failing test
2. GREEN:  Write minimal code to pass the test
3. REFACTOR: Improve code while keeping tests green
```

### Example Workflow

#### Step 1: Pick a Test File

Start with a specific feature, for example technical indicators:

```bash
pytest tests/unit/test_technical_indicators.py -v
```

All tests will **FAIL** initially (this is expected!).

#### Step 2: Pick One Test

Let's implement Simple Moving Average (SMA):

```bash
pytest tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic -v
```

#### Step 3: Create the Module

The test expects a module called `technical_indicators.py`:

```python
# technical_indicators.py
import pandas as pd

def calculate_sma(prices, window):
    """Calculate Simple Moving Average."""
    return prices.rolling(window=window).mean()
```

#### Step 4: Run Test Again

```bash
pytest tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic -v
```

Test should now **PASS** ✅

#### Step 5: Continue with Next Test

Repeat for each test in the file:

```bash
pytest tests/unit/test_technical_indicators.py::TestSimpleMovingAverage -v
```

## Recommended Implementation Order

### Phase 1: Technical Indicators (Unit Tests)

Start here because other features depend on these:

1. **`tests/unit/test_technical_indicators.py`**
   ```bash
   pytest tests/unit/test_technical_indicators.py
   ```

   Implement in this order:
   - `calculate_sma()` - Simple Moving Average
   - `calculate_ema()` - Exponential Moving Average
   - `calculate_rsi()` - Relative Strength Index
   - `calculate_macd()` - MACD indicator
   - `calculate_bollinger_bands()` - Bollinger Bands
   - `calculate_stochastic()` - Stochastic Oscillator
   - `calculate_atr()` - Average True Range
   - `calculate_obv()` - On-Balance Volume
   - `calculate_vwap()` - VWAP

   Create file: `technical_indicators.py`

### Phase 2: Stock Comparison Logic (Unit Tests)

2. **`tests/unit/test_stock_comparison.py`**
   ```bash
   pytest tests/unit/test_stock_comparison.py
   ```

   Implement:
   - `compare_stocks()` - Compare multiple stocks
   - `normalize_prices()` - Normalize for comparison
   - `calculate_percentage_change()` - Returns calculation
   - `calculate_daily_returns()` - Daily returns
   - `calculate_volatility()` - Volatility measure
   - `calculate_sharpe_ratio()` - Risk-adjusted returns
   - `calculate_correlation()` - Correlation analysis
   - `calculate_beta()` - Beta calculation
   - `calculate_max_drawdown()` - Drawdown analysis
   - Portfolio functions
   - Ranking functions
   - Data alignment functions

   Create file: `stock_comparison.py`

### Phase 3: Chart Data Formatting (Unit Tests)

3. **`tests/unit/test_chart_data_formatting.py`**
   ```bash
   pytest tests/unit/test_chart_data_formatting.py
   ```

   Implement:
   - `format_line_chart()` - Line chart formatting
   - `format_multi_stock_chart()` - Multi-stock charts
   - `format_candlestick_chart()` - Candlestick charts
   - `format_with_indicators()` - Indicator overlays
   - `format_bollinger_bands()` - Bollinger Bands display
   - Validation functions
   - Color scheme functions
   - Normalization functions
   - Chart options generation

   Create file: `chart_formatter.py`

### Phase 4: API Integration (Integration Tests)

4. **`tests/integration/test_comparison_api.py`**
   ```bash
   pytest tests/integration/test_comparison_api.py
   ```

   Add to existing `app.py`:
   - `/api/compare` - Compare multiple stocks
   - `/api/stock/<symbol>/indicators` - Get technical indicators
   - `/api/compare/with-indicators` - Compare with indicators
   - `/api/compare/chart-data` - Chart-ready data
   - `/api/compare/metrics` - Performance metrics

## Testing Best Practices

### 1. Run Tests Frequently

```bash
# Run tests after every small change
pytest

# Or use watch mode (requires pytest-watch)
ptw
```

### 2. Focus on One Feature at a Time

```bash
# Work on SMA only
pytest -k "sma"

# Work on API endpoints only
pytest tests/integration/
```

### 3. Use Coverage to Find Gaps

```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

### 4. Test Edge Cases

The test suite includes edge cases:
- Empty data
- Insufficient data
- Invalid inputs
- NaN values
- Negative prices

### 5. Keep Tests Fast

- Unit tests should run in milliseconds
- Use mocked data (provided in `conftest.py`)
- Run slow tests separately: `pytest -m "not slow"`

## Common Commands

```bash
# Run everything
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_technical_indicators.py

# Run specific test class
pytest tests/unit/test_technical_indicators.py::TestSimpleMovingAverage

# Run specific test method
pytest tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic

# Run by marker
pytest -m unit          # Only unit tests
pytest -m integration   # Only integration tests

# Run in parallel (faster)
pytest -n auto

# Stop on first failure
pytest -x

# Show print statements
pytest -s

# Run last failed tests only
pytest --lf

# Coverage report
pytest --cov=. --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=. --cov-report=html
```

## Understanding Test Output

### Success ✅
```
tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic PASSED
```

### Failure ❌
```
tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic FAILED

E   ModuleNotFoundError: No module named 'technical_indicators'
```

This tells you:
1. Which test failed
2. Why it failed (module not found)
3. What you need to do (create `technical_indicators.py`)

### Error ⚠️
```
tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic ERROR

E   ImportError: cannot import name 'calculate_sma' from 'technical_indicators'
```

This tells you:
1. The module exists but function is missing
2. Add the `calculate_sma` function

## Debugging Failed Tests

### 1. Read the Error Message

```bash
pytest tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic -v
```

Error messages show:
- Expected vs actual values
- Line numbers
- Variable values

### 2. Use Print Statements

```python
def test_something(self):
    result = calculate_sma(prices, 5)
    print(f"Result: {result}")  # Will show with pytest -s
    assert result.iloc[4] == 14.0
```

Run with: `pytest -s`

### 3. Use Debugger

```bash
pytest --pdb  # Drop into debugger on failure
```

### 4. Check Test Logic

Read the test to understand what it expects:

```python
def test_sma_calculation_basic(self):
    # Given: A series of prices
    prices = pd.Series([10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
    window = 5

    # When: SMA is calculated
    result = calculate_sma(prices, window)

    # Then: First 4 values should be NaN, then moving averages
    assert pd.isna(result.iloc[0:4]).all()
    assert result.iloc[4] == 14.0  # (10+12+14+16+18)/5
```

## Module Organization

Create these files as you implement:

```
/home/user/try_git/
├── app.py                          # Existing Flask app (extend this)
├── technical_indicators.py         # NEW: Technical indicators
├── stock_comparison.py             # NEW: Comparison logic
├── chart_formatter.py              # NEW: Chart formatting
├── chart_validator.py              # NEW: Data validation
├── requirements.txt                # Updated with test dependencies
└── tests/                          # All tests (already created)
    ├── conftest.py
    ├── unit/
    │   ├── test_technical_indicators.py
    │   ├── test_stock_comparison.py
    │   └── test_chart_data_formatting.py
    └── integration/
        └── test_comparison_api.py
```

## Tips for Success

### 1. Start Small
- Don't try to implement everything at once
- Focus on one test at a time
- Get it working, then move to the next

### 2. Use the Fixtures
- `conftest.py` provides lots of test data
- Use `sample_stock_data`, `mock_yfinance_data`, etc.
- Don't make real API calls in tests

### 3. Follow the Tests
- Tests document the expected behavior
- Read test names - they describe what to implement
- Test assertions show expected outputs

### 4. Keep It Simple
- Write minimal code to pass tests
- Refactor after tests pass
- Don't over-engineer

### 5. Run Tests Often
- After every function you write
- Before committing code
- As you refactor

## Getting Help

### Check Test Documentation
```bash
pytest --markers  # Show available markers
pytest --fixtures # Show available fixtures
```

### View Test Coverage
```bash
pytest --cov=. --cov-report=html
open htmlcov/index.html
```

This shows which lines need tests.

## Next Steps

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Run tests**: `pytest` (they will all fail initially)
3. **Pick first test**: `pytest tests/unit/test_technical_indicators.py::TestSimpleMovingAverage::test_sma_calculation_basic -v`
4. **Implement function**: Create `technical_indicators.py` with `calculate_sma()`
5. **Run test again**: Should pass ✅
6. **Repeat**: Move to next test

Happy Test-Driven Development! 🚀
