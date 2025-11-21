# Stock Price Tracker

A web application that allows you to track stock prices and view financial data from Yahoo Finance.

## Features

- **Stock Symbol Search**: Enter any stock symbol to get real-time financial data
- **Financial Data Table**: View key metrics including current price, volume, market cap, P/E ratio, and more
- **Price Chart**: Interactive 6-month historical price chart
- **CSV Export**: Download historical stock data as CSV file
- **Responsive Design**: Beautiful, modern UI that works on all devices

## Installation

1. Clone the repository or download the files

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter a stock symbol (e.g., AAPL, GOOGL, MSFT, TSLA) and click "Get Stock Data"

4. View the financial data, statistics, and price chart

5. Click "Download CSV" to export the historical data

## Supported Stock Symbols

The application supports any stock symbol available on Yahoo Finance, including:
- US stocks (e.g., AAPL, GOOGL, MSFT)
- International stocks (e.g., TSM, BABA)
- ETFs (e.g., SPY, QQQ)
- And many more!

## API Endpoints

- `GET /` - Main web interface
- `GET /api/stock/<symbol>` - Get stock data as JSON
- `GET /api/download/<symbol>` - Download stock data as CSV

## Technologies Used

- **Backend**: Python, Flask
- **Data Source**: Yahoo Finance (via yfinance library)
- **Frontend**: HTML, CSS, JavaScript
- **Charts**: Chart.js
- **Data Processing**: Pandas

## Notes

- Stock data is fetched in real-time from Yahoo Finance
- Historical data covers the last 6 months
- All prices are in USD
- The application requires an internet connection to fetch stock data

## License

MIT
