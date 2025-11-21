from flask import Flask, render_template, jsonify, request, send_file
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    try:
        # Fetch stock data
        stock = yf.Ticker(symbol.upper())

        # Get historical data (last 6 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return jsonify({'error': 'Invalid stock symbol or no data available'}), 404

        # Get stock info
        info = stock.info

        # Prepare historical data for chart
        chart_data = {
            'dates': hist.index.strftime('%Y-%m-%d').tolist(),
            'prices': hist['Close'].round(2).tolist(),
            'volumes': hist['Volume'].tolist()
        }

        # Prepare key financial metrics
        financial_data = {
            'symbol': symbol.upper(),
            'company_name': info.get('longName', 'N/A'),
            'current_price': round(info.get('currentPrice', hist['Close'].iloc[-1]), 2),
            'previous_close': round(info.get('previousClose', 0), 2),
            'open': round(info.get('open', 0), 2),
            'day_high': round(info.get('dayHigh', 0), 2),
            'day_low': round(info.get('dayLow', 0), 2),
            'volume': info.get('volume', 0),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 'N/A',
            'eps': round(info.get('trailingEps', 0), 2) if info.get('trailingEps') else 'N/A',
            'dividend_yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 'N/A',
            'fifty_two_week_high': round(info.get('fiftyTwoWeekHigh', 0), 2),
            'fifty_two_week_low': round(info.get('fiftyTwoWeekLow', 0), 2),
            'beta': round(info.get('beta', 0), 2) if info.get('beta') else 'N/A',
        }

        return jsonify({
            'financial_data': financial_data,
            'chart_data': chart_data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<symbol>')
def download_csv(symbol):
    try:
        # Fetch stock data
        stock = yf.Ticker(symbol.upper())

        # Get historical data (last 6 months)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return jsonify({'error': 'Invalid stock symbol or no data available'}), 404

        # Prepare CSV data
        hist.index = hist.index.strftime('%Y-%m-%d')
        hist = hist.round(2)

        # Create CSV in memory
        output = io.StringIO()
        hist.to_csv(output)
        output.seek(0)

        # Convert to bytes
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)

        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'{symbol.upper()}_stock_data.csv'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
