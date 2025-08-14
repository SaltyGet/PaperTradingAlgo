import os
import pandas as pd
import numpy as np
import yfinance as yf
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Tuple
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingStrategy:
    def __init__(self):
        # Alpaca API credentials (set as environment variables)
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.secret_key = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = 'https://paper-api.alpaca.markets'  # Paper trading URL
        
        # Initialize Alpaca API
        self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
        
        # Portfolio constraints
        self.max_individual_stock = 0.05  # 5% max for non-tech stocks
        self.max_tech_stock = 0.15  # 15% max for tech stocks
        self.max_sector_allocation = 0.25  # 25% max for non-tech sectors
        self.max_tech_allocation = 0.75  # 75% max for tech sector
        
        # Strategy parameters
        self.lookback_period = 252  # 1 year
        self.rebalance_frequency = 21  # 21 trading days (monthly)
        self.min_volume = 1000000  # Minimum daily volume
        self.min_price = 5.0  # Minimum stock price
        
        # Technical indicators parameters
        self.momentum_window = 63  # 3 months
        self.volatility_window = 21  # 1 month
        self.rsi_window = 14
        
        # Sector mappings (simplified)
        self.sector_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'ADBE', 'CRM'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABT', 'TMO', 'DHR', 'BMY', 'ABBV', 'MDT'],
            'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'SCHW', 'USB'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'PXD', 'OXY'],
            'Industrials': ['BA', 'CAT', 'GE', 'UNP', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'MMM'],
            'Consumer': ['WMT', 'PG', 'KO', 'PEP', 'NKE', 'HD', 'MCD', 'SBUX', 'TGT', 'COST'],
            'Materials': ['LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'IFF'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG', 'EXR', 'AVB', 'VTR', 'WELL', 'O'],
            'Communication': ['VZ', 'T', 'CMCSA', 'DIS', 'CHTR', 'TMUS', 'ATVI', 'EA', 'TTWO', 'NWSA']
        }
        
        # Load NYSE stock universe
        self.stock_universe = self.get_nyse_stocks()
        
    def get_nyse_stocks(self) -> List[str]:
        """Get list of NYSE stocks with sufficient history and volume"""
        try:
            # Get all NYSE stocks from Alpaca
            assets = self.api.list_assets(status='active', asset_class='us_equity')
            nyse_stocks = [asset.symbol for asset in assets if asset.exchange == 'NYSE']
            
            # Filter stocks based on our criteria
            filtered_stocks = []
            for symbol in nyse_stocks[:1200]:  # Limit to avoid rate limits
                try:
                    stock = yf.Ticker(symbol)
                    hist = stock.history(period='2y')
                    
                    if len(hist) > 500:  # At least 2 years of data
                        avg_volume = hist['Volume'].tail(30).mean()
                        avg_price = hist['Close'].tail(30).mean()
                        
                        if avg_volume > self.min_volume and avg_price > self.min_price:
                            filtered_stocks.append(symbol)
                            
                    if len(filtered_stocks) >= 1000:
                        break
                        
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    continue
                    
            logger.info(f"Selected {len(filtered_stocks)} stocks for universe")
            return filtered_stocks
            
        except Exception as e:
            logger.error(f"Error getting NYSE stocks: {e}")
            return []
    
    def get_sector(self, symbol: str) -> str:
        """Determine sector for a given stock symbol"""
        for sector, stocks in self.sector_mapping.items():
            if symbol in stocks:
                return sector
        return 'Other'  # Default sector
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for stock analysis"""
        # Price momentum
        data['momentum'] = data['Close'].pct_change(self.momentum_window)
        
        # Volatility
        data['volatility'] = data['Close'].pct_change().rolling(self.volatility_window).std()
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Moving averages
        data['ma_20'] = data['Close'].rolling(20).mean()
        data['ma_50'] = data['Close'].rolling(50).mean()
        data['ma_200'] = data['Close'].rolling(200).mean()
        
        # Price relative to moving averages
        data['price_ma20_ratio'] = data['Close'] / data['ma_20']
        data['price_ma50_ratio'] = data['Close'] / data['ma_50']
        data['price_ma200_ratio'] = data['Close'] / data['ma_200']
        
        # Volume indicators
        data['volume_ma'] = data['Volume'].rolling(20).mean()
        data['volume_ratio'] = data['Volume'] / data['volume_ma']
        
        return data
    
    def calculate_score(self, symbol: str) -> float:
        """Calculate composite score for stock ranking"""
        try:
            # Get stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period='1y')
            
            if len(hist) < 200:
                return 0
                
            # Calculate technical indicators
            hist = self.calculate_technical_indicators(hist)
            
            # Get latest values
            latest = hist.iloc[-1]
            
            # Scoring components
            momentum_score = latest['momentum'] if not pd.isna(latest['momentum']) else 0
            rsi_score = (50 - abs(latest['rsi'] - 50)) / 50 if not pd.isna(latest['rsi']) else 0
            ma_score = (latest['price_ma20_ratio'] - 1) * 2 if not pd.isna(latest['price_ma20_ratio']) else 0
            volume_score = min(latest['volume_ratio'], 3) / 3 if not pd.isna(latest['volume_ratio']) else 0
            
            # Trend score
            trend_score = 0
            if latest['Close'] > latest['ma_20'] > latest['ma_50'] > latest['ma_200']:
                trend_score = 1
            elif latest['Close'] > latest['ma_20'] > latest['ma_50']:
                trend_score = 0.5
            elif latest['Close'] > latest['ma_20']:
                trend_score = 0.25
                
            # Volatility penalty
            volatility_penalty = -min(latest['volatility'], 0.1) * 10 if not pd.isna(latest['volatility']) else 0
            
            # Combined score
            total_score = (momentum_score * 0.3 + 
                          rsi_score * 0.15 + 
                          ma_score * 0.25 + 
                          volume_score * 0.1 + 
                          trend_score * 0.15 + 
                          volatility_penalty * 0.05)
            
            return total_score
            
        except Exception as e:
            logger.warning(f"Error calculating score for {symbol}: {e}")
            return 0
    
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """Determine optimal portfolio allocation"""
        # Calculate scores for all stocks
        stock_scores = {}
        for symbol in self.stock_universe:
            score = self.calculate_score(symbol)
            if score > 0:
                stock_scores[symbol] = score
        
        # Sort by score
        sorted_stocks = sorted(stock_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Allocate portfolio with constraints
        portfolio = {}
        sector_allocations = {}
        
        for symbol, score in sorted_stocks:
            sector = self.get_sector(symbol)
            
            # Determine max allocation for this stock
            if sector == 'Technology':
                max_stock_allocation = self.max_tech_stock
                max_sector_allocation = self.max_tech_allocation
            else:
                max_stock_allocation = self.max_individual_stock
                max_sector_allocation = self.max_sector_allocation
            
            # Check sector constraint
            current_sector_allocation = sector_allocations.get(sector, 0)
            
            if current_sector_allocation < max_sector_allocation:
                # Calculate allocation based on score and constraints
                remaining_sector_capacity = max_sector_allocation - current_sector_allocation
                allocation = min(max_stock_allocation, remaining_sector_capacity)
                
                # Adjust allocation based on relative score
                total_allocated = sum(portfolio.values())
                if total_allocated < 0.95:  # Leave 5% cash
                    allocation = min(allocation, 0.95 - total_allocated)
                    
                    if allocation > 0.01:  # Minimum 1% allocation
                        portfolio[symbol] = allocation
                        sector_allocations[sector] = current_sector_allocation + allocation
            
            # Stop if portfolio is sufficiently allocated
            if sum(portfolio.values()) >= 0.95:
                break
        
        # Normalize to ensure total allocation <= 100%
        total_allocation = sum(portfolio.values())
        if total_allocation > 0:
            portfolio = {k: v/total_allocation * 0.95 for k, v in portfolio.items()}
        
        logger.info(f"Portfolio allocation: {len(portfolio)} stocks")
        logger.info(f"Sector allocations: {sector_allocations}")
        
        return portfolio
    
    def execute_trades(self, target_portfolio: Dict[str, float]):
        """Execute trades to achieve target portfolio"""
        try:
            # Get current positions
            positions = self.api.list_positions()
            current_positions = {pos.symbol: float(pos.qty) for pos in positions}
            
            # Get account info
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            
            # Calculate target quantities
            target_quantities = {}
            for symbol, allocation in target_portfolio.items():
                try:
                    # Get current price
                    latest_quote = self.api.get_latest_quote(symbol)
                    current_price = float(latest_quote.ask_price)
                    
                    target_value = portfolio_value * allocation
                    target_qty = int(target_value / current_price)
                    
                    if target_qty > 0:
                        target_quantities[symbol] = target_qty
                        
                except Exception as e:
                    logger.warning(f"Error getting price for {symbol}: {e}")
                    continue
            
            # Execute trades
            for symbol, target_qty in target_quantities.items():
                current_qty = current_positions.get(symbol, 0)
                qty_diff = target_qty - current_qty
                
                if abs(qty_diff) > 0:
                    try:
                        if qty_diff > 0:
                            # Buy
                            self.api.submit_order(
                                symbol=symbol,
                                qty=qty_diff,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )
                            logger.info(f"Buying {qty_diff} shares of {symbol}")
                        else:
                            # Sell
                            self.api.submit_order(
                                symbol=symbol,
                                qty=abs(qty_diff),
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                            logger.info(f"Selling {abs(qty_diff)} shares of {symbol}")
                            
                    except Exception as e:
                        logger.error(f"Error executing trade for {symbol}: {e}")
            
            # Close positions not in target portfolio
            for symbol in current_positions:
                if symbol not in target_quantities and current_positions[symbol] > 0:
                    try:
                        self.api.submit_order(
                            symbol=symbol,
                            qty=int(current_positions[symbol]),
                            side='sell',
                            type='market',
                            time_in_force='day'
                        )
                        logger.info(f"Closing position in {symbol}")
                    except Exception as e:
                        logger.error(f"Error closing position for {symbol}: {e}")
                        
        except Exception as e:
            logger.error(f"Error executing trades: {e}")
    
    def get_sp500_performance(self) -> float:
        """Get S&P 500 performance for comparison"""
        try:
            spy = yf.Ticker('SPY')
            hist = spy.history(period='1y')
            return (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
        except Exception as e:
            logger.error(f"Error getting S&P 500 performance: {e}")
            return 0
    
    def log_performance(self):
        """Log portfolio performance vs S&P 500"""
        try:
            account = self.api.get_account()
            
            # Calculate portfolio performance (simplified)
            portfolio_value = float(account.portfolio_value)
            equity = float(account.equity)
            
            # Get S&P 500 performance
            sp500_perf = self.get_sp500_performance()
            
            logger.info(f"Portfolio Value: ${portfolio_value:,.2f}")
            logger.info(f"Equity: ${equity:,.2f}")
            logger.info(f"S&P 500 Performance (1Y): {sp500_perf:.2f}%")
            
        except Exception as e:
            logger.error(f"Error logging performance: {e}")
    
    def run_strategy(self):
        """Main strategy execution loop"""
        logger.info("Starting trading strategy...")
        
        last_rebalance = None
        
        while True:
            try:
                current_time = datetime.now()
                
                # Check if market is open
                clock = self.api.get_clock()
                if not clock.is_open:
                    logger.info("Market is closed. Waiting...")
                    time.sleep(3600)  # Wait 1 hour
                    continue
                
                # Check if rebalancing is needed
                if (last_rebalance is None or 
                    (current_time - last_rebalance).days >= self.rebalance_frequency):
                    
                    logger.info("Rebalancing portfolio...")
                    
                    # Get optimal portfolio allocation
                    target_portfolio = self.get_portfolio_allocation()
                    
                    # Execute trades
                    self.execute_trades(target_portfolio)
                    
                    # Log performance
                    self.log_performance()
                    
                    last_rebalance = current_time
                    
                    # Wait before next check
                    time.sleep(3600)  # Wait 1 hour
                else:
                    # Log current performance
                    self.log_performance()
                    time.sleep(14400)  # Wait 4 hours
                    
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(300)  # Wait 5 minutes before retry

if __name__ == "__main__":
    # Ensure required environment variables are set
    required_env_vars = ['ALPACA_API_KEY', 'ALPACA_SECRET_KEY']
    
    for var in required_env_vars:
        if not os.getenv(var):
            logger.error(f"Environment variable {var} is not set")
            exit(1)
    
    # Initialize and run strategy
    strategy = TradingStrategy()
    strategy.run_strategy()
