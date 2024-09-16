import logging
import os
import time
import json
import pandas as pd
import talib
import requests
from dotenv import load_dotenv
from web3 import Web3

# Load environment variables
load_dotenv()

# Set up logging to a file
logging.basicConfig(filename="trading_bot_log.txt", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", filemode="a")

# Web3 setup for Ethereum network
w3 = Web3(Web3.HTTPProvider(os.getenv('ALCHEMY_URL')))
private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS')

# Uniswap V3 Router address (replace if needed)
uniswap_router_address = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")

# Load Uniswap ABI from JSON file in the project directory
with open('uniswap_abi_v3.json', 'r') as abi_file:
    uniswap_router_abi = json.load(abi_file)

# Load ERC-20 ABI from JSON file in the project directory
with open('erc20_abi.json', 'r') as erc20_file:
    erc20_abi = json.load(erc20_file)

# Load Uniswap Router Contract using the ABI
uniswap = w3.eth.contract(address=uniswap_router_address, abi=uniswap_router_abi)

# Log and print message helper
def log_and_print(message, level="info"):
    if level == "info":
        logging.info(message)
    elif level == "error":
        logging.error(message)
    print(message)

# Fetch OHLCV data (replace with your preferred API)
def fetch_data(symbol):
    token_ids = {
        'ETH/PEPE': 'pepe',
        'ETH/BOBO': 'bobo',
        'ETH/MEME': 'memecoin',
        'ETH/SHIB': 'shiba-inu',
        'ETH/NEIRO': 'neiro-on-eth'
    }
    token_id = token_ids.get(symbol)
    
    url = f"https://api.coingecko.com/api/v3/coins/{token_id}/market_chart"
    params = {'vs_currency': 'usd', 'days': '30', 'interval': 'daily'}
    response = requests.get(url, params=params)
    data = response.json()
    
    if 'prices' in data:
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    else:
        log_and_print(f"Error fetching data for {symbol}: {data}")
        return None

# Calculate moving averages (SMA/FMA)
def calculate_moving_averages(df):
    df['SMA'] = talib.SMA(df['close'], timeperiod=20)  # Slow Moving Average (20 periods)
    df['FMA'] = talib.SMA(df['close'], timeperiod=5)   # Fast Moving Average (5 periods)
    return df

# Check buy/sell signals based on moving averages
def check_signals(df):
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    if prev_row['FMA'] <= prev_row['SMA'] and last_row['FMA'] > last_row['SMA']:
        return 'buy'
    elif prev_row['FMA'] >= prev_row['SMA'] and last_row['FMA'] < last_row['SMA']:
        return 'sell'
    return 'hold'

# Get the balance of a token in the wallet
def get_token_balance(wallet_address, token_address):
    token_contract = w3.eth.contract(address=token_address, abi=erc20_abi)
    balance = token_contract.functions.balanceOf(wallet_address).call()
    return balance

# Get the token address for a given pair
def get_token_address(pair):
    token_addresses = {
        'ETH/PEPU': '0x6982508145454ce325ddbe47a25d4ec3d2311933',  # PEPE
        'ETH/STARS': '0xb90b2a35c65dbc466b04240097ca756ad2005295',  # BOBO
        'ETH/FLOCK': '0xb131f4a55907b10d1f0a50d8ab8fa09ec342cd74',  # MEME
        'ETH/SHIB': '0x95ad61b0a150d79219dcf64e1e6cc01f0b64c4ce',  # SHIB
        'ETH/SPONGEV2': '0xee2a03aa6dacf51c18679c516ad5283d8e7c2637'  # NEIRO
    }
    address = token_addresses.get(pair)
    if address is None:
        raise ValueError(f"Token address for pair {pair} not found.")
    return Web3.to_checksum_address(address)

# Swap ETH for a token using Uniswap V3's exactInputSingle function
def swap_eth_to_token(token_address, eth_amount_in_wei):
    try:
        # Define the function to get the gas price from Etherscan
        def get_etherscan_gas_price():
            # Your implementation here
            pass

        def check_balance_for_gas_and_value(gas_price, eth_amount_in_wei):
            # Your implementation here
            pass

        gas_price = get_etherscan_gas_price()
        
        # Check if we have enough ETH for the transaction and gas
        if not check_balance_for_gas_and_value(gas_price, eth_amount_in_wei):
            log_and_print("Not enough ETH for transaction and gas.")
            return None

        deadline = int(time.time()) + 60  # Transaction deadline in seconds

        # Use Uniswap V3's exactInputSingle for ETH to token swap
        transaction = uniswap.functions.exactInputSingle({
            'tokenIn': Web3.to_checksum_address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"),  # ETH address
            'tokenOut': token_address,
            'fee': 3000,  # Pool fee (adjust as needed, 3000 = 0.3%)
            'recipient': wallet_address,
            'deadline': deadline,
            'amountIn': eth_amount_in_wei,
            'amountOutMinimum': 0,  # No slippage control here
            'sqrtPriceLimitX96': 0  # No price limit
        }).build_transaction({
            'from': wallet_address,
            'value': eth_amount_in_wei,
            'gas': 2000000,
            'gasPrice': w3.to_wei(gas_price, 'gwei'),
            'nonce': w3.eth.get_transaction_count(wallet_address, 'pending')
        })

        signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        log_and_print(f"Swapped ETH for tokens, transaction hash: {tx_hash.hex()}")
        return tx_hash

    except Exception as e:
        log_and_print(f"Error swapping ETH to token: {e}", "error")
        return None

# Swap a token for ETH using Uniswap V3's exactOutputSingle function
def swap_token_to_eth(token_address, token_amount):
    try:
        # Define the function to get the gas price from Etherscan
        def get_etherscan_gas_price():
            # Your implementation here
            pass

        def check_balance_for_gas(gas_price):
            # Your implementation here
            pass

        gas_price = get_etherscan_gas_price()

        # Check if we have enough tokens for the transaction
        if not check_balance_for_gas(gas_price):
            log_and_print("Not enough tokens for transaction.")
            return None

        deadline = int(time.time()) + 60  # Transaction deadline in seconds

        # Use Uniswap V3's exactOutputSingle for token to ETH swap
        transaction = uniswap.functions.exactOutputSingle({
            'tokenIn': token_address,
            'tokenOut': Web3.to_checksum_address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"),  # ETH address
            'fee': 3000,  # Pool fee (adjust as needed, 3000 = 0.3%)
            'recipient': wallet_address,
            'deadline': deadline,
            'amountOut': 0,  # No slippage control here
            'amountInMaximum': token_amount,
            'sqrtPriceLimitX96': 0  # No price limit
        }).build_transaction({
            'from': wallet_address,
            'gas': 2000000,
            'gasPrice': w3.to_wei(gas_price, 'gwei'),
            'nonce': w3.eth.get_transaction_count(wallet_address, 'pending')
        })

        signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)
        log_and_print(f"Swapped tokens for ETH, transaction hash: {tx_hash.hex()}")
        return tx_hash

    except Exception as e:
        log_and_print(f"Error swapping token to ETH: {e}", "error")
        return None

# Prevent repeated buys by ensuring a buy is followed by a sell
def run_bot(pairs, eth_amount_in_wei, stop_loss_percentage):
    token_purchase_prices = {}
    tokens_bought = {}  # Track whether a token has already been bought

    while True:
        for pair in pairs:
            df = fetch_data(pair)
            if df is not None:
                df = calculate_moving_averages(df)
                signal = check_signals(df)
                current_price = df['close'].iloc[-1]

                log_and_print(f"Signal for {pair}: {signal}, Current Price: {current_price}")

                # Handle buy signal
                if signal == 'buy' and not tokens_bought.get(pair):
                    tx = swap_eth_to_token(get_token_address(pair), eth_amount_in_wei)
                    if tx:
                        token_purchase_prices[pair] = current_price
                        tokens_bought[pair] = True
                        log_and_print(f"Bought {pair} at {current_price}")
                    else:
                        log_and_print(f"Failed to execute buy order for {pair}", "error")

                # Handle sell signal
                elif signal == 'sell' and tokens_bought.get(pair):
                    token_balance = get_token_balance(wallet_address, get_token_address(pair))
                    if token_balance > 0:
                        tx = swap_token_to_eth(get_token_address(pair), token_balance)
                        if tx:
                            log_and_print(f"Sold {pair}")
                            tokens_bought[pair] = False
                        else:
                            log_and_print(f"Failed to execute sell order for {pair}", "error")

                # Stop-loss trigger
                if pair in token_purchase_prices:
                    purchase_price = token_purchase_prices[pair]
                    stop_loss_price = purchase_price * (1 - stop_loss_percentage / 100)

                    if current_price <= stop_loss_price:
                        token_balance = get_token_balance(wallet_address, get_token_address(pair))
                        if token_balance > 0:
                            tx = swap_token_to_eth(get_token_address(pair), token_balance)
                            if tx:
                                log_and_print(f"Stop-loss triggered for {pair}. Sold at {current_price}.")
                                tokens_bought[pair] = False
                                del token_purchase_prices[pair]
                            else:
                                log_and_print(f"Failed to execute stop-loss sell for {pair}", "error")

        time.sleep(60)  # Check every minute

# Example usage with ETH-meme coin pairs
pairs = ['ETH/PEPE', 'ETH/BOBO', 'ETH/MEME', 'ETH/SHIB', 'ETH/NEIRO']
eth_amount_in_wei = w3.to_wei(0.001, 'ether')  # Amount of ETH to swap

# Run the bot with a 5% stop-loss
run_bot(pairs, eth_amount_in_wei, stop_loss_percentage=5)