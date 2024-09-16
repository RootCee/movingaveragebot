# Import necessary libraries and modules
import logging
import os
import time
import json
import pandas as pd
import talib
import requests
from urllib3.util.retry import Retry  # Correct import
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv
from web3 import Web3
from flashbots import flashbot
from eth_account import Account  # Flashbots requires eth_account for signing transactions
from eth_account.signers.local import LocalAccount
from colorama import Fore, Style, init

def log_and_print(message, level="info"):
    # Apply different colors for different log levels
    if level == "info":
        logging.info(message)
        print(Fore.GREEN + message)  # Green for info messages
    elif level == "warning":
        logging.warning(message)
        print(Fore.YELLOW + message)  # Yellow for warning messages
    elif level == "error":
        logging.error(message)
        print(Fore.RED + message)  # Red for error messages
    else:
        print(Fore.WHITE + message)  # Default white for other messages

# Modify the section where you log prices, volumes, and other specific details
def log_pair_data(pair, price, liquidity, volume_24h, volume_7d):
    price_str = Fore.CYAN + f"Price: {price}" + Style.RESET_ALL
    liquidity_str = Fore.MAGENTA + f"Liquidity: {liquidity}" + Style.RESET_ALL
    volume_24h_str = Fore.YELLOW + f"24H Volume: {volume_24h}" + Style.RESET_ALL
    volume_7d_str = Fore.BLUE + f"7D Volume: {volume_7d}" + Style.RESET_ALL
    
    log_and_print(f"{pair}: {price_str}, {liquidity_str}, {volume_24h_str}, {volume_7d_str}")


# Load environment variables
load_dotenv()

# Set up logging to a file
logging.basicConfig(filename="trading_bot_log.txt", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", filemode="a")

# Web3 setup for Ethereum network
w3 = Web3(Web3.HTTPProvider(os.getenv('ALCHEMY_URL')))
private_key = os.getenv('PRIVATE_KEY')
wallet_address = os.getenv('WALLET_ADDRESS')
flashbots_key = os.getenv("FLASHBOTS_PRIVATE_KEY")
flashbots_signer = w3.eth.account.from_key(flashbots_key)

# Define the amount of ETH to use for swaps (in wei)
eth_amount_in_wei = w3.to_wei(0.001, 'ether')  # Adjust the value as needed

# Uniswap V3 Router address
uniswap_router_address = Web3.to_checksum_address("0xE592427A0AEce92De3Edee1F18E0157C05861564")

# Load Uniswap ABI from JSON file in the project directory
with open('uniswap_abi_v3.json', 'r') as abi_file:
    uniswap_router_abi = json.load(abi_file)

# Load ERC-20 ABI from JSON file in the project directory
with open('erc20_abi.json', 'r') as erc20_file:
    erc20_abi = json.load(erc20_file)

# Load Uniswap Router Contract using the ABI
uniswap = w3.eth.contract(address=uniswap_router_address, abi=uniswap_router_abi)

# Initialize Flashbots with correct Web3 instance and signer
flashbots_provider = flashbot(w3, flashbots_signer, os.getenv('FLASHBOTS_URL'))

# Uniswap Subgraph API URL
uniswap_api_url = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"


# Cache dictionary to store historical data
historical_data_cache = {}

# Define cache expiration time (e.g., 1 hour)
CACHE_EXPIRATION = 3600  # seconds

# Load cache from file if it exists
if os.path.exists('historical_data_cache.json'):
    with open('historical_data_cache.json', 'r') as cache_file:
        historical_data_cache = json.load(cache_file)

def save_cache_to_file():
    # Save cache to a JSON file
    with open('historical_data_cache.json', 'w') as cache_file:
        json.dump(historical_data_cache, cache_file)


# Create a session with retries and backoff
def create_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


# Example transaction creation
def create_transaction(pair):
    token_pair_address = get_pair_address(pair)
    
    tx = {
        'from': wallet_address,
        'to': token_pair_address,
        'value': w3.to_wei(0.1, 'ether'),
        'gas': 21000,
        'maxFeePerGas': w3.to_wei('20', 'gwei'),
        'maxPriorityFeePerGas': w3.to_wei('2', 'gwei'),
        'nonce': w3.eth.get_transaction_count(wallet_address),
        'chainId': 1
    }
    
    signed_txn = w3.eth.account.sign_transaction(tx, private_key)
    return signed_txn


# Flashbots bundle submission
def send_flashbots_bundle(signed_txn):
    try:
        # Ensure rawTransaction is correctly signed and hex encoded
        if not hasattr(signed_txn, 'rawTransaction'):
            raise ValueError("Signed transaction does not have a rawTransaction attribute.")
        
        bundle = [{
            'tx': signed_txn.rawTransaction.hex()
        }]
        
        current_block = w3.eth.block_number
        target_block_number = current_block + 1  # Target the next block

        # Submit Flashbots bundle with target_block_number
        result = flashbots_provider.flashbots.sendBundle(
    [{"signed_transaction": signed_txn.rawTransaction.hex()}],  # The signed transaction
    target_block_number=target_block_number  # Add target block number
)

        if result and 'bundleHash' in result:
            log_and_print(f"Flashbots bundle sent successfully, bundle hash: {result['bundleHash']}")
            return result['bundleHash']
        else:
            log_and_print("Error submitting Flashbots bundle", "error")
            return None
    except Exception as e:
        log_and_print(f"Error in send_flashbots_bundle: {str(e)}", "error")
        return None
    
   

# Get the token address for a given pair
def get_pair_address(pair):
    pair_addresses = {
        'ETH/PEPE': '0x11950d141EcB863F01007AdD7D1A342041227b58',  # Replace with verified PEPE contract address
        'ETH/BOBO': '0xe945683B3462D2603a18BDfBB19261C6a4f03aD1',  # Replace with verified BOBO contract address
        'ETH/MKR': '0xe8c6c9227491C0a8156A0106A0204d881BB7E531',  # Replace with verified MEME contract address
        'ETH/MINT': '0xF4c5e0F4590b6679B3030d29A84857F226087FeF',  # Replace with verified SHIB contract address
        'ETH/NEIRO': '0x3885fbe4CD8aeD7b7e9625923927Fa1CE30662A3'   # Replace with verified NEIRO contract address
    }
    address = pair_addresses.get(pair)
    if address is None:
        raise ValueError(f"Pair address for {pair} not found.")
    return address

# Example usage of Flashbots bundle for multiple pairs
pairs = ['ETH/PEPE', 'ETH/BOBO', 'ETH/MKR', 'ETH/MINT', 'ETH/NEIRO']  # List of token pairs
   
# Mapping of pairs to CoinGecko token IDs
token_id_mapping = {
    'ETH/PEPE': 'pepe',      # CoinGecko ID for PEPE
    'ETH/BOBO': 'bobo',      # CoinGecko ID for BOBO
    'ETH/MKR': 'mkr',  # Correct CoinGecko ID for MEME
    'ETH/MINT': 'mint',  # Correct CoinGecko ID for SHIB
    'ETH/NEIRO': 'neiro-on-eth'  # Correct CoinGecko ID for NEIRO
}

# Function to find and return the CoinGecko ID for a specific token if missing from the mapping
def get_coingecko_id(token_name, session):
    url = 'https://api.coingecko.com/api/v3/coins/list'
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()
        token_list = response.json()
        for token in token_list:
            if token_name.lower() in token['id'].lower() or token_name.lower() == token['symbol'].lower():
                log_and_print(f"Found CoinGecko ID for {token_name}: {token['id']}")
                return token['id']
        log_and_print(f"CoinGecko ID for {token_name} not found.", "error")
        return None
    except requests.exceptions.RequestException as e:
        log_and_print(f"Error fetching CoinGecko token list: {e}", "error")
        return None


# Fetch data from Dexscreener API with session and retry logic
def fetch_dexscreener_data(pair, session):
    token_address = get_pair_address(pair)
    url = f'https://api.dexscreener.com/latest/dex/pairs/ethereum/{token_address}'
    
    try:
        response = session.get(url, timeout=10)  # Use session with retry mechanism
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        data = response.json()

        if data and isinstance(data, dict) and 'pairs' in data and len(data['pairs']) > 0:
            pair_data = data['pairs'][0]
            price_usd = pair_data.get('priceUsd')
            liquidity_usd = pair_data.get('liquidity', {}).get('usd')

            # Fetching 24H and 7-day volume safely, return None if missing
            volume_24h = pair_data.get('volume', {}).get('usd24h')
            volume_7d = pair_data.get('volume', {}).get('usd7d')

            # Log the information in a colored output
            price_str = Fore.CYAN + f"Price: {price_usd}" + Style.RESET_ALL
            liquidity_str = Fore.MAGENTA + f"Liquidity: {liquidity_usd}" + Style.RESET_ALL
            volume_24h_str = Fore.YELLOW + f"24H Volume: {volume_24h if volume_24h else 'Missing'}" + Style.RESET_ALL
            volume_7d_str = Fore.BLUE + f"7D Volume: {volume_7d if volume_7d else 'Missing'}" + Style.RESET_ALL

            log_and_print(f"{pair}: {price_str}, {liquidity_str}, {volume_24h_str}, {volume_7d_str}")

            # Return the values
            return {
                'price': price_usd,
                'liquidity': liquidity_usd,
                'volume_24h': volume_24h,
                'volume_7d': volume_7d
            }
        else:
            log_and_print(f"No data found for token address {token_address}.", "error")
            return None

    except requests.exceptions.RequestException as e:
        log_and_print(f"Error fetching data from DexScreener: {e}", "error")
        return None



# Function to fetch data from Uniswap Subgraph
def fetch_uniswap_data(pair_address):
    query = f"""
    {{
      pool(id: "{pair_address}") {{
        token0 {{
          symbol
        }}
        token1 {{
          symbol
        }}
        liquidity
        volumeUSD
        token0Price
        token1Price
        dayData(first: 7) {{
          date
          volumeUSD
        }}
      }}
    }}
    """
    response = requests.post('https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3', json={'query': query})
    if response.status_code == 200:
        data = response.json()['data']['pool']
        return {
            'pair': f"{data['token0']['symbol']}/{data['token1']['symbol']}",
            'liquidity': data['liquidity'],
            'volume_24h': data['volumeUSD'],
            'token0Price': data['token0Price'],
            'token1Price': data['token1Price']
        }
    else:
        print("Error fetching data from Uniswap Subgraph")
        return None
    


# Fetch data from SushiSwap using The Graph API
def fetch_sushiswap_data(pair_address):
    query = f"""
    {{
      pair(id: "{pair_address}") {{
        token0 {{
          symbol
        }}
        token1 {{
          symbol
        }}
        reserveUSD
        volumeUSD
      }}
    }}
    """
    response = requests.post('https://api.thegraph.com/subgraphs/name/sushiswap/exchange', json={'query': query})
    if response.status_code == 200:
        data = response.json()['data']['pair']
        return {
            'pair': f"{data['token0']['symbol']}/{data['token1']['symbol']}",
            'liquidity': data['reserveUSD'],
            'volume_24h': data['volumeUSD']
        }
    else:
        print("Error fetching data from SushiSwap")
        return None


# Fetch data from PancakeSwap using The Graph API
def fetch_data_from_dex(pair, dex):
    if dex == 'uniswap':
        return fetch_uniswap_data(pair)
    elif dex == 'sushiswap':
        return fetch_sushiswap_data(pair)
    #elif dex == 'pancakeswap':
        #return fetch_pancakeswap_data(pair)
    else:
        return None

# Fetch data from DeFi Llama API
def fetch_defillama_data(session):
    url = "https://api.llama.fi/dexVolumes"
    
    try:
        response = session.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()

        # Example: Extract relevant data from DeFi Llama response
        # Here, you'll need to filter or find the token pair you're interested in
        # Adjust based on your desired pair and how DeFi Llama structures the data
        dex_data = data.get("protocols", [])
        
        for dex in dex_data:
            if dex["name"].lower() == "uniswap":  # You can adjust this to target a specific DEX
                volume_24h = dex.get("volume_24h")
                volume_7d = dex.get("volume_7d")
                return {
                    "volume_24h": volume_24h,
                    "volume_7d": volume_7d
                }
        return None

    except requests.exceptions.RequestException as e:
        log_and_print(f"Error fetching data from DeFi Llama: {e}", "error")
        return None


# Fetch historical data with retry and dynamic token ID fetching
def fetch_historical_data_with_cache(token_name, session):
    current_time = time.time()

    # Dynamically fetch token ID if not found in mapping
    token_id = token_id_mapping.get(token_name)
    if not token_id:
        token_id = get_coingecko_id(token_name, session)
        if token_id:
            token_id_mapping[token_name] = token_id  # Save the ID to the mapping for future use
        else:
            return None

    # Check if data is in cache and not expired
    if token_id in historical_data_cache:
        cached_data, timestamp = historical_data_cache[token_id]
        if current_time - timestamp < CACHE_EXPIRATION:
            df = pd.DataFrame(cached_data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df

    # Fetch new historical data from CoinGecko if not in cache or expired
    url = f'https://api.coingecko.com/api/v3/coins/{token_id}/market_chart'
    params = {'vs_currency': 'usd', 'days': '30', 'interval': 'daily'}
    try:
        response = session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'prices' in data:
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            historical_data_cache[token_id] = (data, current_time)
            save_cache_to_file()
            return df
        else:
            log_and_print(f"Malformed historical data for {token_name}", "error")
            return None
    except requests.exceptions.RequestException as e:
        log_and_print(f"Error fetching historical data for {token_name}: {e}", "error")
        return None

# Calculate moving averages (SMA/FMA) and RSI
def calculate_indicators(df):
    # Ensure the price column is converted to numeric format
    df['price'] = pd.to_numeric(df['price'], errors='coerce')

    # Drop rows with NaN values before calculating SMA/FMA
    df = df.dropna(subset=['price'])

    # Calculate moving averages
    df['SMA'] = talib.SMA(df['price'].values, timeperiod=21)  # Slow Moving Average
    df['FMA'] = talib.SMA(df['price'].values, timeperiod=7)   # Fast Moving Average

    # Calculate RSI
    df['RSI'] = talib.RSI(df['price'].values, timeperiod=9)  # RSI with a 14-period window

    return df

# Check buy/sell signals based on moving averages and RSI
def check_signals(df):
    if len(df) < 2:
        return 'hold'  # Not enough data

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    # Define signal conditions
    buy_signal = prev_row['FMA'] <= prev_row['SMA'] and last_row['FMA'] > last_row['SMA']
    rsi_buy_condition = last_row['RSI'] < 30  # RSI oversold

    sell_signal = prev_row['FMA'] >= prev_row['SMA'] and last_row['FMA'] < last_row['SMA']
    rsi_sell_condition = last_row['RSI'] > 70  # RSI overbought

    # Colors for buy/sell signals
    fma_str = Fore.LIGHTCYAN_EX + f"FMA = {last_row['FMA']}" + Style.RESET_ALL
    sma_str = Fore.LIGHTMAGENTA_EX + f"SMA = {last_row['SMA']}" + Style.RESET_ALL
    rsi_str = Fore.LIGHTYELLOW_EX + f"RSI = {last_row['RSI']}" + Style.RESET_ALL

    if buy_signal and rsi_buy_condition:
        log_and_print(f"{Fore.GREEN}Buy signal: {fma_str} crossed above {sma_str} and {rsi_str} is oversold{Style.RESET_ALL}")
        return 'buy'
    elif sell_signal and rsi_sell_condition:
        log_and_print(f"{Fore.RED}Sell signal: {fma_str} crossed below {sma_str} and {rsi_str} is overbought{Style.RESET_ALL}")
        return 'sell'
    else:
        # Use distinct colors for the hold signal
        fma_str_hold = Fore.LIGHTBLUE_EX + f"FMA = {last_row['FMA']}" + Style.RESET_ALL
        sma_str_hold = Fore.LIGHTGREEN_EX + f"SMA = {last_row['SMA']}" + Style.RESET_ALL
        rsi_str_hold = Fore.LIGHTRED_EX + f"RSI = {last_row['RSI']}" + Style.RESET_ALL

        log_and_print(f"{Fore.LIGHTWHITE_EX}Hold signal: {fma_str_hold}, {sma_str_hold}, {rsi_str_hold}{Style.RESET_ALL}")
        return 'hold'

# Function to log the pair information in a different color
def log_pair_info(pair, current_price, volume_24h, volume_7d):
    pair_info_str = Fore.LIGHTMAGENTA_EX + f"Pair: {pair}, Price: {current_price}, 24H Volume: {volume_24h}, 7D Volume: {volume_7d}" + Style.RESET_ALL
    log_and_print(pair_info_str)


# Get the balance of a token in the wallet
def get_token_balance(wallet_address, token_address):
    token_contract = w3.eth.contract(address=token_address, abi=erc20_abi)
    balance = token_contract.functions.balanceOf(wallet_address).call()
    return balance

# Check if enough ETH balance is available for the transaction and gas
def check_balance_for_gas_and_value(gas_price_gwei, eth_amount_in_wei):
    try:
        # Convert wallet address to checksum format to avoid issues
        wallet_address_checksum = Web3.to_checksum_address(wallet_address)
        eth_balance_wei = w3.eth.get_balance(wallet_address_checksum)
        gas_limit = 2000000  # Estimated gas limit for the swap transaction
        
        # Calculate total gas cost in wei
        gas_cost_in_wei = w3.to_wei(gas_price_gwei * gas_limit, 'gwei')
        
        # Total ETH required: transaction amount + gas cost
        total_required_wei = eth_amount_in_wei + gas_cost_in_wei
        
        # Check if the wallet has enough ETH for the transaction and gas
        if eth_balance_wei >= total_required_wei:
            return True
        else:
            # Calculate shortfall and log an error message
            shortfall_wei = total_required_wei - eth_balance_wei
            shortfall_eth = w3.from_wei(shortfall_wei, 'ether')
            log_and_print(f"Insufficient funds: Wallet balance is {w3.from_wei(eth_balance_wei, 'ether')} ETH, "
                          f"but {w3.from_wei(total_required_wei, 'ether')} ETH is required for gas and value.")
            log_and_print(f"You are short of {shortfall_eth} ETH.", "error")
            return False
    except Exception as e:
        log_and_print(f"Error checking balance: {e}", "error")
        return False

def estimate_gas(token_address, eth_amount_in_wei, fee_tier):
    try:
        # Build the transaction for gas estimation using the correct parameters for Uniswap V3 exactInputSingle
        transaction = uniswap.functions.exactInputSingle({
            'tokenIn': Web3.to_checksum_address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"),  # ETH address
            'tokenOut': token_address,
            'fee': fee_tier,  # This will now dynamically take the fee for the correct token pair
            'recipient': wallet_address,
            'deadline': int(time.time()) + 60,  # 60-second deadline for the transaction
            'amountIn': eth_amount_in_wei,
            'amountOutMinimum': 0,  # 0 for estimation purposes
            'sqrtPriceLimitX96': 0  # No price limit, estimate assuming best price
        }).build_transaction({
            'from': wallet_address,
            'value': eth_amount_in_wei,  # Sending ETH in the transaction
            'gas': 200000,  # A reasonable default gas limit to help estimation
            'gasPrice': w3.eth.gas_price  # Fetch the current gas price
        })

        # Estimate the gas for the built transaction
        estimated_gas = w3.eth.estimate_gas(transaction)
        return estimated_gas
    except Exception as e:
        log_and_print(f"Error estimating gas: {e}", "error")
        # Set a fallback gas limit if the estimation fails (e.g., 2000000)
        return 2000000

# Swap ETH to token using Flashbots
def swap_eth_to_token_flashbots(token_address, eth_amount_in_wei, slippage_tolerance, fee_tier, retries=3, delay=10):
    for attempt in range(retries):
        try:
            # Fetch gas price and ensure it is a number
            gas_price = float(get_etherscan_gas_price())  # Convert to float if needed

            if not check_balance_for_gas_and_value(gas_price, eth_amount_in_wei):
                log_and_print("Not enough ETH for transaction and gas.")
                return None

            # Ensure slippage tolerance is a float or int
            if not isinstance(slippage_tolerance, (int, float)):
                log_and_print(f"Invalid slippage tolerance type: {type(slippage_tolerance)}", "error")
                return None

            # Fetch expected output amount
            expected_output_amount = 1000  # Example value, ensure this is a number

            # Adjust slippage, ensure amount_out_minimum is a number
            amount_out_minimum = int(expected_output_amount * (1 - slippage_tolerance))
            log_and_print(f"Amount out minimum: {amount_out_minimum}")

            # Estimate gas for the transaction
            estimated_gas = estimate_gas(token_address, eth_amount_in_wei, fee_tier)  # Ensure fee_tier is passed

            # Build transaction
            transaction = uniswap.functions.exactInputSingle({
                'tokenIn': Web3.to_checksum_address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"),  # ETH address
                'tokenOut': token_address,
                'fee': fee_tier,
                'recipient': wallet_address,
                'deadline': int(time.time()) + 60,  # 60-second deadline
                'amountIn': eth_amount_in_wei,
                'amountOutMinimum': amount_out_minimum,  # Ensure it's an integer
                'sqrtPriceLimitX96': 0  # No price limit
            }).build_transaction({
                'from': wallet_address,
                'value': eth_amount_in_wei,
                'gas': estimated_gas,  # Use the estimated gas
                'gasPrice': w3.to_wei(float(gas_price), 'gwei'),  # Ensure gasPrice is a float/int
                'nonce': w3.eth.get_transaction_count(wallet_address, 'pending')
            })

            # Sign the transaction but don't log success yet
            signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)

            # Submit the signed transaction to Flashbots
            current_block = w3.eth.block_number
            target_block_number = current_block + 1

            result = flashbots_provider.flashbots.sendBundle(
                [{"signed_transaction": signed_txn.rawTransaction.hex()}],
                target_block_number
            )

            if 'bundleHash' in result:
                # Wait for the transaction to be mined and confirmed before logging success
                log_and_print(f"Transaction sent via Flashbots, waiting for confirmation (bundle hash: {result['bundleHash']})")
                wait_for_flashbots_confirmation(result['bundleHash'])
                log_and_print(f"Transaction confirmed on chain, bundle hash: {result['bundleHash']}")
                return result['bundleHash']
            else:
                log_and_print(f"Flashbots error: {result}", "error")
                return None
        
        except Exception as e:
            log_and_print(f"Attempt {attempt + 1} failed: {str(e)}", "error")
            time.sleep(delay)  # Wait before retrying

    log_and_print(f"Failed to execute transaction after {retries} attempts.", "error")
    return None


# Swap token to ETH using Flashbots
def swap_token_to_eth_flashbots(token_address, token_amount, slippage_tolerance, retries=3, delay=10):
    for attempt in range(retries):
        try:
            gas_price = get_etherscan_gas_price()
            if not check_balance_for_gas_and_value(gas_price, 0):
                log_and_print("Not enough ETH for transaction and gas.")
                return None

            # Estimate gas for the transaction
            estimated_gas = estimate_gas(token_address, token_amount)

            # Desired ETH output (adjust according to your trade logic)
            desired_eth_output = 0.5 * (10**18)  # Example value, replace with your logic

            transaction = uniswap.functions.exactOutputSingle({
                'tokenIn': token_address,
                'tokenOut': Web3.to_checksum_address("0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"),  # ETH address
                'fee': 3000,  # Pool fee (adjust as needed, 3000 = 0.3%)
                'recipient': wallet_address,
                'deadline': int(time.time()) + 60,
                'amountOut': int(desired_eth_output),  # Desired ETH output
                'amountInMaximum': token_amount,  # Maximum token amount for the swap
                'sqrtPriceLimitX96': 0  # No price limit
            }).build_transaction({
                'from': wallet_address,
                'value': 0,
                'gas': estimated_gas,  # Estimated gas
                'gasPrice': w3.to_wei(gas_price, 'gwei'),
                'nonce': w3.eth.get_transaction_count(wallet_address, 'pending')
            })

            signed_txn = w3.eth.account.sign_transaction(transaction, private_key=private_key)
            current_block = w3.eth.block_number
            target_block_number = current_block + 1

            result = flashbots_provider.flashbots.sendBundle(
                [{"signed_transaction": signed_txn.rawTransaction.hex()}],
                target_block_number
            )

            if 'bundleHash' in result:
                # Now log after confirmation
                log_and_print(f"Transaction sent via Flashbots, waiting for confirmation (bundle hash: {result['bundleHash']})")
                wait_for_flashbots_confirmation(result['bundleHash'])
                log_and_print(f"Transaction confirmed on chain, bundle hash: {result['bundleHash']}")
                return result['bundleHash']
            else:
                log_and_print(f"Flashbots error: {result}", "error")
                return None
        
        except Exception as e:
            log_and_print(f"Attempt {attempt + 1} failed: {str(e)}", "error")
            time.sleep(delay)  # Wait before retrying

    log_and_print(f"Failed to execute transaction after {retries} attempts.", "error")
    return None


# Main bot function to run the trading bot
def run_bot(pairs, eth_amount_in_wei, stop_loss_percentage, slippage_tolerance=0.01, fee_tier_mapping=None):
    session = create_session()  # Initialize session with retries
    token_purchase_prices = {}
    tokens_bought = {}
    price_data = {pair: pd.DataFrame(columns=['price']) for pair in pairs}

    if fee_tier_mapping is None:
        fee_tier_mapping = {}  # Provide a default empty dictionary if none is provided
    
    while True:
        for pair in pairs:
            try:
                # Log the processing pair in a distinct color
                log_and_print(f"{Fore.LIGHTYELLOW_EX}Processing transaction for pair: {pair}{Style.RESET_ALL}")

                # Fetch real-time data from Uniswap Subgraph
                token_address = get_pair_address(pair)
                
                # Fetch real-time data from DexScreener
                real_time_data = fetch_dexscreener_data(pair, session)
                
                # If real-time data is available, log the prices and volumes
                if real_time_data:
                    current_price = float(real_time_data['price'])
                    volume_24h = real_time_data['volume_24h']
                    volume_7d = real_time_data['volume_7d']
                    
                    # Log pair info in a distinct color
                    log_and_print(
                        f"{Fore.LIGHTMAGENTA_EX}Price: {current_price}, 24H Volume: {volume_24h}, 7D Volume: {volume_7d}{Style.RESET_ALL}"
                    )
                
                # Token ID mapping and historical data fetching
                token_id = token_id_mapping.get(pair)
                if not token_id:
                    log_and_print(f"{Fore.RED}Token ID not found for pair {pair}.{Style.RESET_ALL}", "error")
                    continue
                
                # Fetch historical data
                historical_data = fetch_historical_data_with_cache(token_id, session)
                if historical_data is None or historical_data.empty:
                    log_and_print(f"{Fore.RED}Not enough historical data for {pair}.{Style.RESET_ALL}")
                    continue

                # Calculate indicators (SMA, RSI) and get buy/sell signals
                new_row = pd.DataFrame({'timestamp': [pd.Timestamp.now()], 'price': [current_price]})
                historical_data = pd.concat([historical_data, new_row], ignore_index=True)
                historical_data = calculate_indicators(historical_data)
                signal = check_signals(historical_data)

                # Display the signal in its own distinct color
                log_and_print(f"{Fore.LIGHTCYAN_EX}Signal for {pair}: {signal}, Current Price: {current_price}{Style.RESET_ALL}")

                # Handle buy/sell logic based on signal
                fee_tier = fee_tier_mapping.get(pair, 3000)
                
                if signal == 'buy' and not tokens_bought.get(pair):
                    tx = swap_eth_to_token_flashbots(swap_eth_to_token_flashbots, token_address, eth_amount_in_wei, slippage_tolerance, fee_tier)
                    if tx:
                        token_purchase_prices[pair] = current_price
                        tokens_bought[pair] = True
                        log_and_print(f"{Fore.GREEN}Bought {pair} at {current_price}{Style.RESET_ALL}")
                    else:
                        log_and_print(f"{Fore.RED}Failed to execute buy order for {pair}{Style.RESET_ALL}", "error")

                elif signal == 'sell' and tokens_bought.get(pair):
                    token_balance = get_token_balance(wallet_address, token_address)
                    if token_balance > 0:
                        tx = swap_token_to_eth_flashbots(swap_token_to_eth_flashbots, token_address, token_balance, slippage_tolerance)
                        if tx:
                            log_and_print(f"{Fore.GREEN}Sold {pair}{Style.RESET_ALL}")
                            tokens_bought[pair] = False
                        else:
                            log_and_print(f"{Fore.RED}Failed to execute sell order for {pair}{Style.RESET_ALL}", "error")

                # Stop-loss logic
                if pair in token_purchase_prices:
                    purchase_price = token_purchase_prices[pair]
                    stop_loss_price = purchase_price * (1 - stop_loss_percentage / 100)
                    if current_price <= stop_loss_price:
                        token_balance = get_token_balance(wallet_address, token_address)
                        if token_balance > 0:
                            tx = swap_eth_to_token_flashbots(swap_token_to_eth_flashbots, token_address, token_balance, slippage_tolerance)
                            if tx:
                                log_and_print(f"{Fore.RED}Stop-loss triggered for {pair}. Sold at {current_price}.{Style.RESET_ALL}")
                                tokens_bought[pair] = False
                            else:
                                log_and_print(f"{Fore.RED}Failed to execute stop-loss sell for {pair}{Style.RESET_ALL}", "error")

            except Exception as e:
                log_and_print(f"{Fore.RED}Error processing {pair}: {str(e)}{Style.RESET_ALL}", "error")

        # Dynamic sleep time based on price movement volatility
        previous_price = price_data[pair]['price'].iloc[-1] if not price_data[pair].empty else current_price
        sleep_time = 60  # Default 60 seconds
        if abs(current_price - previous_price) / previous_price > 0.02:  # If price moves more than 2%
            sleep_time = 30  # Reduce sleep time to 30 seconds
        time.sleep(sleep_time)


 
                       
# Helper to wait for Flashbots transaction confirmation
def wait_for_flashbots_confirmation(tx_hash, required_confirmations=6, timeout=600, poll_interval=10):
    """
    Waits for the Flashbots transaction to be confirmed with the required number of confirmations.
    
    Args:
        tx_hash (str): Transaction hash.
        required_confirmations (int): Number of confirmations required for the transaction.
        timeout (int): Maximum time to wait (in seconds) for the transaction to be confirmed.
        poll_interval (int): Time interval (in seconds) between each check.
        
    Returns:
        dict or None: Returns the transaction receipt if confirmed, otherwise returns None.
    """
    try:
        start_time = time.time()  # Record the starting time
        while time.time() - start_time < timeout:  # Check until the timeout is reached
            receipt = w3.eth.get_transaction_receipt(tx_hash)
            
            if receipt:
                # Get confirmation count or set to 0 if unavailable
                confirmations = w3.eth.block_number - receipt['blockNumber'] if receipt['blockNumber'] else 0

                if confirmations >= required_confirmations:
                    log_and_print(f"Transaction {tx_hash} confirmed with {confirmations} confirmations!")
                    return receipt
                else:
                    log_and_print(f"Transaction {tx_hash} has {confirmations} confirmations. Waiting for {required_confirmations}.")
            else:
                log_and_print(f"Transaction {tx_hash} not yet mined.")
            
            time.sleep(poll_interval)  # Wait before checking again
        
        log_and_print(f"Timeout: Transaction {tx_hash} not confirmed after {timeout} seconds.", "error")
        return None  # Timeout reached, transaction not confirmed

    except Exception as e:
        log_and_print(f"Error while waiting for confirmations: {str(e)}", "error")
        return None

 # Fetch gas prices from Etherscan API with retry mechanism
def get_etherscan_gas_price():
    session = create_session()
    try:
        api_key = os.getenv('ETHERSCAN_API_KEY')
        url = f'https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={api_key}'
        response = session.get(url, timeout=10)
        gas_data = response.json()

        if gas_data.get("status") == "1":
            # Convert gas prices to float
            low_gas_price = float(gas_data['result']['SafeGasPrice'])  # Convert to float
            avg_gas_price = float(gas_data['result']['ProposeGasPrice'])  # Convert to float
            high_gas_price = float(gas_data['result']['FastGasPrice'])  # Convert to float

            log_and_print(f"Fetched gas prices: Low: {low_gas_price} Gwei, Avg: {avg_gas_price} Gwei, High: {high_gas_price} Gwei")
            return avg_gas_price  # Return avg_gas_price as a float
        else:
            log_and_print(f"Error fetching gas prices: {gas_data['result']}", "error")
            return 20  # Fallback to 20 Gwei in case of error
    except requests.exceptions.RequestException as e:
        log_and_print(f"Error fetching gas prices: {e}", "error")
        return 20  # Fallback value in case of error

# Example usage
pairs = ['ETH/PEPE', 'ETH/BOBO', 'ETH/MKR', 'ETH/MINT', 'ETH/NEIRO']
eth_amount_in_wei = w3.to_wei(0.001, 'ether')
slippage_tolerance = 0.01  # Example: 1% slippage tolerance
fee_tier_mapping = {
    'ETH/PEPE': 3000,  # Uniswap fee tier in basis points (0.3%)
    'ETH/BOBO': 10000,  # Uniswap fee tier in basis points (1%)
    'ETH/MKR': 500,  # Uniswap fee tier in basis points (0.05%)
    'ETH/MINT': 3000,  # Uniswap fee tier in basis points (0.3%)
    'ETH/NEIRO': 3000  # Uniswap fee tier in basis points (0.3%)
}

# Run the bot with a 5% stop-loss
run_bot(pairs, eth_amount_in_wei, stop_loss_percentage=5, slippage_tolerance=0.01, fee_tier_mapping=fee_tier_mapping)

