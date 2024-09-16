# Crypto Trading Bot

## Overview

This is a Python-based cryptocurrency trading bot designed to automatically trade ERC-20 tokens using strategies based on moving averages. The bot supports swapping tokens on Uniswap V2 and includes features such as stop-loss protection and real-time gas price optimization. It can handle trading ETH against popular stablecoins like USDT, USDC, and DAI.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Setting Up](#setting-up)
- [Explanation of Code Elements](#explanation-of-code-elements)
  - [Main Features](#main-features)
  - [Functions Overview](#functions-overview)
    - [Slippage](#slippage)
    - [Gas Fees](#gas-fees)
    - [ETH Spent](#eth-spent)
- [Using the Bot](#using-the-bot)
- [Environment Variables (.env)](#environment-variables-env)
- [Example .env File](#example-env-file)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/crypto-trading-bot.git
   cd crypto-trading-bot

2. Set up a virtual environment:
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the dependencies:
pip install -r requirements.txt
Dependencies

This project requires several Python libraries:

Web3.py: To interact with the Ethereum blockchain.
Requests: For API requests, including fetching gas prices from Etherscan and token prices.
Pandas: For handling and analyzing the OHLCV data.
TA-Lib: To compute technical indicators like moving averages.
dotenv: For managing environment variables.
You can install all required dependencies by running:

pip install -r requirements.txt

Setting Up

Create a .env file in the project root directory to store sensitive information like API keys and private keys.
Add the following variables to your .env file (see the example below for more details):
ALCHEMY_URL: Your Alchemy API URL for connecting to Ethereum nodes.
PRIVATE_KEY: Your wallet's private key.
WALLET_ADDRESS: The address of your Ethereum wallet.
ETHERSCAN_API_KEY: Your API key for fetching gas prices from Etherscan.
Explanation of Code Elements

Main Features
Automated Trading: The bot automatically buys and sells tokens based on the signals generated from the moving average strategy.
Stop-Loss Protection: The bot includes a stop-loss feature, where it automatically sells tokens if their price drops below a predefined threshold.
Gas Price Optimization: The bot fetches gas prices in real time and adjusts the gas fee for transactions accordingly.
Functions Overview
1. Slippage

Slippage refers to the difference between the expected price of a trade and the actual price when the trade is executed. The bot sets slippage tolerance to 0 in swapExactETHForTokens for simplicity, but in real-world scenarios, it's essential to adjust this to account for price fluctuations.

Location in Code:

transaction = uniswap.functions.swapExactETHForTokens(
    0,  # Slippage tolerance set to 0 (you may want to adjust this)
)

2. Gas Fees

Gas fees are the cost paid to the Ethereum network for executing a transaction. The bot fetches real-time gas prices using the Etherscan API and increases the gas price by 10% for faster execution.

Location in Code:

gas_price = w3.to_wei(get_higher_gas_price(current_gas_price), 'gwei')

3. ETH Spent

The ETH spent on transactions is controlled by the amount set in the eth_amount_in_wei variable. You can modify this amount based on the size of the trades you'd like to execute.

Location in Code:

eth_amount_in_wei = w3.to_wei(0.001, 'ether')  # Set the amount of ETH to swap

Functions Explanation:
swap_eth_to_token: Swaps ETH for a specific ERC-20 token using the Uniswap V2 contract.
swap_token_to_eth: Converts ERC-20 tokens back to ETH.
get_etherscan_gas_price: Fetches the latest gas prices from Etherscan to ensure optimized transaction fees.
get_token_balance: Checks the wallet's token balance for a specific token address.
run_bot: The main function that runs the trading strategy by fetching market data, calculating signals, and executing trades.
Using the Bot

Edit the .env file: Ensure your .env file is configured correctly with the necessary API keys and wallet information.
Run the bot:

python moving_average_bot.py

Log File: The bot logs all trading activities (buy, sell, stop-loss triggers, errors) in a file called trading_bot_log.txt.
Environment Variables (.env)

The .env file is used to store sensitive information like API keys and wallet addresses. This file should never be committed to version control.

Environment Variables:
ALCHEMY_URL: Your Alchemy URL to connect to Ethereum nodes.
PRIVATE_KEY: The private key of your Ethereum wallet (ensure this is kept secure!).
WALLET_ADDRESS: Your Ethereum wallet address.
ETHERSCAN_API_KEY: Your Etherscan API key for fetching gas prices.
Example .env File

ALCHEMY_URL=https://eth-mainnet.alchemyapi.io/v2/your-alchemy-api-key
PRIVATE_KEY=your-private-key
WALLET_ADDRESS=your-wallet-address
ETHERSCAN_API_KEY=your-etherscan-api-key

Feel free to modify the settings to suit your trading strategies and always ensure that your private key is securely stored!


### Notes:

- The `README.md` provides clear instructions for setting up the bot after cloning from a repository, including setting up the environment, running the bot, and understanding the key components of the code.
- It explains the crucial elements like slippage, gas fees, ETH spent, and the purpose of the `.env` file for storing sensitive information.
- Make sure to include all the necessary dependencies in a `requirements.txt` file so users can install them easily after cloning the repository.
