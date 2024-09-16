import logging
from crewai import Agent, Task, Crew, Process
from llm_selector import LLMSelector
from tools.coingecko_search_tool import CoinGeckoSearchTool
from tools.chart_tool import chart_tool
from tools.sentiment_analysis_tool import SentimentAnalysisTool
from tools.sentiment_analysis_tool import sentiment_analysis_tool

# Initialize the sentiment analysis tool
sentiment_analysis_tool = SentimentAnalysisTool()
import pandas as pd
import talib

# Hardcoded list of ERC-20 tokens and their contract addresses
erc20_tokens = {
    'USDT': '0xdac17f958d2ee523a2206206994597c13d831ec7',  # Tether USD
    'LINK': '0x514910771af9ca656af840dff83e8264ecf986ca',  # Chainlink
    'UNI': '0x1f9840a85d5af5bf1d1762f925bdaddc4201f984',  # Uniswap
    'DAI': '0x6b175474e89094c44da98b954eedeac495271d0f',  # Dai Stablecoin
    'AAVE': '0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9'   # Aave
}


# Function to calculate technical indicators
def calculate_indicators(price_data):
    # Example price data (replace with real market data)
    df = pd.DataFrame(price_data)

    # Calculate indicators
    df['FMA'] = talib.EMA(df['Close'], timeperiod=12)  # Fast-moving average
    df['SMA'] = talib.SMA(df['Close'], timeperiod=26)  # Simple moving average
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)  # Relative Strength Index
    
    return df

def generate_signal(df):
    # Basic example logic for buy, sell, or hold signals
    if df['RSI'].iloc[-1] > 70:
        return 'Sell'
    elif df['RSI'].iloc[-1] < 30:
        return 'Buy'
    else:
        return 'Hold'
    

# Function to generate a detailed report based on token data and sentiment analysis
def generate_detailed_report(token_data, sentiment_analysis):
    report = []
    
    for token, data in token_data.items():
        # Calculate technical indicators
        indicators = calculate_indicators(data['market_data'])
        signal = generate_signal(indicators)

        # Add market analysis to report
        report.append(f"Token: {token}")
        report.append(f"Fast Moving Average (FMA): {indicators['FMA'].iloc[-1]:.2f}")
        report.append(f"Simple Moving Average (SMA): {indicators['SMA'].iloc[-1]:.2f}")
        report.append(f"Relative Strength Index (RSI): {indicators['RSI'].iloc[-1]:.2f}")
        report.append(f"Suggested Action: {signal}")

        # Add sentiment analysis to report
        report.append(f"Sentiment Analysis: {sentiment_analysis[token]}")
        report.append("\n---\n")
    
    return "\n".join(report)

def save_report_to_file(report):
    with open('token_report.md', 'w') as f:
        f.write(report)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Function to print progress updates
def log_and_print(message):
    logging.info(message)
    print(message)

# Initialize LLM (OpenAI or Groq)
model_choice = input("Choose your LLM (openai/groq): ").lower()
llm_selector = LLMSelector(model_choice=model_choice)
llm = llm_selector.get_llm()

# Token Extractor Agent
log_and_print("Initializing Token Extractor agent...")
token_extractor = Agent(
    role="Token Extractor",
    goal="Extract tokens from the hardcoded list.",
    tools=[],  # No external tools required
    verbose=True,
    backstory="You are responsible for gathering token pairs from the list."
)

# Market Data Collector Agent
log_and_print("Initializing Market Data Collector agent...")
coingecko_search_tool = CoinGeckoSearchTool()  # Initialize the search tool
coingecko_search = Agent(
    role="Market Data Collector",
    goal="Search for token data from the CoinGecko API.",
    tools=[coingecko_search_tool],  # Ensure tool is passed correctly
    verbose=True,
    backstory="You are responsible for retrieving market data for tokens."
)

# Chart Generator Agent
log_and_print("Initializing Chart Generator agent...")
# We'll pass token data to the chart tool later in the script after collecting market data
chart_generator = Agent(
    role="Chart Generator",
    goal="Generate price charts for tokens.",
    tools=[],  # Tool will be initialized with token data after collecting it
    verbose=True,
    backstory="You are responsible for generating charts based on token price data."
)

# Sentiment Analyzer Agent (with selected LLM)
sentiment_analyzer = Agent(
    role="Sentiment Analyzer",
    goal="Perform sentiment analysis on the text related to the tokens.",
    tools=[{"name": "SentimentAnalysisTool", "tool": sentiment_analysis_tool}],
    backstory="You are responsible for analyzing sentiment data based on the token markets."
)


# Report Generator Agent (with selected LLM)
log_and_print("Initializing Report Generator agent...")
report_generator = Agent(
    role="Report Generator",
    goal="Generate a comprehensive report based on the collected data.",
    llm=llm,  # Pass the selected LLM (Groq or OpenAI)
    verbose=True,
    tools=[],
    backstory="You generate detailed reports based on the market data."
)

# Define Tasks
log_and_print("Creating tasks...")

extract_data_task = Task(
    description="Extract tokens from the hardcoded list.",
    expected_output="List of tokens for searching.",
    agent=token_extractor
)

search_data_task = Task(
    description="Search for token data from CoinGecko.",
    expected_output="Market data for tokens.",
    agent=coingecko_search
)

# Define the chart generation task later once we have token data
generate_charts_task = None

perform_sentiment_task = Task(
    description="Perform sentiment analysis on tokens.",
    expected_output="Sentiment analysis data for tokens.",
    agent=sentiment_analyzer
)

generate_report_task = Task(
    description="Generate a comprehensive report including FMA, SMA, RSI, market trends, and sentiment analysis.",
    expected_output="Detailed report on token data, including technical indicators and sentiment analysis.",
    agent=report_generator,
    output_file='token_report.md'
)

# Define the Crew and Process
log_and_print("Assembling the crew...")

crew = Crew(
    agents=[token_extractor, coingecko_search, chart_generator, sentiment_analyzer, report_generator],
    tasks=[extract_data_task, search_data_task, perform_sentiment_task, generate_report_task],
    process=Process.sequential,  # Sequential process for step-by-step execution
    config={}  # Ensure an empty config is passed to prevent 'NoneType' error
)

# Step 1: Run the Crew with the hardcoded ERC-20 tokens
log_and_print("Starting the crew process with hardcoded ERC-20 tokens...")
token_data = crew.kickoff(inputs={"tokens": list(erc20_tokens.keys())})

# Step 2: Ensure token_data is correctly structured
log_and_print("Initializing Chart Generator tool with token data...")
try:
    chart_tool = chart_tool(token_data_list=token_data)  # Handle token data correctly
    chart_generator.tools = [{"name": "chart_tool", "tool": chart_tool}]  # Update the agent's tools

    # Step 3: Create a new task for chart generation
    generate_charts_task = Task(
        description="Generate charts for the tokens based on the collected data.",
        expected_output="List of generated charts for each token.",
        agent=chart_generator
    )

    # Step 4: Append the chart generation task to the crew's tasks
    crew.tasks.append(generate_charts_task)

    # Step 5: Run the Crew again, this time including the chart generation task
    log_and_print("Running the crew process for chart generation...")
    result = crew.kickoff(inputs={"tokens": list(erc20_tokens.keys())})

    log_and_print("Chart generation process complete.")
    print("Result:")
    print(result)

except ValueError as e:
    log_and_print(f"Error initializing chart tool: {e}")
