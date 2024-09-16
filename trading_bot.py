from llm_selector import LLMSelector

def run_trading_bot(llm_choice):
    # Initialize the LLM
    llm = LLMSelector(model_choice=llm_choice)

    # Define a trading prompt
    trading_prompt = "Give me a sentiment analysis of Bitcoin in the current market."
    
    # Generate a response (sentiment analysis)
    sentiment = llm.sentiment_analysis(trading_prompt)
    print(f"Sentiment Analysis from {llm_choice.capitalize()} LLM:\n{sentiment}")

if __name__ == "__main__":
    model_choice = input("Choose your LLM (openai/groq): ").lower()
    run_trading_bot(model_choice)


# Run Bot = python trading_bot.py // Choose your LLM (openai/groq): openai // Sentiment Analysis from Openai LLM: Positive 
#// Choose your LLM (openai/groq): groq // Sentiment Analysis from Groq LLM: Neutral