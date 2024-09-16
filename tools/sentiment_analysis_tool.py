from crewai_tools import tool, BaseTool  # Ensure proper imports
from groq import Groq
import os

# Ensure Groq API Key is set
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq client
groq_client = Groq(api_key=groq_api_key)

class SentimentAnalysisTool(BaseTool):  # Properly inheriting from BaseTool
    """
    A tool that performs sentiment analysis using Groq API.
    """

    name: str = "SentimentAnalysisTool"  # Define required name field
    description: str = "A tool that performs sentiment analysis on provided text using the Groq API."  # Define required description field

    def _run(self, text_data: str) -> str:
        """
        Analyze the sentiment of the given text data using the Groq API.
        
        Args:
            text_data (str): The text data to analyze.
        
        Returns:
            str: Sentiment result (Positive, Negative, Neutral).
        """
        # Build the prompt for sentiment analysis
        prompt = f"Analyze the sentiment of the following text:\n\n{text_data}"
        
        # Call Groq API to perform sentiment analysis
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-groq-70b-8192-tool-use-preview",
            max_tokens=60
        )
        
        # Extract and return the sentiment result
        return response.choices[0].message.content.strip()

# Create the tool instance
sentiment_analysis_tool = tool(SentimentAnalysisTool)
