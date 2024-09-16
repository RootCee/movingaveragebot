import os
import openai
from groq import Groq
from dotenv import load_dotenv
from langchain_core.runnables.base import Runnable
from langchain.prompts.base import StringPromptValue  # Import StringPromptValue to check the input type

# Load environment variables
load_dotenv()

class OpenAILLM(Runnable):
    """
    Simple OpenAI wrapper.
    """
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        openai.api_key = self.api_key
        self.model_params = {}

    def _call(self, prompt: str, stop=None):
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=100,
            stop=stop
        )
        return response.choices[0].text.strip()

    def _generate(self, prompt, stop=None):
        return self._call(prompt, stop)

    def bind(self, stop=None, max_tokens=100, temperature=0.7):
        """
        Bind method to set parameters like stop words, max tokens, etc.
        """
        self.model_params = {
            "stop": stop,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        return self

    def invoke(self, inputs, config=None):
        """
        Method for LangChain to run the model. Updated to handle the StringPromptValue object.
        """
        if isinstance(inputs, StringPromptValue):
            prompt = inputs.to_string()  # Use to_string() to extract the prompt
        elif isinstance(inputs, str):
            prompt = inputs
        else:
            prompt = inputs.get('prompt', '') if isinstance(inputs, dict) else ''
        
        return self._generate(prompt)


class GroqLLM(Runnable):
    """
    Simple Groq wrapper.
    """
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self.client = Groq(api_key=self.api_key)
        self.model_params = {}

    def _call(self, prompt: str, stop=None):
        response = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-groq-70b-8192-tool-use-preview",
            max_tokens=100,
            stop=stop
        )
        return response.choices[0].message.content.strip()

    def _generate(self, prompt, stop=None):
        return self._call(prompt, stop)

    def bind(self, stop=None, max_tokens=100, temperature=0.7):
        """
        Bind method to set parameters like stop words, max tokens, etc.
        """
        self.model_params = {
            "stop": stop,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        return self

    def invoke(self, inputs, config=None):
        """
        Method for LangChain to run the model. Updated to handle the StringPromptValue object.
        """
        if isinstance(inputs, StringPromptValue):
            prompt = inputs.to_string()  # Use to_string() to extract the prompt
        elif isinstance(inputs, str):
            prompt = inputs
        else:
            prompt = inputs.get('prompt', '') if isinstance(inputs, dict) else ''
        
        return self._generate(prompt)


class LLMSelector:
    def __init__(self, model_choice="openai"):
        """
        Initializes the LLM based on user choice (openai or groq).
        """
        self.model_choice = model_choice
        if self.model_choice == "openai":
            self.llm = OpenAILLM()
        elif self.model_choice == "groq":
            self.llm = GroqLLM()
        else:
            raise ValueError("Invalid model choice. Choose 'openai' or 'groq'.")

    def get_llm(self):
        return self.llm
