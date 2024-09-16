from crewai_tools import BaseTool

class CoinGeckoSearchTool(BaseTool):
    name: str = "CoinGeckoSearchTool"  # Explicit type annotation for 'name'
    description: str = "Tool to fetch token data from CoinGecko API."

    def _run(self, token: str) -> str:
        # Here, you would call the CoinGecko API with the token symbol
        # For now, let's simulate the API response with a placeholder result
        return f"Market data for {token}: price, volume, market cap."

    def _run_batch(self, tokens: list) -> list:
        # Batch processing if needed for multiple tokens
        return [self._run(token) for token in tokens]
