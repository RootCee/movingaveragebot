import requests

def get_token_data(contract_address):
    url = f"https://api.coingecko.com/api/v3/coins/ethereum/contract/{contract_address}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def coingecko_search_tool(erc20_tokens):
    token_data_list = []
    for token, address in erc20_tokens.items():
        token_data = get_token_data(address)
        if token_data:
            token_data_list.append(token_data)
    return token_data_list
