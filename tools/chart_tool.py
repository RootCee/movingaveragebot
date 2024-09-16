import matplotlib.pyplot as plt

def generate_chart(token_name, price_data):
    """
    Generates and saves a chart for a given token based on price data.
    price_data must contain 'time' and 'prices' as lists.
    """
    if not isinstance(price_data, dict) or 'time' not in price_data or 'prices' not in price_data:
        raise ValueError("price_data must be a dictionary with 'time' and 'prices' as keys")
    
    # Plot the price data
    plt.figure(figsize=(10, 6))
    plt.plot(price_data['time'], price_data['prices'], label=f'{token_name} Prices')

    # Add chart labels and title
    plt.title(f'{token_name} Price Chart')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    # Save the chart as a PNG file
    plt.savefig(f'{token_name}_price_chart.png')
    plt.close()  # Close the figure to avoid memory issues

def chart_tool(token_data_list):
    """
    Generates charts for a list of tokens.
    Each token's data should be in a dictionary format with 'name' and 'price_data' (which contains 'time' and 'prices').
    """
    for token_data in token_data_list:
        if isinstance(token_data, dict):  # If token_data is a dictionary
            token_name = token_data.get('name')
            price_data = token_data.get('price_data')  # Assume price_data is a dictionary with 'time' and 'prices'
        elif isinstance(token_data, tuple):  # If token_data is a tuple
            token_name = token_data[0]
            price_data = token_data[1]
        else:
            raise ValueError("Unexpected data format for token data")

        # Validate price_data format
        if not isinstance(price_data, dict) or 'time' not in price_data or 'prices' not in price_data:
            raise ValueError("price_data must be a dictionary with 'time' and 'prices' as keys")

        # Ensure price_data is in the correct format
        generate_chart(token_name, price_data)
