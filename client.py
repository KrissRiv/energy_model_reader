import requests
import json

def predict_gdp(coal_consumption, gas_consumption, oil_consumption, renewables_consumption, nuclear_consumption):
    """
    Sends a POST request to the Flask API to predict GDP based on energy consumption data.

    Args:
        coal_consumption: Amount of coal consumption.
        gas_consumption: Amount of gas consumption.
        oil_consumption: Amount of oil consumption.
        renewables_consumption: Amount of renewable energy consumption.
        nuclear_consumption: Amount of nuclear energy consumption.


    Returns:
        The predicted GDP value as a float, or None if an error occurs.
    """

    url = 'http://127.0.0.1:5000/predict' # Replace with your API endpoint
    data = {
        'gdp': 100,
        'coal_consumption': coal_consumption,
        'gas_consumption': gas_consumption,
        'nuclear_consumption': nuclear_consumption,
        'oil_consumption': oil_consumption,
        'renewables_consumption': renewables_consumption,
    }

    headers = {'Content-Type': 'application/json'}

    try:
        print(f"json.dumps(data) {json.dumps(data)}")
        print(f"request: {requests.post(url, data=json.dumps(data), headers=headers)}")
        response = requests.post(url, data=json.dumps(data), headers=headers)
        print(f"Response {response}")
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        result = response.json()
        if 'prediction' in result:
            return result['prediction'][0]  # Assuming a single prediction
        elif 'error' in result:
            print(f"API Error: {result['error']}")
            return None
        else:
            print("API returned an unexpected response.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to the API: {e}")
        return None


# Example usage:
predicted_gdp = predict_gdp(coal_consumption=100, gas_consumption=200, oil_consumption=150, renewables_consumption=50, nuclear_consumption=25)

if predicted_gdp is not None:
    print(f"Predicted GDP: {predicted_gdp}")
