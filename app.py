import pickle
from flask import Flask, request, jsonify

# Load the saved model
filename = 'best_model_rf.pkl'  # Replace with the actual filename
loaded_model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        print(f"Data: {data}")

        # Check if the necessary features are present
        required_features = ['coal_consumption', 'gas_consumption', 'nuclear_consumption', 'oil_consumption', 'renewables_consumption']
        print(f" Required: {required_features}")
        if not all(feature in data for feature in required_features):
            print("error")
            return jsonify({'error': 'Missing required features'}), 400

        # Create a DataFrame from the input data
        import pandas as pd
        input_df = pd.DataFrame([data])
        print(f"Input_df {input_df}")

        # Make predictions
        print(f"loaded_model {loaded_model}")
        prediction = loaded_model.predict(input_df)

        # Return the prediction
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
