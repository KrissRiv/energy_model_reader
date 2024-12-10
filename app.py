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

        # Check if the necessary features are present
        required_features = ['coal_consumption', 'gas_consumption', 'nuclear_consumption', 'oil_consumption', 'renewables_consumption']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing required features'}), 400

        # Create a DataFrame from the input data
        import pandas as pd
        input_df = pd.DataFrame([data])

        # Make predictions
        prediction = loaded_model.predict(input_df)

        # Return the prediction
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
