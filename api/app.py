import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Load the saved model
model_rf = 'best_model_rf.pkl'  # Replace with the actual filename
model_gr = 'best_model_gr.pkl'  # Replace with the actual filename

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()

        # Check if the necessary features are present
        required_features = ['coal_consumption', 'gas_consumption', 'hydro_consumption', 'renewables_consumption', 'solar_consumption', 'wind_consumption']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing required features'}), 400
        
        modelo = data['model']
        loaded_model = pickle.load(open(modelo, 'rb'))

        # Create a DataFrame from the input data
        input_df = pd.DataFrame([data])

        # Make predictions
        prediction = loaded_model.predict(input_df)

        # Return the prediction
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 443)))
