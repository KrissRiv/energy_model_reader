import os
import pickle
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd

# Correct file paths
model_rf_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_rf.pkl')
model_gr_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model_gr.pkl')

app = Flask(__name__)

@app.route('/')
def serve_index():
    return send_from_directory('../static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:

        data = request.get_json()

        # Check if the necessary features are present
        required_features = ['coal_consumption', 'gas_consumption', 'hydro_consumption', 'renewables_consumption', 'solar_consumption', 'wind_consumption']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing required features'}), 400
        
        modelo = data['model']
        if modelo == 'best_model_rf':
            model_path = model_rf_path
        elif modelo == 'best_model_gr':
            model_path = model_gr_path
        else:
            return jsonify({'error': 'Invalid model selected'}), 400
        
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)

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
