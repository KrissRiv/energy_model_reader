from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Cargar el modelo desde un archivo pickle
def load_model():
    with open('sa_energy_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model

model = load_model()

@app.route('/')
def home():
    return "Bienvenido a la API de Consumo Energético de Sudamérica"

@app.route('/predict', methods=['POST'])
def predict():
    # Esperamos un JSON con los datos de entrada para la predicción
    data = request.json
    try:
        # Supongamos que el modelo espera tres características de entrada
        features = data['features']  # por ejemplo, [feature1, feature2, feature3]
        features_array = np.array(features).reshape(1, -1)  # Convertir a un array numpy

        # Realizar la predicción
        prediction = model.predict(features_array)

        return jsonify({'prediction': prediction.tolist()})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
