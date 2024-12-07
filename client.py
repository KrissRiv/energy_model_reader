import requests

def get_prediction(features):
    # URL de la API de Flask
    api_url = "http://127.0.0.1:5000/predict"
    
    # Datos a enviar en formato JSON
    data = {"features": features}

    try:
        # Hacemos una solicitud POST a la API con los datos
        response = requests.post(api_url, json=data)

        # Revisamos si la solicitud fue exitosa
        if response.status_code == 200:
            # Imprimir la respuesta de la API
            prediction = response.json().get('prediction')
            if prediction is not None:
                print(f"Predicción recibida: {prediction}")
            else:
                print("Respuesta inesperada, no se encontró una predicción.")
        else:
            print(f"Solicitud fallida con estatus: {response.status_code}")
            print(f"Mensaje de error: {response.json().get('error')}")

    except requests.exceptions.RequestException as e:
        # En caso de excepción al realizar la solicitud
        print(f"Error al conectar con la API: {e}")

if __name__ == "__main__":
    # Ejemplo de características de entrada para la predicción
    # Asegúrate de que estas características coincidan con lo que espera tu modelo
    example_features = [100, 50, 75, 25, 100]  # Reemplaza con las características reales

    # Llamar a la función para obtener la predicción
    get_prediction(example_features)
