# Importar las bibliotecas necesarias
from flask import Flask, request, jsonify
import pandas as pd
from train import clean_df as clean_df_train
from predict import clean_df as clean_df_predict
from sklearn.neighbors import KNeighborsRegressor
import pickle
from loguru import logger
from datetime import datetime
from sklearn.model_selection import train_test_split

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Cargar el modelo preentrenado
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Ruta para realizar predicciones
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos de entrada del cuerpo de la solicitud
        data = request.get_json()

        # Convertir los datos a un DataFrame de pandas
        input_data = pd.DataFrame(data, index=[0])

        # Verificar la presencia de las claves necesarias en los datos de entrada
        required_keys = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
        if not all(key in input_data.columns for key in required_keys):
            return jsonify({'error': 'Datos de entrada incompletos o incorrectos'}), 400

        # Limpiar y procesar los datos de entrada
        cleaned_data = clean_df_predict(input_data)
        # Realizar la predicción con el modelo
        predictions = model.predict(cleaned_data)
        # Devolver las predicciones como respuesta en formato JSON
        return jsonify({'predictions': predictions.tolist()})
    except AttributeError as e:
        logger.error(f"Error en la ruta /predict: {str(e)}")
        return jsonify({'error': f'Ocurrió un error en la predicción: {str(e)}'}), 500


# Ruta para realizar el entrenamiento del modelo
@app.route('/train', methods=['POST'])
def train():
    try:
        # Obtener los datos de entrenamiento del cuerpo de la solicitud
        train_data = request.get_json()

        # Convertir los datos a un DataFrame
        df = pd.DataFrame(train_data)

        # Realizar la limpieza de datos y entrenar el modelo
        df = clean_df_train(df)
        

        # Dividir los datos en conjuntos de entrenamiento y prueba
        X = df.drop(columns=['fare_amount'])
        y = df['fare_amount']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crear una nueva instancia del modelo para entrenar desde cero
        model = KNeighborsRegressor(n_neighbors=10)  # Puedes ajustar el número de vecinos según sea necesario

        # Entrenar el modelo
        model.fit(X_train, y_train)

        # Guardar el modelo entrenado (opcional)
        model_file = 'model.pkl'  # Especifica el nombre del archivo de modelo
        with open(model_file, 'wb') as model_file:
            pickle.dump(model, model_file)

        return jsonify({'message': 'Modelo entrenado exitosamente'})
    except Exception as e:
        logger.error(f"Error en la ruta /train: {str(e)}")
        return jsonify({'error': 'Ocurrió un error en el entrenamiento del modelo'}), 500


# Punto de entrada principal
if __name__ == '__main__':
    # Configurar la aplicación para ejecutarse en el puerto 5000
    app.run(port=5000, debug=True)