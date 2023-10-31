# Importar las bibliotecas necesarias
import argparse  # Para manejar argumentos de la línea de comandos
import pickle  # Para serializar/deserializar modelos
import pandas as pd  # Para trabajar con datos en formato tabular
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.neighbors import KNeighborsRegressor  # Para el modelo KNN
from sklearn.metrics import mean_squared_error  # Para calcular la métrica RMSE
import numpy as np  # Para operaciones matemáticas
from datetime import datetime  # Para trabajar con fechas y horas
from loguru import logger

# Definir los valores por defecto para los argumentos
DEFAULT_TRAIN_FILE = 'train.csv'  # Nombre del archivo CSV de datos de entrenamiento
DEFAULT_MODEL_FILE = 'model.pkl'  # Nombre del archivo para guardar el modelo entrenado

# Configurar los argumentos de la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--train_file', type=str, default=DEFAULT_TRAIN_FILE, help='a CSV file with train data')  # Argumento para el archivo de datos de entrenamiento
parser.add_argument('--model_file', type=str, default=DEFAULT_MODEL_FILE, help='where the trained model will be stored')  # Argumento para el archivo de modelo entrenado

args = parser.parse_args()  # Analizar los argumentos de la línea de comandos
train_file = args.train_file  # Nombre del archivo de datos de entrenamiento
model_file_arg = args.model_file  # Nombre del archivo para guardar el modelo entrenado (renombrado para evitar conflictos)

# Función para limpiar el DataFrame de datos
def clean_df(df):
    """
    Limpia el DataFrame de datos de entrenamiento.

    Parameters:
    df (pd.DataFrame): El DataFrame de datos de entrenamiento.

    Returns:
    pd.DataFrame: El DataFrame de datos limpios.
    """
    # Selecciona solo los primeros 15 caracteres de la columna 'pickup_datetime'
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 15)
    # Convierte la columna 'pickup_datetime' a formato datetime en UTC
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    # Corrige incorrectamente las coordenadas de longitud/latitud
    df = df.assign(rev=df.dropoff_latitude < df.dropoff_longitude)
    idx = (df['rev'] == 1)
    df.loc[idx, ['dropoff_longitude', 'dropoff_latitude']] = df.loc[idx, ['dropoff_latitude', 'dropoff_longitude']].values
    df.loc[idx, ['pickup_longitude', 'pickup_latitude']] = df.loc[idx, ['pickup_latitude', 'pickup_longitude']].values
    
    # Filtra y limpia los puntos de datos fuera de los rangos apropiados
    criteria = (
        " 0 < fare_amount <= 500"
        " and 0 < passenger_count <= 6 "
        " and -75 <= pickup_longitude <= -72 "
        " and -75 <= dropoff_longitude <= -72 "
        " and 40 <= pickup_latitude <= 42 "
        " and 40 <= dropoff_latitude <= 42 "
    )
    df = (df
          .dropna()
          .query(criteria)
          .reset_index()
          .drop(columns=['rev', 'index'])
         )
    return df

# Cargar los datos de entrenamiento desde un archivo CSV
def load_data(file_path):
    """
    Carga los datos de entrenamiento desde un archivo CSV y aplica la limpieza.

    Args:
        file_path (str): Ruta al archivo CSV que contiene los datos de entrenamiento.

    Returns:
        pandas.DataFrame: El DataFrame de datos limpios y procesados.
    """
    cols = ['fare_amount', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    train = pd.read_csv(file_path, usecols=cols)
    # Realiza la limpieza de los datos
    train = clean_df(train)
    
    return train

# Modelo K-Nearest Neighbors (KNN)
def knn_model(x_train, x_test, y_train, y_test, neighbors):
    """
    Entrena un modelo K-Nearest Neighbors (KNN) y realiza predicciones.

    Args:
        x_train (pandas.DataFrame): Conjunto de entrenamiento de características.
        x_test (pandas.DataFrame): Conjunto de prueba de características.
        y_train (pandas.Series): Etiquetas de entrenamiento.
        y_test (pandas.Series): Etiquetas de prueba.
        neighbors (list of int): Lista de números de vecinos para probar.

    Returns:
        tuple: Un tuple que contiene el modelo KNN entrenado, el RMSE mínimo y las mejores predicciones.
    """
    min_rmse = 1000

    # Convierte la columna 'pickup_datetime' a valores numéricos
    x_train['pickup_datetime'] = (x_train['pickup_datetime'].dt.tz_localize(None) - datetime(1970, 1, 1)).dt.total_seconds().astype(float)
    x_test['pickup_datetime'] = (x_test['pickup_datetime'].dt.tz_localize(None) - datetime(1970, 1, 1)).dt.total_seconds().astype(float)
    print("RMSE mínimo y las mejores predicciones:")  # Imprimir un mensaje
    for n in neighbors:
        knn = KNeighborsRegressor(n_neighbors=n)
        knn.fit(x_train, y_train)
        pred = knn.predict(x_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        if rmse < min_rmse:
            min_rmse = rmse
            model = knn
            best_pred = pred
        print('Neighbours', n, 'RMSE', rmse)
    return model, min_rmse, best_pred

# Construir el modelo y realizar el entrenamiento con KNN
def build_and_train_model(train, model_fn):
    """
    Construye un modelo y realiza el entrenamiento con KNN.

    Args:
        train (pandas.DataFrame): DataFrame de datos de entrenamiento.
        model_fn (function): Función que entrena un modelo KNN.

    Returns:
        None
    """
    logger.info(f"Reescribiendo modelo existente en archivo {model_file_arg}")# Imprimir un mensaje
    X = train.drop(columns=['fare_amount'])
    y = train['fare_amount']

    # Divide los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Especifica una lista de números de vecinos para probar
    neighbors = [10, 20]

    # Entrenamiento de modelos KNN con diferentes números de vecinos
    logger.info(f"Ajustando modelo.")# Imprimir un mensaje
    model, rmse, pred = model_fn(x_train, x_test, y_train, y_test, neighbors)

    # Guardar el modelo entrenado en el archivo especificado
    with open(model_file_arg, 'wb') as model_file:
        pickle.dump(model, model_file)
    logger.info(f"Modelo guardado en {model_file.name}")# Imprimir un mensaje

if __name__ == "__main__":
    
    train_data = load_data(train_file)  # Cargar los datos de entrenamiento desde el archivo CSV
    logger.info(f"Datos de entrenamiento cargados desde {train_file}")# Imprimir un mensaje
    
    # Llamar a la función build_and_train_model para entrenar el modelo KNN
    build_and_train_model(train_data, knn_model)
    