import argparse  # Importar la biblioteca argparse para procesar argumentos de línea de comandos
import pickle  # Importar la biblioteca pickle para trabajar con serialización de objetos
import pandas as pd  # Importar la biblioteca pandas para el manejo de datos en formato tabular
import os  # Importar la biblioteca os para operaciones relacionadas con el sistema operativo
from sklearn.neighbors import KNeighborsRegressor  # Importar el regresor KNeighbors de scikit-learn
from datetime import datetime  # Agregar esta línea para manejar la columna pickup_datetime

# Definir los valores por defecto para los argumentos
DEFAULT_INPUT_FILE = 'test.csv'  # Nombre de archivo de entrada por defecto
DEFAULT_PREDICTIONS_FILE = 'predictions.csv'  # Nombre de archivo de predicciones por defecto
DEFAULT_MODEL_FILE = 'model.pkl'  # Nombre de archivo de modelo por defecto

# Configurar el análisis de argumentos de la línea de comandos
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default=DEFAULT_INPUT_FILE, type=str, help='un archivo CSV con datos de entrada (sin objetivos)')  # Argumento para el archivo de entrada
parser.add_argument('--predictions_file', default=DEFAULT_PREDICTIONS_FILE, type=str, help='un archivo CSV donde se guardarán las predicciones')  # Argumento para el archivo de predicciones
parser.add_argument('--model_file', default=DEFAULT_MODEL_FILE, type=str, help='un archivo .pkl con un modelo previamente almacenado (ver train.py)')  # Argumento para el archivo de modelo

args = parser.parse_args()  # Analizar los argumentos de la línea de comandos
model_file = args.model_file  # Nombre del archivo de modelo especificado en los argumentos
input_file = args.input_file  # Nombre del archivo de entrada especificado en los argumentos
predictions_file = args.predictions_file  # Nombre del archivo de predicciones especificado en los argumentos

if not os.path.isfile(model_file):
    print(f"El archivo del modelo {model_file} no existe")  # Imprimir un mensaje si el archivo del modelo no existe
    exit(-1)  # Salir del programa con un código de error

if not os.path.isfile(input_file):
    print(f"El archivo de entrada {input_file} no existe")  # Imprimir un mensaje si el archivo de entrada no existe
    exit(-1)  # Salir del programa con un código de error

def clean_df(df):
    """
    Limpia un DataFrame de datos de entrada.

    Args:
        df (pandas.DataFrame): El DataFrame que contiene los datos a limpiar.

    Returns:
        pandas.DataFrame: El DataFrame limpio y procesado.
    """
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 15)  # Seleccionar solo los primeros 15 caracteres de la columna 'pickup_datetime'
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')  # Convertir la columna 'pickup_datetime' a formato datetime en UTC
    
    # Convertir 'pickup_datetime' a una representación numérica
    df['pickup_datetime'] = (df['pickup_datetime'].dt.tz_localize(None) - datetime(1970, 1, 1)).dt.total_seconds().astype(float)
    
    # Invertir incorrectamente las coordenadas de longitud/latitud
    df = df.assign(rev=df['dropoff_latitude'] < df['dropoff_longitude'])
    idx = (df['rev'] == 1)
    df.loc[idx, ['dropoff_longitude', 'dropoff_latitude']] = df.loc[idx, ['dropoff_latitude', 'dropoff_longitude']].values
    df.loc[idx, ['pickup_longitude', 'pickup_latitude']] = df.loc[idx, ['pickup_latitude', 'pickup_longitude']].values
    
    # Eliminar puntos de datos fuera de rangos apropiados
    criteria = (
        "0 < passenger_count <= 6 "
        "and -75 <= pickup_longitude <= -72 "
        "and -75 <= dropoff_longitude <= -72 "
        "and 40 <= pickup_latitude <= 42 "
        "and 40 <= dropoff_latitude <= 42"
    )
    df = (df
          .dropna()
          .query(criteria)
          .reset_index()
          .drop(columns=['rev', 'index'])          
    )
    return df

print("Cargando datos de entrada")  # Imprimir un mensaje
cols = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
Xts = pd.read_csv(input_file, usecols=cols)  # Leer el archivo de entrada y seleccionar columnas específicas
Xts = clean_df(Xts)  # Llamar a la función para limpiar los datos

# Cargar el modelo preentrenado
model = pickle.load(open(model_file, 'rb'))

print("Realizando predicciones")  # Imprimir un mensaje
predictions = model.predict(Xts)  # Realizar predicciones con el modelo

# Imprimir las predicciones o guardarlas en el archivo de predicciones
for prediction in predictions:
    print(f'Predicción: {prediction}')

print(f"Guardando predicciones en {predictions_file}")  # Imprimir un mensaje con el nombre del archivo de predicciones
pd.DataFrame(predictions.reshape(-1, 1), columns=['predicciones']).to_csv(predictions_file, index=False)  # Guardar las predicciones en un archivo CSV