import pickle
from flask import jsonify, request
from loguru import logger
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def clean_df(df):
    """
    Limpia un DataFrame de datos de entrada.

    Args:
        df (pandas.DataFrame): El DataFrame que contiene los datos a limpiar.

    Returns:
        pandas.DataFrame: El DataFrame limpio y procesado.
    """
    # Selecciona solo los primeros 15 caracteres de la columna 'pickup_datetime'
    df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 15)
    # Convierte la columna 'pickup_datetime' a formato datetime en UTC
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
    
    df = df.assign(rev=df['dropoff_latitude'] < df['dropoff_longitude'])
    idx = (df['rev'] == 1)
    df.loc[idx, ['dropoff_longitude', 'dropoff_latitude']] = df.loc[idx, ['dropoff_latitude', 'dropoff_longitude']].values
    df.loc[idx, ['pickup_longitude', 'pickup_latitude']] = df.loc[idx, ['pickup_latitude', 'pickup_longitude']].values
    
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
