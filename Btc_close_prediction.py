# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 21:35:01 2025

@author: Richar
"""

#Clase 3, proyecto propia y github

#Git es un software de control de versiones 
#version 1, version 2, .... de un codigo que estemos trabajando 
#git sistema que registra cambios en archivos a lo largo del tiempo
#FUNCIONES DESTACADAS
#File revisions_ Historial de versiones de archivos
#Remote shared repositoy: repositorio central
#push envia cambios pull descarga actualizaciones
#Cada repositorio sera un proyecto, se puede subir todo lo que acompaña a tu codigo
#puedes subir tu codigo y tu base de datos, se debe explicar bien que hacen los datos
#pero no subir la base de datos porque puede ser robado el trabajo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import os
os.chdir('C:/Users/Richar/Desktop/ciencia de datos')


df=pd.read_csv('bitcoin_usd.csv')
df.head()
df.describe()

print(df.isna().sum())

print(df.dtypes) 

print(df.info()) 
#Checar si hay nas con ?. o cosas asi para cualquier columna
print(df.iloc[:, 8].unique())
df['datetime'] = pd.to_datetime(df['Unnamed: 0'], format='%Y-%m-%d')
#La base de datos esta limpia
df = df.drop('Unnamed: 0', axis=1)  # Eliminamos la columna original
df.set_index('datetime', inplace=True)  # Establecemos fecha como índice
print(df.dtypes) 
#Para facilidad
df = df.rename(columns={
    'open_USD': 'open',
    'high_USD': 'high',
    'low_USD': 'low',
    'close_USD': 'close'
})
#Para escoger que se usa para entrenar se usara la correlación
corr_matrix = df.corr()
cancer_corr = corr_matrix[['close']]
sns.heatmap(cancer_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()
#se toma las de mayor correlación, tomamos de 0.24 a mas 









train_data_or = df[['open','high','low']]  

imputer=SimpleImputer(strategy='mean')
train_data_or=pd.DataFrame(imputer.fit_transform(train_data_or), columns=train_data_or.columns, index=train_data_or.index)

scaler=MinMaxScaler(feature_range=(0,1)) #Transforma los datos en ceros y unos para convergencia del modelo
train_scaled=scaler.fit_transform(train_data_or) #aplica la transformacion a train

#Crear secuencias temporales, datos para el entrenamiento

def create_dataset(dataset, look_back=60): #cada 60 pasos
    X, Y= [], []
    for i in range(len(dataset)-look_back): 
        a=dataset[i:(i+look_back),:]
        X.append(a) #x se crea con esos 60 pasos  de look back
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

#aqui se usa el lookback en el train 
X_train, y_train =create_dataset(train_scaled, 60)


#Modelo LSTM
model=Sequential([
    LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True), #estado oculto de 100 dimensiones, return sequence devuelve las salidas de cada paso
    #El el X_train.shape[1] es valor de look back cuantos pasos de tiempo historicos se van a considerar
    #El X_train.shape[2] es para las caracteristicas que tenemos
    Dropout(0,2), #evita sobreajuste, apaga aleatoriamente el 20% de neuronas durante entrenamiento, mas recomendable de 0.2 a 0.3 de 20 a 30%
    LSTM(100, return_sequences=False), #return sequence de segunda capa false, solo queremos la final
    Dense(1) #da una neurona que da un unico valor de la demanda en el paso del tiempo, siempre se deja en 1
    ])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error') 
#compilar el modelo con Adam de taasa de aprendizaje, MSE para problemas de 
#regresion, penaliza errores grandes al ser cuadratica

 

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1) #50 epocas

#Preparar los datos reales para compararlos
random_sample = df.sample(n=150, random_state=42)

real_data=random_sample[['open', 'high', 'low']]
real_data = pd.DataFrame(imputer.transform(real_data), columns=real_data.columns, index=real_data.index)
real_scaled = scaler.transform(real_data)


# Crear datos de entrada para predicción
X_real, _ = create_dataset(np.vstack([train_scaled[-60:], real_scaled]), 60)
#' Predicciones
predictions_scaled = model.predict(X_real)
predictions = scaler.inverse_transform(np.concatenate([predictions_scaled, np.zeros((len(predictions_scaled), 2))], axis=1))[:, 0]
#np.zeros((len(predictions_scaled), 2)) el 2 se debe a la cantidad de ceros añadida
# al ser 3 variables y usar una se ocupa poner 2 ceros
# Datos reales para comparar
real_demand = scaler.inverse_transform(real_scaled)[:, 0]

# Graficar
plt.figure(figsize=(14, 7))
plt.plot(real_data.index, real_demand, label='Datos Reales', color='blue')
plt.plot(real_data.index[:len(predictions)], predictions, label='Predicciones', color='red')
plt.title('Predicciones vs Datos Reales Bitcoin')
plt.xlabel('Fechas')
plt.ylabel('Valor close')
plt.legend()
plt.grid(True)
plt.show()






import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import r2_score

# Asegurarse de que tanto 'real_demand' como 'predictions' tienen el mismo índice de tiempo
real_demand = pd.Series(real_demand[:len(predictions)], index=real_data.index[:len(predictions)])
predictions_series = pd.Series(predictions, index=real_data.index[:len(predictions)])

# Crear una lista para almacenar los R-squared de cada mes
r_squared_monthly = []

# Agrupar los datos por mes y calcular el R-squared para cada mes
for month, group in real_demand.groupby(real_demand.index.month):
    # Extraer las predicciones correspondientes para ese mes
    pred_for_month = predictions_series[group.index]
    
    # Calcular el R-squared
    r_squared = r2_score(group, pred_for_month)
    
    # Almacenar el resultado
    r_squared_monthly.append((month, r_squared))
    print(f'R-squared para el mes {month}: {r_squared}')

# Crear la figura usando Plotly para una mejor interactividad y presentación
fig = go.Figure()

# Añadir los datos reales a la gráfica
fig.add_trace(go.Scatter(x=real_data.index, y=real_demand, mode='lines', name='Datos Reales 2024', line=dict(color='blue')))

# Añadir las predicciones a la gráfica
fig.add_trace(go.Scatter(x=real_data.index[:len(predictions)], y=predictions, mode='lines', name='Predicciones 2024', line=dict(color='red')))

# Configuración adicional del gráfico
fig.update_layout(
    title='Comparación de Predicciones vs Datos Reales para ORI 2024',
    xaxis_title='Fecha',
    yaxis_title='Estimación de Demanda por Balance (MWh)',
    legend_title='Leyenda'
)

# Mostrar la gráfica
fig.show()

# Imprimir los R-squared mensuales
for month, r_squared in r_squared_monthly:
    print(f"R-squared para el mes {month}: {r_squared:.4f}")





















