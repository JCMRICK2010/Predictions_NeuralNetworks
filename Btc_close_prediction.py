# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 21:35:01 2025

@author: Richar
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
os.chdir('C:/Users/Richar/Desktop/ciencia de datos')

df = pd.read_csv('btc_1h.csv')
df.head()
df.describe()

print(df.isna().sum())
print(df.dtypes) 
print(len(df))
print(df.columns)
print(df.info()) 

print(df.iloc[:, 1].unique())
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

real_data = df.loc['2021-12':'2022-01']
df = df.drop(real_data.index)

df.set_index('datetime', inplace=True)
real_data.set_index('datetime', inplace=True)

print(df.dtypes) 

corr_matrix = df.corr()
btc_corr = corr_matrix[['close']]
sns.heatmap(btc_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.show()

train_data_or = df[['open','high','low']]  

imputer = SimpleImputer(strategy='mean')
train_data_or = pd.DataFrame(imputer.fit_transform(train_data_or), columns=train_data_or.columns, index=train_data_or.index)

scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_data_or)

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset)-look_back): 
        a = dataset[i:(i+look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

X_train, y_train = create_dataset(train_scaled, 60)

model = Sequential([
    LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

real_data = real_data.drop(['close', 'volume'], axis=1)
real_data = pd.DataFrame(imputer.transform(real_data), columns=real_data.columns, index=real_data.index)
real_scaled = scaler.transform(real_data)

X_real, _ = create_dataset(np.vstack([train_scaled[-60:], real_scaled]), 60)
predictions_scaled = model.predict(X_real)
predictions = scaler.inverse_transform(np.concatenate([predictions_scaled, np.zeros((len(predictions_scaled), 2))], axis=1))[:, 0]

real_demand = scaler.inverse_transform(real_scaled)[:, 0]

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

real_demand = pd.Series(real_demand[:len(predictions)], index=real_data.index[:len(predictions)])
predictions_series = pd.Series(predictions, index=real_data.index[:len(predictions)])

r_squared_monthly = []

for month, group in real_demand.groupby(real_demand.index.month):
    pred_for_month = predictions_series[group.index]
    r_squared = r2_score(group, pred_for_month)
    r_squared_monthly.append((month, r_squared))
    print(f'R-squared para el mes {month}: {r_squared}')

fig = go.Figure()

fig.add_trace(go.Scatter(x=real_data.index, y=real_demand, mode='lines', name='Datos Reales 2024', line=dict(color='blue')))
fig.add_trace(go.Scatter(x=real_data.index[:len(predictions)], y=predictions, mode='lines', name='Predicciones 2024', line=dict(color='red')))

fig.update_layout(
    title='Comparación de Predicciones vs Datos Reales para ORI 2024',
    xaxis_title='Fecha',
    yaxis_title='Estimación de Demanda por Balance (MWh)',
    legend_title='Leyenda'
)

fig.show()

for month, r_squared in r_squared_monthly:
    print(f"R-squared para el mes {month}: {r_squared:.4f}")


















