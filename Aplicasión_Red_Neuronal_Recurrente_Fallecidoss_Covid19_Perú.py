# -*- coding: utf-8 -*-


# **REDES NEURONALES RECURRENTES PARA PREDECIR LA CANTIDAD DE FALLECIOS EN PERÚ POR COVID19**

- Autor: José Espinoza
- Email: jose.espinoza.l@uni.pe
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)

"""Subiendo archivos:"""

from google.colab import drive
drive.mount('/content/drive')

"""Mostrando lista de archivos:"""

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# ls -l /content/drive/My\ Drive/Maestria-UNI
# 
# # ls -l /content/drive/My\ Drive/Maestria-UNI

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# %matplotlib inline 
import tensorflow as tf
## math library for mathematical function
import math
## data reader od panda is used fetch the data from web
import pandas_datareader as web
## numpy is used to create multi dimensional array
import numpy as np
## tensorflow is uded to create DL model and wrapping the other libraries

## sklearn is providing ultility functions for standerdizing or scaling data
from sklearn.preprocessing import MinMaxScaler
## keras is a neural network library
from keras.layers import LSTM     #GRU
from keras.layers import Dense
from keras.models import Sequential
## it is uded to create plotting area
import matplotlib.pyplot as mtlplt
## feature scaling distribution
from matplotlib import rcParams

"""

---

 ## PARTE I - PREPROCESAMIENTO DE DATOS
 

---

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Path principal
path_dataset = '/content/drive/My Drive/WORKSHOP_SERIES_TIEMPO/dataset/'

# Importando dataset
dataset_train = pd.read_excel("".join([path_dataset,"covid_train.xlsx"]))

dataset_test = pd.read_excel("".join([path_dataset,"covid_test.xlsx"]))

dataset_train.head(2)

df_train_test = pd.concat((dataset_train['Total'], dataset_test['Total']), axis = 0)

df_train_test = df_train_test.reset_index()

df_train_test['Total']

df_train_test.shape

## Visulaizing close price on graph from historical data
mtlplt.figure(figsize=(25,9))
mtlplt.title('FALLECIDOS POR COVID19')
mtlplt.plot(df_train_test['Total'])
mtlplt.xticks(range(0,dataset_train.shape[0],10),dataset_train['Fecha'].loc[::10],rotation=45) ### de 10 en 10 resumirá las fechas
mtlplt.xlabel('FECHA', fontsize=20)
mtlplt.ylabel('CANTIDAD',fontsize=20)
mtlplt.show

## Visulaizing close price on graph from historical data
mtlplt.figure(figsize=(18,9))
mtlplt.title('FALLECIDOS POR COVID19')
mtlplt.plot(dataset_train['Total'])
mtlplt.xticks(range(0,dataset_train.shape[0],10),dataset_train['Fecha'].loc[::10],rotation=45) ### de 10 en 10 resumirá las fechas
mtlplt.xlabel('FECHA', fontsize=20)
mtlplt.ylabel('CANTIDAD',fontsize=20)
mtlplt.show

dataset_train.shape

# dataset_train.iloc[:,1:2].values

# Obteniendo el campo 'Total fallecidos'
training_set = dataset_train.iloc[:,1:2].values
# print(training_set)

# Normalización: (x-min)/(max-min) --> feature_range [0 - 1]
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Estructura de 7 datos de entrada (t) y 1 dato de salida (t+1)  X[1-7] --> y[8]
train_len = len(dataset_train)
amplitude = 7   ###   ELIGEN LA AMPLITUD 

print("Número de elementos de la serie (Train): ", train_len)
print("Amplitud: ", amplitude)

X_train = []
y_train = []

for i in range(amplitude, train_len):
    X_train.append(training_set_scaled[i-amplitude:i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Redimensionando X_train (train_len,amplitude,1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print ("Dimensión de X_train: ",X_train.shape)

"""

---

## PARTE II - CONSTRUYENDO LA RED NEURONAL RECURRENTE
 

---

"""

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# inicializando the RNN
rnn_model = Sequential()

# 1ra capa LSTM
# input_shape (amplitude,1)
rnn_model.add(LSTM(units = 7, return_sequences = True, input_shape = (X_train.shape[1], 1)))  #lstm es un tipo de red neuronal recurrente , tambien está el GRU

# 2da capa LSTM
rnn_model.add(LSTM(units = 7, return_sequences = True))

# 3ra capa LSTM
rnn_model.add(LSTM(units = 7, return_sequences = True))

# 4ta capa LSTM
rnn_model.add(LSTM(units = 7, return_sequences = False))

# Output layer
rnn_model.add(Dense(units = 1))

# Compiling the RNN
rnn_model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# help(rnn_model)

# dataset_train.head(2)

dataset_train.shape

# help(rnn_model)

#Parámetros
batch_size = 5 # va tomar de 5 en 5 al train 
epochs = 150 

# Entrenamiento de la RNN con nuestro Training set
history  =  rnn_model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size)

# history = model.fit(x_train, y_train, batch_size=30, epochs=TRAIN_ITER, verbose=1,validation_split=0.01)

# Loss
loss = history.history['loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.title('Training validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""---

 ## PARTE III - PREDICCIONES Y VISUALIZACIÓN DE RESULTADOS
 

---



"""

dataset_test = pd.read_excel("".join([path_dataset,"covid_test.xlsx"]))
dataset_test

dataset_test.iloc[:, 1:2].values

real_covid = dataset_test.iloc[:, 1:2].values
# print(real_covid)

test_len = len(dataset_test)
print("Número de elementos de la serie (Test): ", test_len)

# Valores pronosticados
# Concatenar en un solo vector (train + test)
dataset_total = pd.concat((dataset_train['Total'], dataset_test['Total']), axis = 0)

# Obteniendo los (amplitude) datos anteriores al primer elemento del Test set
inputs = dataset_total[len(dataset_total) - test_len - amplitude:].values

# Redimensionando de (x,) --> (x,1)
inputs = inputs.reshape(-1,1)

# Feature Scaling
inputs = sc.transform(inputs)

X_test = []

for i in range(0, test_len):
    X_test.append(inputs[i:i+amplitude, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print ("Dimensión de X_test: ",X_test.shape)

predicted_mort = rnn_model.predict(X_test)
predicted_mort = sc.inverse_transform(predicted_mort)
predicted_mort = [np.round(x) for x in predicted_mort]
# print(predicted_mort)

predicted_mort_train = rnn_model.predict(X_train)
predicted_mort_train = sc.inverse_transform(predicted_mort_train)
predicted_mort_train = [np.round(x) for x in predicted_mort_train]

training_set = dataset_train.iloc[:,1:2].values

# training_set

# training_set[7:,]

plt.figure(figsize=(20,7),dpi=80)
plt.plot(training_set[7:,], color = 'blue', label = 'REAL')
plt.plot(predicted_mort_train, color = 'red', label = 'PRONÓSTICO)', dashes=[6, 2])
plt.title('Entrenamiento de la RNA de la cantidad de fallecidos en Perú')
plt.xlabel('Tiempo')
plt.ylabel('cantidad')
plt.legend()
plt.show()

plt.figure(figsize=(20,7),dpi=80)
plt.plot(training_set[7:,], color = 'blue', label = 'REAL')
plt.plot(predicted_mort_train, color = 'red', label = 'PRONÓSTICO)', dashes=[6, 2])
plt.title('Entrenamiento de la RNA de la cantidad de fallecidos en Perú')
plt.xlabel('Tiempo')
plt.ylabel('cantidad')
plt.legend()
plt.show()

plt.figure(figsize=(20,7),dpi=80)
plt.plot(training_set[7:,], color = 'blue', label = 'REAL')
plt.plot(predicted_mort_train, color = 'red', label = 'PRONÓSTICO)', dashes=[6, 2])
plt.title('Entrenamiento de la RNA de la cantidad de fallecidos en Perú')
plt.xlabel('Tiempo')
plt.ylabel('cantidad')
plt.legend()
plt.show()

# y_train = sc.inverse_transform(y_train)

"""---

 #### VISUALIZACION DE RESULTADOS
 

---
"""

# Visualizando resultados
plt.figure(figsize=(20,6),dpi=80)
plt.plot(real_covid, color = 'blue', label = 'fallecidos - (Real)')
plt.plot(predicted_mort, color = 'red', label = 'fallecidos - (Pronóstico)', dashes=[6, 2])
plt.title('Predicción de fallecidos por COVID19-Perú (16 al 22 de enero)')
plt.xlabel('Tiempo')
plt.ylabel('Mortalidad')
plt.legend()
plt.show()

# Visualizando resultados
plt.figure(figsize=(20,6),dpi=80)
plt.plot(real_covid, color = 'blue', label = 'fallecidos - (Real)')
plt.plot(predicted_mort, color = 'red', label = 'fallecidos - (Pronóstico)', dashes=[6, 2])
plt.title('Predicción de fallecidos por COVID19-Perú (16 al 22 de enero)')
plt.xlabel('Tiempo')
plt.ylabel('Mortalidad')
plt.legend()
plt.show()

"""---

 #### RMSE
 

---
"""

# RMSE
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(real_covid, predicted_mort)
rmse = math.sqrt(mse)

import numpy as np #creamos func MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

MAPE = mean_absolute_percentage_error(real_covid, predicted_mort)
MAE = mean_absolute_error(real_covid, predicted_mort)
mse = mean_squared_error(real_covid, predicted_mort)
rmse = math.sqrt(mse)
print ("mse: ", mse)
print ("rmse: ", rmse) 
print ("MAPE: ", MAPE)
print ("MAE: ", MAE)

MAPE = mean_absolute_percentage_error(real_covid, predicted_mort)
MAE = mean_absolute_error(real_covid, predicted_mort)
mse = mean_squared_error(real_covid, predicted_mort)
rmse = math.sqrt(mse)
print ("mse: ", mse)
print ("rmse: ", rmse) 
print ("MAPE: ", MAPE)
print ("MAE: ", MAE)









