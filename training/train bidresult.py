import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import glob
import tensorflow as tf

df = pd.read_csv('./sample_data/bidresult.csv', parse_dates=True)
df = df.set_index(['time'])
# print(df)
data_train = df['target_price']
data_train = np.array(data_train)
print(data_train)
x_train, y_train = [], []

for i in range(7, len(data_train) -7):
    print(i)
    x_train.append(data_train[i - 7: i])
    y_train.append(data_train[i: i + 7])

x_train, y_train = np.array(x_train), np.array(y_train)
# Normalize dataset between 0 and 1 with MinMaxScaler
x_scaler = MinMaxScaler()
x_train = x_scaler.fit_transform(x_train)

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)

x_train = x_train.reshape(x_train.shape[0], 7, 1)

reg = Sequential()
reg.add(LSTM(units=200, activation='relu', input_shape=(7, 1)))
reg.add(Dense(7))

reg.compile(loss='mse', optimizer='adam')

reg.fit(x_train, y_train, epochs=1)

reg.save('bidresult_model.h5')
