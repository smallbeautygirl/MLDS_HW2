import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import nan
import os
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import MinMaxScaler
import glob

training_data_dir = './training/training_data'

all_files = glob.glob(os.path.join(training_data_dir, "*.csv"))

df = pd.concat((pd.read_csv(f, parse_dates=True)
               for f in all_files), ignore_index=True)
df.set_index(['time'])
print(df.head())
print(df.tail())
print(type(df))
# columns=['time','generation','consumption']
# for csv_file in os.listdir(training_data_dir):
#     data = pd.read_csv(os.path.join(training_data_dir, csv_file),
#                        sep=';', parse_dates=True, low_memory=False)
print(df.describe())
print(123)
print(df.head())
print(df.info())
print(df.shape)

df.to_csv(os.path.join(training_data_dir, 'cleaned_data.csv'))
# print(df.loc[:'generation'])
# print(df['2018-01-01 00:00:00'])
data_train = df.loc[:'2018-08-31 23:00:00':]['consumption']
data_train = np.array(data_train)

data_train2 = df.loc[:'2018-08-31 23:00:00':]['generation']
data_train2 = np.array(data_train)

x_train, y_train = [], []
x_train2, y_train2 = [], []

# 24 hours
for i in range(24, len(data_train) - 24):
    x_train.append(data_train[i - 24: i])
    y_train.append(data_train[i: i + 24])
    
for i in range(24, len(data_train2) - 24):
    x_train2.append(data_train2[i - 24: i])
    y_train2.append(data_train2[i: i + 24])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train2, y_train2 = np.array(x_train2), np.array(y_train2)
print(x_train.shape, y_train.shape)

# Normalize dataset between 0 and 1 with MinMaxScaler
# x_scaler = MinMaxScaler()
# X_train = x_scaler.fit_transform(x_train)

# y_scaler = MinMaxScaler()
# y_train = y_scaler.fit_transform(y_train)

reg = Sequential()
reg.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
reg.add(Dense(1))

reg.compile(loss='mse', optimizer='adam')

reg.fit(x_train, y_train, epochs=1)

reg.save('consumption_model.h5')

reg2 = Sequential()
reg2.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
reg2.add(Dense(1))

reg2.compile(loss='mse', optimizer='adam')

reg2.fit(x_train2, y_train2, epochs=1)

reg2.save('generation_model.h5')
