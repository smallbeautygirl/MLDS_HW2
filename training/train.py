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


def training_consumption(df):
    data_train_c = df['consumption']
    data_train_c = np.array(data_train_c)

    x_train, y_train = [], []

    # 24*7 = 168 hours
    for i in range(24, len(data_train_c) - 24):
        x_train.append(data_train_c[i - 24: i])
        y_train.append(data_train_c[i: i + 24])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Normalize dataset between 0 and 1 with MinMaxScaler
    x_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)

    x_train = x_train.reshape(x_train.shape[0], 24, 1)

    # training
    reg = Sequential()
    reg.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
    reg.add(Dense(24))

    reg.compile(loss='mse', optimizer='adam')

    reg.fit(x_train, y_train, epochs=1)

    reg.save('consumption_model.h5')


def training_generation(df):
    data_train_g = df['generation']
    data_train_g = np.array(data_train_g)

    x_train, y_train = [], []

    # 24*7 = 168 hours
    for i in range(24, len(data_train_g) - 24):
        x_train.append(data_train_g[i - 24: i])
        y_train.append(data_train_g[i: i + 24])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Normalize dataset between 0 and 1 with MinMaxScaler
    x_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)

    x_train = x_train.reshape(x_train.shape[0], 24, 1)

    # training
    reg2 = Sequential()
    reg2.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
    reg2.add(Dense(24))

    reg2.compile(loss='mse', optimizer='adam')

    reg2.fit(x_train, y_train, epochs=1)

    reg2.save('generation_model.h5')


if __name__ == "__main__":
    # training_data_dir = './training/training_data'

    # all_files = glob.glob(os.path.join(training_data_dir, "*.csv"))

    # df = pd.concat((pd.read_csv(f, parse_dates=True)
    #                for f in all_files), ignore_index=True)
    # df.set_index(['time'])

    # print(df.head())
    # print(df.tail())
    # print(type(df))
    # # columns=['time','generation','consumption']
    # # for csv_file in os.listdir(training_data_dir):
    # #     data = pd.read_csv(os.path.join(training_data_dir, csv_file),
    # #                        sep=';', parse_dates=True, low_memory=False)
    # print(df.describe())
    # print(123)
    # print(df.head())
    # print(df.info())
    # print(df.shape)

    # df.to_csv('./training/cleaned_data.csv')
    df = pd.read_csv('./training/cleaned_data.csv', parse_dates=True)
    df = df.set_index(['time'])

    training_consumption(df)
    training_generation(df)
