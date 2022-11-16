import numpy as np
from numpy import nan
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def fill_missing(data):
    one_day = 24*60
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if np.isnan(data[row, col]):
                data[row, col] = data[row-one_day, col]

def evaluate_model(y_true, y_predicted):
    scores = []

    # calculate scores for each day
    for i in range(y_true.shape[1]):
        mse = mean_squared_error(y_true[:, i], y_predicted[:, i])
        rmse = np.sqrt(mse)
        scores.append(rmse)

    # calculate score for whole prediction
    total_score = 0
    for row in range(y_true.shape[0]):
        for col in range(y_predicted.shape[1]):
            total_score = total_score + (y_true[row, col] - y_predicted[row, col])**2
    total_score = np.sqrt(total_score//(y_true.shape[0]*y_predicted.shape[1]))

    return total_score, scores
# data = pd.read_csv('household_power_consumption.txt', sep=';', parse_dates=True, low_memory=False)

data2 = pd.read_csv('sample_data/consumption.csv', sep=',', parse_dates=True, low_memory=False)

data2.set_index(['time'],inplace=True)
data2= data2.astype('float')

# print(np.isnan(data).sum())

# fill_missing(data.values)
fill_missing(data2.values)
# data.to_csv('cleaned_data.csv')
# dataset = pd.read_csv('cleaned_data.csv', parse_dates=True,
#                       index_col='date_time', low_memory=False)
# data = dataset.resample('D').sum()
# data_train = data.loc[:'2009-12-31', :]['Global_active_power']
# print(data_train)
# print('---------------------')
data_train = data2.loc[:, :]['consumption']
# print(data_train)
# data_test = data['2010']['Global_active_power']

data_train = np.array(data_train)

# data_test = np.array(data_test)
X_train, y_train = [], []
# X_test, y_test = [], []

for i in range(24, len(data_train) - 24):
    X_train.append(data_train[i - 24:i])
    y_train.append(data_train[i:i + 24])
    
# for i in range(7, len(data_test) - 7):
#     X_test.append(data_test[i - 7:i])
#     y_test.append(data_test[i:i + 7])


X_train, y_train = np.array(X_train), np.array(y_train)
# X_test, y_test = np.array(X_test), np.array(y_test)

print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

x_scaler = MinMaxScaler()
X_train = x_scaler.fit_transform(X_train)

y_scaler = MinMaxScaler()
y_train = y_scaler.fit_transform(y_train)

# X_test = x_scaler.transform(X_test)
# y_test = y_scaler.transform(y_test)

print(X_train)
X_train = X_train.reshape(120, 24, 1)
print('------------------')
print(X_train)
reg = Sequential()
reg.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
reg.add(Dense(24))

reg.compile(loss='mse', optimizer='adam')
reg.fit(X_train, y_train, epochs=100)


# # # 將模型儲存至 HDF5 檔案中
reg.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# X_test = X_test.reshape(331, 7, 1)

# y_pred = reg.predict(X_test)

# y_pred = y_scaler.inverse_transform(y_pred)
# print(y_pred)
# print('==================')
# y_true = y_scaler.inverse_transform(y_test)
# print(y_true)
# print('==================')
# evaluate_model(y_true, y_pred)

# print(np.std(y_true[0]))
