import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import random
from datetime import datetime, timedelta
#################################################################################################################################
# You should not modify this part.


def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv",
                        help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv",
                        help="input the generation data path")
    parser.add_argument(
        "--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv",
                        help="output the bids path")

    return parser.parse_args()


def output(path, data):
    import pandas as pd

    df = pd.DataFrame(
        data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return
# You should not modify this part.
#################################################################################################################################


def evaluate_model(last_data, y_true_c, y_predicted_c, y_true_g, y_predicted_g):
    scores = []

    output_data = []
    action = None
    # calculate scores for each day
    for i in range(y_true_c.shape[1]):
        mse_c = mean_squared_error(y_true_c[:, i], y_predicted_c[:, i])
        mse_g = mean_squared_error(y_true_g[:, i], y_predicted_g[:, i])
        rmse_c = np.sqrt(mse_c)
        # rmse_g = np.sqrt(mse_g)
        # scores.append(rmse_c)
        rmse_g = random.uniform(0.0, 3.2)

        if rmse_g > rmse_c or (i >= 10 and i <= 15):
            action = "sell"
            price = random.uniform(0.7, 2.5)
        else:
            action = "buy"
            price = random.uniform(
                1.15, 2.7)

        output_data.append([
            f"{last_data+timedelta(days=1)+timedelta(hours=i)}", action, price, abs(
                rmse_g-rmse_c)
        ])

    return output_data
    # calculate score for whole prediction
    # total_score = 0
    # for row in range(y_true_c.shape[0]):
    #     for col in range(y_predicted_c.shape[1]):
    #         total_score = total_score + \
    #             (y_true_c[row, col] - y_predicted_c[row, col])**2
    # total_score = np.sqrt(total_score//(y_true_c.shape[0]*y_predicted_c.shape[1]))

    # return total_score, scores


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
    # reg = Sequential()
    # reg.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
    # reg.add(Dense(24))

    # reg.compile(loss='mse', optimizer='adam')

    # reg.fit(x_train, y_train, epochs=1)

    # reg.save('consumption_model.h5')
    return x_scaler, y_scaler


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
    # reg2 = Sequential()
    # reg2.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
    # reg2.add(Dense(24))

    # reg2.compile(loss='mse', optimizer='adam')

    # reg2.fit(x_train, y_train, epochs=1)

    # reg2.save('generation_model.h5')
    return x_scaler, y_scaler


def test_consumption(consumption_data, x_scaler, y_scaler):
    x_test_c, y_test_c = [], []

    # Split test data by day (24 hours)
    for i in range(24, len(consumption_data)-24):
        x_test_c.append(consumption_data[i-24:i])
        y_test_c.append(consumption_data[i:i+24])

    x_test_c, y_test_c = np.array(x_test_c), np.array(y_test_c)

    x_test_c = x_scaler.transform(x_test_c)
    y_test_c = y_scaler.transform(y_test_c)

    x_test_c = x_test_c.reshape(x_test_c.shape[0], 24, 1)

    consumption_model = tf.keras.models.load_model(
        'consumption_model_ep_60.h5')
    print(consumption_model.summary())
    y_pred_c = consumption_model.predict(x_test_c)
    y_pred_c = y_scaler.inverse_transform(y_pred_c)
    y_true_c = y_scaler.inverse_transform(y_test_c)

    return y_pred_c, y_true_c


def test_generation(generation_data, x_scaler, y_scaler):
    x_test_g, y_test_g = [], []
    # Split test data by day (24 hours)
    for i in range(24, len(generation_data)-24):
        x_test_g.append(generation_data[i-24:i])
        y_test_g.append(generation_data[i:i+24])

    x_test_g, y_test_g = np.array(x_test_g), np.array(y_test_g)

    x_test_g = x_scaler.transform(x_test_g)
    y_test_g = y_scaler.transform(y_test_g)

    x_test_g = x_test_g.reshape(x_test_g.shape[0], 24, 1)

    generation_model = tf.keras.models.load_model('generation_model_ep_60.h5')

    y_pred_g = generation_model.predict(x_test_g)
    y_pred_g = y_scaler.inverse_transform(y_pred_g)
    y_true_g = y_scaler.inverse_transform(y_pred_g)

    return y_pred_g, y_true_g


if __name__ == "__main__":
    args = config()

    df = pd.read_csv('./training/cleaned_data.csv', parse_dates=True)
    df = df.set_index(['time'])

    x_scaler_c, y_scaler_c = training_consumption(df)
    x_scaler_g, y_scaler_g = training_generation(df)

    # consumption
    consumption_data = pd.read_csv(
        args.consumption, parse_dates=True)
    consumption_data = consumption_data.set_index(['time'])
    consumption_data = consumption_data['consumption']
    consumption_data = np.array(consumption_data)

    # generation
    generation_data = pd.read_csv(
        args.generation, parse_dates=True)

    print(generation_data.iloc[-1].tolist())
    last_data = generation_data.iloc[-1].tolist()[0]
    print(last_data.split(' ')[0])
    last_data = datetime.strptime(last_data.split(' ')[0], "%Y-%m-%d")
    print(last_data)
    generation_data = generation_data.set_index(['time'])
    generation_data = generation_data['generation']

    generation_data = np.array(generation_data)

    y_pred_c, y_true_c = test_consumption(
        consumption_data, x_scaler_c, y_scaler_c)
    y_pred_g, y_true_g = test_generation(
        generation_data, x_scaler_g, y_scaler_g)

    result = evaluate_model(last_data, y_true_c, y_pred_c, y_true_g, y_pred_g)
    output(args.output, result)
    # result = evaluate_model(y_true_g, y_pred_g)
    # print(result)

    # print(np.std(y_true_c[0]))
    # print(len(result[1]))

    # print(np.std(y_true_g[0]))
    # print(len(result[1]))
