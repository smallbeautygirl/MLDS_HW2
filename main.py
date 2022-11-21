import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
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
        rmse_g = np.sqrt(mse_g)
        # scores.append(rmse_c)
        action
        if rmse_g > rmse_c:
            action = "sell"
            price = random.uniform(1.15, 2.7)
        else:
            action = "buy"
            price = random.uniform(
                1.15, 2.7)
        hour = f'0{i}' if len(str(i)) == i else f'{i}'

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


if __name__ == "__main__":
    args = config()

    df = pd.read_csv('./training/cleaned_data.csv', parse_dates=True)
    df = df.set_index(['time'])

    data_train = df['consumption']
    data_train = np.array(data_train)

    data_train2 = df['generation']
    data_train2 = np.array(data_train2)

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

    # Normalize dataset between 0 and 1 with MinMaxScaler
    x_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)

    x_train = x_train.reshape(x_train.shape[0], 24, 1)

    ###
    x_scaler2 = MinMaxScaler()
    x_train2 = x_scaler2.fit_transform(x_train2)

    y_scaler2 = MinMaxScaler()
    y_train2 = y_scaler2.fit_transform(y_train2)

    x_train2 = x_train2.reshape(x_train2.shape[0], 24, 1)

# reg = Sequential()
# reg.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
# reg.add(Dense(24))

# reg.compile(loss='mse', optimizer='adam')

# reg.fit(x_train, y_train, epochs=1)

# reg.save('consumption_model.h5')

    consumption_data = pd.read_csv(
        args.consumption, parse_dates=True)
    consumption_data = consumption_data.set_index(['time'])
    consumption_data = consumption_data['consumption']

    consumption_data = np.array(consumption_data)

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

    x_test_c, y_test_c = [], []

    # Split test data by day (24 hours)
    for i in range(24, len(consumption_data)-24):
        x_test_c.append(consumption_data[i-24:i])
        y_test_c.append(consumption_data[i:i+24])

    x_test_c, y_test_c = np.array(x_test_c), np.array(y_test_c)

    x_test_c = x_scaler.transform(x_test_c)
    y_test_c = y_scaler.transform(y_test_c)

    x_test_c = x_test_c.reshape(x_test_c.shape[0], 24, 1)

    x_test_g, y_test_g = [], []
    # Split test data by day (24 hours)
    for i in range(24, len(generation_data)-24):
        x_test_g.append(generation_data[i-24:i])
        y_test_g.append(generation_data[i:i+24])

    x_test_g, y_test_g = np.array(x_test_g), np.array(y_test_g)

    x_test_g = x_scaler2.transform(x_test_g)
    y_test_g = y_scaler2.transform(y_test_g)

    x_test_g = x_test_g.reshape(x_test_g.shape[0], 24, 1)

    consumption_model = tf.keras.models.load_model('consumption_model.h5')
    generation_model = tf.keras.models.load_model('generation_model.h5')
    # Show the model architecture
    # print(consumption_model.summary())
    y_pred_c = consumption_model.predict(x_test_c)
    y_pred_g = generation_model.predict(x_test_g)

    y_pred_c = y_scaler.inverse_transform(y_pred_c)
    y_pred_g = y_scaler2.inverse_transform(y_pred_g)

    y_true_c = y_scaler.inverse_transform(y_test_c)
    y_true_g = y_scaler2.inverse_transform(y_pred_g)

    result = evaluate_model(last_data, y_true_c, y_pred_c, y_true_g, y_pred_g)
    print(result)
    output(args.output, result)
    # result = evaluate_model(y_true_g, y_pred_g)
    # print(result)

    # print(np.std(y_true_c[0]))
    # print(len(result[1]))

    # print(np.std(y_true_g[0]))
    # print(len(result[1]))
