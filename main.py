import pandas as pd
import numpy as np
from datetime import datetime,timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
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
            total_score = total_score + \
                (y_true[row, col] - y_predicted[row, col])**2
    total_score = np.sqrt(total_score//(y_true.shape[0]*y_predicted.shape[1]))

    return total_score, scores


if __name__ == "__main__":
    args = config()
    df = pd.read_csv('./training/cleaned_data.csv', parse_dates=True)
    df = df.set_index(['time'])
    print(df.head())
    print(df['consumption'])
    data_train = df['consumption']
    print(data_train)
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
    print(x_train.shape, y_train.shape)
    print(x_train2.shape, y_train2.shape)

    # Normalize dataset between 0 and 1 with MinMaxScaler
    x_scaler = MinMaxScaler()
    x_train = x_scaler.fit_transform(x_train)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(y_train)

    print(x_train.shape[0])
    x_train = x_train.reshape(x_train.shape[0], 24, 1)
    print(x_train.shape)
    ###
    x_scaler2 = MinMaxScaler()
    x_train2 = x_scaler2.fit_transform(x_train2)

    y_scaler2 = MinMaxScaler()
    y_train2 = y_scaler2.fit_transform(y_train2)

    x_train2 = x_train2.reshape(x_train2.shape[0], 24, 1)
    print(x_train2.shape)

# reg = Sequential()
# reg.add(LSTM(units=200, activation='relu', input_shape=(24, 1)))
# reg.add(Dense(24))

# reg.compile(loss='mse', optimizer='adam')

# reg.fit(x_train, y_train, epochs=1)

# reg.save('consumption_model.h5')

    consumption_data = pd.read_csv(
        args.consumption, parse_dates=True)
    generation_data = pd.read_csv(
        args.generation, parse_dates=True)
    consumption_data = consumption_data.set_index(['time'])
    consumption_data = consumption_data['consumption']
    
    generation_data_time=generation_data['time']
    generation_data = generation_data.set_index(['time'])
    generation_data = generation_data['generation']
    # print('consumption_data',consumption_data)
    # print('generation_data',generation_data)
    consumption_data = np.array(consumption_data)
    generation_data = np.array(generation_data)
    x_test_c, y_test_c = [], []
    x_test_g, y_test_g = [], []

    for i in range(24, len(consumption_data)-24):
        x_test_c.append(consumption_data[i-24:i])
        y_test_c.append(consumption_data[i:i+24])
        
    for i in range(24, len(generation_data)-24):
        x_test_g.append(generation_data[i-24:i])
        y_test_g.append(generation_data[i:i+24])

    x_test_c = x_scaler.transform(x_test_c)
    y_test_c = y_scaler.transform(y_test_c)
    
    x_test_g = x_scaler2.fit_transform(x_test_g)
    y_test_g = y_scaler2.fit_transform(y_test_g)

    x_test_c = x_test_c.reshape(x_test_c.shape[0], 24, 1)
    x_test_g = x_test_g.reshape(x_test_g.shape[0], 24, 1)

    consumption_model = tf.keras.models.load_model('consumption_model.h5')
    generation_model = tf.keras.models.load_model('generation_model.h5')
    # Show the model architecture
    print(consumption_model.summary())
    print('156---------------------------')
    print('x_test_c',x_test_c)
    print('x_test_g',x_test_g)
    y_pred_c = consumption_model.predict(x_test_c)
    y_pred_g = generation_model.predict(x_test_g)
    print('162---------------------------')
    print('y_pred_c',y_pred_c)
    print('y_pred_g',y_pred_g)

    y_pred = y_scaler.inverse_transform(y_pred_c)
    y_pred2 = y_scaler2.inverse_transform(y_pred_g) #這邊的訓練模型有錯 所以會跑不過
    print('---------------------------')
    print('y_pred_c',y_pred_c)
    print('y_pred_g',y_pred_g)

    y_true = y_scaler.inverse_transform(y_test_c)
    y_true2 = y_scaler2.inverse_transform(y_test_g)
    # print('---------------------------')
    # print('y_true',y_true)
    # print('y_true2',y_true2)

    # print('---------------------------')
    # print(evaluate_model(y_true, y_pred))
    # print(evaluate_model(y_true2, y_pred2))

    print('---------------------------')
    print('y_true',np.std(y_true[0])) #預測會用多少電
    print('y_true2',np.std(y_true2[0])) #預測會產多少電

    last_day =generation_data_time[len(generation_data_time)-1]
    date = last_day.split(' ')
    day = datetime.strptime(date[0], '%Y-%m-%d').date()+timedelta(days=1)
    day_s = datetime.strftime(day,'%Y-%m-%d') #歷史資料的最後一天+1

    data = []
    for i in range(1,24):
        print("y_true[i]",np.std(y_true[i]))
        print("y_true2[i]",np.std(y_true2[i]))
        # 用電大於產電，要買電
        if float(np.std(y_true[i]))>float(np.std(y_true2[i])):
            data.append([day_s+" "+str(i)+":00:00", "buy", 1.8, float(np.std(y_true[i]))-float(np.std(y_true2[i]))])
        elif float(np.std(y_true[i]))<float(np.std(y_true2[i])):
            data.append([day_s+" "+str(i)+":00:00", "sell", 2, float(np.std(y_true2[i]))-float(np.std(y_true[i]))])

    output(args.output, data)
