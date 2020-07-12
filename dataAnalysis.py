import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, Embedding, TimeDistributed, LeakyReLU, Attention, RepeatVector
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from test import AttentionDecoder
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as signal
import pywt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def wavelet(data):
    # Approximation and detail coefficients.
    (_, cD) = pywt.dwt(data, "db1")
    return cD


def normalize(data):
    data = data.apply(lambda x: wavelet((x - np.mean(x)) / (np.max(x) - np.min(x))))

    return data


def define_bp(data, bp_name, start_frequency, end_frequency):    
    data[bp_name] = 0
    for frequency in range(start_frequency, end_frequency + 1, 1):
        data[bp_name] += data["{}Hz".format(frequency)]

    return data


def get_data(file_path):
    data = pd.read_csv(file_path)
    data.rename({ "0Hz": "label" }, axis=1, inplace=True)
    
    data = data.iloc[1:]
    data = data[int(data.shape[0] * 0.2): int(data.shape[0] * 0.8)]

    # data = clean_outlier(data)
    data = normalize(data)
    
    # selected band
    brainwaves = [
        ["theta", 4, 7],
        ["low_alpha", 8, 9],
    ]

    for brainwave in brainwaves:
        data = define_bp(data, brainwave[0], brainwave[1], brainwave[2])

    data = data[["theta", "low_alpha"]]

    data[abs(data["theta"]) > 0.6] *= 6
    data[abs(data["low_alpha"]) > 0.6] *= 6
    return data

def offset_data(data, offset_index):
    offset_data = []
    for start_index in range(0, data.shape[0] - (data.shape[0] % offset_index), offset_index):
        offset_data.append(np.array(data[start_index:start_index + offset_index]))

    return np.array(offset_data)


def shuffle_data_and_label(data, label):
    np.random.seed(15)
    random_list = np.arange(data.shape[0])
    np.random.shuffle(random_list)
    
    return data[random_list], label[random_list]


def get_train_and_test_data(unfocused_data, focused_data, offset_index):
    unfocused_data = offset_data(unfocused_data, offset_index)
    focused_data = offset_data(focused_data, offset_index)

    data = np.vstack((unfocused_data, focused_data))

    onehotencoder = OneHotEncoder()
    label = onehotencoder.fit_transform([[0]] * unfocused_data.shape[0] + [[1]] * focused_data.shape[0]).toarray()

    data, label = shuffle_data_and_label(data, label)

    split_index = int(data.shape[0] * 0.8)
    x_train = data[:split_index]
    x_test = data[split_index:]

    y_train = label[:split_index]
    y_test = label[split_index:]

    return x_train, y_train, x_test, y_test


def get_model(input_shape):
    model = Sequential()
    model.add(Conv1D(32, 3, input_shape=input_shape[1:]))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(AttentionDecoder(64, 2))
    # model.add(LSTM(64))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Attention([]))
    # model.add(LSTM(64))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.3))
    # # model.add(Dense(128))
    # # model.add(LeakyReLU(alpha=0.4))
    # model.add(Dense(64))
    # model.add(LeakyReLU(alpha=0.4))
    # model.add(Dense(16))
    # model.add(LeakyReLU(alpha=0.4))
    # model.add(Dense(2, activation="softmax", kernel_initializer="he_normal"))
    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
    return model


if __name__ == "__main__":
    names = ["darkfanxing", "kuo", "lin", "tsai"]
    
    unfocused_data, focused_data = np.zeros((1, 2)), np.zeros((1, 2))
    for count in range(1, 3 + 1, 1):
        for name in names:
            unfocused_data = np.vstack((unfocused_data, get_data("./data/unfocused/{}_{}.csv".format(name, count)).values))
            focused_data = np.vstack((focused_data, get_data("./data/focused/{}_{}.csv".format(name, count)).values))

    unfocused_data = np.vstack((unfocused_data, get_data("./data/unfocused/darkfanxing_4.csv").values))
    focused_data = np.vstack((focused_data, get_data("./data/focused/darkfanxing_4.csv").values))
    unfocused_data, focused_data = unfocused_data[1:], focused_data[1:]

    # plt.subplot(411)
    # sns.lineplot(x=[i for i in range(unfocused_data.shape[0])], y=unfocused_data[:, 0])

    # plt.subplot(412)
    # sns.lineplot(x=[i for i in range(unfocused_data.shape[0])], y=unfocused_data[:, 1])
    
    # plt.subplot(413)
    # sns.lineplot(x=[i for i in range(focused_data.shape[0])], y=focused_data[:, 0])
    
    # plt.subplot(414)
    # sns.lineplot(x=[i for i in range(focused_data.shape[0])], y=focused_data[:, 1])
    
    # plt.show()
    # plt.clf()

    x_train, y_train, x_test, y_test = get_train_and_test_data(unfocused_data, focused_data, 10)
    model = get_model(x_train.shape)
    callback = EarlyStopping(monitor="loss", patience=10, verbose=2, mode="auto")
    model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[callback], verbose=2)
    
    model.evaluate(x_test, y_test)
    # predictions = np.argmax(model.predict(x_test), axis=-1)
    
    # accuracy = get_accuracy(predictions, y_test)
    
    # print(accuracy)