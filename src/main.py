import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from get_model import get_cnn_rnn_model
import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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
    np.random.seed(20)
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
    x_train, y_train = data[:split_index], label[:split_index]
    x_test, y_test = data[split_index:], label[split_index:]

    return x_train, y_train, x_test, y_test



    # model = Sequential()
    # model.add(Conv1D(64, 5, activation="relu"))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Dense(128))
    # model.add(LeakyReLU(alpha=0.4))
    # model.add(LSTM(64))
    # # model.add(LeakyReLU(alpha=0.2))
    
    # model.add(Dropout(0.3))
    # model.add(Dense(128))
    # model.add(LeakyReLU(alpha=0.4))
    # model.add(Dense(64))
    # model.add(LeakyReLU(alpha=0.4))
    # model.add(Dense(16))
    # model.add(LeakyReLU(alpha=0.4))
    # model.add(Dense(2, activation="softmax", kernel_initializer="he_normal"))
    # model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001, decay=1e-6), metrics=['accuracy'])
    # return model

def get_accuracy(predictions, label):
    count = 0
    for index in range(len(label)):
        if label[0, predictions[index]] == 1:
            count += 1

    return count / len(label)


def get_unfocused_and_focused_data():
    names = ["darkfanxing", "kuo", "lin", "tsai"]

    unfocused_data, focused_data = np.zeros((1, 2)), np.zeros((1, 2))
    for count in range(1, 3 + 1, 1):
        for name in names:
            unfocused_data = np.vstack((unfocused_data, get_data("./data/unfocused/{}_{}.csv".format(name, count)).values))
            focused_data = np.vstack((focused_data, get_data("./data/focused/{}_{}.csv".format(name, count)).values))

    unfocused_data = np.vstack((unfocused_data, get_data("./data/unfocused/darkfanxing_4.csv").values))
    focused_data = np.vstack((focused_data, get_data("./data/focused/darkfanxing_4.csv").values))
    unfocused_data, focused_data = unfocused_data[1:], focused_data[1:]

    return unfocused_data, focused_data


def save_model(model_name):
    model_path = os.getcwd() + "/model/"
    architecture_name = "{}.json".format(model_name)

    if architecture_name not in os.listdir(model_path):
        with open(model_path + architecture_name, "w") as f:
            f.write(model.to_json())

    model.save_weights('model/{}-{}({}).h5'.format(model_name, int(accuracy), random.randint(-999, 999)))

if __name__ == "__main__":
    unfocused_data, focused_data = get_unfocused_and_focused_data()
    x_train, y_train, x_test, y_test = get_train_and_test_data(unfocused_data, focused_data, 10)

    model_name, model = get_cnn_rnn_model()
    callback = EarlyStopping(monitor="loss", patience=25, verbose=2, mode="auto")
    model.fit(x_train, y_train, epochs=1000, batch_size=32, callbacks=[callback], verbose=1)
    
    predictions = np.argmax(model.predict(x_test), axis=-1)
    accuracy = round(get_accuracy(predictions, y_test), 2) * 100
    print("Model accuracy is {}%".format(accuracy))

    save_model(model_name)