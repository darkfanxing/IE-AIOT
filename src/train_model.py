import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from get_model import get_cnn_rnn_model, get_cnn_model
import random
from tensorflow.python.framework import ops
from keras import backend as K
from utility import get_data, change_to_sequence_data, signal_filter

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

def shuffle_data_and_label(data, label):
    np.random.seed(20)
    random_list = np.arange(data.shape[0])
    np.random.shuffle(random_list)
    
    return data[random_list], label[random_list]

def get_theta_and_low_alpha_band_data(data):
    data = data.reshape((data.shape[0] * 16, 16))
    fft_data = np.abs(np.fft.fft(data)).astype(int)
    
    band_data = np.zeros((fft_data.shape[0], 2))
    for i in range(fft_data.shape[0]):
        band_data[i, 0] = fft_data[i, 4:7 + 1].sum()
        band_data[i, 1] = fft_data[i, 8:9 + 1].sum()

    return band_data


def get_train_and_test_data(unfocused_data, focused_data, offset_index):
    for i in range(unfocused_data.shape[0]):
        unfocused_data[i] = signal_filter(unfocused_data[i])
    
    for i in range(focused_data.shape[0]):
        focused_data[i] = signal_filter(focused_data[i])

    unfocused_data = unfocused_data.reshape((unfocused_data.shape[0] * 3, 256))
    focused_data = focused_data.reshape((focused_data.shape[0] * 3, 256))

    unfocused_data = get_theta_and_low_alpha_band_data(unfocused_data)
    focused_data = get_theta_and_low_alpha_band_data(focused_data)

    unfocused_data = change_to_sequence_data(unfocused_data, offset_index)
    focused_data = change_to_sequence_data(focused_data, offset_index)

    data = np.vstack((unfocused_data, focused_data))


    label = np.array([0] * unfocused_data.shape[0] + [1] * focused_data.shape[0])

    data, label = shuffle_data_and_label(data, label)

    split_index = int(data.shape[0] * 0.85)
    x_train, y_train = data[:split_index], label[:split_index]
    x_test, y_test = data[split_index:], label[split_index:]

    return x_train, y_train, x_test, y_test


def get_accuracy(predictions, label):
    test = predictions
    test[test >= 0.5] = 1
    test[test < 0.5] = 0
    count = 0
    for index in range(len(label)):
        if label[index] == test[index][0]:
            count += 1

    return count / len(label)


def get_unfocused_and_focused_data_files_path():
    focused_files_path = os.listdir(os.getcwd() + "/data/new_focused/")
    unfocused_files_path = os.listdir(os.getcwd() + "/data/new_unfocused/")
    
    return focused_files_path, unfocused_files_path

def get_unfocused_and_focused_data():
    unfocused_data, focused_data = np.zeros((1, 768)), np.zeros((1, 768))
    focused_files_path, unfocused_files_path = get_unfocused_and_focused_data_files_path()
    
    for file_path in unfocused_files_path:
        data = get_data("./data/new_unfocused/{}".format(file_path))
        unfocused_data = np.vstack((unfocused_data, data))
    for file_path in focused_files_path:
        focused_data = np.vstack((focused_data, get_data("./data/new_focused/{}".format(file_path))))

    unfocused_data, focused_data = unfocused_data[1:], focused_data[1:]
    return unfocused_data, focused_data


def save_model(model_name, accuracy):
    model_path = os.getcwd() + "/model/"
    architecture_name = "{}.json".format(model_name)

    if architecture_name not in os.listdir(model_path):
        with open(model_path + architecture_name, "w") as f:
            f.write(model.to_json())

    model.save_weights('model/{}-{}({}).h5'.format(model_name, int(accuracy), random.randint(-9999, 9999)))

if __name__ == "__main__":
    unfocused_data, focused_data = get_unfocused_and_focused_data()
    x_train, y_train, x_test, y_test = get_train_and_test_data(unfocused_data, focused_data, 16)

    for _ in range(50):
        model_name, model = get_cnn_rnn_model(x_train.shape[1:])

        callback = EarlyStopping(monitor="loss", patience=20, verbose=2, mode="auto")
        model.fit(x_train, y_train, epochs=100, batch_size=64, callbacks=[callback], verbose=1)
        
        predictions = model.predict(x_test)
        accuracy = round(get_accuracy(predictions, y_test), 2) * 100
        print("Model accuracy is {}%".format(accuracy))

        model_path = os.getcwd() + "/model/"
        architecture_name = "{}.json".format(model_name)

        if architecture_name not in os.listdir(model_path):
            with open(model_path + architecture_name, "w") as f:
                f.write(model.to_json())

        model.save_weights('model/{}-{}({}).h5'.format(model_name, int(accuracy), random.randint(-9999, 9999)))

    #     # # save_model(model_name, accuracy)
    #     # K.clear_session()
    #     # ops.reset_default_graph()

    #     # print(predictions[:20])

    #     # sns.lineplot(x=[time for time in range(predictions.shape[0])], y=predictions[:, 0]).set(xlabel="time(s)", ylabel="focused_score")
    #     # plt.show()