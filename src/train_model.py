import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping
from get_model import get_cnn_rnn_model, get_cnn_model
import random
from tensorflow.python.framework import ops
from keras import backend as K
from utility import get_data, change_to_sequence_data

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


def get_train_and_test_data(unfocused_data, focused_data, offset_index):
    unfocused_data = change_to_sequence_data(unfocused_data, offset_index)
    focused_data = change_to_sequence_data(focused_data, offset_index)

    data = np.vstack((unfocused_data, focused_data))

    onehotencoder = OneHotEncoder()
    label = onehotencoder.fit_transform([[0]] * unfocused_data.shape[0] + [[1]] * focused_data.shape[0]).toarray()

    data, label = shuffle_data_and_label(data, label)

    split_index = int(data.shape[0] * 0.8)
    x_train, y_train = data[:split_index], label[:split_index]
    x_test, y_test = data[split_index:], label[split_index:]

    return x_train, y_train, x_test, y_test


def get_accuracy(predictions, label):
    count = 0
    for index in range(len(label)):
        if label[0, predictions[index]] == 1:
            count += 1

    return count / len(label)


def get_unfocused_and_focused_data_files_path():
    focused_files_path = os.listdir(os.getcwd() + "/data/focused/")
    unfocused_files_path = os.listdir(os.getcwd() + "/data/unfocused/")
    
    return focused_files_path, unfocused_files_path

def get_unfocused_and_focused_data():
    unfocused_data, focused_data = np.zeros((1, 2)), np.zeros((1, 2))
    focused_files_path, unfocused_files_path = get_unfocused_and_focused_data_files_path()
    
    for file_path in unfocused_files_path:
        unfocused_data = np.vstack((unfocused_data, get_data("./data/unfocused/{}".format(file_path), True)))
    for file_path in focused_files_path:
        focused_data = np.vstack((focused_data, get_data("./data/focused/{}".format(file_path), True)))

    unfocused_data, focused_data = unfocused_data[1:], focused_data[1:]
    return unfocused_data, focused_data


def save_model(model_name):
    model_path = os.getcwd() + "/model/"
    architecture_name = "{}.json".format(model_name)

    if architecture_name not in os.listdir(model_path):
        with open(model_path + architecture_name, "w") as f:
            f.write(model.to_json())

    model.save_weights('model/{}-{}({}).h5'.format(model_name, int(accuracy), random.randint(-9999, 9999)))

if __name__ == "__main__":
    unfocused_data, focused_data = get_unfocused_and_focused_data()
    print(unfocused_data[:2])
    print(unfocused_data.shape, focused_data.shape)
    x_train, y_train, x_test, y_test = get_train_and_test_data(unfocused_data, focused_data, 10)

    for _ in range(50):
        model_name, model = get_cnn_rnn_model(x_train.shape[1:])
        callback = EarlyStopping(monitor="loss", patience=30, verbose=2, mode="auto")
        model.fit(x_train, y_train, epochs=1000, batch_size=64, callbacks=[callback], verbose=1)
        
        predictions = np.argmax(model.predict(x_test), axis=-1)
        accuracy = round(get_accuracy(predictions, y_test), 2) * 100
        print("Model accuracy is {}%".format(accuracy))

        save_model(model_name)
        K.clear_session()
        ops.reset_default_graph()