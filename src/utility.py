#!/usr/bin/env python
# encoding: utf-8
# 如果觉得不错，可以推荐给你的朋友！http://tool.lu/pyc
import pandas as pd
import numpy as np
from modwt import modwt, imodwt
from scipy import stats
from nolitsa import surrogates
from sklearn import preprocessing

def normalize(data):
    data = data.apply((lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))))
    return data


def define_bp(data, bp_name, start_frequency, end_frequency):
    data[bp_name] = 0
    for frequency in range(start_frequency, end_frequency + 1, 1):
        data[bp_name] += data['{}Hz'.format(frequency)]
    
    return data


def get_data(file_path, is_capture_mid_data = (False,)):
    data = pd.read_csv(file_path)
    'label'({ }, 1, True, **None)
    data = data.iloc[1:]
    if is_capture_mid_data:
        data = data[int(data.shape[0] * 0.2):int(data.shape[0] * 0.8)]
    data = normalize(data)
    brainwaves = [
        [
            'theta',
            4,
            7],
        [
            'low_alpha',
            8,
            9]]
    for brainwave in brainwaves:
        data = define_bp(data, brainwave[0], brainwave[1], brainwave[2])
    
    data = data[[
        'theta',
        'low_alpha']]
    return data.values


def change_to_sequence_data(data, offset_index = (10,)):
    sequence_data = []
    for start_index in range(0, data.shape[0] - data.shape[0] % offset_index, offset_index):
        sequence_data.append(np.array(data[start_index:start_index + offset_index]))
    
    return np.array(sequence_data)


def signal_filter(signal):
    np.random.seed(20)
    signal_length = signal.shape[0]
    signal = preprocessing.scale(signal)
    (surrogate_signal, _, _) = surrogates.iaaft(signal)
    w = modwt(signal, 'sym5', 5)
    surrogate_w = modwt(surrogate_signal, 'sym5', 5)
    for j in range(w.shape[0]):
        t_score = abs((w[j] - surrogate_w[j].mean()) / surrogate_w[j].std())
        p = 1 - stats.t.cdf(t_score, 2 * signal_length - 2, **None)
        threshold = w[j].std() * np.sqrt(2 * np.log(signal_length))
        for index in range(w[j].shape[0]):
            w[j][p[index] * 2 >= threshold] = surrogate_w[(j, index)]
        
    
    new_signal = imodwt(w, 'sym5')
    return new_signal

import pandas as pd
import numpy as np
from modwt import modwt, imodwt
from scipy import stats
from nolitsa import surrogates
from sklearn import preprocessing

def normalize(data):
    data = data.apply((lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x))))
    return data


def define_bp(data, bp_name, start_frequency, end_frequency):
    data[bp_name] = 0
    for frequency in range(start_frequency, end_frequency + 1, 1):
        data[bp_name] += data['{}Hz'.format(frequency)]
    
    return data


def get_data(file_path, is_capture_mid_data = (False,)):
    data = pd.read_csv(file_path)
    'label'({ }, 1, True, **None)
    data = data.iloc[1:]
    if is_capture_mid_data:
        data = data[int(data.shape[0] * 0.2):int(data.shape[0] * 0.8)]
    data = normalize(data)
    brainwaves = [
        [
            'theta',
            4,
            7],
        [
            'low_alpha',
            8,
            9]]
    for brainwave in brainwaves:
        data = define_bp(data, brainwave[0], brainwave[1], brainwave[2])
    
    data = data[[
        'theta',
        'low_alpha']]
    return data.values


def change_to_sequence_data(data, offset_index = (10,)):
    sequence_data = []
    for start_index in range(0, data.shape[0] - data.shape[0] % offset_index, offset_index):
        sequence_data.append(np.array(data[start_index:start_index + offset_index]))
    
    return np.array(sequence_data)


def signal_filter(signal):
    np.random.seed(20)
    signal_length = signal.shape[0]
    signal = preprocessing.scale(signal)
    (surrogate_signal, _, _) = surrogates.iaaft(signal)
    w = modwt(signal, 'sym5', 5)
    surrogate_w = modwt(surrogate_signal, 'sym5', 5)
    for j in range(w.shape[0]):
        t_score = abs((w[j] - surrogate_w[j].mean()) / surrogate_w[j].std())
        p = 1 - stats.t.cdf(t_score, 2 * signal_length - 2, **None)
        threshold = w[j].std() * np.sqrt(2 * np.log(signal_length))
        for index in range(w[j].shape[0]):
            w[j][p[index] * 2 >= threshold] = surrogate_w[(j, index)]
        
    
    new_signal = imodwt(w, 'sym5')
    return new_signal

