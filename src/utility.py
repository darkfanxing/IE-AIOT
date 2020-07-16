import pandas as pd
import numpy as np
import pywt

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


def get_data(file_path, is_capture_mid_data=False):
    data = pd.read_csv(file_path)
    data.rename({ "0Hz": "label" }, axis=1, inplace=True)
    
    data = data.iloc[1:]
    if is_capture_mid_data:
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
    return data.values

def change_to_sequence_data(data, offset_index=10):
    sequence_data = []
    for start_index in range(0, data.shape[0] - (data.shape[0] % offset_index), offset_index):
        sequence_data.append(np.array(data[start_index:start_index + offset_index]))

    return np.array(sequence_data)