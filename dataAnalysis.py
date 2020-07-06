import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')

def define_bp(data, bp_name, start_frequency, end_frequency):
    for frequency in range(start_frequency, end_frequency + 1, 1):
        data[bp_name] = 0
        data[bp_name] += data['{}Hz'.format(frequency)]
    
    return data

def get_data(file_path):
    data = pd.read_csv(file_path)
    data.rename({ '0Hz': 'label' }, axis=1, inplace=True)
    
    data = data.iloc[1:]
    data = data[int(data.shape[0] * 0.15): int(data.shape[0] * 0.85)]

    brainwaves = [
        ['theta', 4, 7],
        ['low_alpha', 8, 9],
        ['middle_alpha', 9, 12],
        ['high_alpha', 12, 14],
        ['low_beta', 13, 16],
        ['middle_beta', 17, 20],
        ['high_beta', 21, 28],
    ]
    
    for brainwave in brainwaves:
        data = define_bp(data, brainwave[0], brainwave[1], brainwave[2])

    # data['theta'] = (1 / (1 + np.exp(-0.2 * (data['theta'] / 10 ** 3))) * 2 - 2) * -1 
    return data / 1000

# for i in range(20):
unfocused_data = get_data('./data/unfocused/lin_1.csv')
focused_data = get_data('./data/focused/lin_1.csv')
unfocused_data['label'] = 1
focused_data['label'] = 2

unfocused_data = unfocused_data[['theta', 'low_alpha']].values
focused_data = focused_data[['theta', 'low_alpha']].values


svdd = OneClassSVM(gamma='auto')
svdd.fit(unfocused_data)

count = 0
score = svdd.predict(focused_data)
for i in score:
    if i > 0:
        count += 1

print(count / score.shape[0])

# count = 0
# score = svdd.score_samples(unfocused_data)
# for i in score:
#     if i > 0:
#         count += 1

# print(score.shape)
# print(count)

# data = unfocused_data.append(focused_data)
# data = data[int(data.shape[0] * 0.2): int(data.shape[0] * 0.8)]
# brainwave_data = data[['theta', 'low_alpha', 'middle_alpha', 'high_alpha', 'middle_beta', 'high_beta']].values
# label = data['label'].values

# test_unfocused_data = get_data('./data/unfocused/lin_2.csv')
# test_focused_data = get_data('./data/focused/lin_2.csv')
# test_unfocused_data['label'] = 1
# test_focused_data['label'] = 2
# test_data = test_unfocused_data.append(test_focused_data)
# test_data = test_data[int(test_data.shape[0] * 0.2): int(test_data.shape[0] * 0.8)]
# test_brainwave_data = test_data[['theta', 'low_alpha', 'middle_alpha', 'high_alpha', 'middle_beta', 'high_beta']].values
# test_label = test_data['label'].values

# print(data.info())

# print(test_data.info())

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(brainwave_data, label)

# print(knn.score(test_brainwave_data, test_label))
    # sns.barplot(x='label', y='theta', data=data)
    # plt.show()
    # plt.clf()