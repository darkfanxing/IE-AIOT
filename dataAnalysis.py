import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
plt.style.use('ggplot')

def get_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    data.rename({ '0Hz': 'label' }, axis=1, inplace=True)
    psd_data = (data / 10 ** 3) ** 2

    for frequency in range(4, 8 + 1, 1):
        psd_data['theta'] = 0
        psd_data['theta'] += psd_data['{}Hz'.format(frequency)]

    return psd_data

unfocused_data = get_data('./data/Mocx_0.csv')[5000:7500] / 1e7
focused_data = get_data('./data/Mocx_1.csv')[5000:7500] / 1e7
unfocused_data['label'] = 0


data = unfocused_data.append(focused_data) 

sns.barplot(x='label', y='theta', data=data)
# sns.scatterplot(x='low_beta', y='high_beta', hue='label', data=data)
plt.show()
# plt.clf()
# sns.scatterplot(x='low_alpha', y='high_alpha', hue='label', data=data)
# plt.show()


# fig = plt.figure()
# for row_index in range(psd_data.shape[0]):
#     fig.clf()
#     plt.ylim(0, 60000000)
#     psd_data.iloc[row_index].plot(kind='bar')
#     plt.draw()
#     plt.pause(0.00000001)