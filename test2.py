import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = np.genfromtxt("src/data/ttt.csv", delimiter=",")

# fftx
fft_y = np.abs(np.fft.fft(data)).astype(int)

x = np.arange(len(data))

# plt.plot(x[18:22], fft_y[18:22])
# plt.show()