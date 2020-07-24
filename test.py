from src.utility import signal_filter
import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


x = np.linspace(0, 10, 100, endpoint=False)
y = np.cos(-x ** 2 / 6.0)

f = signal.resample(y, 25)
xnew = np.linspace(0, 10, 25, endpoint=False)

plt.plot(x, y, 'go-', xnew, f, '.-', 10, y[0], 'ro')
plt.show()