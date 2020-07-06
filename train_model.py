import tensorflow as tf
import numpy as np

param_count = 5
test_count = 10

t_x = np.floor(1000 * np.random.random([test_count,param_count]),dtype=np.float32)

t_w = np.floor(1000 * np.random.random([param_count,1]),dtype=np.float32)

#根据公式 t_y = t_x * t_w 算出值 t_y
t_y = t_x.dot(t_w)

print(t_y)