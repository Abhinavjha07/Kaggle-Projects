import pandas as pd
import numpy as np
from keras.utils import np_utils

data = pd.read_csv('train.csv',skiprows = 1)
data = np.array(data)
Y = np.array(data[:,:1])
X = np.array(data[:,1:])

Y = np_utils.to_categorical(Y)
print(Y.shape)
X = np.reshape(X,(41999,28,28))
np.save('train_X.npy',X)
np.save('train_Y.npy',Y)
