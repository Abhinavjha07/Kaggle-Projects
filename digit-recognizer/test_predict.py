import pandas as pd
import numpy as np
from keras.models import load_model

data = pd.read_csv('test.csv',skiprows = 0)
data = np.array(data)

X = np.array(data)
print(X.shape)
X = np.reshape(X,(-1,28,28,1))
np.save('test_X.npy',X)
X = X.reshape([-1,28,28,1])

model = load_model('my_model')
Y = model.predict_classes(X)
print(Y)
    
Y = np.array(Y)
with open('output.csv','w') as f:
    line = 'ImageId'+ ','+'Label\n'
    f.write(line)
    c = 1
    for i in Y:
        line = str(c)+','+str(i)+'\n'
        f.write(line)
        c+=1
