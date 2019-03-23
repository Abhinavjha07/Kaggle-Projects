import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils
from random import shuffle
columns = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Survived']
data = pd.read_csv(r'train.csv',usecols = columns)

#print(data)
sex = []
for i in data['Sex']:
    if i == 'male':
        sex.append(0)
    else:
        sex.append(1)

embarkment = []
c1,c2,c3=0,0,0
for x in data['Embarked']:
    if x == 'S':
        c1+=1
        embarkment.append(1)
    elif x == 'C':
        c2+=1
        embarkment.append(2)
    elif x == 'Q':
        c3+=1
        embarkment.append(3)
    elif x is np.NaN:
        embarkment.append(3)

Y = np.array(data['Survived'])
Y = Y.reshape(891,1)
Y = np_utils.to_categorical(Y)
print(Y)

sex = np.array(sex)
sex = np.reshape(sex,(891,1))
embarkment = np.array(embarkment)
embarkment = np.reshape(embarkment,(891,1))

train_data = np.array(np.concatenate((sex,embarkment,np.array(data['Fare']).reshape(891,1),
                                      np.array(data['Pclass']).reshape(891,1), np.array(data['Age']).reshape(891,1), np.array(data['SibSp']).reshape(891,1),
                                       np.array(data['Parch']).reshape(891,1)),axis=1))

col_mean = np.nanmean(train_data,axis=0)


#print(col_mean)

inds = np.where(np.isnan(train_data))
#print(inds)
train_data[inds] = np.take(col_mean,inds[1])


X = preprocessing.scale(train_data)
print(X.shape[1],Y.shape[1])
np.save('train_X.npy',X)
np.save('train_y.npy',Y)

