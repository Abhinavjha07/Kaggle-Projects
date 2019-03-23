import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from keras.utils import np_utils
from keras.models import load_model

columns = ['PassengerId','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
data = pd.read_csv(r'test.csv',usecols = columns)

#print(data)
sex = []
for i in data['Sex']:
    if i == 'male':
        sex.append(0)
    else:
        sex.append(1)
p_id = data['PassengerId']
p_id = np.array(p_id)
p_id = np.reshape(p_id,(p_id.shape[0],1))
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
sex = np.array(sex)

sex = np.reshape(sex,(sex.shape[0],1))
embarkment = np.array(embarkment)
embarkment = np.reshape(embarkment,(embarkment.shape[0],1))

test_data = np.array(np.concatenate((sex,embarkment,np.array(data['Fare']).reshape(418,1),
                                      np.array(data['Pclass']).reshape(418,1), np.array(data['Age']).reshape(418,1), np.array(data['SibSp']).reshape(418,1),
                                       np.array(data['Parch']).reshape(418,1)),axis=1))

col_mean = np.nanmean(test_data,axis=0)




inds = np.where(np.isnan(test_data))
#print(inds)
test_data[inds] = np.take(col_mean,inds[1])


X = preprocessing.scale(test_data)


model = load_model('my_model')
Y = model.predict_classes(X)
print(Y)
Y = np.array(Y)

with open('output.csv','w') as f:
    line = 'PassengerId'+ ','+'Survived\n'
    f.write(line)
    
    for i in range(418):
        line = str(p_id[i][0])+','+str(Y[i])+'\n'
        f.write(line)



    
