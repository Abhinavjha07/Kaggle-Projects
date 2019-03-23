import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils

batch_size=128
num_classes = 10
epochs=12

(train_X, train_Y), (test_X, test_Y) = mnist.load_data()
test_Y = np_utils.to_categorical(test_Y)
train_Y = np_utils.to_categorical(train_Y)
#print(train_X.shape, train_Y.shape,test_X.shape,test_Y.shape)
train_Y = np.reshape(train_Y,(-1,10))
test_Y = np.reshape(test_Y,(-1,10))
#print(train_X.shape, train_Y.shape,test_X.shape,test_Y.shape)
train_X = np.append(train_X,test_X)
train_Y = np.append(train_Y,test_Y)


train_X = np.reshape(train_X,(-1,28,28,1))
train_Y = np.reshape(train_Y,(-1,10))

#train_X = np.load('train_X.npy')
#train_Y = np.load('train_Y.npy')
#train_X = np.reshape(train_X,(-1,28,28,1))
print(train_X.shape)
print(train_Y.shape)
model = Sequential()
shape = (28,28,1)
model.add(Conv2D(32,kernel_size=(3,3),activation = 'relu',input_shape = shape))
model.add(MaxPooling2D(pool_size = (3,3)))

model.add(Dropout(0.2))

model.add(Conv2D(64,kernel_size=(3,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3,3)))

model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(128,activation='relu'))

model.add(Dense(num_classes,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print('Training : ')
model.fit(train_X,train_Y,epochs=10,batch_size=batch_size,validation_split=0.001,shuffle=True)
model.save('my_model')
