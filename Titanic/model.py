from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

import numpy as np
from random import shuffle

X_train = np.load('train_X.npy')
Y_train =np.load('train_y.npy')

input_dim = X_train.shape[1]
n_classes = Y_train.shape[1]

model = Sequential()
model.add(Dense(128,input_dim = input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
print('Training : ')
model.fit(X_train,Y_train,epochs=200,batch_size=256,validation_split=0.1,verbose=2,shuffle = True)
model.save('my_model')
