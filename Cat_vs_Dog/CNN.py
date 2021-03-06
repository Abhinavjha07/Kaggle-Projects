import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt

TRAIN_DIR = '/home/abhinav/MLProjects/Tensorflow/CatVSDog_Classifier/train'
TEST_DIR = '/home/abhinav/MLProjects/Tensorflow/CatVSDog_Classifier/test'
LR = 1e-3
IMG_SIZE = 50

MODEL_NAME = 'dogsVScats-{}-{}.model'.format(LR,'2-conv-basic')

train_data = np.load('train_data.npy')
test_data = np.load('test_data.npy')

convnet = input_data(shape = [None,IMG_SIZE,IMG_SIZE,1],name = 'input')
convnet = conv_2d(convnet,32,[3,3],activation = 'relu')
convnet = max_pool_2d(convnet,[3,3])

convnet = conv_2d(convnet,64,[5,5],activation = 'relu')
convnet = max_pool_2d(convnet,[5,5])

convnet = fully_connected(convnet,1024,activation = 'relu')
convnet = dropout(convnet,0.8)
convnet = fully_connected(convnet,84,activation = 'relu')

convnet = fully_connected(convnet,2,activation = 'softmax')
convnet = regression(convnet,optimizer = 'adam' ,learning_rate = 0.01,loss = 'categorical_crossentropy',name = 'targets')
model = tflearn.DNN(convnet)


if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model_loaded!!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input':X},{'targets':Y},n_epoch=2,validation_set=({'input':test_x},{'targets':test_y}),snapshot_step =500,show_metric = True,run_id = MODEL_NAME)
model.save(MODEL_NAME)

#Increase the accuracy by increasing the number of layers

'''
To see the accuracy of our network
#tf.reset_default_graph()

test_data = np.load('test_data.npy')

fig = plt.figure()
for num,data in enumerate(test_data[:15]):
    img_num = data[1]
    img_data = data[0]
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(50,50,1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1: str_label ='Dog'
    else : str_label = 'Cat'
    y.imshow(orig,cmap = 'gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.show()

with open('submission_file.csv','w') as f:
    f.write('id,label\n')

with open('submission_file.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        y = fig.add_subplot(3,4,num+1)
        orig = img_data
        data = img_data.reshape(50,50,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))

'''





