import numpy as np
import csv
from sys import argv
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Input,LeakyReLU,InputLayer
from keras.layers import Conv2D, MaxPooling2D, Flatten,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image
import keras.backend as K
import os
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception
from keras.models import Model
 

THRESHOLD = 0.15
split=10 #單位%
epochs=500
batch=64#batch size
category=28#number of classification
pic_dim=256#picture dimension
channel_num=4
SHAPE = (pic_dim, pic_dim,channel_num)
model_filepath="temp/focal.hdf5"#save model
data_filepath="train_resize/"

def f1(y_true, y_pred):
    y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    #f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1-K.mean(f1)



file=[]
csv=np.genfromtxt('train.csv',delimiter=',',dtype=str)
for i in range(1,len(csv)):
    line=csv[i][0]
    name=join(data_filepath, line)
    file.append(name)
file=np.array(file)


#label
#compute class weight
count = 0
y=np.zeros((len(csv)-1)*category)
y=y.reshape(len(csv)-1,category)
class_weight=np.zeros(category)
for i in range(1,len(csv)):
    line=csv[i][1]
    line=line.split(" ");
    for label in line:
        class_weight[int(label)]+=1
        y[count][int(label)]=1
    count += 1
    print('\rfile: ' + repr(count), end='', flush=True)
print('', flush=True)


#以class[0]為1當基準
class_weight=1/class_weight*12588
weight={
    0:class_weight[0],
    1:class_weight[1],
    2:class_weight[2],
    3:class_weight[3],
    4:class_weight[4],
    5:class_weight[5],
    6:class_weight[6],
    7:class_weight[7],
    8:class_weight[8],
    9:class_weight[9],
    10:class_weight[10],
    11:class_weight[11],
    12:class_weight[12],
    13:class_weight[13],
    14:class_weight[14],
    15:class_weight[15],
    16:class_weight[16],
    17:class_weight[17],
    18:class_weight[18],
    19:class_weight[19],
    20:class_weight[20],
    21:class_weight[21],
    22:class_weight[22],
    23:class_weight[23],
    24:class_weight[24],
    25:class_weight[25],
    26:class_weight[26],
    27:class_weight[27]
}


#random data
rng_state = np.random.get_state()
np.random.shuffle(file)
np.random.set_state(rng_state)
np.random.shuffle(y)
#split data
train_file=file[len(file)*split//100:len(file)]
y_train=y[len(file)*split//100:len(file)]
test_file=file[0:len(file)*split//100]
y_test=y[0:len(file)*split//100]

#get test data from disk
"""
temp=[]
x_test=np.array([])
count=0
for i in range(0,len(test_file)):
    R = Image.open(test_file[i] + '_red.png')
    G = Image.open(test_file[i] + '_green.png')
    B = Image.open(test_file[i] + '_blue.png')
    Y = Image.open(test_file[i] + '_yellow.png')
    im = np.stack((
            np.asarray(R,dtype=np.float32), 
            np.asarray(G,dtype=np.float32), 
            np.asarray(B,dtype=np.float32),
            np.asarray(Y,dtype=np.float32)), -1)
    im=im/255
    temp.append(im)
    count+=1
    print('\rtest: ' + repr(count), end='', flush=True)
print('', flush=True)
x_test=np.append(x_test,temp)
x_test=x_test/255
x_test=x_test.reshape(len(file)//split,pic_dim,pic_dim,4)
"""
model=Sequential()
model.add(InputLayer(SHAPE))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))



model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))

#model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))

model.add(GlobalAveragePooling2D())
#model.add(Flatten())
#model.add(Dense(input_dim=pic_dim*pic_dim,units=256,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(28, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(Dense(28, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())

model.add(Dense(category,init='uniform'))
model.add(Activation('sigmoid'))
model.summary()

#use pretrained model
"""
base_model =Xception(include_top=False, weights='imagenet',  input_shape=(pic_dim,pic_dim,3))
count=0
for layer in base_model.layers:
    count+=1
    layer.trainable = False

model = Sequential()
#model.add(Conv2D(3, (1, 1), input_shape=(pic_dim,pic_dim, 1), activation='relu', padding='same'))
model.add(InputLayer(SHAPE))
model.add(BatchNormalization())
#model.add(BatchNormalization(axis=-1))
model.add(Model(inputs=base_model.input, outputs=base_model.output))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(512, (7, 7), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(1024, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
#model.add(Flatten())
#model.add(Dense(input_dim=pic_dim*pic_dim,units=256,activation='relu'))
model.add(Dense(category,kernel_initializer='uniform'))
model.add(Activation('sigmoid'))
model.summary()
"""





adam = Adam(lr=0.001)
model.compile(loss=f1_loss,optimizer=adam,metrics=[f1])
es = EarlyStopping(monitor='val_f1', patience=6, verbose=1, mode='max')
lr_function=ReduceLROnPlateau(monitor='val_f1',patience=3,verbose=1,factor=0.5,min_lr=0.00001, mode='max')
checkpoint = ModelCheckpoint(model_filepath, monitor='val_f1', verbose=1, save_best_only=True, mode='max')

#get training data form disk
def imageLoader_train(files, batch_size):
    L = len(files)
    temp_x=np.array([])
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            temp_x=[]
            X=[]
            limit = min(batch_end, L)
            for i in range(batch_start,limit):
                R = Image.open(files[i] + '_red.png')
                G = Image.open(files[i] + '_green.png')
                B = Image.open(files[i] + '_blue.png')
                if(channel_num==4):
                    Y = Image.open(files[i] + '_yellow.png')
                    im = np.stack((
                            np.asarray(R,dtype=np.float32), 
                            np.asarray(G,dtype=np.float32), 
                            np.asarray(B,dtype=np.float32),
                            np.asarray(Y,dtype=np.float32)), -1)
                else:
                    im = np.stack((
                            np.asarray(R,dtype=np.float32), 
                            np.asarray(G,dtype=np.float32), 
                            np.asarray(B,dtype=np.float32)), -1)
                X.append(im)
            temp_x=np.append(temp_x,X)
            temp_x=temp_x/255
            temp_x=temp_x.reshape(limit-batch_start,pic_dim,pic_dim,channel_num)
            Y=[]
            Y.append(y_train[batch_start:limit])
            temp_y=np.array(Y,dtype=np.float32)
            temp_y=temp_y.reshape(limit-batch_start,category)

            yield temp_x,temp_y
            batch_start += batch_size   
            batch_end += batch_size

def imageLoader_test(files, batch_size):
    L = len(files)
    temp_x=np.array([])
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            temp_x=[]
            X=[]
            limit = min(batch_end, L)
            for i in range(batch_start,limit):
                R = Image.open(files[i] + '_red.png')
                G = Image.open(files[i] + '_green.png')
                B = Image.open(files[i] + '_blue.png')
                if(channel_num==4):
                    Y = Image.open(files[i] + '_yellow.png')
                    im = np.stack((
                            np.asarray(R,dtype=np.float32), 
                            np.asarray(G,dtype=np.float32), 
                            np.asarray(B,dtype=np.float32),
                            np.asarray(Y,dtype=np.float32)), -1)
                else:
                    im = np.stack((
                            np.asarray(R,dtype=np.float32), 
                            np.asarray(G,dtype=np.float32), 
                            np.asarray(B,dtype=np.float32)), -1)
                X.append(im)
            temp_x=np.append(temp_x,X)
            temp_x=temp_x/255
            temp_x=temp_x.reshape(limit-batch_start,pic_dim,pic_dim,channel_num)
            Y=[]
            Y.append(y_test[batch_start:limit])
            temp_y=np.array(Y,dtype=np.float32)
            temp_y=temp_y.reshape(limit-batch_start,category)

            yield temp_x,temp_y
            batch_start += batch_size   
            batch_end += batch_size


"""
gen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    rotation_range=8,
    width_shift_range=0.08,
    shear_range=0.3,
    height_shift_range=0.08,
    zoom_range=0.08,
    data_format="channels_last",
    horizontal_flip=True)
"""

model.fit_generator(imageLoader_train(train_file,batch),steps_per_epoch=len(train_file)//batch+1,epochs=epochs,verbose=1,validation_steps=len(test_file)//batch+1,validation_data=imageLoader_test(test_file,batch),callbacks=[lr_function,checkpoint])
