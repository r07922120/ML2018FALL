import numpy as np
import csv
from sys import argv
import os
from keras.models import load_model
import sys
from os import listdir
from os.path import isfile, join
from PIL import Image
import keras.backend as K
import tensorflow as tf

THRESHOLD = 0.15
epochs=10
batch=25
category=28
pic_dim=256
kernel_num=4
path="test_resize" 
def f1(y_true, y_pred):
    #y_pred = K.round(y_pred)
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
path_list=os.listdir(path)
path_list.sort(key = str.lower)
i=0
id_name=[]
for ff in path_list:
    if(i%4==1):
        name=join("test_resize/", ff)
        name=name.rstrip("_green.png/")
        file.append(name)
    i+=1
test_file=np.array(file)

model = load_model("temp/model5.hdf5",custom_objects={'f1': f1,"f1_loss":f1_loss})


temp=[]
count=0
result=[]
"""
for i in range(0,11000):
    R = Image.open(test_file[i] + '_red.png')
    G = Image.open(test_file[i] + '_green.png')
    B = Image.open(test_file[i] + '_blue.png')
    Y = Image.open(test_file[i] + '_yellow.png')
    if(kernel_num==4):
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
    if(i%1000==999):
        x_test=np.asarray(temp)
        x_test=x_test/255
        x_test=x_test.reshape(1000,4,pic_dim,pic_dim)
        result.append(model.predict(x_test, batch_size = 50, verbose = 1))
        temp=[]
    print('\rtest: ' + repr(i), end='', flush=True)
print('', flush=True)
"""
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
                a=np.asarray(R,dtype=np.float32)
                if(kernel_num==4):
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
            temp_x=temp_x.reshape(limit-batch_start,pic_dim,pic_dim,kernel_num)
            yield temp_x
            batch_start += batch_size   
            batch_end += batch_size
"""
x_test=np.asarray(temp)
x_test=x_test/255
x_test=x_test.reshape(len(x_test),4,pic_dim,pic_dim)
"""
result.append(model.predict_generator(imageLoader_test(test_file,batch),steps=len(test_file)//batch+1, verbose = 1))


with open("ans.csv", 'w') as f:
    f.write('id,Predicted\n')
    for i in range(len(result)):
        for j in range(len(result[i])):
            name=file[i].lstrip("test_resize/")
            #name=name.rstrip("_green.png")
            f.write(name+',')
            m=0
            for k in range(len(result[i][j])):
                if(result[i][j][k]>THRESHOLD):
                    if(m==0):
                        f.write(str(k))
                        m+=1
                    else:
                        f.write(' '+str(k))
            if(m==0):
                predict = np.argmax(result[i][j])
                f.write(str(predict))
            f.write("\n")
