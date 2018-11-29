import numpy as np
import csv
from sys import argv
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import sys

norm=1
n_fold_num=1
pic_num=28709
pic_id=np.arange(0,pic_num)
#random.shuffle(people_id)#random for folds
pic_dim=48
category=7
epochs=100
batch=300
#load data
x, y = np.array([]), np.array([])
temp=[]
with open(argv[1], 'r') as f:
	count = 0
	for line in list(csv.reader(f))[1:]:
		y=np.append(y,float(line[0]))
		temp.append([float(a) for a in line[1].split()])
		count += 1
		print('\rX_train: ' + repr(count), end='', flush=True)
	print('', flush=True)
y=np_utils.to_categorical(y, category)
x=np.append(x,temp)
x=x.reshape(pic_num,pic_dim,pic_dim,1)
y=y.reshape(pic_num,category)
x=x/255
rng_state = np.random.get_state()
np.random.shuffle(x)
np.random.set_state(rng_state)
np.random.shuffle(y)
x_train=x[2870:28709]
y_train=y[2870:28709]
x_test=x[0:2870]
y_test=y[0:2870]
#CNN
model=Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))
model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.3))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(BatchNormalization(axis=-1))

model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.3))


model.add(Flatten())
#model.add(Dense(input_dim=pic_dim*pic_dim,units=256,activation='relu'))


model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
#model.add(Dropout(0.1))
model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
model.add(BatchNormalization())
#model.add(Dropout(0.1))
model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

es = EarlyStopping(monitor='val_acc', patience=6, verbose=1, mode='auto')
#model.fit(x, y, batch_size=batch, epochs=epochs, verbose=1, validation_split=0.1, callbacks=[es])


gen = ImageDataGenerator(featurewise_center=False,
	samplewise_center=False,
	rotation_range=8,
	width_shift_range=0.08,
	shear_range=0.3,
	height_shift_range=0.08,
	zoom_range=0.08,
	data_format="channels_last",
	horizontal_flip=True)
gen.fit(x_train)
train_gen=gen.flow(x_train,y_train,batch_size=batch)

lr_function=ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)

filepath="temp/model5.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

model.fit_generator(train_gen,steps_per_epoch=batch,epochs=epochs,verbose=1,validation_data=(x_test,y_test),callbacks=[es,lr_function,checkpoint])


result=model.evaluate(x,y)
print("acc:",result[1])

