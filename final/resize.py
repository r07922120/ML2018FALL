import cv2
import numpy as np
from os.path import isfile, join
from PIL import Image
import os

csv=np.genfromtxt('train.csv',delimiter=',',dtype=str)
shape=256
for i in range(1,len(csv)):
    line=csv[i][0]
    name=join("train/", line)
    name_save=join("train_resize/", line)
    fileR = name + "_red.png"
    fileG = name + "_green.png"
    fileB = name + "_blue.png"
    fileY = name + "_yellow.png"
    R=Image.open(fileR)
    G=Image.open(fileG)
    B=Image.open(fileB)
    Y=Image.open(fileY)
    cv2.waitKey(0)
    R.thumbnail( (shape,shape) )
    G.thumbnail( (shape,shape) )
    B.thumbnail( (shape,shape) )
    Y.thumbnail( (shape,shape) )
    fileR = name_save + "_red.png"
    fileG = name_save + "_green.png"
    fileB = name_save + "_blue.png"
    fileY = name_save + "_yellow.png"
    R.save( fileR )
    G.save( fileG )
    B.save( fileB )
    Y.save( fileY )
    print('\rtrain: ' + repr(i), end='', flush=True)
print('', flush=True)
path="test" 
path_list=os.listdir(path)
path_list.sort(key = str.lower)
file=[]
i=0
for ff in path_list:
    if(i%4==1):
        name=ff.rstrip("_green.png/")
        file.append(name)
    i+=1
    print('\rtest: ' + repr(i), end='', flush=True)
print('', flush=True)
test_file=np.array(file)
i=0
for line in test_file:
    name=join("test/", line)
    name_save=join("test_resize/", line)
    fileR = name + "_red.png"
    fileG = name + "_green.png"
    fileB = name + "_blue.png"
    fileY = name + "_yellow.png"
    R=Image.open(fileR)
    G=Image.open(fileG)
    B=Image.open(fileB)
    Y=Image.open(fileY)
    R.thumbnail( (shape,shape) )
    G.thumbnail( (shape,shape) )
    B.thumbnail( (shape,shape) )
    Y.thumbnail( (shape,shape) )
    fileR = name_save + "_red.png"
    fileG = name_save + "_green.png"
    fileB = name_save + "_blue.png"
    fileY = name_save + "_yellow.png"
    R.save( fileR )
    G.save( fileG )
    B.save( fileB )
    Y.save( fileY )
    i+=1
    print('\rtest: ' + repr(i), end='', flush=True)
print('', flush=True)
