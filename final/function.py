import numpy as np
from PIL import Image


epochs=100
batch=4
category=28
pic_dim=512
def imageLoader(files, batch_size):
    L = len(files)
    L=10   
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            X=[]
            limit = min(batch_end, L)
            for i in range(batch_start,limit):
                for j in range(3):
                    if(j==0):
                        name=(files[i]+ "_red.png")
                    elif(j==1):
                        name=(files[i]+"_green.png")
                    elif(j==2):
                        name=(files[i]+ "_blue.png")
                    img=Image.open(name)
                    a=np.asarray(img)
                    X.append(a)
            temp_x=np.array(X)
            temp_x=temp_x.reshape(limit-batch_start,3,pic_dim,pic_dim)
            temp_x = temp_x
            Y=[]
            for i in range(batch_start,limit):
                for j in range(1*1):
                    Y.append(y_train[i])
            temp_y=np.array(Y)
            temp_y=temp_y.reshape(limit-batch_start,category)
            

            yield (temp_x, temp_y)    

            batch_start += batch_size   
            batch_end += batch_size
