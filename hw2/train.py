import numpy as np
import math
import training_gen
import random


norm=1
n_fold_num=1
feature_num_start=23
#feature=[0,1,2,4,5,11,12,13,14,15,16,17,18,19]
feature=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
feature_num=len(feature)
people_num=20000
people_id=np.arange(0,people_num)
random.shuffle(people_id)#random for folds
ratio=0.5

trainx=np.genfromtxt('train_x.csv',delimiter=',',dtype=float,encoding="big5")
trainy=np.genfromtxt('train_y.csv',delimiter=',',dtype=float,encoding="big5")
a=np.array([])
b=np.array([])


#input data
for n in range(1,people_num+1):
	a=np.append(a,trainx[n][feature])
	b=np.append(b,trainy[n])
a=a.reshape(people_num,feature_num)
b=b.reshape(people_num,1)






train_x=np.array([])
val_x=np.array([])
train_y=np.array([])
val_y=np.array([])

#n_fold train set and validation set
for f in range(n_fold_num):
	val_x=np.append(val_x,a[people_id[people_num//n_fold_num*f:people_num//n_fold_num*(f+1)]])
	train_x=np.append(train_x,a[people_id[0:people_num//n_fold_num*f]])
	train_x=np.append(train_x,a[people_id[people_num//n_fold_num*(f+1):people_num]])
	val_y=np.append(val_y,b[people_id[people_num//n_fold_num*f:people_num//n_fold_num*(f+1)]])
	train_y=np.append(train_y,b[people_id[0:people_num//n_fold_num*f]])
	train_y=np.append(train_y,b[people_id[people_num//n_fold_num*(f+1):people_num]])

train_x=train_x.reshape(n_fold_num,people_num//n_fold_num*(n_fold_num-1),feature_num)
val_x=val_x.reshape(n_fold_num,people_num//n_fold_num,feature_num)
train_y=train_y.reshape(n_fold_num,people_num//n_fold_num*(n_fold_num-1))
val_y=val_y.reshape(n_fold_num,people_num//n_fold_num)


if (n_fold_num==1):
	temp=train_x
	train_x=val_x
	val_x=temp
	temp=train_y
	train_y=val_y
	val_y=temp


#train the folds
acc=0.0
for f in range(n_fold_num):
	b_sol,w_sol,mean,sigma=training_gen.training(train_x,train_y,f,feature_num,n_fold_num,people_num,feature,norm)
	if(f==0):
		w_his1=w_sol
		b_his1=[b_sol]

	else:
		w_his1=np.append(w_his1,w_sol)
		b_his1=np.append(b_his1,b_sol)
	if (n_fold_num!=1):
		acc_temp=training_gen.validation(val_x,val_y,w_sol,b_sol,f,feature_num,feature,norm,mean,sigma,ratio)
		print(acc_temp)
		acc=acc+acc_temp


#print(acc/n_fold_num)
w_his1=w_his1.reshape(n_fold_num,feature_num*norm)
if (n_fold_num==1):
	np.savetxt("w_gen.csv", w_his1, delimiter=",")
	np.savetxt("b_gen.csv", b_his1, delimiter=",")
	np.savetxt("mean_gen.csv", mean, delimiter=",")
	np.savetxt("sigma_gen.csv", sigma, delimiter=",")


