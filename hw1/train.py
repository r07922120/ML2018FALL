import numpy as np
import math
import train1
import random


month = [1, 2, 3, 4, 5, 6,7,8,9,10,11,12]
random.shuffle(month)#random for folds




feature=[1,2,7,8,9,12]
n_fold_num=1
train_hour=5#how many hour for trainning
day=240
month_day=20
day_real=240
hour=24
feature_num_start=18
feature_num=len(feature)
for i in range(len(feature)):
	if(feature[i]==9):
		pm25_pos=i



csv=np.genfromtxt('train.csv',delimiter=',',dtype=float,encoding="big5")
a=np.array([])



#input data
for i in range(0,feature_num_start):
	if(i in feature):
		for j in range(0,day_real):
			a=np.append(a,csv[i+1+j*feature_num_start][3:27])

a=a.reshape(feature_num,day*hour)



train=np.array([])
val=np.array([])

#n_fold train set and validation set
for i in range(n_fold_num):
	for j in range(0,feature_num):
		for k in range(len(month)//n_fold_num*i,len(month)//n_fold_num*(i+1)):	
			val=np.append(val,a[j][(month[k]-1)*month_day*hour:month[k]*month_day*hour])
		for k in range(0,len(month)//n_fold_num*i):	
			train=np.append(train,a[j][(month[k]-1)*month_day*hour:month[k]*month_day*hour])
		for k in range(len(month)//n_fold_num*(i+1),len(month)):	
			train=np.append(train,a[j][(month[k]-1)*month_day*hour:month[k]*month_day*hour])
train=train.reshape(n_fold_num,feature_num,day*hour//n_fold_num*(n_fold_num-1))
val=val.reshape(n_fold_num,feature_num,day*hour//n_fold_num)

if (n_fold_num==1):
	temp=train
	train=val
	val=temp


#train the folds
err=0.0
for i in range(n_fold_num):
	b_sol,w_sol=train1.training(train,i,feature_num,pm25_pos,train_hour,n_fold_num)

	if(i==0):
		w_his1=w_sol
		b_his1=[b_sol]
	else:
		w_his1=np.append(w_his1,w_sol)
		b_his1=np.append(b_his1,b_sol)
	if (n_fold_num!=1):
		print(train1.validation(val,w_sol,b_sol,i,feature_num,pm25_pos,train_hour),month[i*(len(month)//n_fold_num):(i+1)*(len(month)//n_fold_num)])
		err=err+train1.validation(val,w_sol,b_sol,i,feature_num,pm25_pos,train_hour)
print(err/n_fold_num)
w_his1=w_his1.reshape(n_fold_num,feature_num*train_hour)
if (n_fold_num==1):
	np.savetxt("w.csv", w_his1, delimiter=",")
	np.savetxt("b.csv", b_his1, delimiter=",")
	print(w_his1)
	print(b_his1)

