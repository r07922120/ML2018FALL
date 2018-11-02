import numpy as np
import math
import csv
import function






def validation(val_x,val_y,w,b,fold_num,feature_num,feature,norm,mean,sigma,ratio):
	acc_sum=0.0
	num=len(val_y[fold_num])
	val_x=val_x[fold_num]
	val_y=val_y[fold_num]
	#print(val_x[0][11:16])
	val_x=function.correct(val_x,mean,num,feature_num,feature)

	#dimension
	a_temp=np.array([])
	for n in range(num):
		for i in range(norm):
			a_temp=np.append(a_temp,function.re(val_x[n],feature)**(i+1))
	val_x=a_temp

	val_x=val_x.reshape(num,feature_num*norm)
	val_y=val_y.reshape(num)
	#print(mean,sigma)
	for i in range(num):
		val_x[i]=(val_x[i]-mean)/sigma
	
	#compute accuracy
	r1=0
	r0=0
	for n in range(num):
		#y=b
		y=b
		for j in range(feature_num*norm):
			y=y+w[j]*val_x[n][j]
		r=1/(1+math.exp(-y))
		#print(r)
		if(r>=ratio):
			#print(r)
			r=1

		else:
			r=0
		if(r==val_y[n]):
			r=1
		else:
			if(r==0):
				r0+=1
			else:
				r1+=1
			r=0
		acc_sum=acc_sum+r
	print(r0)
	print(r1)
		
	return acc_sum/num




def training(train_x,train_y,fold_num,feature_num,n_fold_num,people_num,feature,norm):
	if(n_fold_num!=1):
		train_x=train_x.reshape(n_fold_num,people_num//n_fold_num*(n_fold_num-1),feature_num)
		train_y=train_y.reshape(n_fold_num,people_num//n_fold_num*(n_fold_num-1))
		num=people_num//n_fold_num*(n_fold_num-1)
	else:
		train_x=train_x.reshape(n_fold_num,people_num,feature_num)
		train_y=train_y.reshape(n_fold_num,people_num)
		num=people_num
	a_temp=np.array([])
	b_temp=np.array([])
	count=0
	for n in range(num):
		if(function.wrong(train_x[fold_num][n],feature)==0):
			for i in range(norm):
				a_temp=np.append(a_temp,function.re(train_x[fold_num][n],feature)**(i+1))
			b_temp=np.append(b_temp,train_y[fold_num][n])
		else:
			count+=1
	train_x=a_temp
	train_y=b_temp
	num-=count

	train_x=train_x.reshape(num,feature_num*norm)
	train_y=train_y.reshape(num)
	train_x,mean,sigma=function.scaling(train_x,feature_num*norm,num)
	train_x=train_x.reshape(num,feature_num*norm)
	
	sum_all=np.zeros((feature_num*norm), dtype=float)
	sum0=np.zeros((feature_num*norm), dtype=float)
	sum1=np.zeros((feature_num*norm), dtype=float)
	mean0= np.zeros((feature_num*norm), dtype=float)
	mean1= np.zeros((feature_num*norm), dtype=float)
	mean_all= np.zeros((feature_num*norm), dtype=float)
	N0=0.0
	N1=0.0
	sigma_co=np.zeros((feature_num*norm*feature_num*norm), dtype=float)
	sigma_co=sigma_co.reshape(feature_num*norm,feature_num*norm)
	xi_mean_temp=np.zeros((feature_num*norm), dtype=float)
	

	for n in range (num):
		sum_all=sum_all+train_x[n]
		if(train_y[n]==0):
			sum0=sum0+train_x[n]
			N0+=1.0
		else:
			sum1=sum1+train_x[n]
			N1+=1.0
	mean0=sum0/N0
	mean1=sum1/N1
	mean_all=sum_all/(N1+N0)
	mean0=mean0.reshape(feature_num*norm,1)
	mean1=mean1.reshape(feature_num*norm,1)
	for n in range (num):
		xi_mean_temp=train_x[n]-mean_all
		xi_mean_T=xi_mean_temp.reshape(1,feature_num*norm)
		xi_mean=xi_mean_temp.reshape(feature_num*norm,1)
		sigma_co+=np.dot(xi_mean,xi_mean_T)
	
	sigma_co=sigma_co/num
		


	w_arr=np.zeros((feature_num*norm), dtype=float)
	b=np.array([0.0])

	mean0_temp=np.dot(np.transpose(mean0),np.linalg.inv(sigma_co))
	mean1_temp=np.dot(np.transpose(mean1),np.linalg.inv(sigma_co))
	w_arr=np.dot(np.transpose(mean1-mean0),np.linalg.inv(sigma_co))
	b=1/2*np.dot(mean0_temp,mean0)-1/2*np.dot(mean1_temp,mean1)+math.log(N1/N0)
	b=b[0]
	w_arr=w_arr.reshape(feature_num*norm)
	return b,w_arr,mean,sigma
