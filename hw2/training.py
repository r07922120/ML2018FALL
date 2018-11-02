import numpy as np
import math
import csv
import function






def validation(val_x,val_y,w,b,fold_num,feature_num,feature,norm,mean,sigma,ratio):
	print(ratio)
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
	val_x,feature_num=function.encoding(val_x,feature_num*norm,feature,num)
	val_y=val_y.reshape(num)
	#print(mean,sigma)
	for i in range(num):
		val_x[i]=(val_x[i]-mean)/sigma#feature scaling
	#compute accuracy
	r1=0
	r0=0
	for n in range(num):
		#y=b
		y=b
		for j in range(feature_num*norm):
			y=y+w[j]*val_x[n][j]
		r=1/(1+math.exp(-y))
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
	train_x,feature_num=function.encoding(train_x,feature_num*norm,feature,num)

	mean= np.zeros((feature_num), dtype=float)
	sigma= np.zeros((feature_num), dtype=float)
	train_x,mean,sigma=function.scaling(train_x,feature_num*norm,num)
	train_x=train_x.reshape(num,feature_num*norm)
	
	
	
	
	
	lr=0.01#inital learning rate
	iteration=1300
	b=0.0#initial b
	w=0.0#initial w
	w_arr=np.array([])
	for i in range(feature_num*norm):
		w_arr=np.append(w_arr,w)
	w_arr=w_arr.reshape(feature_num*norm)
	m= np.zeros((feature_num*norm), dtype=float)
	v= np.zeros((feature_num*norm), dtype=float)
	m_temp= np.zeros((feature_num*norm), dtype=float)
	v_temp= np.zeros((feature_num*norm), dtype=float)
	bm=0.0
	bv=0.0
	bm_temp=0.0
	bv_temp=0.0
	p=0.9
	beta1=0.9
	beta2=0.999
	err=0.00000001


	#training
	acc=np.array([])
	for i in range(iteration):
		b_grad=0.0
		w_grad= np.zeros((feature_num*norm), dtype=float)
		acc_sum=0.0
		for n in range(len(train_y)):
			wx=(w_arr*train_x[n]).sum()
			fx=wx+b
			r=1/(1+math.exp(-fx))
			if(r>=0.5):
				s=1
			else:
				s=0
			if(s==train_y[n]):
				s=1
			else:
				s=0
			acc_sum=acc_sum+s
			w_grad=w_grad-(train_y[n]-r)*train_x[n]
			b_grad=b_grad-(train_y[n]-r)#-lamb*(b**norm)/len(y_data)
		#print(acc_sum/len(train_y))
		acc=np.append(acc,acc_sum/len(train_y))
		m=beta1*m+(1-beta1)*w_grad
		v=beta2*v+(1-beta2)*(w_grad**2)
		m_temp=m/(1-(beta1**(i+1)))
		v_temp=v/(1-(beta2**(i+1)))
		w_arr=w_arr-lr*m_temp/(np.sqrt(v_temp)+err)

		bm=beta1*bm+(1-beta1)*b_grad
		bv=beta2*bv+(1-beta2)*(b_grad**2)
		bm_temp=bm/(1-(beta1**(i+1)))
		bv_temp=bv/(1-(beta2**(i+1)))
		b=b-lr*bm_temp/(np.sqrt(bv_temp)+err)

		
		
		#if(i%1000==0):
		#	print(w_arr,b)
	#print("feature:",feature,"norm: ",norm)
	np.savetxt("acc_no.csv", acc, delimiter=",")
	return b,w_arr,mean,sigma,feature_num
