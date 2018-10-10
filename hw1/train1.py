import numpy as np
import math
import csv


feature_num_start=18
hour=24
day=240

#interpolation wrong data
def inter(train_set,k,set_num):
	low=0
	low_index=0
	high=0
	high_index=0
	
	for i in range(1,1):
		if((k-i)<0 or k+i>=len(train_set[set_num])):
			break
		else:	
			if(k%(day*hour)-i<0 or k%(day*hour)+i>=day*hour ):
				break
			else:
				if(train_set[set_num][k-i]>0 and train_set[set_num][k-i]<=500):
					if(low_index==0):
						low_index=i
						low=train_set[set_num][k-i]
				if(train_set[set_num][k+i]>0 and train_set[set_num][k+i]<=500):
					if(high_index==0):
						high_index=i
						high=train_set[set_num][k+i]
	if(high_index!=0 and low_index!=0):
		train_set[set_num][k]=low_index/(high_index+low_index)*(high-low)+low
		#arr[k]=low/(high+low)*(high_index-low_index)+low_index
	return train_set


def validation(val_set,w,b,set_num,feature_num,pm25_pos,train_hour):
	x_data=np.array([])
	y_data=np.array([])
	error_sum=0.0
	#get the first nine hours and last our
	for j in range(0,feature_num):
		for i in range(0,len(val_set[set_num][j])):
			if(i%(20*24)<(20*24-train_hour)):
				
				#delete wrong data
				check=0
				for k in range(train_hour+1):
					for f in range(0,feature_num):
						if(val_set[set_num][f][i+k]<=0 or val_set[set_num][f][i+k]>300):
							check=1
							break
				if(check==1):
					continue
				
				x_data=np.append(x_data,val_set[set_num][j][i:i+train_hour])
				if(j%feature_num==pm25_pos):
					y_data=np.append(y_data,val_set[set_num][j][i+train_hour])
	x_data=x_data.reshape(feature_num,len(y_data),train_hour)

	#compute error
	for i in range(len(y_data)):
		#y=b
		y=0
		for f in range(0,feature_num):
				for h in range(0,train_hour):
					y=y+w[f][h]*x_data[f][i][h]
		error_sum=error_sum+(y-y_data[i])**2
	return math.sqrt(error_sum/len(y_data))




def training(train_set,set_num,feature_num,pm25_pos,train_hour,n_fold_num):
	if(n_fold_num!=1):
		train_set=train_set.reshape(n_fold_num,feature_num*(day*hour//n_fold_num*(n_fold_num-1)))
	else:
		train_set=train_set.reshape(n_fold_num,feature_num*day*hour)
	for i in range(len(train_set[set_num])):
		if(train_set[set_num][i]<=0 or train_set[set_num][i]>500):
			train_set=inter(train_set,i,set_num)
	if(n_fold_num!=1):
		train_set=train_set.reshape(n_fold_num,feature_num,(day*hour//n_fold_num*(n_fold_num-1)))
	else:
		train_set=train_set.reshape(n_fold_num,feature_num,day*hour)



	x_data=np.array([])
	y_data=np.array([])

	#feature scaling
	temp_set=train_set
	for j in range(0,len(train_set[set_num][0])):
		mean=0.0
		sigma=0.0
		for i in range(0,feature_num):
			for j in  range(feature_num):
				mean=mean+train_set[set_num][i][j]
			mean=mean/feature_num
			for j in  range(feature_num):
				sigma=(train_set[set_num][i][j]-mean)**2+sigma
			sigma=math.sqrt(sigma/feature_num)
			if(sigma==0.0):
				sigma=1.0	
			for j in  range(feature_num):
				train_set[set_num][i][j]=(train_set[set_num][i][j]-mean)/sigma
	
		
	#get training data	
	for j in range(0,feature_num):
		for i in range(0,len(train_set[set_num][j])):
			if(i%(20*24)<(20*24-train_hour)):

				#delete wrong data
				check=0
				for k in range(train_hour+1):
					for f in range(0,feature_num):
						if(temp_set[set_num][f][i+k]<=0 or temp_set[set_num][f][i+k]>300):
							check=1
							break
				if(check==1):
					continue
				
				x_data=np.append(x_data,train_set[set_num][j][i:i+train_hour])
				if(j%feature_num==pm25_pos):
					y_data=np.append(y_data,train_set[set_num][j][i+train_hour])
	x_data=x_data.reshape(feature_num,len(y_data),train_hour)
	y_data=y_data.reshape(len(y_data),1)

	"""
	x_data_temp=np.array([])
	for i in range(len(y_data)):
		for j in range(feature_num):
			for k in range(train_hour):
				x_data_temp=np.append(x_data_temp,x_data[j][i][k])
	x_data_temp=x_data_temp.reshape(len(y_data),feature_num*train_hour)
	"""
	
	
	
	lr=0.01#inital learning rate
	iteration=100000
	b=0.0#initial b
	w=0.0#initial w
	w_arr=np.array([])
	w_grad=np.array([])
	for i in range(feature_num*train_hour):
		w_arr=np.append(w_arr,w)
	w_arr=w_arr.reshape(feature_num*train_hour)
	w_grad= np.zeros((feature_num*train_hour), dtype=float)
	m= np.zeros((feature_num*train_hour), dtype=float)
	v= np.zeros((feature_num*train_hour), dtype=float)
	m_temp= np.zeros((feature_num*train_hour), dtype=float)
	v_temp= np.zeros((feature_num*train_hour), dtype=float)
	p=0.9
	beta1=0.9
	beta2=0.999
	err=0.00000001
	
	"""
	print(len(y_data))
	w_arr=w_arr.reshape(feature_num*train_hour,1)
	xt=x_data_temp.T
	xtx=np.dot(xt,x_data_temp)
	xtx_1=np.linalg.inv(xtx)
	xtx_1xt=np.dot( xtx_1,xt)
	w_arr=np.dot(xtx_1xt,y_data)
	w_arr=w_arr.reshape(feature_num,train_hour)
	#w_arr=(xTx)(-1)xTy
	#iterations
	"""

	x_data_temp=np.array([])
	for i in range(len(y_data)):
		for j in range(feature_num):
			for k in range(train_hour):
				x_data_temp=np.append(x_data_temp,x_data[j][i][k])
	x_data_temp=x_data_temp.reshape(len(y_data),feature_num*train_hour)
	x_data=x_data_temp
	#training
	#print(len(y_data))
	for i in range(iteration):
		b_grad=0.0
		w_grad= np.zeros((feature_num*train_hour), dtype=float)
		for n in range(len(y_data)):
			wx=(w_arr*x_data_temp[n]).sum()
			w_grad=w_grad-2.0*(y_data[n]-wx)*x_data_temp[n]

		#adam gradient decent
		m=beta1*m+(1-beta1)*w_grad
		v=beta2*v+(1-beta2)*(w_grad**2)
		m_temp=m/(1-(beta1**(i+1)))
		v_temp=v/(1-(beta2**(i+1)))
		w_arr=w_arr-lr*m_temp/(np.sqrt(v_temp)+err)
		

		if(i%1000==0):
			print("f",feature_num," h ",train_hour,"inter 1",i, "lr: ",lr,"no b adam")
			print(w_arr,b)
	w_arr=w_arr.reshape(feature_num,train_hour)
	
	return 0,w_arr
	#return w_arr
