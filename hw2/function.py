import numpy as np
import math
import csv

# check the wrong data
def wrong(arr,feature):
	for i in range(len(arr)):
		if(feature[i]==0 and arr[i]<=0):
			return 1
		elif(feature[i]==1 and (arr[i]<1 or arr[i]>2)):
			return 1
		elif(feature[i]==2 and (arr[i]<1 or arr[i]>4)):
			return 1
		elif(feature[i]==3 and (arr[i]<1 or arr[i]>3)):
			return 1
		elif(feature[i] in [4,11,12,13,14,15,16,17,18,19,20,21,22] and arr[i]<0):
			return 1
	return 0
#correct the PAYMONTH
def re(arr,feature):
	for i in range(len(arr)):
		if(feature[i] in [5,6,7,8,9,10]):
			if(arr[i]<0):
				arr[i]=0
			elif(arr[i]>9):
				arr[i]=9
			else:
				arr[i]=arr[i]
	return arr
#feature scaling
def scaling(arr,feature_num,num):
	mean= np.zeros((feature_num), dtype=float)
	sigma= np.zeros((feature_num), dtype=float)
	for i in range(feature_num):
		sum_temp=0.0
		for j in range(num):
			sum_temp+=arr[j][i]
		mean[i]=sum_temp/num
		for j in range(num):
			sigma[i]+=(arr[j][i]-mean[i])**2
		sigma[i]=math.sqrt(sigma[i]/num)
		if(sigma[i]==0.0):
			sigma[i]=1.0	
		for j in range(num):
				arr[j][i]=(arr[j][i]-mean[i])/sigma[i]
	return arr,mean,sigma





#correct the wrong data
def correct(arr,mean,num,feature_num,feature):
	for i in range(num):
		for j in range(feature_num):
			if(feature[j]==0 and arr[i][j]<=0):
				arr[i][j]=mean[j]
			elif(feature[j]==1 and (arr[i][j]<1 or arr[i][j]>2)):
				arr[i][j]=round(mean[j],0)
			elif(feature[j]==2 and (arr[i][j]<1 or arr[i][j]>4)):
				arr[i][j]=round(mean[j],0)
			elif(feature[j]==3 and (arr[i][j]<1 or arr[i][j]>3)):
				arr[i][j]=round(mean[j],0)
			elif(feature[j] in [4,11,12,13,14,15,16,17,18,19,20,21,22] and arr[i][j]<0):
				arr[i][j]=mean[j]
	return arr

#one hot encoding
def encoding(train_x,feature_num,feature,num):
	male= np.zeros((num), dtype=float)
	female= np.zeros((num), dtype=float)
	graduate= np.zeros((num), dtype=float)
	university= np.zeros((num), dtype=float)
	high= np.zeros((num), dtype=float)
	education_other= np.zeros((num), dtype=float)
	marry= np.zeros((num), dtype=float)
	single= np.zeros((num), dtype=float)
	matiral_others= np.zeros((num), dtype=float)
	for i in range(num):
		for j in range(feature_num):
			if(feature[j] == 1):
				if(train_x[i][j]==1):
					male[i]=1
				else:
					female[i]=1
				train_x[i][j]=0
			elif(feature[j] == 2):
				if(train_x[i][j]==1):
					graduate[i]=1
				elif(train_x[i][j]==2):
					university[i]=1
				elif(train_x[i][j]==3):
					high[i]=1
				else:
					education_other[i]=1
				train_x[i][j]=0
			elif(feature[j] == 3):
				if(train_x[i][j]==1):
					marry[i]=1
				elif(train_x[i][j]==2):
					single[i]=1
				else:
					matiral_others[i]=1
				train_x[i][j]=0
	feature_num+=9
	train_x=np.c_[train_x,male]
	train_x=np.c_[train_x,female]
	train_x=np.c_[train_x,graduate]
	train_x=np.c_[train_x,university]
	train_x=np.c_[train_x,high]
	train_x=np.c_[train_x,education_other]
	train_x=np.c_[train_x,marry]
	train_x=np.c_[train_x,single]
	train_x=np.c_[train_x,matiral_others]
	return train_x,feature_num
