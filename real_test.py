import numpy as np
import csv
import sys

feature_num_start=18

feature=[1,2,7,8,9,12]
#feature=[8,9]
train_hour=5 #how many hour for trainning
for i in range(len(feature)):
	if(feature[i]==9):
		pm25_pos=i
id_total=260
feature_num=len(feature)
a=np.array([])

try:
    if (sys.argv[1] == '-'):
        f = sys.stdin.read().splitlines()
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        #input_f = open(input_file, 'r')
        #output_f = open(output_file, 'r')
except :
    print ("Error Reading from file:")
file=np.genfromtxt(input_file,delimiter=',',dtype=float,encoding="big5")
b=np.genfromtxt('b.csv',delimiter=',',dtype=float)
w=np.genfromtxt('w.csv',delimiter=',',dtype=float)


#interpolation wrong data
def inter(test_set,id_num,f,h):
	low=0
	low_index=0
	high=0
	high_index=0
	
	for i in range(1,10):
		if((h-i)<0):
			break
		else:	
			if(test_set[id_num][f][h-i]>0 and test_set[id_num][f][h-i]<=500):
				if(low_index==0):
					low_index=i
					low=test_set[id_num][f][h-i]
	for i in range(1,10):
		if(h+i>=len(test_set[id_num][f])):
			break
		else:	
			if(test_set[id_num][f][h+i]>0 and test_set[id_num][f][h+i]<=500):
				if(high_index==0):
					high_index=i
					high=test_set[id_num][f][h+i]
	if(high_index!=0 and low_index!=0):
		test_set[id_num][f][h]=low_index/(high_index+low_index)*(high-low)+low
	elif(high_index!=0 and low_index==0):
		test_set[id_num][f][h]=high
	elif(high_index==0 and low_index!=0):
		test_set[id_num][f][h]=low

	if(h>=train_hour and f==pm25_pos):
		y_cal=b
		for f_num in range(0,feature_num):
			for h_num in range(h-train_hour,h):
				y_cal=y_cal+w[f_num][h_num]*test_set[id_num][f_num][h_num]
		test_set[id_num][f][k]=y_cal
	return test_set



for i in range(0,id_total):
	for j in range(0,feature_num_start):
		if(j in feature):
			a=np.append(a,file[i*feature_num_start+j][11-train_hour:11])
a=a.reshape(id_total,feature_num,train_hour)


for i in range(id_total):
	for f in range(feature_num):
		for h in range(train_hour):
			if(a[i][f][h]<=0 or a[i][f][h]>=500):
				a=inter(a,i,f,h)

w=w.reshape(feature_num,train_hour)


y=np.array([])


x_data=np.array([])
with open(output_file, 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(['id', 'value'])
	for n in range(id_total):
		y_cal=b
		for f in range(0,feature_num):
			for h in range(0,train_hour):
				y_cal=y_cal+w[f][h]*a[n][f][h]

		writer.writerow(['id_%d'%(n), y_cal])

