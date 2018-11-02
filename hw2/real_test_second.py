import numpy as np
import csv
import sys
import math
import function

norm=1
feature_num_start=23
#feature=[0,1,2,3,4,5,11,12,13,14,15,16,17,18,19]
feature=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
feature_num=len(feature)
num=10000
people_id=np.arange(0,num)
ratio=0.475
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
b=np.genfromtxt('b_gen.csv',delimiter=',',dtype=float)
w=np.genfromtxt('w_gen.csv',delimiter=',',dtype=float)
mean=np.genfromtxt('mean_gen.csv',delimiter=',',dtype=float)
sigma=np.genfromtxt('sigma_gen.csv',delimiter=',',dtype=float)


for n in range(1,num+1):
	a=np.append(a,file[n][feature])
a=a.reshape(num,feature_num)

a=function.correct(a,mean,num,feature_num,feature)

a_temp=np.array([])
for n in range(num):
	for i in range(norm):
		a_temp=np.append(a_temp,function.re(a[n],feature)**(i+1))
a=a_temp

a=a.reshape(num,feature_num)
#a,feature_num=function.encoding(a,feature_num*norm,feature,num)
for i in range(num):
	a[i]=(a[i]-mean)/sigma

w=w.reshape(feature_num*norm)



with open(output_file, 'w') as csvfile:
	writer = csv.writer(csvfile, delimiter=',')
	writer.writerow(['id', 'value'])
	for n in range(num):
		y_cal=b
		for f in range(0,feature_num):
			y_cal=y_cal+w[f]*a[n][f]
		
		r=1/(1+math.exp(-y_cal))
		if(r>=ratio):
			r=1
		else:
			r=0
		writer.writerow(['id_%d'%(n), r])

