import numpy as np
import csv
from sys import argv
import os
from keras.models import load_model
import jieba
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
# argv: [1]test.csv [2]predict.csv [3]model.h5

jieba.load_userdict(argv[2])
model_vec = Word2Vec.load("model.bin")
words = list(model_vec.wv.vocab)


text = []
with open(argv[1], 'r') as f:
	count = 0
	for line in list(csv.reader(f))[1:]:
		#y=np.append(y,float(line[0]))
		seg_list=jieba.cut(line[1], cut_all=False)
		temp=[]
		for s in seg_list: 
			temp.append(s)
		text.append(temp)
		count += 1
		print('\rX_train: ' + repr(count), end='', flush=True)
	print('', flush=True)

new_sentences = []
count = 0
max_sentence_len=-1
for sen in text:
	new_sen = []
	count_len=0
	for word in sen:
		#print(model_vec.wv.vocab[word].index)
		count_len+=1
		try:
			new_sen.append(model_vec.wv.vocab[word].index)
			#new_sen.append(model_vec.wv[word])
		except:
			new_sen.append(0)
		if(count_len>2000):
			break
	if(max_sentence_len<count_len):
		max_sentence_len=count_len
	new_sentences.append(new_sen)
	#print(new_sen)
	count += 1
	print('\rSentence ' + repr(count), end='', flush=True)
x_test=np.array(new_sentences[0:len(new_sentences)])
x_test = pad_sequences(x_test, maxlen=2001)

#labels = [int(round(x[0])) for x in model.predict(x_test) ]

model = load_model("temp/model1.hdf5")
result = model.predict(x_test, batch_size = 400, verbose = 1)
with open(argv[3], 'w') as f:
	f.write('id,label\n')
	for i in range(len(result)):
		predict = int(round(result[i][0]))
		f.write(repr(i) + ',' + repr(predict) + '\n')



