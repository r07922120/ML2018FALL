import jieba
import numpy as np
import csv
from sys import argv
import os
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model,to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from gensim.models import Word2Vec
import logging

jieba.load_userdict(argv[4])



text = []  # list of text samples
label = []  # list of label ids
"""
for name in sorted(os.listdir(TEXT_DATA_DIR)):
	path = os.path.join(TEXT_DATA_DIR, name)
	if os.path.isdir(path):
		label_id = len(labels_index)
		labels_index[name] = label_id
		for fname in sorted(os.listdir(path)):
			if fname.isdigit():
				fpath = os.path.join(path, fname)
				if sys.version_info < (3,):
					f = open(fpath)
				else:
					f = open(fpath, encoding='latin-1')
				t = f.read()
				i = t.find('\n\n')  # skip header
				if 0 < i:
					t = t[i:]
				texts.append(t)
				f.close()
				labels.append(label_id)
"""
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
train_len=len(text)
with open(argv[3], 'r') as f:
	count = 0
	for line in list(csv.reader(f))[1:]:
		#y=np.append(y,float(line[0]))
		seg_list=jieba.cut(line[1], cut_all=False)
		temp=[]
		for s in seg_list: 
			temp.append(s)
		text.append(temp)
		count += 1
		print('\rX_test: ' + repr(count), end='', flush=True)
	print('', flush=True)
with open(argv[2], 'r') as f:
	count = 0
	for line in list(csv.reader(f))[1:]:
		#y=np.append(y,float(line[0]))
		label.append(int(line[1]))
		count += 1
		print('\rY_train: ' + repr(count), end='', flush=True)
	print('', flush=True)




# train model
model_vec = Word2Vec(text, min_count=1)
# summarize the loaded model
#print(model)
# summarize vocabulary
#words = list(model_vec.wv.vocab)
#print(words)
# access vector for one word
#print(model['東施'])
# save model
model_vec.save('model.bin')

pretrained_weights = model_vec.wv.syn0
vocab_size, emdedding_size = pretrained_weights.shape

vocab_dim = 100
batch_size = 400
n_epoch = 10


text=text[0:train_len]
new_sentences = []
count = 0
max_sentence_len=-1
repeat_len=0
for sen in text:
	new_sen = []
	count_len=0
	repeat_len=0
	for word in sen:
		#print(model_vec.wv.vocab[word].index)
		count_len+=1
		new_sen.append(model_vec.wv.vocab[word].index)
		new_sen.append(model_vec.wv.vocab[word].index)
		if(count_len>2000):
			break
	if(max_sentence_len<count_len):
		max_sentence_len=count_len
	new_sentences.append(new_sen)
	#print(new_sen)
	count += 1
	print('\rSentence ' + repr(count), end='', flush=True)
#print(new_sentences)
rng_state = np.random.get_state()
np.random.shuffle(new_sentences)
np.random.set_state(rng_state)
np.random.shuffle(label)
x_train=np.array(new_sentences[3000:len(new_sentences)])
x_test=np.array(new_sentences[0:3000])
y_train=np.array(label[3000:len(new_sentences)])
y_test=np.array(label[0:3000])
x_train =pad_sequences(x_train, maxlen=max_sentence_len)
x_test = pad_sequences(x_test, maxlen=max_sentence_len)


"""
train_x = np.zeros([len(new_sentences), max_sentence_len], dtype=np.int32)
for i, sentence in enumerate(new_sentences):
	for t, word in enumerate(new_sentences[:-1]):
		#word2idx(word)
		print(word)
		#train_x[i, t] = word2idx(word)
"""


model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=emdedding_size, weights=[pretrained_weights],input_length=max_sentence_len))
#model.add(LSTM(units=emdedding_size))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.summary()
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

filepath="temp/model1.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

"""
model.add(BatchNormalization(axis=-1))
model.add(Dense(units=vocab_size))
model.add(BatchNormalization(axis=-1))
"""
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,validation_data=(x_test, y_test),callbacks=[checkpoint])
