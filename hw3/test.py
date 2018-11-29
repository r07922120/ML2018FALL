import numpy as np
import csv
from sys import argv
import os
from keras.models import load_model


# argv: [1]test.csv [2]predict.csv [3]model.h5

data = []
with open(argv[1], 'r') as f:
	for line in list(csv.reader(f))[1:]:
		data.append( [float(x) for x in line[1].split()] )
x = np.array(data)
x = x/255
x = x.reshape(x.shape[0], 48, 48, 1)

model = load_model("temp/model0.hdf5")
result = model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model1.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model2.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model3.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model4.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model5.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model6.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model7.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model8.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)
model = load_model("temp/model9.hdf5")
result = result+model.predict(x, batch_size = 100, verbose = 1)


with open(argv[2], 'w') as f:
	f.write('id,label\n')
	for i in range(len(result)):
		predict = np.argmax(result[i])
		f.write(repr(i) + ',' + repr(predict) + '\n')



