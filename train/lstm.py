import time
import cv2
import numpy as np
import math
from os import listdir, path
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

# LSTM model import
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.optimizers import Adam
# Timer
start_time = time.time()

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

# Global variables
curr_dir = path.dirname(__file__)
root_dir = path.dirname(curr_dir)
n_steps = 1
train_len = 332

# Load array
dataset = np.load(path.join(root_dir, 'data', 'reconstructed_gray', 'lstm.npy'))

# data = dataset[:,0:3]

train = dataset[0:train_len,]
test = dataset[(train_len - n_steps):342,]

# Split
X_train, y_train = split_sequences(train, n_steps)
X_test, y_test = split_sequences(test, n_steps)
print(y_test.shape)

n_features = X_train.shape[2]

# Define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_features))

opt = Adam(learning_rate=0.001)
model.compile(optimizer = opt, loss = 'mse')

# Fit model
model.fit(X_train, y_train, epochs=500, verbose=2)

# Make prediction
trainPredict = model.predict(X_train)
# testPredict = model.predict(X_test)
# print(trainPredict.shape)
# print(testPredict.shape)

testPredict = np.zeros(shape=(10,(64*512)))
testPredict[0] = model.predict(X_test[0].reshape(1,1,(64*512)))

for i in range(9):
	new_X_pred = testPredict[i].reshape(1,(64*512))
	testPredict[i+1] = model.predict(new_X_pred.reshape(1,1,(64*512)))

# rescale
y_train, trainPredict, y_test, testPredict = y_train * 255, trainPredict * 255, y_test * 255, testPredict * 255

for i in range(342 - train_len):
	print('---------------------------------------')
	print('Pred pic',i,':')
	# calculate root mean squared error
	# trainScore = math.sqrt(mean_squared_error(y_train[:,0], trainPredict[:,0]))
	# print('Train Score: %.2f RMSE' % (trainScore))
	testScore = math.sqrt(mean_squared_error(y_test[i,:], testPredict[i,:]))
	print('Test Score: %.2f RMSE' % (testScore))

	# trainScore = ssim(y_train[:,0], trainPredict[:,0])
	# print('Train Score: %.2f SSIM' % (trainScore))
	y_test_ssim = y_test[i,:].reshape(64,512)
	testPredict_ssim = testPredict[i,:].reshape(64,512)
	testScore = ssim(y_test_ssim, testPredict_ssim, data_range = (testPredict_ssim.max() - testPredict_ssim.min()))
	print('Test Score: %.2f SSIM' % (testScore))
	



# for i in range(testPredict.shape[0]):
# 	temp = testPredict[i]
# 	temp = temp.reshape(64, 512)

# 	name = str(i) + '.png'
# 	cv2.imwrite(path.join(curr_dir, 'test_pred_gray', 'LSTM', name), temp)

# # plot baseline and predictions
# plt.plot(y_test[:,0])
# plt.plot(testPredict[:,0])
# # plt.plot(testPredictPlot)
# plt.show()

print('Time: %s seconds.' % (time.time() - start_time))