import time
import cv2
import pandas as pd
from os import listdir, path
from matplotlib import pyplot as plt

# Timer
start_time = time.time()

# Get all file names in images folder
curr_path = path.dirname(__file__)
name_list = listdir(curr_path + '\images')

# Downscale variables
horizontal = 512
vertical = 256

# Create RGB multiindex table
# Create columns
cols = []
for i in range(vertical):
    for j in range(horizontal):
        _ = (str(i) + ',' + str(j))
        cols.append(_)

# Create table
columns = pd.MultiIndex.from_product([cols, ['R', 'G', 'B']], names=['pixel', 'color'])
data = pd.DataFrame(None, columns = columns)

# Get RGB of each image
for i in range(1):
    # Get file path
    file_path = path.join(curr_path, 'images', name_list[i])

    # Read image, resize to 1024x512 and convert to RGB
    img_src = cv2.imread(file_path)
    img_resize = cv2.resize(img_src, (horizontal, vertical))
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    # Load into table
    for a in range(vertical):
        for b in range(horizontal):
            curr_col = (str(a) + ',' + str(b))
            data.loc[i,[curr_col]] = img_rgb[a][b]
print(data)
print('Time: %s seconds.' % (time.time() - start_time))


"""
for i in range(362,365):
    # Get file path
    file_path = path.join(curr_path, 'images', name_list[i])

    # Read image, resize to 1024x512 and convert to RGB
    img_src = cv2.imread(file_path)
    img_resize = cv2.resize(img_src, (128, 64))
    img_resize = cv2.resize(img_resize, (4096, 2048))
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.show()
"""




#--------------------------------------------------------
from keras.layers import Input, Dense
from keras.models import Model

# this is the size of our encoded representations
encoding_dim = 524288  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(8388608,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(8388608, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# -------------------------------------------------------------
import time
import cv2
import numpy as np
from os import listdir, path
from matplotlib import pyplot as plt
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# Timer
start_time = time.time()

# Get all file names in images folder
curr_path = path.dirname(__file__)
name_list = listdir(curr_path + '\images')

row = 1024
column = 512

image_array = []

for i in range(1):
    # Get file path
    file_path = path.join(curr_path, 'images', name_list[i])

    # Read image and convert to RGB
    img = cv2.imread(file_path)
    img = cv2.resize(img, (row, column))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.asarray(img)
    img_array = img_array.flatten()
    img_array = img_array.astype(np.float64)
    img_array = (img_array / 255)
    image_array.append(img_array)

image_array = np.array(image_array)
print(image_array.shape)

# Fit regression DNN model.
print("Creating/Training Neural Network")
model = Sequential()
model.add(Dense(50, input_dim=image_array.shape[1], activation='relu'))

model.add(Dense(image_array.shape[1], activation = 'sigmoid'))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(image_array,image_array,verbose=0,epochs=100)

print("Done Training - Predicting / Scoring")
pred = model.predict(image_array)
print('Time: %s seconds.' % (time.time() - start_time))

for i in range(len(pred)):
    print(pred[i])
    img_array2 = pred[i].reshape(column,row,3)
    img_array2 = (img_array2 * 255)
    img_array2 = img_array2.astype(np.uint8)
    plt.imshow(img_array2)
    plt.show()