import time
import cv2
import numpy as np
from os import listdir, path
from matplotlib import pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model

# Timer
start_time = time.time()

# Get all file names in images folder
curr_path = path.dirname(__file__)
encode_path = path.join(curr_path, 'autoencoded_img')
name_list = listdir(curr_path + '\images')

row = 1024
column = 512

table = []
image_array = []
img_name = []

for i in range(20):
    # Get file path
    file_path = path.join(curr_path, 'images', name_list[i])

    # Read image and convert to RGB
    img = cv2.imread(file_path)
    img = cv2.resize(img, (row, column))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.asarray(img)
    img_array = img_array.flatten()
    img_array = img_array.astype(np.float64)
    img_array = (img_array / 255)
    image_array.append(img_array)
    img_name.append(name_list[i])

image_array = np.array(image_array)
print(image_array.shape)

    # this is the size of our encoded representations
print("Creating/Training Neural Network")
encoding_dim = 64
# this is our input placeholder
input_img = Input(shape=(image_array.shape[1],))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(image_array.shape[1], activation='sigmoid')(encoded)

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

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(image_array, image_array,
                epochs=50,
                shuffle=False)

encoded_imgs = encoder.predict(image_array)
decoded_imgs = decoder.predict(encoded_imgs)
table.append(encoded_imgs)

for i in range(image_array.shape[0]):
    print(encoded_imgs[i])
    print(decoded_imgs[i])
    img_array2 = decoded_imgs[i].reshape(column, row, 3)
    img_array2 = (img_array2 * 255)
    img_array2 = img_array2.astype(np.uint8)
    file_path = path.join(encode_path, img_name[i])
    cv2.imwrite(file_path, img_array2)
    print('Time: %s seconds.' % (time.time() - start_time))

with open('out.txt', 'wb') as outfile:
    for data_slice in table:
        np.savetxt(outfile, data_slice, fmt='%4.3f')
