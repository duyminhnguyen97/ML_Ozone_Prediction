import time
import cv2
import numpy as np
from os import listdir, path
from matplotlib import pyplot as plt
# Timer
start_time = time.time()

# Get all file names in images folder
curr_path = path.dirname(__file__)
name_list = listdir(curr_path + '\images')
name_list.remove('.gitignore') # Remove .gitignore file

# Resize dimension
horizontal = 2048
vertical = 1024
block = 4

# Random Weight
weight = []
np.random.seed(1)
for i in range(block):
    weight.append(np.random.uniform(-1, 1))
print('Weight is:',weight)

b = []

# Main
for i in range(10):
    # Get img RGB
    file_path = path.join(curr_path, 'images', name_list[i])
    img = cv2.imread(file_path)
    #print(img.shape)
    img = cv2.resize(img, (horizontal, vertical))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    img = img.astype(np.float64)
    #img = (img / 255)

    # Split into 3 array RGB
    rgb = np.dsplit(img, 3)
    green = np.asarray(rgb[1]).reshape(1024, 2048)

    temp_arr = []

    for j in range(16):
        for k in range(16):
            temp_arr.append(green[j][k])
    temp_arr = np.asarray(temp_arr).reshape(16,16)
    b.append(temp_arr)

b = np.asarray(b)
print(b)
print(b.shape)