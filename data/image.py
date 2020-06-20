import time
import cv2
import numpy as np
from scipy import linalg as la
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
print('Weight is:',weight,'\n')

b = []

# Main
for i in range(1):
    # Get img RGB
    file_path = path.join(curr_path, 'images', name_list[i])
    img = cv2.imread(file_path)
    #print(img.shape)
    img = cv2.resize(img, (horizontal, vertical))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    img = img.astype(np.float64)
    #img = (img / 255)
    print(img)
    # plt.imshow(img, cmap="gray")
    # plt.show()
    # Split into 3 array RGB
    # rgb = np.dsplit(img, 3)
    # green = np.asarray(rgb[1]).reshape(1024, 2048)

    temp_arr = []

    for j in range(16):
        for k in range(16):
            temp_arr.append(img[j][k])
    temp_arr = np.asarray(temp_arr).reshape(16,16)
    b.append(temp_arr)

b = np.asarray(b)
print(b)
#print(b.shape)

# a = []
# for i in range(1, 7):
#     temp = []
#     for j in range(1,7):
#         temp.append(i)
#     a.append(temp)
# a = np.asarray(a)
# print(a,'\n')

# rank = np.linalg.matrix_rank(b)

# print(rank)

# (P, L, U) = la.lu(a)
# D = np.diag(np.diag(U))
# U /= np.diag(U)[:, None]

# print(P,'\n')
# print(L,'\n')
# print(D,'\n')
# print(U,'\n')
# a_new = P.dot(L.dot(D.dot(U)))
# print(a_new)