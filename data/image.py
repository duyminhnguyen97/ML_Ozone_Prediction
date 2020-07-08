import time
import cv2
import numpy as np
import pickle as pk
from os import listdir, path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# Timer
start_time = time.time()

# Get all file names in images folder
curr_path = path.dirname(__file__)
name_list = listdir(curr_path + '\images')
name_list.remove('.gitignore') # Remove .gitignore file

# Resize dimension
horizontal = 2048
vertical = 1024
ncomp = 150

pca_matrix = []

# Main
pca_X = PCA(ncomp)

for i in range(len(name_list)):
    # Get img RGB
    file_path = path.join(curr_path, 'images', name_list[i])
    img = cv2.imread(file_path)
    #print(img.shape)
    img = cv2.resize(img, (horizontal, vertical))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    img = img.astype(np.float64)

    # plt.imshow(img, cmap="gray")
    # plt.show()

    img = np.asarray(img)

    X_proj = pca_X.fit_transform(img)
    print(X_proj)
    print('-------------------------------------------')
    X_inv_proj = pca_X.inverse_transform(X_proj)
    # print(np.cumsum(pca_X.explained_variance_ratio_))

    cv2.imwrite(path.join(curr_path, 'reconstructed_img', name_list[i]), X_inv_proj)

    pca_matrix.append(X_proj.flatten())

pca_matrix = np.asarray(pca_matrix)
pca_matrix = pca_matrix.transpose()
print(pca_matrix)

# sample_arr = pca_matrix[0:5,]
# print(sample_arr.shape)
np.save(path.join(curr_path, 'pca'), pca_matrix)

pkl_filename = path.join(curr_path, 'models', 'model_gray.pkl')
with open(pkl_filename, 'wb') as file:
    pk.dump(pca_X, file, protocol=4)

print('Time: %s seconds.' % (time.time() - start_time))




# Old code for whole dataset
""" import time
import cv2
import numpy as np
import pickle as pk
from os import listdir, path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# Timer
start_time = time.time()

# Get all file names in images folder
curr_path = path.dirname(__file__)
name_list = listdir(curr_path + '\images')
name_list.remove('.gitignore') # Remove .gitignore file

# Resize dimension
horizontal = 2048
vertical = 1024
ncomp = len(name_list)

img_gray = []

# Main
for i in range(len(name_list)):
    # Get img RGB
    file_path = path.join(curr_path, 'images', name_list[i])
    img = cv2.imread(file_path)
    #print(img.shape)
    img = cv2.resize(img, (horizontal, vertical))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    img = img.astype(np.float64)

    # plt.imshow(img, cmap="gray")
    # plt.show()

    img = img.flatten()
    img_gray.append(img)

img_gray = np.asarray(img_gray)
print(img_gray.shape)

pca_X = PCA(ncomp)

# X_proj has np columns with main components
X_proj = pca_X.fit_transform(img_gray)
print(X_proj.shape)

X_inv_proj = pca_X.inverse_transform(X_proj) #reshaping
print(X_inv_proj.shape)
print(np.cumsum(pca_X.explained_variance_ratio_))

for i in range(X_inv_proj.shape[0]):
    img_new = X_inv_proj[i].reshape(vertical, horizontal)
    cv2.imwrite(path.join(curr_path, 'reconstructed_img', name_list[i]), img_new)

pkl_filename = path.join(curr_path, 'models', 'model_gray.pkl')
with open(pkl_filename, 'wb') as file:
    pk.dump(pca_X, file, protocol=4)

# for i in range(X_inv_proj.shape[0]):
#     img_inv = X_inv_proj[i].reshape(vertical, horizontal)
#     plt.imshow(img_inv, cmap="gray")
#     plt.show()

print('Time: %s seconds.' % (time.time() - start_time)) """