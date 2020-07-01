import time
import cv2
import numpy as np
from os import listdir, path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# Timer
start_time = time.time()

# Get all file names in images folder
curr_path = path.dirname(__file__)
name_list = listdir(path.join(curr_path, 'images'))
name_list.remove('.gitignore') # Remove .gitignore file

# Resize dimension
horizontal = 1024
vertical = 512
ncomp = len(name_list)

r = []

# Main
for i in range(len(name_list)):
    # Get img RGB
    file_path = path.join(curr_path, 'images', name_list[i])
    img = cv2.imread(file_path)
    # print(img.shape)
    img = cv2.resize(img, (horizontal, vertical))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    img = img.astype(np.float64)
    img = img.flatten()
    r.append(img)
    # plt.imshow(img, cmap="gray")
    # plt.show()

r = np.asarray(r)
print(r)
print(r.shape)


pca_X = PCA(ncomp)

# X_proj has np columns with main components
X_proj = pca_X.fit_transform(r)
print(X_proj.shape)

X_inv_proj = pca_X.inverse_transform(X_proj) #reshaping
print(X_inv_proj.shape)
print(np.cumsum(pca_X.explained_variance_ratio_))

for i in range(X_inv_proj.shape[0]):
    img_new = X_inv_proj[i].reshape(vertical, horizontal,3)
    cv2.imwrite(path.join(curr_path, 'reconstructed_rgb', name_list[i]), img_new)

# for i in range(X_inv_proj.shape[0]):
#     img_inv = X_inv_proj[i].reshape(vertical, horizontal)
#     plt.imshow(img_inv, cmap="gray")
#     plt.show()

print('Time: %s seconds.' % (time.time() - start_time))