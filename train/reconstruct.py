import time
import cv2
import numpy as np
import pickle as pk
from os import listdir, path
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

# Timer
start_time = time.time()

# Global var
curr_dir = path.dirname(__file__)
root_dir = path.dirname(curr_dir)

# Load PCA Prediction
pca_pred = np.load(path.join(curr_dir, 'pca_pred.npy'))
pca_pred = pca_pred.transpose().reshape(6, 1024, 150)

# Load PCA model
pkl_filename = path.join(root_dir, 'data', 'models', 'model_gray.pkl')
pca_X = pk.load(open(pkl_filename, 'rb'))

for i in range(pca_pred.shape[0]):
    temp = pca_pred[i]
    X_inv_proj = pca_X.inverse_transform(temp)
    print(X_inv_proj.shape)

    name = str(i) + '.png'
    cv2.imwrite(path.join(curr_dir, 'pred_gray_img', name), X_inv_proj)
    
print('Time: %s seconds.' % (time.time() - start_time))