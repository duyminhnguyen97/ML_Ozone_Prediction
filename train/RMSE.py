import numpy as np
import math
import time
import cv2
from os import listdir, path
from sklearn.metrics import mean_squared_error

# Timer
curr_time = time.time()

# Global Variable
curr_dir = path.dirname(__file__)
root_dir = path.dirname(curr_dir)

name_list = listdir(path.join(root_dir, 'data', 'images'))
name_list.remove('.gitignore')
name_list_test = listdir(path.join(root_dir, 'train', 'test_pred_rgb', 'north'))
print(name_list_test)


# Resize dimension
horizontal = 512
vertical = 256

file_path_ori = path.join(root_dir, 'data', 'reconstructed_rgb', 'north', name_list[len(name_list) - 1])
img = cv2.imread(file_path_ori)
print(img.shape)
img = cv2.resize(img, (horizontal, vertical))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.asarray(img)
img = img.astype(np.float32)
img = np.asarray(img)
north = img[0:int(vertical/4),] / 255

##############################################
file_path_new = path.join(curr_dir, 'test_pred_rgb', 'north', name_list_test[len(name_list_test) - 1])
img = cv2.imread(file_path_new)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

testScore = math.sqrt(mean_squared_error((north), (img)))
print(testScore)