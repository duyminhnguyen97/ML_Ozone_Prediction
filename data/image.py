import time
import cv2
import numpy as np
from os import listdir, path

# Timer
start_time = time.time()

# Get all file names in images folder
curr_dir = path.dirname(__file__)
name_list = listdir(path.join(curr_dir, 'images'))
name_list.remove('.gitignore')  # Remove .gitignore file

# Resize dimension
horizontal = 512
vertical = 256

# Main
north_matrix = []
south_matrix = []

for i in range(len(name_list)):
    # Get img RGB
    file_path = path.join(curr_dir, 'images', name_list[i])
    img = cv2.imread(file_path)
    #print(img.shape)
    img = cv2.resize(img, (horizontal, vertical))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.asarray(img)
    img = img.astype(np.float32)

    # plt.imshow(img, cmap="gray")
    # plt.show()

    north = img[0:int(vertical / 4),:] / 255
    south = img[int(vertical * 3 / 4):int(vertical),:] / 255


    cv2.imwrite(path.join(curr_dir, 'reconstructed_gray', 'north', name_list[i]), north * 255)
    cv2.imwrite(path.join(curr_dir, 'reconstructed_gray', 'south', name_list[i]), south * 255)

    north_matrix.append(north.flatten())
    south_matrix.append(south.flatten())
    
north_matrix = np.asarray(north_matrix).transpose()
print(north_matrix.shape)

south_matrix = np.asarray(south_matrix).transpose()
print(south_matrix.shape)

np.save(path.join(curr_dir, 'reconstructed_gray', 'north', 'gray_north'), north_matrix)
np.save(path.join(curr_dir, 'reconstructed_gray', 'south', 'gray_south'), south_matrix)

print('Time: %s seconds.' % (time.time() - start_time))