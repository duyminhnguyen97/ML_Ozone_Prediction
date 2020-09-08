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
north_R = []
north_G = []
north_B = []

south_R = []
south_G = []
south_B = []

for i in range(len(name_list)):
    # Get img RGB
    file_path = path.join(curr_dir, 'images', name_list[i])
    img = cv2.imread(file_path)
    #print(img.shape)
    img = cv2.resize(img, (horizontal, vertical))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img)
    img = img.astype(np.float32)

    # plt.imshow(img, cmap="gray")
    # plt.show()

    img = np.asarray(img)
    north = img[0:int(vertical/4),] / 255
    south = img[int(vertical - vertical/4):int(vertical),] / 255

    cv2.imwrite(path.join(curr_dir, 'reconstructed_rgb', 'north', name_list[i]), cv2.cvtColor(north * 255, cv2.COLOR_BGR2RGB))
    cv2.imwrite(path.join(curr_dir, 'reconstructed_rgb', 'south', name_list[i]), cv2.cvtColor(south * 255, cv2.COLOR_BGR2RGB))

    nR = north[:,:,0]
    nG = north[:,:,1]
    nB = north[:,:,2]

    sR = south[:,:,0]
    sG = south[:,:,1]
    sB = south[:,:,2]

    north_R.append(nR.flatten())
    north_G.append(nG.flatten())
    north_B.append(nB.flatten())

    south_R.append(sR.flatten())
    south_G.append(sG.flatten())
    south_B.append(sB.flatten())

north_R = np.asarray(north_R).transpose()
print(north_R.shape)
north_G = np.asarray(north_G).transpose()
print(north_G.shape)
north_B = np.asarray(north_B).transpose()
print(north_B.shape)

south_R = np.asarray(south_R).transpose()
print(south_R.shape)
south_G = np.asarray(south_G).transpose()
print(south_G.shape)
south_B = np.asarray(south_B).transpose()
print(south_B.shape)

np.save(path.join(curr_dir, 'reconstructed_rgb', 'north', 'R'), north_R)
np.save(path.join(curr_dir, 'reconstructed_rgb', 'north', 'G'), north_G)
np.save(path.join(curr_dir, 'reconstructed_rgb', 'north', 'B'), north_B)

np.save(path.join(curr_dir, 'reconstructed_rgb', 'south', 'R'), south_R)
np.save(path.join(curr_dir, 'reconstructed_rgb', 'south', 'G'), south_G)
np.save(path.join(curr_dir, 'reconstructed_rgb', 'south', 'B'), south_B)

print('Time: %s seconds.' % (time.time() - start_time))