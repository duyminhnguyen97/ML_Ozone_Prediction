# Execute this file to update images from NOAA folder

import time
import os, os.path
from ftplib import FTP
from contextlib import closing

# Timer
start_time = time.time()

# Connect to directory
ftp = FTP('public.sos.noaa.gov')
ftp.login()
ftp.cwd('rt/ozone/4096')

# List of files
name_list = []
ftp.retrlines("NLST", name_list.append)

# Download images
for i in range(len(name_list)):
     # Set save directory
    save_path = os.path.dirname(__file__)
    file_name = os.path.join(save_path, 'images', name_list[i])

    # Check if file already exist
    if os.path.exists(file_name) == True:
        continue

            # Download file
    else:
        with open(file_name, 'wb') as f:
            ftp.retrbinary('RETR ' + name_list[i], f.write)
            f.close()


print('Time: %s seconds.' % (time.time() - start_time))
