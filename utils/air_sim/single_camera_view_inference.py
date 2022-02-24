# Author: Trevor Sherrard
# Since: Feb. 23, 2021 
# Purpose: Demonstrate built in image segmentation 
#          capabilities of airsim enviornment. 

import airsim
import cv2
import numpy as np

# attempt to get client connection to airsim
client = airsim.MultirotorClient()
client.confirmConnection()

# set all template cube meshes to a segmentation object id of 20
print("setting cube segmentation ID")
found = client.simSetSegmentationObjectID("templatecube[\w]*", 20, True)
print("Done: %r" % found)

# get images from client
responses = client.simGetImages(
    [airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False)]
    )  #scene vision image in uncompressed RGBA array

print('Retrieved images: %d', len(responses))

if(len(responses) == 1):
    response = responses[0]

    # convert to numpy image
    img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
    img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
    
    # save frame for veiwing
    cv2.imwrite("segmentation_test.png", img_rgb)
