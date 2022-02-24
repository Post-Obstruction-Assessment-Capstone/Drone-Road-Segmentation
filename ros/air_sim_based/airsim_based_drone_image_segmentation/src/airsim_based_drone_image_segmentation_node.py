# Author: Trevor Sherrard
# Since: Feburary 24, 2022
# Project: UAV Based Post Obstruction Assesment Captsone Project
# Purpose: This file contains the implementation of a ROS node that
#          attempts to extract images from a drone within the airsim simulator instance
#          and perform semantic segmentation on said images using methods built into airsim.

import rospy
import airsim
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class AirsimSemanticSegmentation:
    def __init__(self):
        self.image_pub_topic_pat = "airsim_based_drone_image_segmentation/vn_{}/cam_{}"
        self.drone_name_list = list()
        self.pub_dict = dict()

        # set this as a parameter in the launch file later. Here for testing
        self.drone_name_list.append("drone_1")

        # set image update period
        self.img_update_period = 0.25

        # create cv bridge object
        self.bridge = CvBridge()

    def start_node(self):
        # start ROS node
        rospy.init_node("airsim_based_drone_image_segmentation_node")
        rospy.loginfo("started airsim_based_drone_image_segmentation_node!")

        # create publishers for drone images
        for drone_name in self.drone_name_list:
            cam_pub_dict = dict()

            # format topic names
            cam_0_pub_name = self.image_pub_topic_pat.format(drone_name, str(0))
            cam_1_pub_name = self.image_pub_topic_pat.format(drone_name, str(1))
            cam_2_pub_name = self.image_pub_topic_pat.format(drone_name, str(2))

            # create publisher objects
            cam_0_pub = rospy.Publisher(cam_0_pub_name, Image, queue_size=1)
            cam_1_pub = rospy.Publisher(cam_1_pub_name, Image, queue_size=1)
            cam_2_pub = rospy.Publisher(cam_2_pub_name, Image, queue_size=1)

            # add publisher objects to dictionary
            cam_pub_dict["0"] = cam_0_pub
            cam_pub_dict["1"] = cam_1_pub
            cam_pub_dict["2"] = cam_2_pub

            # add cam publisher dictionary to larger dictionary
            self.pub_dict[drone_name] = cam_pub_dict

        # create airsim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        # set all object segmentation ID's to 0 to start
        rospy.loginfo("setting segmentation IDs all objects to zero")
        found = self.client.simSetSegmentationObjectID("[\w]*", 0, True);
        rospy.loginfo("Done: %r" % (found))

        # set all template cube meshes to a segmentation object id of 20
        rospy.loginfo("setting segmentation IDs for template cubes")
        found = self.client.simSetSegmentationObjectID("templatecube[\w]*", 40, True)
        rospy.loginfo("Done: %r" % found)

    def run_node(self):
        while(not rospy.is_shutdown()):
            self.get_sem_images_from_drones()

    def get_sem_images_from_drones(self):
        drone_resp_dict = dict()
        for drone_name in self.drone_name_list:
            responses = self.client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Segmentation, False, False),
                 airsim.ImageRequest("1", airsim.ImageType.Segmentation, False, False),
                 airsim.ImageRequest("2", airsim.ImageType.Segmentation, False, False)],
                vehicle_name=drone_name
            )  # scene vision image in uncompressed RGBA array

            cam_id = 0
            single_drone_resp_dict = dict()
            for resp in responses:
                img1d = np.fromstring(resp.image_data_uint8, dtype=np.uint8)
                img_rgb = img1d.reshape(resp.height, resp.width, 3)  # reshape array to 3 channel image array H X W X 3

                # put image into dictionary
                single_drone_resp_dict[str(cam_id)] = img_rgb

                # increment cam id
                cam_id += 1

            # add single drone image dictionary to larger dictionary
            drone_resp_dict[drone_name] = single_drone_resp_dict

            # publish custom image topic using this data
            self.pub_images(drone_resp_dict)

    def pub_images(self, drone_image_dict):
        # get images dict for a single drone
        for drone_name in drone_image_dict:
            image_dict = drone_image_dict[drone_name]

            # get all images for a single drone
            for cam_id in image_dict:
                img = image_dict[cam_id]

                # convert image to ROS message
                img_msg = self.bridge.cv2_to_imgmsg(img, encoding="rgb8")

                # extract publisher object for cooresponding camera
                temp_pub = self.pub_dict[drone_name][cam_id]
                temp_pub.publish(img_msg)

if(__name__ == "__main__"):
    airsim_seg = AirsimSemanticSegmentation()
    airsim_seg.start_node()
    airsim_seg.run_node()