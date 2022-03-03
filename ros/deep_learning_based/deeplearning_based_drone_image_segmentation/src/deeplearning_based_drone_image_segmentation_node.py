#!/usr/bin/env python3

# Author: Trevor Sherrard
# Since: March 2nd, 2022
# Project: UAV Based Post Obstruction Assesment Capstone Project
# Purpose: This file contains the implementation of a ROS node that
#          attempts to extract images from a drone within the airsim simulator instance
#          and perform semantic segmentation on said images making use of a trained UNET model.
import airsim
import rospy
import numpy as np
import rospkg
import tensorflow as tf
import keras

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class DeepLearningBasedSegmentaion:
    def __init__(self):
        self.pkg_path = ""
        self.image_pub_topic_pat = "deeplearning_based_drone_image_segmentation/seg_image/cam_{}"
        self.camera_name_list = list()
        self.pub_dict = dict()

        # create placeholder for UNET model object
        self.model = None

        # set image height and width
        self.img_height = 800
        self.img_width = 1200

        # declare image type to be used.
        self.image_type = airsim.ImageType.Scene

        # create cv bridge object
        self.bridge = CvBridge()

    def start_node(self):
        # start ROS node
        rospy.init_node("deeplearning_based_drone_image_segmentation_node")
        rospy.loginfo("started deeplearning_based_drone_image_segmentation_node!")

        # get ros params
        camera_name_list_str = rospy.get_param("~camera_name_list")

        # convert drone list string representation to string
        self.camera_name_list = camera_name_list_str.strip("][").split(", ")

        if(len(self.camera_name_list) == 0):
            rospy.logerr("need at least one camera name to extract images for inference!")
            return False

        for camera_name in self.camera_name_list:
            # format topic name
            temp_topic_name = self.image_pub_topic_pat.format(camera_name)

            # create publisher object
            temp_pub = rospy.Publisher(temp_topic_name, Image, queue_size=1)

            # add to dictionary, keyed by camera name
            self.pub_dict[camera_name] = temp_pub

        rospy.loginfo("created publishers")

        # get package path
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path("deeplearning_based_drone_image_segmentation")
        model_path = pkg_path + "/models/saved_unet_model.h5"

        # load UNET model
        self.model = keras.models.load_model(model_path)

        # create airsim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        rospy.loginfo("connected to airsim")

        # return True if we get this far
        return True

    def grab_images_and_infer(self):
        # loop through all cameras provided and try to get images
        # of the sceene.
        for camera_name in self.camera_name_list:
            # try and grab a single image
            responses = self.client.simGetImages([airsim.ImageRequest(camera_name, self.image_type, False, False)])

            if(len(responses) == 0):
                rospy.loginfo("could not get an image from camera: {}".format(camera_name))
                continue

            # try to extract response and generate numpy image
            raw_image = responses[0]
            img1d = np.fromstring(raw_image.image_data_uint8, dtype=np.uint8)

            # reshape array to 3 channel image array H X W X 3
            img_rgb = img1d.reshape(raw_image.height, raw_image.width, 3)

            # perform image inference
            seg_image = self.image_inference(img_rgb)

            # publish inference image
            self.pub_inference_image(seg_image, camera_name)

    def image_inference(self, image):
        # load image as dataset
        image = tf.convert_to_tensor(image)
        image = tf.reshape(image, (self.img_height, self.img_width, 3))
        dataset = tf.data.Dataset.from_tensors(image)
        dataset = dataset.batch(1)

        # try to make prediction
        predictions = self.model.predict(dataset)
        predictions = np.argmax(predictions, axis=3)

        # convert dtype from int64 to uint8
        single_channel_pred = predictions[0]
        single_channel_pred = single_channel_pred.astype("uint8")

        return single_channel_pred

    def pub_inference_image(self, infer_image, camera_name):
        # try to convert image to ROS message format and publish
        try:
            infer_image_msg = self.bridge.cv2_to_imgmsg(infer_image, encoding="mono8")
            self.pub_dict[camera_name].publish(infer_image_msg)
        except CvBridgeError as cv_err:
            rospy.logerr("could not generated ros message for segmentation image "
                         "obtained from camera {}: {}".format(camera_name, cv_err))

    def run_node(self):
        while(not rospy.is_shutdown()):
            self.grab_images_and_infer()

if(__name__ == "__main__"):
    DL_seg = DeepLearningBasedSegmentaion()

    status = DL_seg.start_node()

    if(status):
        DL_seg.run_node()
    else:
        rospy.logerr("could not start deeplearning_based_drone_image_segmentation_node!")