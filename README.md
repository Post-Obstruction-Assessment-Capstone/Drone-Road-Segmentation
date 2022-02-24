# Drone-Road-Segmentation
This repository contains source code needed to train, evaluate and deploy a U-NET based segmentation model
that can be used for roadway identification and segmentation.

## Deep Learning Segmentation Implementation

### Tensorflow Version
This setup was verified to work with tensorflow 2.8.0 and Keras 2.8.0.

### Getting Training Data
The training dataset can be obtained here: https://www.kaggle.com/bulentsiyah/semantic-drone-dataset?select=dataset

Please download both the dataset.zip file and the class_dict_seg.csv file. Extract the zip file and ensure
that your project directory structure has the following form:

* Drone-Road-Segmentation
  * dataset
    * semantic_drone_dataset
      * label_images_semantic
      * original_images
      * class_dict_seg.csv

### Dataset Format
The dataset itself is mostly comprised of images taken from the 'Aerial Semantic Segmentation Dataset' hosted on 
Kaggle. The dataset is split into original images (see original_images directory in semantic_drone_dataset), 
and labeled semantic images (see label_images_semantic in semantic_drone_dataset). For any original image, the
corresponding labeled image has the same base filename (i.e. original_images/000.jpg and label_images_semantic/000.jpg).

The labeled images are constructed as masks of the original images. Each of these sub-masks in the image take
on a value corresponding to the particular class said labeled object/region belongs to. The mapping between

### Model Training and Evaluation

### Deployment Via ROS node

## Airsim Based Segmentation Implementation
The airsim simulation environment provides functionality to extract ground truth segmentation images from cameras
on the UAV itself. Meshes to be segmented in images are selected by their names via a regular expression. A ROS node 
was implemented to extract the segmented images from the simulation environment, and publish them as ROS image topics.
This node can be launched by running the following command:

```commandline
roslaunch airsim_based_drone_image_segmentation airsim_based_drone_image_segmentation.launch 
```

in the launch file itself, the mesh name matching regex and names of vehicles to extract images from are set as ROS params. 
Please change these as necessary. 