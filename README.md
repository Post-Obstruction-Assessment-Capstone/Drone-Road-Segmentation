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
The U-NET model can be trained by executing the following command:

```commandline
python3 train_model.py
```
Once the training process is complete, the model will be saved to the models directory. Please change training
hyperparamters defined at the top of the script if needed.

### Deployment Via ROS node
To run the model trained in the previous step within a ROS node, please run the following command:

```commandline
roslaunch deeplearning_based_drone_image_segmentation deeplearning_based_drone_image_segmentation.launch
```

This will start a node that will continually grab frames from the cameras specified within the launch file. 
Please change these camera names as needed. **NOTE:** this launch file assumes the model to load for inference
purposes is stored in the models directory under the deeplearning_based_drone_image_segmentation ROS package
structure. If you train a new model, make sure to move a copy of it to this aforementioned directory.

### Published Data Format
For the deep learning implementation, the node will publish segmentation image results to topics with the pattern:

```commandline
deeplearning_based_drone_image_segmentation/seg_image/cam_{cam_name}
```

Where cam_name is one of the camera names defined in deeplearning_based_drone_image_segmentation.launch.

These segmentation images are images in which each pixel value corresponds to a particular class detected within the image itself. 
For the deep learning implementation, this class mapping can be seen in the table below:

| Class       | Pixel Value |
|-------------|-------------|
| Unlabeled   | 0           |
| Paved-Area  | 1           |
| Dirt        | 2           |
| Grass       | 3           |
| Gravel      | 4           |
| Water       | 5           |
| Rocks       | 6           |
| Pool        | 7           |
| Vegetation  | 8           |
| Roof        | 9           |
| Wall        | 10          |
| Window      | 11          |
| Door        | 12          |
| Fence       | 13          |
| Fence-pole  | 14          |
| Person      | 15          |
| Dog         | 16          |
| Car         | 17          |
| Bicycle     | 18          |
| Tree        | 19          |
| Bald-tree   | 20          |
| AR-marker   | 21          |
| Obstacle    | 22          |
| Conflicting | 23          |

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

### Published Data Format
For the airsim based implementation, the node will publish segmentation images to topics with the pattern:

```commandline
airsim_based_drone_image_segmentation/vn_{vehicle_name}/cam_{cam_name}
```

Where vehicle_name is one of the vehicle names set in airsim_based_drone_image_segmentation.launch, and cam_name
is set to 0, 1 or 2. The images published will have a color mapping for each of the classes detected within the image. 
This color mapping will need to be manually set by the user by assigning a segmentation ID in the script
for individual meshes that should be detected. More to come on this.