## Project: Perception Pick & Place


---

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup 

### Exercise 1, 2 and 3 pipeline implemented
#### 1. Pipeline for filtering and RANSAC plane fitting implemented.

The first step for this perception pipeline is to extract the objects from the input point cloud data of RGBD Camera. To extract the required subset of point cloud, I followed the steps mentioned below:

1. Converting Point cloud data from ROS format to PCL format.
2. Removal of noise from RGBD camera using a Statistical Outlier Removal. I have used mean as 50 and standard deviation as 1. 
3. Downsample the data which is done by using a Voxel Grid Filter. This helps in improving computation time. I tried 0.01 and 0.005 values. I got better recognition results with 0.005 as with noisy data we need more pixels related to object for better recognition.
4. Extracting a Region of Interest. This is done using a passthrough filter. I have used two filters with z and y filter axis. z-axis filter values are 0.6 to 1.1. This is estimated by finding the height of objects and table from gazebo environment. y-axis filter values are -0.5 to 0.5. This filter is used to remove the point cloud data from adjacent tables that were in view of pr2 in environment.
5. The last step for extrating the region of objects is RANSAC plane fitting. This method allows us to extract the points belonging to one plane. In our case, by using this we can seperate the point cloud data of table from the object cluster. I have used maximum distance of 0.04. That worked best for me. The data of inliers and outliers are published on ROS topic pcl_objects and pcl_table.    

#### 2. Pipeline including clustering for segmentation implemented.  

The next process is clustering the objects. This is done by using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) or Euclidean Clustering. This is done using function provided by PCL library. There are few parameters that can help to cluster the objects properly. Based on the voxel filter leaf size, the parameters I have choosen are as follows-

Cluster Tolerance- 0.03
Minimum Cluster Size- 30
Maximum Cluster Size- 3000

For the visualization purposes, all the clusters are shown in different color. The colored point cloud data after converting to ROS format from pcl format is published on topic pcl_cluster.

#### 3. Features extracted and SVM trained.  Object recognition implemented.

The next step towards object recognition is to collect the training dataset and train a SUpport Vector Machine to label the clusters properly. For capturing features, histograms of surface normals and colors are used. No. of bins, I have used is 48. I tried using difent bin sizes but it doesnot have much effect after increasing more than 48 in final object recgnition, only slight improvements in confusion matrix. 

For the range, I have used (0,256). For normal, I initially tried using (-1,1) but I didn't get good results, so I switched back to (0,256). The code is in features.py . 

In capture_features.py, I have used 100 poses per model to train. I tried using 25,50 but these give good results in confusion matrix but not in actual recognition. Those values probably overfit the data.

Another approach, I tried is to change few parameters in svm model such as increasing cache size from default of 200Mb to 500Mb and changing C from 1 to 0.5. But the changes in accuracy are not that significant.

I also tried changing Kernel type from linear to poly, sigmoid and rbf. But in our case, linear kernel gives the best accuracy among all the kernel types.

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

After getting a trained model, the next step is to use that model to recognize objects and show the labels. After recognizing, the centroid from each cluster of recognized objects is calculated by taking mean. The datatype of centroid is changed from numpy float to simple float. After that, a dictionary is created using make_yaml_dict() function which requires the input of place , pick and test num scene. Pick position is based on the centroid of the cloud data and place position is based on the dropbox.

The output of all the 3 world_scenes are shown below:

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

I was able to detect all the objects except glue. I guess increasing training data to 500 might help. For now, my glue is recognized as biscuits. But still I was able to recognize 7/8 objects in world3, 4/5 objects in world2 and 3/3 in world1.


  



