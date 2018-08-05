#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    #Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # Statistical outlier filter
    so_filter = cloud.make_statistical_outlier_filter()
    so_filter.set_mean_k(50)
    so_filter.set_std_dev_mul_thresh(1)
    cloud = so_filter.filter()

    # Voxel Grid Downsampling
    vox = cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()

   
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()

    # PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()

   
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.5
    axis_max = 0.5
    passthrough.set_filter_limits(axis_min, axis_max)

    # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()
   
    #RANSAC Plane Segmentation

    # Create the segmentation object
    seg = cloud_filtered.make_segmenter()

    # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # Max distance for a point to be considered fitting the model
    # Experiment with different values for max_distance 
    # for segmenting the table
    max_distance = 0.04
    seg.set_distance_threshold(max_distance)

    # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # Extract inliers and outliers
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(30)
    ec.set_MaxClusterSize(3000)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract()

    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])

    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    
    # Convert PCL data to ROS messages
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cloud_table = pcl_to_ros(extracted_inliers)

    # Publish ROS messages
    pub_cluster.publish(ros_cluster_cloud)
    pub_objects.publish(ros_cloud_objects)
    pub_table.publish(ros_cloud_table)

# Exercise-3 TODOs: 

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    for index,pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster
        pcl_cluster = extracted_outliers.extract(pts_list)
        sample_cloud = pcl_to_ros(pcl_cluster)
        # Compute the associated feature vector
        chists = compute_color_histograms(sample_cloud, using_hsv=True)
        normals = get_normals(sample_cloud)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = sample_cloud
        detected_objects.append(do)


    rospy.loginfo('Detected {} objects:'.format(len(detected_objects_labels)))

    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover(detected_objects)
    except rospy.ROSInterruptException:
        pass

# function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Initialize variables
    test_scene = 2
    object_name = String()
    object_group = String()
    test_scene_num = Int32()
    arm_name = String()
    dictionaries = []
    labels = []
    centroids = []
    pick_pose = Pose()
    place_pose = Pose()
    new_list = []

    # Get/Read parameters
    object_list_param = rospy.get_param('/object_list')
    dropbox = rospy.get_param('/dropbox')

    # Parse parameters into individual variables
    
    test_scene_num.data = test_scene
    num_object_list = len(object_list_param)

                
    rospy.loginfo('Detected {} objects'.format(len(object_list)))

    for objects in object_list:
        #Get the PointCloud for a given object and obtain it's centroid
        labels.append(objects.label)
        points_arr = ros_to_pcl(objects.cloud).to_array()
        numpy_centroid = np.mean(points_arr, axis=0)[:3]
        centroid = [np.asscalar(element) for element in numpy_centroid]
        centroids.append(centroid)

    
    # TODO: Rotate PR2 in place to capture side tables for the collision map

    # Loop through the pick list
    for i in range(num_object_list):
        object_name.data = object_list_param[i]['name']
        object_group = object_list_param[i]['group']
        # TODO: Create 'place_pose' for the object
        for j in range(0,len(labels)):
            if object_name.data == labels[j]:
                pick_pose.position.x = centroids[j][0]
                pick_pose.position.y = centroids[j][1]
                pick_pose.position.z = centroids[j][2]

        for j in range(0,len(dropbox)):
            if object_group == dropbox[j]['group']:
                place_pose.position.x = dropbox[j]['position'][0]
                place_pose.position.y = dropbox[j]['position'][1]
                place_pose.position.z = dropbox[j]['position'][2]

        
        
        # Assign the arm to be used for pick_place
        arm_name.data = 'right' if object_group == 'green' else 'left'

        
        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        dictionaries.append(make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose))


    # Output your request parameters into output yaml file
    file_name = "output_{}.yaml".format(test_scene)
    send_to_yaml(file_name, dictionaries)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('perception',anonymous=True)

    #Create Subscribers
    sub = rospy.Subscriber("/pr2/world/points",pc2.PointCloud2,pcl_callback,queue_size=10)

    #Create Publishers for objects and table
    pub_objects = rospy.Publisher("/pcl_objects",PointCloud2,queue_size=1)
    pub_table = rospy.Publisher("/pcl_table",PointCloud2,queue_size=1)
    pub_cluster = rospy.Publisher("/pcl_cluster",PointCloud2,queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects",DetectedObjectsArray,queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers",Marker,queue_size=1) 
    
    #Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
    	rospy.spin()
