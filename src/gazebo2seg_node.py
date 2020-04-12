#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import imp
import rospkg 
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge
import cv2


rospack = rospkg.RosPack()

packadge_path = rospack.get_path('gazebo2real')
checkpoint_dir = packadge_path+'/src/segmantation_model'
model = tf.keras.models.load_model(checkpoint_dir)


bridge = CvBridge()
semantic_pub = rospy.Publisher('/gazebo2seg/semantic',Image)





if __name__ == '__main__':
    rospy.init_node('gazebo2real_node')
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        scan_cv_msg = rospy.wait_for_message('/camera/image_raw', Image)
        scan_cv_img = bridge.imgmsg_to_cv2(scan_cv_msg,"rgb8")
        scan_cv_img = cv2.normalize(scan_cv_img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        scan_cv_img = tf.image.resize(scan_cv_img, (256,256))
        
        semantic_output = model.predict(np.expand_dims(scan_cv_img,0))[0]
        
        semantic_output = semantic_output*255.0
        semantic_msg_frame = bridge.cv2_to_imgmsg(semantic_output.astype(np.uint8))
        semantic_msg_frame.encoding = 'rgb8'
        semantic_msg_frame.header.frame_id = 'semantic'
        semantic_pub.publish(semantic_msg_frame)
    
        
        
        rate.sleep()
