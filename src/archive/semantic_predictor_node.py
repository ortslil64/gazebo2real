#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import imp
from cgan import Generator, Discriminator
import rospkg 
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge
import cv2


rospack = rospkg.RosPack()
generator = Generator()

packadge_path = rospack.get_path('fast-scnn-ros')
checkpoint_dir = packadge_path+'/src/semantic_predictor_weights/model'
generator.load_weights(checkpoint_dir)


bridge = CvBridge()
image_pub = rospy.Publisher('semantics',Image)





if __name__ == '__main__':
    rospy.init_node('semantics_node')
    rate = rospy.Rate(1)
    

    while not rospy.is_shutdown():
        scan_cv_msg = rospy.wait_for_message('/camera/image_raw', Image)
        scan_cv_img = bridge.imgmsg_to_cv2(scan_cv_msg,"rgb8")
        scan_cv_img = cv2.normalize(scan_cv_img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        scan_cv_img = tf.image.resize(scan_cv_img, (256,256))
        gen_output = generator(np.expand_dims(scan_cv_img,0))
        img = gen_output[0].numpy()
        img = img*255
        msg_frame = bridge.cv2_to_imgmsg(img.astype(np.uint8))
        msg_frame.encoding = 'rgb8'
        msg_frame.header.frame_id = 'semantics'
        image_pub.publish(msg_frame)
        rate.sleep()
