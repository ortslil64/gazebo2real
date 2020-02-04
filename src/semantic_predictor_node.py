#!/usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cgan
from cgan import Generator, Discriminator
import rospkg 
from sensor_msgs.msg import Image
import rospy
from cv_bridge import CvBridge

rospack = rospkg.RosPack()
generator = Generator()

packadge_path = rospack.get_path('fast-scnn-ros')
checkpoint_dir = packadge_path+'/src/weights/model'
generator.load_weights(checkpoint_dir)


bridge = CvBridge()
image_pub = rospy.Publisher('semantics',Image)





if __name__ == '__main__':
    rospy.init_node('semantics_node')
    rate = rospy.Rate(2)
    

    while not rospy.is_shutdown():
        scan_cv_msg = rospy.wait_for_message('/camera/image_raw', Image)
        scan_cv_img = bridge.imgmsg_to_cv2(scan_cv_msg,"rgb8")
        scan_cv_img = tf.image.resize(scan_cv_img, (256,256))
        scan_cv_img = scan_cv_img/255.0
        gen_output = generator.predict(scan_cv_img[tf.newaxis,...])
        img = gen_output[0]
        img = img*255
        msg_frame = bridge.cv2_to_imgmsg(img.astype(np.uint8))
        msg_frame.encoding = 'rgb8'
        msg_frame.header.frame_id = 'semantics'
        image_pub.publish(msg_frame)
        rate.sleep()