#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import cv2
from skimage import io
from glob import glob
from tqdm import tqdm
from gazebo2real_model import Gazebo2Real
import matplotlib.pyplot as plt
import pathlib





dataset_path='datasets/corridor'

train_images=[]
train_masks=[]
val_images=[]
val_masks=[]
def load_images(path):
    temp_img=[]
    images=glob(os.path.join(path,'*.ppm'))
    for i in tqdm(images):
        i=cv2.imread(i)
        i=cv2.normalize(i,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        img = cv2.resize(i,(256,256))
        temp_img.append(img)
    return np.array(temp_img)

     
            

images =  load_images(dataset_path)

model = Gazebo2Real()


epochs = 100

for epoch in range(epochs):
    for ii in tqdm(range(len(images))):
        x = images[ii]
        model.train(x)
    ii = np.random.choice(len(images))
    x_test = images[ii]
    semantic_hat = model.predict_semantic(x_test)
    real_hat = model.predict_real(x_test)
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3)
    ax1.imshow(x_test)
    ax1.set_title("real")
    ax2.imshow(semantic_hat)
    ax2.set_title("semantic_hat")
    ax3.imshow(real_hat)
    ax3.set_title("real_hat")
    fig.show()
    plt.show()
    
    


model.save_weights('real2segmantation_model') 
		

