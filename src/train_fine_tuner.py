import tensorflow as tf
import numpy as np
import os
import cv2
from skimage import io
from glob import glob
from tqdm import tqdm
from cgan import Generator, Discriminator, fit_fine

def load_images(path):
    temp_img,temp_mask=[],[]
    images=glob(os.path.join(path,'*.jpg'))
    for i in tqdm(images):
        i=cv2.imread(i)
        i=cv2.normalize(i,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        img=i[:,:256]
        msk=i[:,256:]  
        temp_img.append(img)
        temp_mask.append(msk)
    return temp_img,temp_mask



initial_learning_rate = 2e-4
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.97,
    staircase=True)


semantic_generator = Generator()
semantic_generator.load_weights('cgan_weights/model')

scene_generator = Generator()
scene_generator.load_weights('semantic_predictor_weights/model')

fine_generator = Generator() 
fine_discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(lr_schedule, beta_1=0.5)


train_path='cityscapes_data/cityscapes_data/train/'
val_path='cityscapes_data/cityscapes_data/val/'

train_images=[]
train_masks=[]
val_images=[]
val_masks=[]

train_images,train_masks=load_images(train_path)
val_images,val_masks=load_images(val_path)


fit_fine(x_train = train_images,
    x_test = val_images,
    batch_size = 1,
    epochs = 200, 
    semantic_generator = semantic_generator,
    scene_generator = scene_generator,
    fine_generator = fine_generator,
    fine_discriminator = fine_discriminator,
    generator_optimizer = generator_optimizer,
    discriminator_optimizer = discriminator_optimizer)

generator.save_weights('fine_tuner_weights/model')

