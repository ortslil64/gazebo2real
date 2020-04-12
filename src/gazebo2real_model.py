#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from models import DeepFilter, normalize, smooth_var, gen_sample
import rospkg 


def normalize_v(x):
    cmax = np.max(x)
    cmin = np.min(x)
    cscale = cmax - cmin
    x_out = (x * 1.0 - cmin) / cscale 
    x_out = np.cast[np.float32](x_out)
    return x_out

class Gazebo2semantic2Real():
    def __init__(self, image_shape = (256,256,3)):
        self.real2semantic = DeepFilter(input_shape = image_shape,
                                        output_shape = image_shape,
                                        lr = 1e-4,
                                        n0_filters = 64,
                                        max_filters = 1024)
	
        self.semantic2real = DeepFilter(input_shape = image_shape,
                                        output_shape = image_shape,
                                        lr = 1e-4,
                                        n0_filters = 64,
                                        max_filters = 1024)
        
        
    def train(self, real_image, semantic_image):
        real_image = np.expand_dims(real_image,0)
        semantic_image = np.expand_dims(semantic_image,0)
        self.real2semantic.train_step(real_image, semantic_image, L = 100)
        self.semantic2real.train_step(semantic_image, real_image, L = 100)
        
    def predict_real(self, semantic_image):
        semantic_image = np.expand_dims(semantic_image,0)
        real_image_hat = self.semantic2real.generator(semantic_image, training=False).numpy()[0]
        return real_image_hat

    def predict_semantic(self, real_image):
        real_image = np.expand_dims(real_image,0)
        semantic_image_hat = self.real2semantic.generator(real_image, training=False).numpy()[0]
        return semantic_image_hat
    
    def convert(self, gazebo_image):
        semantic_image_hat = self.predict_semantic(gazebo_image)
        real_image_hat = self.predict_real(semantic_image_hat)
        return real_image_hat
       
    def save_weights(self, path):
        real2semantic_path = path + '/real2semantic'
        semantic2real_path = path + '/semantic2real'
        self.real2semantic.generator.save_weights(real2semantic_path)
        self.semantic2real.generator.save_weights(semantic2real_path)
       
    def load_weights(self, path):
        real2semantic_path = path + '/real2semantic'
        semantic2real_path = path + '/semantic2real'
        self.real2semantic.generator.load_weights(real2semantic_path)
        self.semantic2real.generator.load_weights(semantic2real_path)
        
 

class Gazebo2Real():
    def __init__(self, image_shape = (256,256,3)):
		rospack = rospkg.RosPack()
		packadge_path = rospack.get_path('gazebo2real')
		checkpoint_dir = packadge_path+'/src/segmantation_model'
		self.real2semantic = tf.keras.models.load_model(checkpoint_dir)
        
	
        self.semantic2real = DeepFilter(input_shape = image_shape,
                                        output_shape = image_shape,
                                        lr = 1e-4,
                                        n0_filters = 64,
                                        max_filters = 1024)
        
        
    def train(self, real_image,):
        real_image = np.expand_dims(real_image,0)
        semantic_image = self.real2semantic.predict(real_image)
        self.semantic2real.train_step(semantic_image, real_image, L = 100)
        
    def predict_real(self, gazebo_image):
        gazebo_image = np.expand_dims(gazebo_image,0)
        semantic_image_hat = self.real2semantic.predict(gazebo_image)
        real_image_hat = self.semantic2real.generator(semantic_image_hat, training=False).numpy()[0]
        return real_image_hat

    def predict_semantic(self, gazebo_image):
        gazebo_image = np.expand_dims(gazebo_image,0)
        semantic_image_hat = self.real2semantic.predict(gazebo_image)[0]
        return semantic_image_hat
       
    def save_weights(self, path):
        semantic2real_path = path + '/semantic2real'
        self.semantic2real.generator.save_weights(semantic2real_path)
       
    def load_weights(self, path):
        semantic2real_path = path + '/semantic2real'
        self.semantic2real.generator.load_weights(semantic2real_path)
    
    
