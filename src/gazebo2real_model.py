#!/usr/bin/env python3
import numpy as np
from models import DeepFilter, normalize, smooth_var, gen_sample

def normalize_v(x):
    cmax = np.max(x)
    cmin = np.min(x)
    cscale = cmax - cmin
    x_out = (x * 1.0 - cmin) / cscale 
    x_out = np.cast[np.float32](x_out)
    return x_out

class Gazebo2Real():
    def __init__(self, hist = 4, image_shape = (256,256,3)):
        self.real2semantic = DeepFilter(input_shape = image_shape,
                                    output_shape = image_shape,
                                    lr = 1e-5,
                                    max_filters = 512)
	
	self.semantic2real = DeepFilter(input_shape = image_shape,
                                    output_shape = image_shape,
                                    lr = 1e-5,
                                    max_filters = 512)
        
        
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
    
