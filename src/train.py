import tensorflow as tf
import numpy as np
import os
import cv2
from skimage import io
from glob import glob
from tqdm import tqdm
from gazebo2real_model import Gazebo2Real
import matplotlib.pyplot as plt





train_path='cityscapes_data/train/'
val_path='cityscapes_data/val/'

train_images=[]
train_masks=[]
val_images=[]
val_masks=[]
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

train_images,train_masks=load_images(train_path)
val_images,val_masks=load_images(val_path)

model = Gazebo2Real()
epochs = 200
n_train = len(train_images)
n_test = len(val_images)

for epoch in tqdm(range(epochs)):
	for ii in range(n_train):
		model.train(train_images[ii], train_masks[ii])
	
	idx = np.random.choice(n_test)
	fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
	real = val_images[idx]
	semantic = val_masks[idx]
	real_hat = model.predict_real(semantic)
	semantic_hat = model.predict_semantic(real)
	ax1.imshow(real)
	ax1.set_title("real")
	ax2.imshow(semantic)
	ax2.set_title("semantic")
	ax3.imshow(real_hat)
	ax3.set_title("real_hat")
	ax4.imshow(semantic_hat)
	ax4.set_title("semantic_hat")
	fig.show()
	plt.show()

model.save_weights(".weights")
		

