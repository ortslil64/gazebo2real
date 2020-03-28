import tensorflow as tf
import numpy as np
import os
import cv2
from skimage import io
from glob import glob
from tqdm import tqdm
from gazebo2real_model import Gazebo2Real
import matplotlib.pyplot as plt
import segmentation_models as sm







train_path_cityscapes='cityscapes_data/train/'
val_path_cityscapes='cityscapes_data/val/'

train_path_corridors='ADE20K/training/c/corridor'
val_path_corridors='ADE20K/validation/c/corridor'

train_images=[]
train_masks=[]
val_images=[]
val_masks=[]
def load_images_cityscapes(path):
    temp_img,temp_mask=[],[]
    images=glob(os.path.join(path,'*.jpg'))
    for i in tqdm(images):
        i=cv2.imread(i)
        i=cv2.normalize(i,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        img=i[:,:256]
        msk=i[:,256:]  
        temp_img.append(img)
        temp_mask.append(msk)
    return np.array(temp_img), np.array(temp_mask)

def load_images_ADE20K(path):
    temp_img,temp_mask=[],[]
    images=glob(os.path.join(path,'*.jpg'), recursive = True)
    for i in tqdm(images):
        j = i[:-4] + "_seg.png"
        
        img=cv2.imread(i)
        img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        img = cv2.resize(img,(256,256))
        
        msk=cv2.imread(j)
        msk=cv2.normalize(msk,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        msk = cv2.resize(msk,(256,256))
        
        
        temp_img.append(img)
        temp_mask.append(msk)

    return np.array(temp_img), np.array(temp_mask)



# train_images,train_masks=load_images_cityscapes(train_path_cityscapes)
# val_images,val_masks=load_images_cityscapes(val_path_cityscapes)


train_images,train_masks=load_images_ADE20K(train_path_corridors)
val_images,val_masks=load_images_ADE20K(val_path_corridors)

model = Gazebo2Real()
epochs = 200
n_train = len(train_images)
n_test = len(val_images)

for epoch in tqdm(range(epochs)):
    idxses = np.random.choice(n_train, n_train, replace = False)
    for ii in range(n_train):
        model.train(train_images[idxses[ii]], train_masks[idxses[ii]])
	
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
		

