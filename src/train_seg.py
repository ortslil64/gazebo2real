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
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import binary_crossentropy
from segmentation_models.metrics import iou_score
import pathlib




train_path_cityscapes='cityscapes_data/train/'
val_path_cityscapes='cityscapes_data/val/'

#train_path_corridors='ADE20K/training/**/'
#val_path_corridors='ADE20K/validation/**/'

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

class ADE20K_object():
    def __init__(self, train_path, test_path):
        self.train_images = glob(os.path.join(train_path,'*.jpg'), recursive = True)
        self.test_images = glob(os.path.join(test_path,'*.jpg'), recursive = True)
        self.n_train = len(self.train_images)
        self.n_test = len(self.test_images)
        self.train_idx = 0
        self.train_idxs = np.random.choice(self.n_train, self.n_train, replace = False)
        
    def get_training(self, idx = None):
        if idx is None:
            if self.train_idx >= self.n_train:
                self.train_idx = 0
            i = self.train_images[self.train_idxs[self.train_idx]]
            j = i[:-4] + "_seg.png"
            img=cv2.imread(i)
            img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
            img = cv2.resize(img,(256,256))
            
            msk=cv2.imread(j)
            msk=cv2.normalize(msk,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
            msk = cv2.resize(msk,(256,256))
            self.train_idx += 1
            return np.expand_dims(img,0), np.expand_dims(msk,0)
        else:
            i = self.train_images[idx]
            j = i[:-4] + "_seg.png"
            img=cv2.imread(i)
            img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
            img = cv2.resize(img,(256,256))
            
            msk=cv2.imread(j)
            msk=cv2.normalize(msk,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
            msk = cv2.resize(msk,(256,256))
            
            return img, msk
        
    def get_test(self):
        idx = np.random.choice(self.n_test)
        i = self.test_images[idx]
        j = i[:-4] + "_seg.png"
        img=cv2.imread(i)
        img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        img = cv2.resize(img,(256,256))
        
        msk=cv2.imread(j)
        msk=cv2.normalize(msk,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
        msk = cv2.resize(msk,(256,256))
        return np.expand_dims(img,0), np.expand_dims(msk,0)
        
    def shuffle(self):
        self.train_idxs = np.random.choice(self.n_train, self.n_train, replace = False)
        self.train_idx = 0
        
        
class ADE20K_objectB():
    def __init__(self, train_path, test_path):
        self.train_images = glob(os.path.join(train_path,'*.jpg'), recursive = True)
        self.test_images = glob(os.path.join(test_path,'*.jpg'), recursive = True)
        self.n_train = len(self.train_images)
        self.n_test = len(self.test_images)
        self.train_idx = 0
        self.train_idxs = np.random.choice(self.n_train, self.n_train, replace = False)
        
    def get_training(self, idx = None):
        if idx is None:
            if self.train_idx >= self.n_train:
                self.train_idx = 0
            i = self.train_images[self.train_idxs[self.train_idx]]
            j = i[:-4] + "_seg.png"
            img=cv2.imread(i)
            img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
            img = cv2.resize(img,(256,256))
            
            msk=cv2.imread(j)
            msk=cv2.normalize(msk,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
            msk = cv2.resize(msk,(256,256))
            self.train_idx += 1
            return np.expand_dims(img,0), np.expand_dims(msk,0)
        else:
            i = self.train_images[idx]
            j = i[:-4] + "_seg.png"
            img=cv2.imread(i)
            img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
            img = cv2.resize(img,(256,256))
            
            msk=cv2.imread(j)
            msk=cv2.normalize(msk,None,0,1,cv2.NORM_MINMAX,cv2.CV_32F)
            msk = cv2.resize(msk,(256,256))
            
            return img, msk
        
    
      
        
    def shuffle(self):
        self.train_idxs = np.random.choice(self.n_train, self.n_train, replace = False)
        self.train_idx = 0
            
            
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

def train_step(model, x, y , optimizer):
        with tf.GradientTape() as tape:
            output = model([x], training=True)
            loss = tf.reduce_mean(tf.abs(y - output))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))



#train_images,train_masks=load_images_cityscapes(train_path_cityscapes)
#val_images,val_masks=load_images_cityscapes(val_path_cityscapes)


# train_images,train_masks=load_images_ADE20K(train_path_corridors)
# val_images,val_masks=load_images_ADE20K(val_path_corridors)

optimizer = tf.keras.optimizers.Adam(1e-4,  beta_1=0.5)
BACKBONE = 'resnet34'


model = sm.Unet('resnet34', classes=3, activation='relu')


dataset = ADE20K_object(train_path_corridors, val_path_corridors)
epochs = 100

for epoch in range(epochs):
    dataset.shuffle()
    for ii in tqdm(range(dataset.n_train)):
        x, y = dataset.get_training()
        train_step(model, x, y, optimizer)
    x, y = dataset.get_test()
    val_hat = model.predict(x)[0]
    val = y[0]
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    ax1.imshow(val_hat)
    ax1.set_title("val_hat")
    ax2.imshow(val)
    ax2.set_title("val")
    fig.show()
    plt.show()
    
    


#model.compile('Adam', loss='mean_absolute_error', metrics=['accuracy'])

# fit model
# model.fit(
#     x=train_images,
#     y=train_masks,
#     batch_size=16,
#     epochs=200,
#     validation_data=(val_images, val_masks),
# )

idx = np.random.choice(len(val_images))
val_hat = model.predict(np.expand_dims(val_images[idx],0))[0]
val = val_masks[idx]
fig, ((ax1, ax2)) = plt.subplots(1, 2)
ax1.imshow(val_hat)
ax1.set_title("val_hat")
ax2.imshow(val)
ax2.set_title("val")
fig.show()
plt.show()



model.save('segmantation_model/my_model') 
		

