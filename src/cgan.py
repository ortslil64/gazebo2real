from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
from sklearn.utils import shuffle
import pandas as pd
import matplotlib.pyplot as plt

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generate_images(model, test_input, tar):
  prediction = model([test_input], training=False)
  prediction = prediction[0].numpy()
  ground_trouth = tar[0]
  plt.subplot(1,2,1)
  plt.imshow(prediction)
  plt.subplot(1,2,2)
  plt.imshow(ground_trouth)
  plt.show()

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target, LAMBDA):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss

def Generator():
  down_stack = [
    downsample(64, 4, apply_batchnorm=False), # (bs, 64, 64, 64)
    downsample(128, 4), # (bs, 32, 32, 128)
    downsample(256, 4), # (bs, 8, 8, 256)
    downsample(512, 4), # (bs, 4, 4, 512)
    downsample(512, 4), # (bs, 2, 2, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
    downsample(512, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(512, 4), # (bs, 16, 16, 1024)
    upsample(256, 4), # (bs, 32, 32, 512)
    upsample(128, 4), # (bs, 64, 64, 512)
    upsample(64, 4), # (bs, 64, 64, 512)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(3, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 64, 64, 3)

  concat = tf.keras.layers.Concatenate()
  
  img_inputs = tf.keras.layers.Input(shape=[256,256,3], name='gen_input_image')
  x = img_inputs
  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)
  skips = reversed(skips[:-1])
  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = concat([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=img_inputs, outputs=x)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256,256,3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256,256,3], name='target_image')
  x = tf.keras.layers.concatenate([inp, tar], axis = 3) # (bs, 64, 64, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 32, 32, 64)
  down2 = downsample(128, 4)(down1) # (bs, 16, 16, 128)
  down3 = downsample(256, 4)(down2) # (bs, 8, 8, 256)
  #down4 = downsample(256, 4)(down3) # (bs, 8, 8, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 10, 10, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 11, 11, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)



def train_step(input_image, target, generator, discriminator, generator_optimizer, discriminator_optimizer):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator([input_image], training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_loss = generator_loss(disc_generated_output, gen_output, target, LAMBDA = 50)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  

def fit(x_train, y_train, x_test, y_test, batch_size, epochs, generator, discriminator, generator_optimizer, discriminator_optimizer):
  for epoch in range(epochs):
    start = time.time()
    # Train
    #x_train_shuf, y_train_shuf = shuffle(x_train, y_train)
    N = len(x_train)//batch_size
    for ii in range(N):
        x_batch, y_batch = np.array(x_train[ii*batch_size:batch_size*(ii+1)]), np.array(y_train[ii*batch_size:batch_size*(ii+1)])
        train_step(x_batch, y_batch, generator, discriminator, generator_optimizer, discriminator_optimizer)
    
    if epoch % 1 == 0:
        ri = np.random.choice(len(x_test))
        generate_images(generator, x_test[ri][tf.newaxis,...], y_test[ri][tf.newaxis,...])
    clear_output(wait=True)

   

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))


