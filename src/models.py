from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from components import downsample, upsample, discriminator_loss, generator_loss
from tensorflow.keras import regularizers
from scipy import signal
    

def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y
    

def normalize(x,cmin, cmax):
    cscale = cmax - cmin
    x_out = (x * 1.0 - cmin) / cscale 
    x_out = np.cast[np.float32](x_out)
    return x_out

def unnormalize(x,cmin, cmax):
    cscale = cmax - cmin
    x_out = x * cscale + cmin
    x_out = np.cast[np.float32](x_out)
    return x_out


def preprocess_data(x):
    d = int(np.sqrt(len(x)))
    img = np.reshape(x, (d,d))
    img = np.expand_dims(img,0)
    return img

def preprocess_Bayesian_data(x_old, z_new):
    d = int(np.sqrt(len(x_old)))
    x = np.reshape(x_old, (d,d))
    x = np.expand_dims(x,2)
    z = np.reshape(z_new, (d,d))
    z = np.expand_dims(z,2)
    return np.concatenate((z,x),axis = 2)

def smooth_var(z, var_len = 128*128//16):
    n = len(z)//var_len
    z_sigma = np.empty_like(z)
    for ii in range(n):
        sigma = np.var(z[(ii*var_len):((ii+1)*var_len)])
        z_sigma[(ii*var_len):((ii+1)*var_len)] = sigma
    return z_sigma

def gen_sample(v_size = 128*128, p_len = 16):
    n = v_size//p_len
    z_sigma = np.empty(v_size)
    for ii in range(p_len):
        z_sigma[(ii*n):((ii+1)*n)] = np.random.rand()
    return z_sigma

def preprocess_Bayesian_data_fft(x_old_real, x_old_imag, x_new_real, x_new_imag):
    d = int(np.sqrt(len(x_old_real)))
    x_old_real = np.reshape(x_old_real, (d,d))
    x_old_real = np.expand_dims(x_old_real,2)
    x_old_imag = np.reshape(x_old_imag, (d,d))
    x_old_imag = np.expand_dims(x_old_imag,2)
    x_new_real = np.reshape(x_new_real, (d,d))
    x_new_real = np.expand_dims(x_new_real,2)
    x_new_imag = np.reshape(x_new_imag, (d,d))
    x_new_imag = np.expand_dims(x_new_imag,2)
    
    return np.concatenate((x_old_real, x_old_imag, x_new_real, x_new_imag),axis = 2)

    


class DeepFilter():
    def __init__(self, input_shape, output_shape, lr = 2e-4, n0_filters = 32, max_filters = 512):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.generator_optimizer = tf.keras.optimizers.Adam(lr,  beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr,  beta_1=0.5)
        # ---- compiling the generator ---- #
        down_stack = []
        up_stack = []
        down_steps = int(np.log2(self.input_shape[0]))
        up_steps = int(np.log2(self.output_shape[0]))
        for ii in range(down_steps):
            n_filters = n0_filters*(2**ii)
            if n_filters > max_filters:
                n_filters = max_filters
            if ii == 0:
                down_stack.append(downsample(n_filters, 4, apply_batchnorm=False, apply_dropout=True))
            else:
                down_stack.append(downsample(n_filters, 4, apply_batchnorm=False, apply_dropout=True))
        for ii in range(up_steps):
            n_filters = n0_filters*(2**ii)
            if n_filters > max_filters:
                n_filters = max_filters
            if ii == 0:
                up_stack.append(upsample(n_filters, 4, apply_batchnorm=False, apply_dropout=True))
            else:
                up_stack.append(upsample(n_filters, 4, apply_batchnorm=False, apply_dropout=True))
            
        up_stack = reversed(up_stack)
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_shape[2], 4, strides=2, padding='same', kernel_initializer=initializer,kernel_regularizer=regularizers.l2(0.001),  activation='tanh')
        concat = tf.keras.layers.Concatenate()
        generator_input = tf.keras.layers.Input(shape=self.input_shape)
        x = generator_input
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
        self.generator = tf.keras.Model(inputs=generator_input, outputs=x)
        
        # ---- compiling the discriminator ---- #
        discriminator_inp = tf.keras.layers.Input(shape=self.input_shape, name='input_image')
        discriminator_tar = tf.keras.layers.Input(shape=self.output_shape, name='target_image')
        if down_steps == up_steps:
            x = tf.keras.layers.concatenate([discriminator_inp, discriminator_tar], axis=3)
        elif down_steps>up_steps:
            n_steps = down_steps - up_steps
            tar_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(discriminator_tar)
            for ii in range(n_steps-1):
                tar_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(tar_x)
            x = tf.keras.layers.concatenate([discriminator_inp, tar_x], axis=3)
        elif down_steps<up_steps:
            n_steps = up_steps - down_steps
            inp_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(discriminator_inp)
            for ii in range(n_steps-1):
                inp_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(inp_x)
            x = tf.keras.layers.concatenate([inp_x, discriminator_tar], axis=3)
        x = downsample(64, 4, apply_batchnorm=False, apply_dropout=True)(x)  
        x = downsample(128, 4, apply_batchnorm=False, apply_dropout=True)(x) 
        x = tf.keras.layers.ZeroPadding2D()(x) # (bs, 10, 10, 256)
        x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.ZeroPadding2D()(x) 
        x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001))(x) 
        self.discriminator = tf.keras.Model(inputs=[discriminator_inp, discriminator_tar], outputs=x)
        
    def train_step(self, x, y, L = 30):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator([x], training=True)
            disc_real_output = self.discriminator([x, y], training=True)
            disc_generated_output = self.discriminator([x, gen_output], training=True)
            gen_loss = generator_loss(disc_generated_output, gen_output, y, LAMBDA = L)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,  self.discriminator.trainable_variables))
        
        
class DeepBayesianFilter():
    def __init__(self):
        self.Predictor = DeepFilter(input_shape = (128,128,1), output_shape = (128,128,1))
        self.InvertPredictor = DeepFilter(input_shape = (128,128,1), output_shape = (128,128,1))
        self.Updator = DeepFilter(input_shape = (128,128,2), output_shape = (128,128,1))
    def train_predictor(self, x_old, x_new):
        cmin_x = np.min(x_old) - 3*np.std(x_old)
        cmax_x = np.max(x_old) + 3*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        self.Predictor.train_step(np.expand_dims(x_old,3), np.expand_dims(x_new,3), L = 50)
    def train_invert_predictor(self, x_old, x_new):
        cmin_x = np.min(x_new) - 3*np.std(x_new)
        cmax_x = np.max(x_new) + 3*np.std(x_new)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        self.InvertPredictor.train_step(np.expand_dims(x_new,3), np.expand_dims(x_old,3), L = 50)
    def train_updator(self,x_old, x_new, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        cmin_x = np.min(x_old) - 3*np.std(x_old)
        cmax_x = np.max(x_old) + 3*np.std(x_old)
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        self.Updator.train_step(np.expand_dims(input_data_update,0), np.expand_dims(x_new,3), L = 100)
    def filter_mean(self, x_old,z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        cmin_x = np.min(x_old) - 3*np.std(x_old)
        cmax_x = np.max(x_old) + 3*np.std(x_old)
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = preprocess_data(x_old)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
        x_new_update = np.reshape(x_new_update, (-1))
        x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
        return x_new_update
    def filter_var(self, x_old,z_new,iters = 10, beta = 0.01):
        x_old_hat = x_old
        x_hat_stack = []
        x_hat_weights = []
        for ii in range(iters):
            x_hat = self.filter_mean(x_old_hat,z_new)
            x = x_hat
            w = np.exp(-beta*np.linalg.norm(x_old_hat - x_old))
            if np.isnan(w).sum() > 0 or np.isnan(x).sum() > 0 :
                break
            else:
                x_hat_stack.append(x)
                x_hat_weights.append(w)
                cmin_x = np.min(x_hat) - 3*np.std(x_hat)
                cmax_x = np.max(x_hat) + 3*np.std(x_hat)
                x_hat = normalize(x_hat, cmin_x, cmax_x)
                x_hat = preprocess_data(x_hat)
                x_old_hat = self.InvertPredictor.generator(np.expand_dims(x_hat,3), training=False)[0].numpy()
                x_old_hat = np.reshape(x_old_hat, (-1))
                x_old_hat = unnormalize(x_old_hat, cmin_x, cmax_x)
        W = np.array(x_hat_weights)
        W = W/np.sum(W)
        if np.isnan(W).sum() > 0:
            W = np.ones(len(W))
            W = W/np.sum(W)
        X = np.array(x_hat_stack)
        x_hat = np.sum(np.diag(W).dot(X), axis =0)
        p_hat = np.sum(np.diag(W).dot(np.multiply(X,X)), axis = 0) -np.multiply(x_hat,x_hat)
        return x_hat_stack[0], p_hat
    
    

    
class DeepFusion():
    def __init__(self):
        self.fuser = DeepFilter(input_shape = (128,128,2), output_shape = (128,128,1))
    def train(self, x_old, x_new, x_tar):
        tot = np.concatenate((x_old,x_new))
        cmin_x = np.min(tot) - 3*np.std(tot)
        cmax_x = np.max(tot) + 3*np.std(tot)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_tar = normalize(x_tar, cmin_x, cmax_x)
        x_tar = preprocess_data(x_tar)
        input_data_update = preprocess_Bayesian_data(x_old, x_new)
        self.fuser.train_step(np.expand_dims(input_data_update,0), np.expand_dims(x_tar,3), L = 50)
    def filter_step(self, x_old,x_new):
        tot = np.concatenate((x_old,x_new))
        cmin_x = np.min(tot) - 3*np.std(tot)
        cmax_x = np.max(tot) + 3*np.std(tot)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_old = normalize(x_old, cmin_x, cmax_x)
        input_data_update = preprocess_Bayesian_data(x_old, x_new)
        x_new_update = self.fuser.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
        x_new_update = np.reshape(x_new_update, (-1))
        x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
        return x_new_update
    
class DeepPredictor():
    def __init__(self):
        self.Predictor = DeepFilter(input_shape = (128,128,1), output_shape = (128,128,1))
    def train(self, x_old, x_new):
        cmin_x = np.min(x_old) - 3*np.std(x_old)
        cmax_x = np.max(x_old) + 3*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        self.Predictor.train_step(np.expand_dims(x_old,3), np.expand_dims(x_new,3), L = 100)
    def predict(self, x_old):
        cmin_x = np.min(x_old) - 3*np.std(x_old)
        cmax_x = np.max(x_old) + 3*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = preprocess_data(x_old)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        x_new_hat = unnormalize(x_new_hat, cmin_x, cmax_x)
        return x_new_hat
    
    
class DeepPredictorFFT():
    def __init__(self):
        self.Predictor = DeepFilter(input_shape = (128,128,1), output_shape = (128,128,1))
    def train(self, x_old, x_new):
        x_old_mat = preprocess_data(x_old)
        x_new_mat = preprocess_data(x_new)
        x_old_mat_fft = np.fft.fft(x_old_mat,axis = 1)
        x_new_mat_fft = np.fft.fft(x_new_mat,axis = 1)
        x_old_fft = x_old_mat_fft.reshape((-1))
        x_new_fft = x_new_mat_fft.reshape((-1))
        
        cmin_x = np.min(x_old_fft.real) - 3*np.std(x_old_fft.real)
        cmax_x = np.max(x_old_fft.real) + 3*np.std(x_old_fft.real)
        x_old_real = normalize(x_old_fft.real, cmin_x, cmax_x)
        x_new_real = normalize(x_new_fft.real, cmin_x, cmax_x)
        x_old_input = preprocess_data(x_old_real)
        x_new_input = preprocess_data(x_new_real)
        self.Predictor.train_step(np.expand_dims(x_old_input,3), np.expand_dims(x_new_input,3), L = 50)
    def predict(self, x_old):
        x_old_mat = preprocess_data(x_old)
        x_old_mat_fft = np.fft.fft(x_old_mat,axis = 1)
        x_old_fft = x_old_mat_fft.reshape((-1))
        cmin_x = np.min(x_old_fft.real) - 3*np.std(x_old_fft.real)
        cmax_x = np.max(x_old_fft.real) + 3*np.std(x_old_fft.real)
        x_old_real = normalize(x_old_fft.real, cmin_x, cmax_x)
        x_old_input = preprocess_data(x_old_real)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old_input,3), training=False)[0].numpy()
        x_new_hat_real = np.reshape(x_new_hat, (-1))
        x_new_hat_real = unnormalize(x_new_hat_real, cmin_x, cmax_x)
        x_new_mat = preprocess_data(x_new_hat_real)
        x_new_hat = np.fft.ifft(x_new_mat, axis = 1)
        x_new_hat = x_new_hat.reshape((-1))
        return x_new_hat.real
    

class DeepFusionFFT():
    def __init__(self):
        self.fuser = DeepFilter(input_shape = (128,128,4), output_shape = (128,128,2))
    def train(self, x_old, x_new, x_tar):
        x_old_mat = preprocess_data(x_old)
        x_new_mat = preprocess_data(x_new)
        x_tar_mat = preprocess_data(x_tar)
        x_old_mat_fft = np.fft.fft(x_old_mat,axis = 1)
        x_new_mat_fft = np.fft.fft(x_new_mat,axis = 1)
        x_tar_mat_fft = np.fft.fft(x_tar_mat,axis = 1)
        x_old_fft = x_old_mat_fft.reshape((-1))
        x_new_fft = x_new_mat_fft.reshape((-1))
        x_tar_fft = x_tar_mat_fft.reshape((-1))
    
        tot = np.concatenate((x_old_fft.real,x_old_fft.imag, x_new_fft.real,x_new_fft.imag))
        cmin_x = np.min(tot) - 3*np.std(tot)
        cmax_x = np.max(tot) + 3*np.std(tot)
        
        x_old_real = normalize(x_old_fft.real, cmin_x, cmax_x)
        x_old_imag = normalize(x_old_fft.imag, cmin_x, cmax_x)
        x_new_real = normalize(x_new_fft.real, cmin_x, cmax_x)
        x_new_imag = normalize(x_new_fft.imag, cmin_x, cmax_x)
        x_tar_real = normalize(x_tar_fft.real, cmin_x, cmax_x)
        x_tar_imag = normalize(x_tar_fft.imag, cmin_x, cmax_x)
        
        
        x_tar_input = preprocess_Bayesian_data(x_tar_real, x_tar_imag)
        
        input_data_update = preprocess_Bayesian_data_fft(x_old_real, x_old_imag, x_new_real, x_new_imag)
        
        self.fuser.train_step(np.expand_dims(input_data_update,0), np.expand_dims(x_tar_input,0), L = 50)
        
    def filter_step(self, x_old,x_new):
        tot = np.concatenate((x_old,x_new))
        cmin_x = np.min(tot) - 3*np.std(tot)
        cmax_x = np.max(tot) + 3*np.std(tot)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_old = normalize(x_old, cmin_x, cmax_x)
        input_data_update = preprocess_Bayesian_data(x_old, x_new)
        x_new_update = self.fuser.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
        x_new_update = np.reshape(x_new_update, (-1))
        x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
        return x_new_update
    
    
class DeepBayesianFilterMV():
    def __init__(self, state_d = 1, obs_d = 1):
        self.Predictor = DeepFilter(input_shape = (128,128,state_d), output_shape = (128,128,state_d), lr = 1e-5, max_filters = 512)
        self.Updator = DeepFilter(input_shape = (128,128,state_d + obs_d), output_shape = (128,128,state_d), lr = 1e-5, max_filters = 512)
        self.state_d = state_d
        self.obs_d = obs_d
    def train_predictor(self, x_old, x_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        x_new_input = np.empty((1,128,128,self.state_d))
        for d in range(self.state_d):
            cmin_x = np.min(x_old[:,d]) - 3*np.std(x_old[:,d])
            cmax_x = np.max(x_old[:,d]) + 3*np.std(x_old[:,d])
            x_old_temp = normalize(x_old[:,d], cmin_x, cmax_x)
            x_new_temp = normalize(x_new[:,d], cmin_x, cmax_x)
            x_new_temp = preprocess_data(x_new_temp)
            x_old_temp = preprocess_data(x_old_temp)
            x_old_input[0,:,:,d] = x_old_temp
            x_new_input[0,:,:,d] = x_new_temp
        self.Predictor.train_step(x_old_input, x_new_input, L = 100)
        
    def train_updator(self,x_old, x_new, z_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        x_new_input = np.empty((1,128,128,self.state_d))
        combined_input = np.empty((1,128,128,self.state_d+self.obs_d))
        for d in range(self.state_d):
            cmin_x = np.min(x_old[:,d]) - 3*np.std(x_old[:,d])
            cmax_x = np.max(x_old[:,d]) + 3*np.std(x_old[:,d])
            x_new_temp = normalize(x_new[:,d], cmin_x, cmax_x)
            x_new_temp = preprocess_data(x_new_temp)
            x_old_temp = normalize(x_old[:,d], cmin_x, cmax_x)
            x_old_temp = preprocess_data(x_old_temp)
            x_old_input[0,:,:,d] = x_old_temp
            x_new_input[0,:,:,d] = x_new_temp
        x_new_hat = self.Predictor.generator(x_old_input, training=False).numpy()
        combined_input[0,:,:,0:self.state_d] = x_new_hat
        for d in range(self.obs_d):
            cmin_z = np.min(z_new[:,d]) - 3*np.std(z_new[:,d])
            cmax_z = np.max(z_new[:,d]) + 3*np.std(z_new[:,d])
            z_new_temp = normalize(z_new[:,d], cmin_z, cmax_z)
            z_new_temp = preprocess_data(z_new_temp)
            combined_input[0,:,:,self.state_d + d] = z_new_temp
        self.Updator.train_step(combined_input, x_new_input, L = 100)
                                
    def filter_step(self, x_old,z_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        combined_input = np.empty((1,128,128,self.state_d+self.obs_d))
        output_x = np.empty_like(x_old)
        cmin_x = np.empty(self.state_d)
        cmax_x = np.empty(self.state_d)
        cmin_z = np.empty(self.obs_d)
        cmax_z = np.empty(self.obs_d)
        for d in range(self.state_d):
            cmin_x[d] = np.min(x_old[:,d]) - 3*np.std(x_old[:,d])
            cmax_x[d] = np.max(x_old[:,d]) + 3*np.std(x_old[:,d])
            x_old_temp = normalize(x_old[:,d], cmin_x[d], cmax_x[d])
            x_old_temp = preprocess_data(x_old_temp)
            x_old_input[0,:,:,d] = x_old_temp
        x_new_hat = self.Predictor.generator(x_old_input, training=False).numpy()
        combined_input[0,:,:,0:self.state_d] = x_new_hat
        for d in range(self.obs_d):
            cmin_z[d] = np.min(z_new[:,d]) - 3*np.std(z_new[:,d])
            cmax_z[d] = np.max(z_new[:,d]) + 3*np.std(z_new[:,d])
            z_new_temp = normalize(z_new[:,d], cmin_z[d], cmax_z[d])
            z_new_temp = preprocess_data(z_new_temp)
            combined_input[0,:,:,self.state_d + d] = z_new_temp
            
        x_new_update = self.Updator.generator(combined_input, training=False)[0].numpy()   
        for d in range(self.state_d):
            output_x[:,d] = unnormalize(np.reshape(x_new_update[:,:,d], (-1)), cmin_x[d], cmax_x[d])

        return output_x
        
    
class DeepFusionFilterMV():
    def __init__(self, state_d = 1, obs_d = 1):
        self.Updator = DeepFilter(input_shape = (128,128,state_d + obs_d), output_shape = (128,128,state_d), lr = 1e-5, max_filters = 512)
        self.state_d = state_d
        self.obs_d = obs_d        
    def train_updator(self,x_old, x_new, z_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        x_new_input = np.empty((1,128,128,self.state_d))
        combined_input = np.empty((1,128,128,self.state_d+self.obs_d))
        for d in range(self.state_d):
            cmin_x = np.min(x_old[:,d]) - 3*np.std(x_old[:,d])
            cmax_x = np.max(x_old[:,d]) + 3*np.std(x_old[:,d])
            x_new_temp = normalize(x_new[:,d], cmin_x, cmax_x)
            x_new_temp = preprocess_data(x_new_temp)
            x_old_temp = normalize(x_old[:,d], cmin_x, cmax_x)
            x_old_temp = preprocess_data(x_old_temp)
            x_old_input[0,:,:,d] = x_old_temp
            x_new_input[0,:,:,d] = x_new_temp
        combined_input[0,:,:,0:self.state_d] = x_old_input
        for d in range(self.obs_d):
            cmin_z = np.min(z_new[:,d]) - 3*np.std(z_new[:,d])
            cmax_z = np.max(z_new[:,d]) + 3*np.std(z_new[:,d])
            z_new_temp = normalize(z_new[:,d], cmin_z, cmax_z)
            z_new_temp = preprocess_data(z_new_temp)
            combined_input[0,:,:,self.state_d + d] = z_new_temp
        self.Updator.train_step(combined_input, x_new_input, L = 100)
                                
    def filter_step(self, x_old,z_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        combined_input = np.empty((1,128,128,self.state_d+self.obs_d))
        output_x = np.empty_like(x_old)
        cmin_x = np.empty(self.state_d)
        cmax_x = np.empty(self.state_d)
        cmin_z = np.empty(self.obs_d)
        cmax_z = np.empty(self.obs_d)
        for d in range(self.state_d):
            cmin_x[d] = np.min(x_old[:,d]) - 3*np.std(x_old[:,d])
            cmax_x[d] = np.max(x_old[:,d]) + 3*np.std(x_old[:,d])
            x_old_temp = normalize(x_old[:,d], cmin_x[d], cmax_x[d])
            x_old_temp = preprocess_data(x_old_temp)
            x_old_input[0,:,:,d] = x_old_temp
        combined_input[0,:,:,0:self.state_d] = x_old_input
        for d in range(self.obs_d):
            cmin_z[d] = np.min(z_new[:,d]) - 3*np.std(z_new[:,d])
            cmax_z[d] = np.max(z_new[:,d]) + 3*np.std(z_new[:,d])
            z_new_temp = normalize(z_new[:,d], cmin_z[d], cmax_z[d])
            z_new_temp = preprocess_data(z_new_temp)
            combined_input[0,:,:,self.state_d + d] = z_new_temp
            
        x_new_update = self.Updator.generator(combined_input, training=False)[0].numpy()   
        for d in range(self.state_d):
            output_x[:,d] = unnormalize(np.reshape(x_new_update[:,:,d], (-1)), cmin_x[d], cmax_x[d])

        return output_x
    
    
class DeepNoisyBayesianFilter():
    def __init__(self, H = None):
        self.Predictor = DeepFilter(input_shape = (128,128,1), output_shape = (128,128,1))
        self.NoisePredictor = DeepFilter(input_shape = (128,128,1), output_shape = (128,128,1), lr = 1e-5, max_filters = 512)
        self.Updator = DeepFilter(input_shape = (128,128,2), output_shape = (128,128,1))
        self.H = H
    def train_predictor(self, x_old, x_new):
        cmin_x = np.min(x_old) - 10*np.std(x_old)
        cmax_x = np.max(x_old) + 10*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        self.Predictor.train_step(np.expand_dims(x_old,3), np.expand_dims(x_new,3), L = 50)
    def train_noise_predictor(self, z_new):
        z_sigma = smooth_var(z_new)
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_output = z_new - np.mean(z_new)
        cmin_z = np.min(z_output) - 3*np.std(z_output)
        cmax_z = np.max(z_output) + 3*np.std(z_output)
        z_output_normalized = normalize(z_output, cmin_z, cmax_z)
        z_sigma_normalized = preprocess_data(z_sigma_normalized)
        z_output_normalized = preprocess_data(z_output_normalized)
        self.NoisePredictor.train_step(np.expand_dims(z_sigma_normalized,3), np.expand_dims(z_output_normalized,3), L = 30)
    def train_updator(self,x_old, x_new, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        self.Updator.train_step(np.expand_dims(input_data_update,0), np.expand_dims(x_new,3), L = 100)
        
    def predict_var(self, x_old,z_new):
        cmin_z = np.min(z_new-np.mean(z_new)) - 3*np.std(z_new)
        cmax_z = np.max(z_new-np.mean(z_new)) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        x_old_normalized = normalize(x_old, cmin_x, cmax_x)
        x_old_normalized = preprocess_data(x_old_normalized)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old_normalized,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        x_stack = []
        for ii in range(50):
            z_sigma = gen_sample()
            cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
            cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
            
            
            
            z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
            z_sigma_normalized = preprocess_data(z_sigma_normalized)
            noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
            noise_hat = np.reshape(noise_hat, (-1))
            noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
            noise_hat = smooth(noise_hat)
            z_new_noisy = z_new + noise_hat
            cmin_z_n = np.min(z_new_noisy) - 3*np.std(z_new_noisy)
            cmax_z_n = np.max(z_new_noisy) + 3*np.std(z_new_noisy)
            z_new_noisy_normalized = normalize(z_new_noisy, cmin_z_n, cmax_z_n)
            
            
            input_data_update = preprocess_Bayesian_data(x_new_hat, z_new_noisy_normalized)
            x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
            x_new_update = np.reshape(x_new_update, (-1))
            x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
            x_stack.append(x_new_update)
        x_stack = np.array(x_stack)
        var = np.var(x_stack, axis = 0)
        return var
        
    def predict_mean(self, x_old,z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = preprocess_data(x_old)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
        x_new_update = np.reshape(x_new_update, (-1))
        x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
        return x_new_update
    
    def gen_noise(self, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        z_sigma = gen_sample()
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_sigma_normalized = preprocess_data(z_sigma_normalized)
        noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
        noise_hat = np.reshape(noise_hat, (-1))
        noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
        noise_hat = smooth(noise_hat)
        return noise_hat
    

class DeepNoisyBayesianFilterMV():
    def __init__(self, H = None, state_d = 1, obs_d = 1):
        self.Predictor = DeepFilter(input_shape = (128,128,state_d), output_shape = (128,128,state_d))
        self.NoisePredictor = DeepFilter(input_shape = (128,128,obs_d), output_shape = (128,128,obs_d), lr = 1e-5, max_filters = 512)
        self.Updator = DeepFilter(input_shape = (128,128,state_d + obs_d), output_shape = (128,128,state_d)) 
        self.state_d = state_d
        self.obs_d = obs_d 
        self.H = H
        
    def train_predictor(self, x_old, x_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        x_new_input = np.empty((1,128,128,self.state_d))
        for d in range(self.state_d):
            cmin_x = np.min(x_old[:,d]) - 10*np.std(x_old[:,d])
            cmax_x = np.max(x_old[:,d]) + 10*np.std(x_old[:,d])
            x_old_temp = normalize(x_old[:,d], cmin_x, cmax_x)
            x_new_temp = normalize(x_new[:,d], cmin_x, cmax_x)
            x_new_temp = preprocess_data(x_new_temp)
            x_old_temp = preprocess_data(x_old_temp)
            x_old_input[0,:,:,d] = x_old_temp
            x_new_input[0,:,:,d] = x_new_temp
        self.Predictor.train_step(x_old_input, x_new_input, L = 50)

    def train_noise_predictor(self, z_new):
        z_new_input = np.empty((1,128,128,self.obs_d))
        z_new_output = np.empty((1,128,128,self.obs_d))
        for d in range(self.obs_d):
            z_sigma = smooth_var(z_new[:,d])
            cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
            cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
            z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
            z_output = z_new[:,d] - np.mean(z_new[:,d])
            cmin_z = np.min(z_output) - 3*np.std(z_output)
            cmax_z = np.max(z_output) + 3*np.std(z_output)
            z_output_normalized = normalize(z_output, cmin_z, cmax_z)
            z_sigma_normalized = preprocess_data(z_sigma_normalized)
            z_output_normalized = preprocess_data(z_output_normalized)
            z_new_input[0,:,:,d] = z_sigma_normalized
            z_new_output[0,:,:,d] = z_output_normalized
        self.NoisePredictor.train_step(z_new_input, z_new_output, L = 30)
        
    def train_updator(self,x_old, x_new, z_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        x_new_output = np.empty((1,128,128,self.state_d))
        combined_input = np.empty((1,128,128,self.state_d+self.obs_d))
        for d in range(self.state_d):
            cmin_z = np.min(z_new[:,d]) - 3*np.std(z_new[:,d])
            cmax_z = np.max(z_new[:,d]) + 3*np.std(z_new[:,d])
            if self.H is None:
                cmin_x = np.min(x_old[:,d]) - 10*np.std(x_old[:,d])
                cmax_x = np.max(x_old[:,d]) + 10*np.std(x_old[:,d])
            else:
                cmin_x = cmin_z/self.H
                cmax_x = cmax_z/self.H
            z_new_temp = normalize(z_new[:,d], cmin_z, cmax_z)
            x_old_temp = normalize(x_old[:,d], cmin_x, cmax_x)
            x_new_temp = normalize(x_new[:,d], cmin_x, cmax_x)
            x_new_temp = preprocess_data(x_new_temp)
            x_old_temp = preprocess_data(x_old_temp)
            x_old_input[0,:,:,d] = x_old_temp
            x_new_output[0,:,:,d] = x_new_temp
        x_new_hat = self.Predictor.generator(x_old_input, training=False).numpy()
        combined_input[0,:,:,0:self.state_d] = x_new_hat
        for d in range(self.obs_d):
            cmin_z = np.min(z_new[:,d]) - 3*np.std(z_new[:,d])
            cmax_z = np.max(z_new[:,d]) + 3*np.std(z_new[:,d])
            z_new_temp = normalize(z_new[:,d], cmin_z, cmax_z)
            z_new_temp = preprocess_data(z_new_temp)
            combined_input[0,:,:,self.state_d + d] = z_new_temp
        self.Updator.train_step(combined_input, x_new_output, L = 100)
        
    def predict_var(self, x_old,z_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        combined_input = np.empty((1,128,128,self.state_d+self.obs_d))
        output_x = np.empty_like(x_old)
        conditioned_z = np.empty_like(z_new)
        z_new_noisy = np.empty_like(z_new)
        cmin_x = np.empty(self.state_d)
        cmax_x = np.empty(self.state_d)
        cmin_z = np.empty(self.obs_d)
        cmax_z = np.empty(self.obs_d)
        cov_output = []
        for d in range(self.state_d):
            cmin_z[d] = np.min(z_new[:,d]-np.mean(z_new[:,d])) - 3*np.std(z_new[:,d])
            cmax_z[d] = np.max(z_new[:,d]-np.mean(z_new[:,d])) + 3*np.std(z_new[:,d])
            if self.H is None:
                cmin_x[d] = np.min(x_old[:,d]) - 10*np.std(x_old[:,d])
                cmax_x[d] = np.max(x_old[:,d]) + 10*np.std(x_old[:,d])
            else:
                cmin_x[d] = cmin_z[d]/self.H
                cmax_x[d] = cmax_z[d]/self.H
            x_old_normalized = normalize(x_old[:,d], cmin_x[d], cmax_x[d])
            x_old_normalized = preprocess_data(x_old_normalized)
            x_old_input[0,:,:,d] = x_old_normalized
        x_new_hat = self.Predictor.generator(x_old_input, training=False)[0].numpy()
        combined_input[0,:,:,0:self.state_d] = x_new_hat
        x_new_hat = np.reshape(x_new_hat, (-1))
        x_stack = []
        for ii in range(50):
            for d in range(self.obs_d):
                z_sigma = gen_sample()
                cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
                cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
                z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
                z_sigma_normalized = preprocess_data(z_sigma_normalized)
                conditioned_z[0:,:,d] = z_sigma_normalized
                
            noise_hat = self.NoisePredictor.generator(conditioned_z, training=False)[0].numpy()
            for d in range(self.obs_d):
                noise_hat_temp = unnormalize(noise_hat[:,d], cmin_z[d], cmax_z[d])
                noise_hat_temp = smooth(noise_hat_temp)
                z_new_temp = preprocess_data(z_new[:,d])
                z_new_noisy[0:,:,d] = z_new_temp + noise_hat_temp
     
                z_new_noisy[0:,:,d] = normalize(z_new_noisy[0:,:,d], cmin_z[d], cmax_z[d])
            
            combined_input[0,:,:,self.state_d + d] = z_new_noisy
            x_new_update = self.Updator.generator(combined_input, training=False)[0].numpy()
            for d in range(self.state_d):
                output_x[:,d] = unnormalize(np.reshape(x_new_update[:,:,d], (-1)), cmin_x[d], cmax_x[d])
            x_stack.append(output_x)
        x_stack = np.array(x_stack)
        for ii in range(x_stack.shape[1]):
            cov_output.append(np.cov(x_stack[:,ii,:].T))
        return cov_output
        
    def predict_mean(self, x_old,z_new):
        x_old_input = np.empty((1,128,128,self.state_d))
        combined_input = np.empty((1,128,128,self.state_d+self.obs_d))
        output_x = np.empty_like(x_old)
        cmin_x = np.empty(self.state_d)
        cmax_x = np.empty(self.state_d)
        cmin_z = np.empty(self.obs_d)
        cmax_z = np.empty(self.obs_d)
        for d in range(self.state_d):
            cmin_x[d] = np.min(x_old[:,d]) - 3*np.std(x_old[:,d])
            cmax_x[d] = np.max(x_old[:,d]) + 3*np.std(x_old[:,d])
            x_old_temp = normalize(x_old[:,d], cmin_x[d], cmax_x[d])
            x_old_temp = preprocess_data(x_old_temp)
            x_old_input[0,:,:,d] = x_old_temp
        x_new_hat = self.Predictor.generator(x_old_input, training=False).numpy()
        combined_input[0,:,:,0:self.state_d] = x_new_hat
        for d in range(self.obs_d):
            cmin_z[d] = np.min(z_new[:,d]) - 3*np.std(z_new[:,d])
            cmax_z[d] = np.max(z_new[:,d]) + 3*np.std(z_new[:,d])
            z_new_temp = normalize(z_new[:,d], cmin_z[d], cmax_z[d])
            z_new_temp = preprocess_data(z_new_temp)
            combined_input[0,:,:,self.state_d + d] = z_new_temp
            
        x_new_update = self.Updator.generator(combined_input, training=False)[0].numpy()   
        for d in range(self.state_d):
            output_x[:,d] = unnormalize(np.reshape(x_new_update[:,:,d], (-1)), cmin_x[d], cmax_x[d])

        return output_x
    
    def gen_noise(self, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        z_sigma = gen_sample()
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_sigma_normalized = preprocess_data(z_sigma_normalized)
        noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
        noise_hat = np.reshape(noise_hat, (-1))
        noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
        noise_hat = smooth(noise_hat)
        return noise_hat
   

        
       





  



