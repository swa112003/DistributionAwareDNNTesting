'''
Code is implemented over the baseline provided in keras git repo
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate SVHN inputs by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Lambda, Input, Dense, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Flatten
from keras.layers import BatchNormalization, LeakyReLU, ReLU
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam
from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
import argparse
import os
import utils

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def Vae_SVHN(input_tensor=None, train=False):
    np.random.seed(0)

    if train:
        # network parameters
        input_shape = (image_size, image_size, image_chn)
        input_tensor = Input(shape=input_shape)

        batch_size = 128
        epochs = 200

    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()

    latent_dim = 512
    
    # VAE model = encoder + decoder
    # build encoder model
    x = Conv2D(8, (5, 5), padding='same', strides=(2, 2), activation='relu')(input_tensor)
    x = Conv2D(16, (5, 5), padding='same', activation='relu')(x)
    x = Conv2D(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = Conv2D(64, (5, 5), padding='same', activation='relu')(x)
    x = Conv2D(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
      
    
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(input_tensor, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)

    x = Conv2DTranspose(64, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = Conv2DTranspose(64, (5, 5), padding='same', activation='relu')(x)
    x = Conv2DTranspose(32, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    x = Conv2DTranspose(16, (5, 5), padding='same', activation='relu')(x)
    x = Conv2DTranspose(8, (5, 5), padding='same', strides=(2, 2), activation='relu')(x)
    pos_mean = Conv2DTranspose(3, (5, 5), padding='same', name='pos_mean')(x)
    pos_log_var = Conv2DTranspose(3, (5, 5), padding='same', name='pos_log_var')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, [pos_mean, pos_log_var], name='decoder')
    #decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(input_tensor)[2])
    vae = Model(input_tensor, outputs, name='vae_svhn')
    #vae.summary()
    
    if train:
        # VAE loss = reconstruction_loss + kl_loss
        loss_a = float(np.log(2 * np.pi)) + outputs[1]
        loss_m = K.square(outputs[0] - input_tensor) / K.exp(outputs[1])
        reconstruction_loss = -0.5 * K.sum((loss_a + loss_m), axis=[-1,-2, -3])

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(-reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer="adam")
        vae.summary()
        vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None), verbose=2)
        # save model
        vae.save_weights('./vae_svhn.h5')
    else:
        vae.load_weights('./vae_svhn.h5')
        print('VAE loaded')
    return vae

    
image_size = 32
image_chn = 3
#Load SVHN data
train_raw = loadmat('./dataset/train_32x32.mat')
train_data = np.array(train_raw['X'])
train_data = np.moveaxis(train_data, -1, 0)
test_raw = loadmat('./dataset/test_32x32.mat')
test_data = np.array(test_raw['X'])
test_data = np.moveaxis(test_data, -1, 0)

x_train = np.reshape(train_data, [-1, image_size, image_size, image_chn])
x_test = np.reshape(test_data, [-1, image_size, image_size, image_chn])

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255


if __name__ == '__main__':
    vae = Vae_SVHN(train=True)
