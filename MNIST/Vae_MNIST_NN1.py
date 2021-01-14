'''
Code is implemented over the baseline provided in keras git repo
https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
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
from keras.models import Model
from keras.datasets import mnist, fashion_mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.optimizers import Adam
from keras.models import model_from_json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import multivariate_normal

import argparse
import os
from utils import *

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

def Vae_MNIST_NN1(input_tensor=None, train=False):
    np.random.seed(0)
    # MNIST dataset
    image_size = 28
    if train:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = np.reshape(x_train, [-1, image_size, image_size, 1])
        x_test = np.reshape(x_test, [-1, image_size, image_size, 1])
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255

        # network parameters
        input_shape = (image_size, image_size, 1)
        input_tensor = Input(shape=input_shape)

        batch_size = 128
        epochs = 50

    elif input_tensor is None:
        print('you have to proved input_tensor when testing')
        exit()

    latent_dim = 200
    intermediate_dims = np.array([400])

    # VAE model = encoder + decoder
    # build encoder model
    original_dim = image_size * image_size
    inputs = Reshape((original_dim,), name='encoder_input')(input_tensor)
    x = Dense(intermediate_dims[0], activation='relu')(inputs)
    for i in range(intermediate_dims.shape[0]):
        if i != 0:
            x = Dense(intermediate_dims[i], activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(input_tensor, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()

    # build decoder model
    intermediate_dims = np.flipud(intermediate_dims)
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dims[0], activation='relu')(latent_inputs)
    for i in range(intermediate_dims.shape[0]):
        if i != 0:
            x = Dense(intermediate_dims[i], activation='relu')(x)
    pos_mean = Dense(original_dim, name='pos_mean')(x)
    pos_log_var = Dense(original_dim, name='pos_log_var')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, [pos_mean, pos_log_var], name='decoder')
    #decoder.summary()

    # instantiate VAE model
    outputs = decoder(encoder(input_tensor)[2])
    vae = Model(input_tensor, outputs, name='vae_mlp')
    #vae.summary()
    if train:
        # VAE loss = reconstruction_loss + kl_loss
        loss_a = float(np.log(2 * np.pi)) + outputs[1]
        loss_m = K.square(outputs[0] - inputs) / K.exp(outputs[1])
        reconstruction_loss = -0.5 * K.sum((loss_a + loss_m), axis=-1)

        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(-reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer="adam")
        vae.summary()
        vae.add_metric(reconstruction_loss, "reconstruct")
        vae.add_metric(kl_loss, "kl")
        vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))
        # save model
        vae.save_weights('./vae_mnist_nn1.h5')
    else:
        vae.load_weights('./vae_mnist_nn1.h5')
        print('VAE loaded')
    return vae

if __name__ == '__main__':
    Vae_MNIST_NN1(train=True)
