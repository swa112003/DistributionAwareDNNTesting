import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model

import os
import glob
import math

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape((x.shape[1], x.shape[2], x.shape[3]))  # original shape (1,img_rows, img_cols,3)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    #from imagenet grad_mean = 1e4 * np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False

def delete_files_from_dir(dirPath, ext):
    # eg input = /tmp/*.txt
    fileFormat = dirPath + '*.' + ext
    files = glob.glob(fileFormat)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
            
            
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

    
def _reconstruction_probability_svhn(decoder, z_mean, z_log_var, X):
    """
    :param decoder: decoder model
    :param z_mean: encoder predicted mean value
    :param z_log_var: encoder predicted sigma square value
    :param X: input data
    :return: reconstruction probability of input
            calculated over L samples from z_mean and z_log_var distribution
    """
    
    
    L = 1000
    start_index = 0
    stop_index = 1000
    if X.shape[0] < 1000:
        stop_index = X.shape[0]
    result =  np.array([])
    for i in range(math.ceil(X.shape[0]/1000)): 
        reconstructed_prob = np.zeros((stop_index - start_index,), dtype='float32')
        z_mean_part = z_mean[start_index:stop_index,:]
        z_log_var_part = z_log_var[start_index:stop_index,:]
        # print(start_index,stop_index)
        for l in range(L):
            sampled_zs = sampling([z_mean_part, z_log_var_part])
            
            #commented steps=1
            mu_hat, log_sigma_hat = decoder.predict(sampled_zs, steps=1)
            log_sigma_hat = np.float64(log_sigma_hat)
            sigma_hat = np.exp(log_sigma_hat) + 0.00001

            loss_a = np.log(2 * np.pi * sigma_hat)
            loss_m = np.square(mu_hat - X[start_index:stop_index,:,:,:]) / sigma_hat
            reconstructed_prob += -0.5 * np.sum(loss_a + loss_m, axis=(-1,-2, -3))
        reconstructed_prob /= L
        result = np.concatenate((result, reconstructed_prob))
        start_index += 1000
        if (stop_index+1000) < X.shape[0]:
            stop_index += 1000
        else:
            stop_index = X.shape[0]
    # print("from utils recon prob shape ", result.shape)
    return result


def reconstruction_probability_svhn(x_target, vae):
    z_mean, z_log_var, _ = vae.get_layer('encoder').predict(x_target, batch_size=128)
    reconstructed_prob_x_target = _reconstruction_probability_svhn(vae.get_layer('decoder'), z_mean, z_log_var, x_target)
    return reconstructed_prob_x_target
    
    
def isInvalid(gen_img, vae, vae_threshold):
    gen_img_density = reconstruction_probability_svhn(gen_img, vae)
    if gen_img_density < vae_threshold or math.isnan(gen_img_density):
        return True
    else:
        return False
