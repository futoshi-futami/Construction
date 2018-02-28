# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:36:47 2018

@author: Futami
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from keras.layers import Input, Dense, Lambda, Add, Multiply, PReLU
from keras.models import Model
from keras import backend as K
from keras import metrics
import math
from keras.datasets import mnist

batch_size = 100
original_dim = 784
latent_dim = 30
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean_0 = Dense(latent_dim)(h)
z_std_0 = Dense(latent_dim, activation="softplus")(h)
hp = Dense(latent_dim)(h)
encoder = Model(x, [z_mean_0,z_std_0, hp])

class MaskedDense(Dense):
    """A dense layer with a masking possibilities"""

    def __init__(self, units, mask, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None,
                 transpose=False, **kwargs):
        super(MaskedDense, self).__init__(units, bias_initializer=bias_initializer,
                                          activation=activation, kernel_initializer=kernel_initializer,
                                          kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                                          kernel_constraint=kernel_constraint, bias_constraint=bias_constraint,
                                          use_bias=use_bias, **kwargs)
        if not transpose:
            self.mask = K.variable(mask)
        else:
            self.mask = K.variable(mask.T)

    def call(self, x, mask=None):
        output = K.dot(x, Multiply()([self.kernel, self.mask]))
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
    

def _mask_matrix_made(dim):
    """A generator of masks for two-layered MADE model (see https://arxiv.org/pdf/1502.03509.pdf)"""
    mask_vector = np.random.randint(1, dim, dim)
    mask_matrix0 = np.fromfunction(lambda k, d: mask_vector[k] >= d, (dim, dim), dtype=int).astype(np.int32).astype(np.float32)
    mask_matrix1 = np.fromfunction(lambda d, k: d > mask_vector[k], (dim, dim), dtype=int).astype(np.int32).astype(np.float32)
    return mask_matrix0, mask_matrix1


def MADE(mask_matrix0, mask_matrix1, latent_dim):
    """A 2-layered MADE model (https://arxiv.org/pdf/1502.03509.pdf)"""
    def f(x):
        hl = MaskedDense(latent_dim, mask=mask_matrix0)(x)
        hl = PReLU()(hl)
        std = MaskedDense(latent_dim, mask=mask_matrix1, activation="softplus")(hl)
        mean = MaskedDense(latent_dim, mask=mask_matrix1, activation=None)(hl)
        return mean, std

    return f

n_latent = 10  # the number of IAF transform you want to apply
latent_models = []

masks = [_mask_matrix_made(latent_dim) for k in range(n_latent)]
    
for k in range(n_latent):
    latent_input = Input(shape=(latent_dim,), batch_shape=(batch_size, latent_dim))

    mask0, mask1 = masks[k]
    mean, std = MADE(mask0, mask1, latent_dim)(latent_input)

    latent_model = Model(latent_input, [mean, std])
    latent_models.append(latent_model)
    
def sample_eps(batch_size, latent_dim, epsilon_std):
    """Create a function to sample N(0, epsilon_std) vectors"""
    return lambda args: K.random_normal(shape=(batch_size, latent_dim),
                                        mean=0.,
                                        stddev=epsilon_std)

def sample_z0(args):
    """Sample from N(mu, sigma) where sigma is the stddev !!!"""
    z_mean, z_std, epsilon = args
    z0 = z_mean + K.exp(K.log(z_std + 1e-8)) * epsilon
    return z0

def iaf_transform_z(args):
    """Apply the IAF transform to input z (https://arxiv.org/abs/1606.04934)"""
    z, mean, std = args
    z_ = z
    z_ -= mean
    z_ /= std
    return z_

eps = Lambda(sample_eps(batch_size, latent_dim, epsilon_std), name='sample_eps')([z_mean_0, z_std_0])
z0 = Lambda(sample_z0, name='sample_z0')([z_mean_0, z_std_0, eps])

z_means = [z_mean_0]
z_stds = [z_std_0]
zs = [z0]
for latent_model in latent_models:
    zz = Add()([Dense(latent_dim, activation='relu')(hp), zs[-1]])
    z_mean, z_std = latent_model(zz)
    z_means.append(z_mean)
    z_stds.append(z_std)
    z = Lambda(iaf_transform_z)([zs[-1], z_mean, z_std])
    zs.append(z)
z = zs[-1]

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

def log_stdnormal(x):
    """log density of a standard gaussian"""
    c = - 0.5 * math.log(2*math.pi)
    result = c - K.square(x) / 2
    return result

def log_normal2(x, mean, log_var):
    """log density of N(mu, sigma)"""
    c = - 0.5 * math.log(2*math.pi)
    result = c - log_var/2 - K.square(x - mean) / (2 * K.exp(log_var) + 1e-8)
    return result

n_sample = 20


def vae_loss(x, x_decoded_mean):
    """
    Variationnal lower bound
    This is the cross entropy minus the KL(Q(.|z)||P(.))
    The latter term is estimated by Monte Carlo sampling
    """
    xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)

    # kl divergence
    # sampling for estimating the expectations
    for k in range(n_sample):
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=1.0)  # used for every z_i sampling
        z0_ = z_mean_0 + z_std_0 * epsilon
        z_ = z0_
        for z_mean, z_std in zip(z_means[1:], z_stds[1:]):
            z_ = iaf_transform_z([z_, z_mean, z_std])

        try:
            loss += K.sum(log_normal2(z0_, z_mean_0, 2 * K.log(z_std_0 + 1e-8)), -1)
        except NameError:
            loss = K.sum(log_normal2(z0_, z_mean_0, 2 * K.log(z_std_0 + 1e-8)), -1)
        loss -= K.sum(log_stdnormal(z_), -1)
    # don't forget the log_std_sum.
    # BE CAUTIOUS!! THE LOG_STD_SUM_0 HAS ALREADY BEEN TAKEN INTO ACCOUNT IN LOSS
    kl_loss = loss / n_sample
    for z_std in z_stds[1:]:
        kl_loss += K.sum(K.log(1e-8 + z_std), -1)

    return xent_loss + kl_loss