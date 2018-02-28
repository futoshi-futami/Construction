# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 13:06:54 2018

@author: Futami
"""

import numpy as np
import tensorflow as tf
rng = np.random.RandomState(1234)

from collections import OrderedDict

from tensorflow.python.framework import ops
from tensorflow.python.layers import core as layers
from tensorflow.python.ops import template as template_ops
from tensorflow.python.ops import control_flow_ops

############ functions related to making grid samoles (U)
def grid_samples_simple(D,Grid_num):
    Comb=np.arange(1,Grid_num)[:,None]
    np.random.shuffle(Comb)
    for i in range(D-1):
        a=np.arange(1,Grid_num)[:,None]
        np.random.shuffle(a)
        Comb=np.concatenate((Comb,a),1)
    Comb=Comb/Grid_num
    Batch_U=Comb.shape[0]
    return Comb,Batch_U

def grid_samples(D,Grid_num):
    import itertools
    list1=range(1,Grid_num)
    Combination=list(itertools.product(list1, repeat=3))
    Comb=np.array(Combination)
    Comb=Comb/Grid_num
    np.random.shuffle(Comb)
    Batch_U=Comb.shape[0]
    return Comb,Batch_U

def grid_samples_simple2(D,Grid_num):
    Comb=np.arange(400,600)[:,None]
    np.random.shuffle(Comb)
    for i in range(D-1):
        a=np.arange(400,500)[:,None]
        np.random.shuffle(a)
        Comb=np.concatenate((Comb,a),1)
    Comb=Comb/Grid_num
    Batch_U=Comb.shape[0]
    return Comb,Batch_U




################functions related to placeholders


def make_placeholders(n_ph,D):
    dic = OrderedDict()
    for i in range(n_ph):
        dic["U_" + str(i)] = tf.placeholder(tf.float32,[1,D],name = "U_placeholder"+ str(i))
    return dic

def make_placeholders_ones(n_ph,D):
    '''
    This is for mean field placeholders.
    D : all the dimensions of the model
    '''
    dic = OrderedDict()
    for i in range(n_ph*D):
        dic["U_" + str(i)] = tf.placeholder(tf.float32,[1,1],name = "U_placeholder"+ str(i))
    return dic

def make_placeholders_ones_test_data(num_sample,D):
    '''
    This is the place holder for predicting the test data
    num_sample : how many samples we use when predicting
    '''
    dic = OrderedDict()
    for i in range(D):
        dic["Ut_" + str(i)] = tf.placeholder(tf.float32,[num_sample,1],name = "U_placeholder"+ str(i))
    return dic








################  functions related to constructing trial functions

def make_dense_function_G(units,name,activation=tf.nn.sigmoid):
    '''
    This is the function constructing one hidden layer dense NN.
    Network archtecture is D-actvation-units-activation-D
    '''
    def _fn(x):
        with ops.name_scope(name, "trial_function"):
            layer = layers.Dense(
                    units,
                    activation=activation,
                    name=name,
                    _scope=name)
            
            layer2 = layers.Dense(
                    units=1,
                    #activation=activation,
                    name=name,
                    _scope=name+'2')
            '''
            layer3 = layers.Dense(
                    units=1,
                    activation=activation,
                    name=name,
                    _scope=name+'3')
            '''            
            return layer2.apply(layer.apply(x))
            #return layer3.apply(layer2.apply(layer.apply(x)))
    return template_ops.make_template("trial_function", _fn)

def make_trials_G(D):
    '''
    This is the function for preparing trial functions when MF approximation.
    We prepare #D trial functions which we use one hidden layer dense NN.
    '''
    return [make_dense_function_G(30,"U_" + str(i)) for i in range(D)]





def make_dense_function(units,name,D,activation=tf.nn.sigmoid,MF=True):
    '''
    This is the function constructing one hidden layer dense NN.
    Network archtecture is D-actvation-units-activation-D
    '''
    if MF:
        output=1,
    else:
        output=D
    def _fn(x):
        with ops.name_scope(name, "trial_function"):
            layer = layers.Dense(
                    units,
                    activation=activation,
                    name=name,
                    _scope=name)
            
            layer2 = layers.Dense(
                    units=output,
                    activation=activation,
                    name=name,
                    _scope=name+'2')
            '''
            layer3 = layers.Dense(
                    units=1,
                    activation=activation,
                    name=name,
                    _scope=name+'3')
            '''            
            return layer2.apply(layer.apply(x))
            #return layer3.apply(layer2.apply(layer.apply(x)))
    return template_ops.make_template("trial_function", _fn)

def make_trials(D,MF=True):
    '''
    This is the function for preparing trial functions when MF approximation.
    We prepare #D trial functions which we use one hidden layer dense NN.
    '''
    if MF:
        return [make_dense_function(8,"U_" + str(i)) for i in range(D)]
    else:
        return make_dense_function(8,"U",D,MF=False)



from MADE import masked_autoregressive_default_template2

##### Trial function 3-20-20-3

#trial_f=masked_autoregressive_default_template2([20,20],shift_only=True)

def make_MADE_trial(H):
    
    def _fn(U0):
        U=minus_inf_inf_to_zero_one(U0)
        trial_f=masked_autoregressive_default_template2([H])
    
        shift,log_scale=trial_f(U)
        sigma1=tf.nn.sigmoid(log_scale)
        _W=U*sigma1+shift*(1.-sigma1)
    
        trial_f2=masked_autoregressive_default_template2([H],name='2')
        shift2,log_scale2=trial_f2(_W)
        sigma2=tf.nn.sigmoid(log_scale2)
        _W2=_W*sigma2+shift2*(1.-sigma2)

        trial_f3=masked_autoregressive_default_template2(hidden_layers=[20],name='3')
        shift3,log_scale3=trial_f3(_W2)
        sigma3=tf.nn.sigmoid(log_scale3)
        _W3=_W2*sigma3+shift3*(1.-sigma3)

        shift4,log_scale4=trial_f(_W3)
        sigma4=tf.nn.sigmoid(log_scale4)
        _W4=_W3*sigma4+shift4*(1.-sigma4)

        W=tf.nn.sigmoid(_W3)

        return W,[sigma1,sigma2,sigma3]#,sigma4]

    return _fn

################### functions related to support transformations


def minus_inf_inf_to_zero_one(variable):
    '''
    This is the function converting the support [0,1] to [-∞,∞] by inverse of sigmoid function
    '''
    return tf.log(variable/(1.-variable))

def minus_inf_inf_to_zero_one_Log_det_J(variable):
    '''
    This is the function of calculating the jacobian of inverse of sigmoid function
    '''
    return tf.log(1./(variable*(1.-variable)))








################# functions related to gradient calculations



def tf_gradient_element_wise(W,U):
    '''
    This is the function of calculating the element wise gradient
    e.g)
    when W is [100,1] and U is [100,3], outputs the [100,3] tensor.
    Be carefull that the axis of 1 of W should be 1, that is the tensor such as [100,3] or [100,None] cannot be treated properly.
    
    If you want to handle the tensor whose axis of 1 is greater than 1, please use Jacobian function or Masked neural net for the trial function
    '''
    num_splits=W.shape[0].value
    _W=tf.split(axis = 0, num_or_size_splits = num_splits, value = W)
    jacobian_row=tf.gradients(_W[0],U)[0][0][:,None]
    jaco_=[tf.gradients(_W[index],U)[0][index][:,None] for index in range(1,num_splits) ]
    for j in jaco_:
        jacobian_row=tf.concat((jacobian_row,j),0)
    return jacobian_row


def _loop(f,iter_num,D):
    event_size = iter_num
    _U=tf.contrib.distributions.Uniform().sample(sample_shape=(1,D))
    U=tf.split(axis = 1, num_or_size_splits = D, value = _U)
    loss = f(U)

    def _loop_body(index, loss):
      _U=tf.contrib.distributions.Uniform().sample(sample_shape=(1,D))
      U=tf.split(axis = 1, num_or_size_splits = D, value = _U)    
      loss_new=loss+f(U)              
      return index + 1, loss_new
  
    _, y = control_flow_ops.while_loop(
        cond=lambda index, _: index < event_size,
        body=_loop_body,
        loop_vars=[0, loss])
    return y






################# functions related to feed_dict



def feed_to_U_MF(dic,dictU,source,D):
    '''
    This is the function preparing the feed_dict for the placeholder. This is used when mean field approximation
    
    dic : this is the ordered dictionary, which already contains data X and Y which we feed to the placeholder.
    dicU : This is the ordered dictionary, which contains the placeholder U.
    source : We will feed placeholder U by using this data.
    D : number of dimensions(number of placeholders when using MF)
    '''
    for ii,j in enumerate(dictU.values()):
        if ii%D == 0:
            U_batch = next(source)[0]
        dic[j] = U_batch[:,ii%D][:,None]
    return dic



def feed_to_U(dic,dictU,source,D):
    '''
    This is the function preparing the feed_dict for the placeholder. This is used when mean field approximation
    
    dic : this is the ordered dictionary, which already contains data X and Y which we feed to the placeholder.
    dicU : This is the ordered dictionary, which contains the placeholder U.
    source : We will feed placeholder U by using this data.
    D : number of dimensions(number of placeholders when using MF)
    '''
    for ii,j in enumerate(dictU.values()):
        U_batch = next(source)[0]
        dic[j] = U_batch
    return dic
