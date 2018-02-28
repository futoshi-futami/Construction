# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 14:40:48 2018

@author: seiic
"""


import numpy as np
import tensorflow as tf
from data_load import generator

rng = np.random.RandomState(1234)
learning_rate = 0.01
feature_dim = 1
n_epochs = 10

# define multimodal distribution

def Gauss(loc,scale,x):
    return tf.exp(-tf.square(x-loc)/np.sqrt((2*np.square(scale))))/(np.sqrt((2*np.pi*np.square(scale))))

def Mixture_Gauss(loc1,loc2,scale1,scale2,x,mix):
    return mix * Gauss(loc1,scale1,x) + (1-mix)* Gauss(loc2,scale2,x)

    
    
    


w_h = tf.Variable(tf.random_normal(shape=[feature_dim, 20], stddev=0.01), name='weights')
b_h = tf.Variable(tf.zeros([1, 20]), name="bias") 
w_h2 = tf.Variable(tf.random_normal(shape=[20, 20], stddev=0.01), name='weights2')
b_h2 = tf.Variable(tf.zeros([1, 20]), name="bias2")
w_h3 = tf.Variable(tf.random_normal(shape=[20, 1], stddev=0.01), name='weights3')
b_h3 = tf.Variable(tf.zeros([1, feature_dim]), name="bias3")

def trial_solution(X):
    H = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h)
    H2 = tf.nn.sigmoid(tf.matmul(H, w_h2) + b_h2)
    W0 = tf.nn.sigmoid(tf.matmul(H2, w_h3) + b_h3)
    return W0

n_u = 10
U = tf.placeholder(tf.float32, [n_u, 1], name='U_placeholder')
W=trial_solution(U)    

Grid_num=100

Comb=np.arange(1,Grid_num)[:,None]
np.random.shuffle(Comb)
Comb=Comb/Grid_num
Uniform = generator([Comb], n_u)


#### Variable Transformation
# Since the support of Gaussian is (-inf,inf), we transform it to the support (0,1) for the numerical boundary condition
def minus_inf_inf_to_zero_one(variable):
    return tf.log(variable/(1-variable))

def minus_inf_inf_to_zero_one_Log_det_J(variable):
    return tf.log(1/variable*(1-variable))


#Transform
params=minus_inf_inf_to_zero_one(W)
#Variable transoformation's Log_det_Jacobian
minus_Log_det_J=-tf.reduce_sum(minus_inf_inf_to_zero_one_Log_det_J(W),1)


minus_Log_p_w = -tf.log(Mixture_Gauss(0,0,1,1,W,0.3))

#Construct Partial differential equation

Posteri_numerator=minus_Log_p_w+minus_Log_det_J

def Jacobian(W,U):
    jacobian_list=[]
    for i in range(feature_dim):
        _W=tf.split(axis = 0, num_or_size_splits = n_u, value = W[:,i])
        jacobian_row=tf.gradients(_W[0],U)[0][0,:][None,:]
        jaco_=[tf.gradients(_W[index],U)[0][index,:][None,:] for index in range(1,n_u) ]
        for j in jaco_:
            jacobian_row=tf.concat((jacobian_row,j),0)#100*1を返す
        jacobian_list.append(jacobian_row[:,:,None])
    J=jacobian_list[0]    
    return J


#[None,1]用の要素微分
#tf_gradient_element_wiseは[100,1]を[100,3]で微分すると[100,3]を出力
def tf_gradient_element_wise(W,U):
    _W=tf.split(axis = 0, num_or_size_splits = n_u, value = W)
    jacobian_row=tf.gradients(_W[0],U)[0][0,:][None,:]
    jaco_=[tf.gradients(_W[index],U)[0][index,:][None,:] for index in range(1,n_u) ]
    for j in jaco_:
        jacobian_row=tf.concat((jacobian_row,j),0)
    return jacobian_row

Posteri_derivative=tf_gradient_element_wise(Posteri_numerator[:,None],U)       


#J=Jacobian(W,U)
J=Jacobian(W,U)
det_J=tf.matrix_determinant(J+tf.eye(feature_dim)*1e-10)[:,None]
Second_derivative=tf_gradient_element_wise(tf.log(tf.abs(det_J)),U)

#What w should be satisfied
G=Second_derivative-Posteri_derivative
#G=Posteri_derivative
loss = tf.reduce_sum(G*G)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())	
    #writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)    
    U_batches = 5#int(Batch_U/minibatch_U)
    for i in range(n_epochs):
        print(i)
        total_loss = 0
        # train the model n_epochs times
        for _ in range(U_batches):
    			     U_batch = next(Uniform)[0]
    			     _, loss_batch = sess.run([optimizer, loss], feed_dict={U:U_batch})
    			     total_loss += loss_batch			     
    			     #print(sess.run([tf.gradients(loss,tf.trainable_variables())], feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
    			     print(sess.run(tf.reduce_mean(tf.trainable_variables()[0]),feed_dict={U:U_batch}))
    			     print('Average loss epoch {0}: {1}'.format(i, total_loss))
    U_batch = next(Uniform)[0]
    print('Optimization Finished!') # should be around 0.35 after 25 epochs 
    #writer.close()
    #n_batches = 5#int(N_train/batch_size)
    
        
    		
		            
    