# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:28:04 2018

@author: Futami
"""


import time
import numpy as np
import tensorflow as tf
rng = np.random.RandomState(1234)
from data_load import generator


####Make Toy_data
N_train=3000
def toy_data3(Num=N_train,d=1):
    data_plus=np.random.multivariate_normal(3.*np.ones(d), 4.*np.eye(d), int(Num/2))
    data_minus=np.random.multivariate_normal(-1.*np.ones(d), 1.*np.eye(d), int(Num/2))
    for i in range(1):
        data_plus=np.hstack((data_plus,np.random.multivariate_normal(3.*np.ones(d), 4*np.eye(d), int(Num/2))))
        data_minus=np.hstack((data_minus,np.random.multivariate_normal(-1.*np.ones(d), 1*np.eye(d), int(Num/2))))
    
    data_plus=np.hstack((data_plus,np.ones(int(Num/2))[:,None]))
    data_minus=np.hstack((data_minus,-np.ones(int(Num/2))[:,None]))
    return np.vstack((data_plus,data_minus))
data_training=toy_data3()
X_train=data_training[:,:-1]
y_train=(data_training[:,-1][:,None]+1)/2

N_test=2000
data_test=toy_data3(N_test)
X_test=data_test[:,:-1]
y_test=(data_test[:,-1][:,None]+1)/2
n_boundary=100
x1_boundary = np.linspace(X_train[:,0].min(), X_train[:,0].max(), n_boundary)
x2_boundary = np.linspace(X_train[:,1].min(), X_train[:,1].max(), n_boundary)
X1, X2= np.meshgrid(x1_boundary, x2_boundary)

Z_0=[]
for i in x1_boundary:
    for j in x2_boundary:
        Z_0.append([i,j])
Z_0=np.array(Z_0)
####

####Make Unit square
#MFなのでばらけさせる
Grid_num=100

Comb=np.arange(1,Grid_num)[:,None]
np.random.shuffle(Comb)
for i in range(2):
    a=np.arange(1,Grid_num)[:,None]
    np.random.shuffle(a)
    Comb=np.concatenate((Comb,a),1)
Comb=Comb/Grid_num
Batch_U=Comb.shape[0]
####

###BC
BC0_data=np.array([[0.]])
BC1_data=np.array([[1.]])

# Define paramaters for the model
learning_rate = 0.1
batch_size = 50
minibatch_U=4
n_epochs = 60

data = generator([X_train, y_train], batch_size)
data2 = generator([X_test, y_test], batch_size)

Uniform = generator([Comb], minibatch_U)


##### Unit sphere
#U = tf.placeholder(tf.float32, [None, 3], name='U_placeholder')
U1 = tf.placeholder(tf.float32, [minibatch_U, 1], name='U1_placeholder')
U2 = tf.placeholder(tf.float32, [minibatch_U, 1], name='U2_placeholder')
U3 = tf.placeholder(tf.float32, [minibatch_U, 1], name='U3_placeholder')
U=[U1,U2,U3]

##### Trial function 3-20-20-3
w_h = tf.Variable(tf.random_normal(shape=[3, 30], stddev=0.01), name='weights')
b_h = tf.Variable(tf.zeros([1, 30]), name="bias") 
wf_h = tf.Variable(tf.random_normal(shape=[30, 30], stddev=0.01), name='fweights')
bf_h = tf.Variable(tf.zeros([1, 30]), name="fbias")
w2_h = tf.Variable(tf.random_normal(shape=[30, 3], stddev=0.01), name='fweights')
b2_h = tf.Variable(tf.zeros([1, 3]), name="fbias")


def trial_solution(X,param_list):
    _X=tf.concat((X[0],X[1],X[2]),1)
    H = tf.nn.sigmoid(tf.matmul(_X, param_list[0]) + param_list[1])
    H2 = tf.nn.sigmoid(tf.matmul(H, param_list[2]) + param_list[3])
    H3 = tf.nn.sigmoid(tf.matmul(H2, param_list[4]) + param_list[5])
    return H3

_W=trial_solution(U,[w_h,b_h,wf_h,bf_h,w2_h,b2_h])

raw_params=[_W[:,0][:,None],_W[:,1][:,None],_W[:,2][:,None]]

#### Variable Transformation
# Since the support of Gaussian is (-inf,inf), we transform it to the support (0,1) for the numerical boundary condition
def minus_inf_inf_to_zero_one(variable):
    return tf.log(variable/(1-variable))

def minus_inf_inf_to_zero_one_Log_det_J(variable):
    return tf.log(1/variable*(1-variable))
#Transform
W=minus_inf_inf_to_zero_one(_W)
w = W[:,:2]
b = W[:,2]
#Variable transoformation's Log_det_Jacobian
minus_Log_det_J=0
for i in raw_params:
    minus_Log_det_J+=-tf.reduce_sum(minus_inf_inf_to_zero_one_Log_det_J(i),1)

X = tf.placeholder(tf.float32, [None, 2], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [None, 1], name='Y_placeholder')

model_output = tf.matmul(X, tf.transpose(w)) + b[None,:]
#Construct Partial differential equation
# 
Minus_Log_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,minibatch_U))),0)
#Minus_Log_likelihood = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,None*にしたかったがエラーに・・・)))[:,0]
Minus_Prior=-tf.reduce_sum(tf.multiply(W, W),1)

Posteri_numerator=Minus_Log_likelihood[:,None]+Minus_Prior+minus_Log_det_J[:,None]

#Trial function term

#[None,1]用の要素微分
#tf_gradient_element_wiseは[100,1]を[100,1]で微分すると[100,1]を出力
def tf_gradient_element_wise(W,U):
    _W=tf.split(axis = 0, num_or_size_splits = minibatch_U, value = W)
    jacobian_row=tf.gradients(_W[0],U)[0][0][:,None]
    jaco_=[tf.gradients(_W[index],U)[0][index][:,None] for index in range(1,minibatch_U) ]
    for j in jaco_:
        jacobian_row=tf.concat((jacobian_row,j),0)
    return jacobian_row

def tf_jacobian(tensor2, tensor1, feed_dict):
    """ Computes the tensor d(tensor2)/d(tensor1) recursively. """
    shape = list(sess.run(tf.shape(tensor2), feed_dict))
    if shape:
        return tf.stack([tf_jacobian(tf.squeeze(M, squeeze_dims = 0), tensor1, feed_dict) 
                         for M in tf.split(axis = 0, num_or_size_splits = shape[0], value = tensor2)]) 
    else:
        grad = tf.gradients(tensor2, tensor1)
        if grad[0] != None:
            return tf.squeeze(grad, squeeze_dims = [0])
        else:
            return tf.zeros_like(tensor1)



dlnp_u1=tf_gradient_element_wise(Posteri_numerator,U1)     
dlnp_u2=tf_gradient_element_wise(Posteri_numerator,U2)
dlnp_u3=tf_gradient_element_wise(Posteri_numerator,U3)
dp_du=[dlnp_u1,dlnp_u2,dlnp_u3]

d2w_du2=[]
for i,j in zip(raw_params,U):
    dw_du=tf_gradient_element_wise(i,j)
    d2w_du2.append(tf_gradient_element_wise(tf.log(tf.abs(dw_du)),j))


#What w should be satisfied
_G=[]
for i,j in zip(d2w_du2,dp_du):
    _G.append((i-j)**2)
L_0=tf.reduce_sum(_G)  

loss = L_0 


#Posterior sampling


# Actual Prediction
A=tf.reduce_mean(tf.sigmoid(model_output),1)
prediction = tf.round(A)
predictions_correct = tf.cast(tf.equal(prediction[:,None], Y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	#writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = 50#int(N_train/batch_size)
	U_batches = 50#int(Batch_U/minibatch_U)
	for i in range(1): # train the model n_epochs times
		print(i)
		for _ in range(40):
			X_batch, Y_batch = next(data)
			total_loss = 0
			for _ in range(U_batches):
			     U_batch = next(Uniform)[0]
			     _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_train, Y:y_train, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None]})
			     total_loss += loss_batch
			     print(sess.run([accuracy], feed_dict={X: X_test, Y:y_test, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None]}))
			     #print(sess.run([tf.gradients(loss,tf.trainable_variables())], feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			     print(sess.run([W], feed_dict={X: X_batch, Y:Y_batch, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None]}))
			print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	#print('Total time: {0} seconds'.format(time.time() - start_time))
	#U_batch = next(Uniform)[0]
	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	#Acc = sess.run([accuracy], feed_dict={X: X_test, Y:y_test, U:U_batch})
	#print('Accuracy {0}'.format(Acc))
	Z=sess.run(tf.sigmoid(model_output), feed_dict={X: Z_0,U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None]}) 

	#writer.close()
    
####
'''
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.gridspec as gridspec

##以下の3つのコマンドは呪文として絶対やること。さもないとlatexにはっつけたときにｐｄｆ変換でフォントがtype3を出しやがるので
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


plus=[]
for i in data_training:
    if i[-1]==1:
       plus.append(i)
plus=np.array(plus)
plus=plus[:,:3]

minus=[]
for i in data_training:
    if i[-1]==-1:
       minus.append(i)
minus=np.array(minus)
minus=minus[:,:3]

Z=np.reshape(np.mean(Z,1), (n_boundary, n_boundary))

plt.contour(X1, X2, Z, levels=[0.5],colors='k')
plt.scatter(plus[:,0],plus[:,1], c='red',s=10.5)
plt.scatter(minus[:,0],minus[:,1], c='blue',s=10.5)
plt.show()
'''
###