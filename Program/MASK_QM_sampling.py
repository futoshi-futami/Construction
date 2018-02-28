# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:20:43 2018

@author: Futami
"""


import time
import numpy as np
import tensorflow as tf
rng = np.random.RandomState(1234)
from data_load import generator
from MADE import masked_autoregressive_default_template2


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
np.random.shuffle(data_training)
X_train=data_training[:,:-1]
y_train=(data_training[:,-1][:,None]+1)/2

N_test=2000
data_test=toy_data3(N_test)
X_test=data_test[:,:-1]
y_test=(data_test[:,-1][:,None]+1)/2
        
####
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
####Make Unit square

import itertools
Grid_num=50
list1=range(1,Grid_num)
Combination=list(itertools.product(list1, repeat=3))
Comb=np.array(Combination)
Comb=Comb/Grid_num
np.random.shuffle(Comb)
Batch_U=Comb.shape[0]

####


# Define paramaters for the model
learning_rate = 0.05
batch_size = 1000
minibatch_U=4
n_epochs = 60

data = generator([X_train, y_train], batch_size)
data2 = generator([X_test, y_test], batch_size)

Uniform = generator([Comb], minibatch_U)


##### Unit sphere
#U = tf.placeholder(tf.float32, [None, 3], name='U_placeholder')
U = tf.placeholder(tf.float32, [minibatch_U, 3], name='U_placeholder')
U_B = tf.placeholder(tf.float32, [minibatch_U, 3], name='U_B_placeholder')

##### Trial function 3-20-20-3

#trial_f=masked_autoregressive_default_template2([20,20],shift_only=True)

trial_f=masked_autoregressive_default_template2([20])
shift,log_scale=trial_f(U)
_W=U*tf.nn.sigmoid(log_scale)+shift*(1-tf.nn.sigmoid(log_scale))
#trial_f2=masked_autoregressive_default_template2([20])
shift2,log_scale2=trial_f(_W)
_W2=_W*tf.nn.sigmoid(log_scale2)+shift2*(1-tf.nn.sigmoid(log_scale2))

shift3,log_scale3=trial_f(_W2)
_W3=_W2*tf.nn.sigmoid(log_scale3)+shift2*(1-tf.nn.sigmoid(log_scale3))

W=tf.nn.sigmoid(_W3)

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

X = tf.placeholder(tf.float32, [None, 2], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [None, 1], name='Y_placeholder')

w = params[:,:2]
b = params[:,2]

model_output = tf.matmul(X, tf.transpose(w)) + b[None,:]

#Construct Partial differential equation
# 
Minus_Log_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,minibatch_U))),0)
#Minus_Log_likelihood = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,None*にしたかったがエラーに・・・)))[:,0]
Minus_Prior=-tf.reduce_sum(tf.multiply(params, params),1)

Posteri_numerator=Minus_Log_likelihood+Minus_Prior+minus_Log_det_J



#Trial function term
#Calculate Jacobian
#Jacobianという関数は[100,3]のテンソルを[100,3]のテンソルで微分すると、100の添え字については同じ添え字同士で合わせてくれて結果として[100,3,3]を出力
def Jacobian(W,U):
    jacobian_list=[]
    for i in range(3):
        _W=tf.split(axis = 0, num_or_size_splits = minibatch_U, value = W[:,i])
        jacobian_row=tf.gradients(_W[0],U)[0][0,:][None,:]
        jaco_=[tf.gradients(_W[index],U)[0][index,:][None,:] for index in range(1,minibatch_U) ]
        for j in jaco_:
            jacobian_row=tf.concat((jacobian_row,j),0)#100*1を返す
        jacobian_list.append(jacobian_row[:,:,None])
    J=jacobian_list[0]
    for i in range(1,3):
        J=tf.concat((J,jacobian_list[i]),2)
    return J

#[None,1]用の要素微分
#tf_gradient_element_wiseは[100,1]を[100,3]で微分すると[100,3]を出力
def tf_gradient_element_wise(W,U):
    _W=tf.split(axis = 0, num_or_size_splits = minibatch_U, value = W)
    jacobian_row=tf.gradients(_W[0],U)[0][0,:][None,:]
    jaco_=[tf.gradients(_W[index],U)[0][index,:][None,:] for index in range(1,minibatch_U) ]
    for j in jaco_:
        jacobian_row=tf.concat((jacobian_row,j),0)
    return jacobian_row

Posteri_derivative=tf_gradient_element_wise(Posteri_numerator[:,None],U)       

J_1=tf_gradient_element_wise(W[:,0][:,None],U)
J_2=tf_gradient_element_wise(W[:,1][:,None],U)
J_3=tf_gradient_element_wise(W[:,2][:,None],U)

det_J=J_1[:,0]*J_2[:,0]*J_3[:,0]
Second_derivative=tf_gradient_element_wise(tf.log(tf.abs(det_J[:,None])),U)

#What w should be satisfied
G=Second_derivative-Posteri_derivative
#G=Posteri_derivative
loss = tf.reduce_sum(G*G)


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
	U_batches = 10#int(Batch_U/minibatch_U)
	for i in range(100): # train the model n_epochs times
		print(i)
		for _ in range(n_batches):
			X_batch, Y_batch = next(data)
			total_loss = 0
			for _ in range(U_batches):
			     U_batch = next(Uniform)[0]
			     _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch, U:U_batch})
			     total_loss += loss_batch
			     print(sess.run([accuracy], feed_dict={X: X_test, Y:y_test, U:U_batch}))
			     #print(sess.run([tf.gradients(loss,tf.trainable_variables())], feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			     #print(sess.run(tf.reduce_mean(tf.trainable_variables()[0]), feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))
	U_batch = next(Uniform)[0]
	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	Acc = sess.run([accuracy], feed_dict={X: X_test, Y:y_test, U:U_batch})
	print('Accuracy {0}'.format(Acc))
	Z=sess.run(tf.sigmoid(model_output), feed_dict={X: Z_0,U:U_batch}) 
	#writer.close()
    
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