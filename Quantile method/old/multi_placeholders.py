# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:47:00 2018

@author: seiic
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 10:27:37 2018

@author: Futami
"""

import time
import numpy as np
import tensorflow as tf
from collections import OrderedDict
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
learning_rate = 0.01
batch_size = 50
minibatch_U=4
n_epochs = 60

data = generator([X_train, y_train], batch_size)
data2 = generator([X_test, y_test], batch_size)

Uniform = generator([Comb], minibatch_U)


def make_placeholders(n_ph,n_u):
    dic = OrderedDict()
    for i in range(n_ph):
        dic["U_" + str(i)] = tf.placeholder(tf.float32,[n_u,3],name = "U_placeholder"+ str(i))
    return dic

def make_placeholders_ones(n_ph,n_u):
    dic = OrderedDict()
    for i in range(n_ph*3):
        dic["U_" + str(i)] = tf.placeholder(tf.float32,[n_u,1],name = "U_placeholder"+ str(i))
    return dic

dictU = make_placeholders(10,minibatch_U)

def minus_inf_inf_to_zero_one(variable):
    return tf.log(variable/(1-variable))

def minus_inf_inf_to_zero_one_Log_det_J(variable):
    return tf.log(1/variable*(1-variable))
def tf_jacobian(tensor2, tensor1, feed_dict):
    """ Computes the tensor d(tensor2)/d(tensor1) recursively. """ 
    shape = list(sess.run(tf.shape(tensor2), feed_dict))
    if shape:
        return tf.stack([tf_jacobian(tf.squeeze(M, squeeze_dims = 0), tensor1, feed_dict) for M in tf.split(axis = 0, num_or_size_splits = shape[0], value = tensor2)]) 
    else:
        grad = tf.gradients(tensor2, tensor1)
        if grad[0] != None:
            return tf.squeeze(grad, squeeze_dims = [0])
        else:
            return tf.zeros_like(tensor1)
def Jacobian(W,U):
    jacobian_list=[]
    for i in range(3):
        _W=tf.split(axis = 0, num_or_size_splits = minibatch_U, value = W[:,i])
        jacobian_row=tf.gradients(_W[0],U)[0][0][:,None]
        jaco_=[tf.gradients(_W[index],U)[0][index][:,None] for index in range(1,minibatch_U) ]
        for j in jaco_:
            jacobian_row=tf.concat((jacobian_row,j),1)
        jacobian_list.append(tf.transpose(jacobian_row)[:,:,None])
    J=jacobian_list[0]
    for i in range(1,3):
        J=tf.concat((J,jacobian_list[i]),2)
    return J


w_h = tf.Variable(tf.random_normal(shape=[3, 20], stddev=0.01), name='weights')
b_h = tf.Variable(tf.zeros([1, 20]), name="bias") 
wf_h = tf.Variable(tf.random_normal(shape=[20, 3], stddev=0.01), name='fweights')
bf_h = tf.Variable(tf.zeros([1, 3]), name="fbias")

def trial_solution(X,param_list):
    H = tf.nn.sigmoid(tf.matmul(X, param_list[0]) + param_list[1])
    H2 = tf.nn.sigmoid(tf.matmul(H, param_list[2]) + param_list[3])
    return H2

def construct_loss(param_list,n_ph,n_u,X,Y,U,dic):
    
    loss = 0
    for  i in range(n_ph):
        U = dic["U_"+ str(i)]  
        W = trial_solution(U,param_list)
        params=minus_inf_inf_to_zero_one(W)
        #Variable transoformation's Log_det_Jacobian
        minus_Log_det_J=-tf.reduce_sum(minus_inf_inf_to_zero_one_Log_det_J(W),1)
        w = params[:,:2]
        b = params[:,2]

        model_output = tf.matmul(X, tf.transpose(w)) + b[None,:]
        #Construct Partial differential equation
        # 
        Minus_Log_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,n_u))),0)
        #Minus_Log_likelihood = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,None*にしたかったがエラーに・・・)))[:,0]
        Minus_Prior=-tf.reduce_sum(tf.multiply(params, params),1)
        
        Posteri_numerator=Minus_Log_likelihood+Minus_Prior+minus_Log_det_J
        Posteri_derivative=tf_gradient_element_wise(Posteri_numerator[:,None],U)       
    
        Det_J=tf.matrix_determinant(Jacobian(W,U))[:,None]
        Second_derivative=tf_gradient_element_wise(tf.log(tf.abs(Det_J)),U)
        
        #What w should be satisfied
        G=Second_derivative-Posteri_derivative
        #G=Posteri_derivative
        G=G
        loss += tf.reduce_sum(G*G)*1e6
    return loss




#### Variable Transformation
# Since the support of Gaussian is (-inf,inf), we transform it to the support (0,1) for the numerical boundary condition

#Transform
w1=minus_inf_inf_to_zero_one(_w1)
w2=minus_inf_inf_to_zero_one(_w2)
b=minus_inf_inf_to_zero_one(_b)

trans_params=[w1,w2,b]

#Variable transoformation's Log_det_Jacobian
minus_Log_det_J=0
for i in raw_params:
    minus_Log_det_J+=-tf.reduce_sum(minus_inf_inf_to_zero_one_Log_det_J(i),1)

X = tf.placeholder(tf.float32, [None, 2], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [None, 1], name='Y_placeholder')

_model_output = X[:,0]*w1+X[:,1]*w2 + b
model_output=tf.transpose(_model_output)
#Construct Partial differential equation
# 
Minus_Log_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,minibatch_U))),0)
#Minus_Log_likelihood = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,None*にしたかったがエラーに・・・)))[:,0]
Minus_Prior=0
for i in trans_params:
    Minus_Prior+=-i*i

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

##Set BC
_w1_B0=trial_solution(U_B,[w_h,b_h,wf_h,bf_h])    
_w2_B0=trial_solution(U_B2,[w_h2,b_h2,wf_h2,bf_h2])
_b_B0=trial_solution(U_B3,[w_h3,b_h3,wf_h3,bf_h3])

_w1_B1=trial_solution(U_B0,[w_h,b_h,wf_h,bf_h])    
_w2_B1=trial_solution(U_B20,[w_h2,b_h2,wf_h2,bf_h2])
_b_B1=trial_solution(U_B30,[w_h3,b_h3,wf_h3,bf_h3])

loss = L_0 + _w1_B0**2 + _w2_B0**2 + _b_B0**2+(_w1_B1-1.)**2 + (_w2_B1-1.)**2 + (_b_B1-1.)**2


#Posterior sampling


# Actual Prediction
A=tf.reduce_mean(tf.sigmoid(model_output),1)
prediction = tf.round(A)
predictions_correct = tf.cast(tf.equal(prediction, Y), tf.float32)
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
		for _ in range(20):
			X_batch, Y_batch = next(data)
			total_loss = 0
			for _ in range(U_batches):
			     U_batch = next(Uniform)[0]
			     _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_train, Y:y_train, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None],U_B:BC0_data,U_B2:BC0_data,U_B3:BC0_data,U_B0:BC1_data,U_B20:BC1_data,U_B30:BC1_data})
			     total_loss += loss_batch
			     print(sess.run([accuracy], feed_dict={X: X_test, Y:y_test, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None],U_B:BC0_data,U_B2:BC0_data,U_B3:BC0_data,U_B0:BC1_data,U_B20:BC1_data,U_B30:BC1_data}))
			     #print(sess.run([tf.gradients(loss,tf.trainable_variables())], feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			     print(sess.run([w1,w2,b], feed_dict={X: X_batch, Y:Y_batch, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None],U_B:BC0_data,U_B2:BC0_data,U_B3:BC0_data,U_B0:BC1_data,U_B20:BC1_data,U_B30:BC1_data}))
			print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	#print('Total time: {0} seconds'.format(time.time() - start_time))
	#U_batch = next(Uniform)[0]
	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	#Acc = sess.run([accuracy], feed_dict={X: X_test, Y:y_test, U:U_batch})
	#print('Accuracy {0}'.format(Acc))
	Z=sess.run(tf.sigmoid(model_output), feed_dict={X: Z_0,U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None],U_B:BC0_data,U_B2:BC0_data,U_B3:BC0_data,U_B0:BC1_data,U_B20:BC1_data,U_B30:BC1_data}) 

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