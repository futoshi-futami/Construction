# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 23:43:48 2018

@author: Futami
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:38:09 2018

@author: Futami
"""

import time
import numpy as np
import tensorflow as tf
rng = np.random.RandomState(1234)
from data_load import generator

from collections import OrderedDict

from utils import grid_samples_simple,grid_samples
from utils import make_placeholders_ones,make_placeholders_ones_test_data
from utils import make_trials_G
from utils import minus_inf_inf_to_zero_one,minus_inf_inf_to_zero_one_Log_det_J
from utils import feed_to_U_MF,_loop

#############Make Toy_data
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


Use_simple=True

# how many parameters are in the model we use
D=3

# Define paramaters for the model
learning_rate = 0.01

'''
the minibatch size of data X and Y. This should be as large values as possible for numerical stability 
'''
batch_size = 1000
minibatch_U=1
n_epochs = 60

#hom many placeholders we use for the loss
Num=50

data = generator([X_train, y_train], batch_size)
data2 = generator([X_test, y_test], batch_size)

#how many samples we use for the predictions


#prepare the placeholders
X = tf.placeholder(tf.float32, [None, 2], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [None, 1], name='Y_placeholder')

#this is the place holder for the prediction

################## Define your model

def Logistic_regression_Model_prior(X,Y,params,minibatch_U=1):
    '''
    This is the function which calculating the minus-log-likelihood and minus-log-prior of logistic-regression.
    
    X,Y : data, which is needed for calculating the likelihood
    params : the parameter of the model, which is already transformed.
    minibatch_U : number of minibatch of U. this should be set to 1.
    '''    
    _model_output = X[:,0]*params[0]+X[:,1]*params[1] + params[2]
    model_output=tf.transpose(_model_output)
    Log_likelihood = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,minibatch_U))),0)
    Prior=0
    for i in params:
        Prior +=-0.5*i*i       
    return Log_likelihood,Prior,model_output

def Logistic_regression_Model_output(X,params):
    '''
    This is the function which calculating the output of logistic regression
    almost same as the function "Logistic_regression_Model_prior", but this is needed for prediction
    '''    
    _model_output = X[:,0]*params[0]+X[:,1]*params[1] + params[2]
    model_output=tf.transpose(_model_output)       
    return model_output
        
def _each_loss(X,Y,Num,Model_prior,D,F,U):
    #U=[tf.contrib.distributions.Uniform().sample(sample_shape=(1,1)) for j in range (D)]
    W=[f(u) for f,u in zip(F,U)]

    #variable transformation        
    #Variable transoformation's Log_det_Jacobian
    #likelihood and prior calculation
    #if you want to consider other than logistic regression, change the model and prior
    Log_likelihood,Prior,_=Model_prior(X,Y,W)

    Posteri_numerator=Log_likelihood[:,None]+Prior

    dp_du=[tf.gradients(Posteri_numerator,u)[0] for u in U]      
    
    d2w_du2=[tf.gradients(tf.log(tf.abs(tf.gradients(_p,__u)[0])),__u)[0] for _p,__u in zip(W,U)]
    
    _G=[(i+j)**2 for i,j in zip(d2w_du2,dp_du)]
    L_0=tf.reduce_sum(_G)
    return L_0

F=make_trials_G(D)

def each_loss(U):
    f=_each_loss(X,Y,Num,Logistic_regression_Model_prior,D,F,U)
    return f

loss = _loop(each_loss,10,3)
'''
loss=0
for _ in range(20):
    _U=tf.contrib.distributions.Uniform().sample(sample_shape=(1,D))
    U=tf.split(axis = 1, num_or_size_splits = D, value = _U)
    loss+=each_loss(U)
'''


# Actual Prediction for test data
_Ut=tf.contrib.distributions.Uniform().sample(sample_shape=(300,D))
Ut=tf.split(axis = 1, num_or_size_splits = D, value = _Ut)

Wt=[f(u) for f,u in zip(F,Ut)]

model_output_t=Logistic_regression_Model_output(X,Wt)
                 
A=tf.reduce_mean(tf.sigmoid(model_output_t),1)
prediction = tf.round(A)
predictions_correct = tf.cast(tf.equal(prediction[:,None], Y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


####sample
_Us=tf.contrib.distributions.Uniform().sample(sample_shape=(10000,D))
Us=tf.split(axis = 1, num_or_size_splits = D, value = _Us)

Ws=[f(u) for f,u in zip(F,Us)]




if True:    
	sess=tf.InteractiveSession()
	#writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = 50#int(N_train/batch_size)
	U_batches = 5#int(Batch_U/minibatch_U)
	for i in range(1000): # train the model n_epochs times
		print(i)
		for _ in range(1):
			X_batch, Y_batch = next(data)
			total_loss = 0
			for _ in range(U_batches):
			     _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_train, Y:y_train})
			     total_loss += loss_batch
			     
			     print(sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}))
			     #print(sess.run([tf.gradients(loss,tf.trainable_variables())], feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			     #print(sess.run([w1,w2,b], feed_dict={X: X_batch, Y:Y_batch, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None]}))
			print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))
	#U_batch = next(Uniform)[0]
	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	Acc = sess.run([accuracy], feed_dict={X: X_test, Y:y_test})
	print('Accuracy {0}'.format(Acc)) 
	Z=sess.run(tf.sigmoid(model_output_t), feed_dict={X: Z_0}) 
	s=sess.run(Ws)
	#writer.close()
    
####
'''
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.gridspec as gridspec
import seaborn

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