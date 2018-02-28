# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 14:05:42 2018

@author: Futami
"""


import time
import numpy as np
import tensorflow as tf
rng = np.random.RandomState(1234)
from data_load import generator,preprocessing2
from collections import OrderedDict

from sklearn import cross_validation

from utils import grid_samples_simple,grid_samples
from utils import make_placeholders_ones,make_placeholders_ones_test_data
from utils import make_trials
from utils import minus_inf_inf_to_zero_one,minus_inf_inf_to_zero_one_Log_det_J
from utils import feed_to_U_MF

############# import data set

import os
os.chdir('C:/Users/Futami/Google ドライブ/Research/beta-divergence/experiment/dataset')

ndarr3 = np.load('credit.npy')
X0=ndarr3[0]['training']
Y0=ndarr3[1]['target']
N,data_dim=X0.shape




####
# Define paramaters for the model
learning_rate = 0.01

'''
the minibatch size of data X and Y. This should be as large values as possible for numerical stability 
'''
batch_size = 1000
minibatch_U=1
n_epochs = 60

#hom many placeholders we use for the loss
Num=4

n_fold = 10 # 交差検定の回数
k_fold = cross_validation.KFold(n=len(X0),n_folds = n_fold,random_state=0)
for train_index, test_index in k_fold:
    X_train, X_test = X0[train_index,:], X0[test_index,:]
    X_train, X_test = preprocessing2(X_train,X_test)
    y_train, y_test = Y0[train_index][:,None], Y0[test_index][:,None]

    data = generator([X_train, y_train], batch_size)


################## Define your model
'''
Bayesian neural net, one hidden layer
'''

Num_of_hidden=20


target_shape=[[Num_of_hidden,data_dim],[Num_of_hidden,1],[1,Num_of_hidden],[1,1]]
target_shape=np.array(target_shape)
a=np.cumprod(target_shape,1)[:,-1]
b=np.cumsum(np.cumprod(target_shape,1)[:,-1])

D=np.cumsum(np.cumprod(target_shape,1)[:,-1])[-1]

def parameter_reshaper(W,b,data_dim,num_of_hidden):
    _W_1=tf.concat(W[0:b[0]],0)
    W1=tf.reshape(_W_1,(data_dim,num_of_hidden))

    _b_1=tf.concat(W[b[0]:b[1]],0)
    b1=tf.reshape(_b_1,(1,num_of_hidden))

    _W_2=tf.concat(W[b[1]:b[2]],0)
    W2=tf.reshape(_W_2,(num_of_hidden,1))

    _b_2=tf.concat(W[b[2]:],0)
    b2=tf.reshape(_b_2,(1,1))

    params=[W1,b1,W2,b2]
    return params

def B_NN_Logistic_regression_Model_prior(X,Y,params):
    '''
    This is the function which calculating the minus-log-likelihood and minus-log-prior of Bayesian NN for logistic-regression.
    
    X,Y : data, which is needed for calculating the likelihood
    params : the parameter of the model, which is already transformed.
    minibatch_U : number of minibatch of U. this should be set to 1.
    '''   

    def neural_network(X):
      h = tf.nn.tanh(tf.matmul(X, params[0]) + params[1])
      h = tf.matmul(h, params[2]) + params[3]
      return h
    #the output model_output is [None,1]

    model_output=neural_network(X)    
    Minus_Log_likelihood = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=Y),0)
    
    Minus_Prior=-tf.reduce_sum([tf.reduce_sum(p*p) for p in params])
               
    return Minus_Log_likelihood,Minus_Prior

def B_NN_Logistic_regression_Model_output(X,params):
    '''
    This is the function which calculating the output of logistic regression
    almost same as the function "Logistic_regression_Model_prior", but this is needed for prediction
    '''    
    def neural_network(X):
      h = tf.nn.tanh(tf.matmul(X, params[0]) + params[1])
      h = tf.matmul(h, params[2]) + params[3]
      return h
    #the output model_output is [None,1]
    model_output=neural_network(X)  
    return model_output
        


Use_simple=True

# how many parameters are in the model we use

# how many pieces we split [0,1]
Grid_num=100
# make grid samples

if Use_simple:
    Comb,Batch_U=grid_samples_simple(D,Grid_num)
else:
    Comb,Batch_U=grid_samples(D,Grid_num)

Uniform = generator([Comb], minibatch_U)

#how many samples we use for the predictions
minibatch_Ut=5
Uniform_t = generator([Comb], minibatch_Ut)

#prepare the placeholders
dictU = make_placeholders_ones(Num,D)
X = tf.placeholder(tf.float32, [None, data_dim], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [None, 1], name='Y_placeholder')

#this is the place holder for the prediction
_Ut=make_placeholders_ones_test_data(minibatch_Ut,D)



def construct_loss(X,Y,dic,Num,Model_prior,D):
    '''
    This is the function calculating the loss function.
    
    X,Y : data, they are place holders
    dic : this is the ordered dictionally which contains placeholders of U
    Num : how many placeholders are included in dic
    Model_prior : what kind of model we use
    D : how many parameters are in the model
    '''
    F=make_trials(D)
    loss = 0
    for  i in range(0,Num*D,D):
        
        U=[dic["U_"+ str(i+j)] for j in range (D)]
        _W=[f(u) for f,u in zip(F,U)]

        #variable transformation        
        W=[minus_inf_inf_to_zero_one(__w) for __w in _W]        
        raw_params=_W
        #Variable transoformation's Log_det_Jacobian
        minus_Log_det_J=0
        for i in raw_params:
            minus_Log_det_J+=-tf.reduce_sum(minus_inf_inf_to_zero_one_Log_det_J(i),1)
        #likelihood and prior calculation
        #if you want to consider other than logistic regression, change the model and prior
        params=parameter_reshaper(W,b,data_dim,Num_of_hidden)
        Minus_Log_likelihood,Minus_Prior=Model_prior(X,Y,params)

        Posteri_numerator=Minus_Log_likelihood[:,None]+Minus_Prior+minus_Log_det_J[:,None]

        dp_du=[tf.gradients(Posteri_numerator,u)[0] for u in U]      
        
        d2w_du2=[tf.gradients(tf.log(tf.abs(tf.gradients(_p,__u)[0])),__u)[0] for _p,__u in zip(raw_params,U)]
        
        _G=[(i-j)**2 for i,j in zip(d2w_du2,dp_du)]
        L_0=tf.reduce_sum(_G)
        
    loss +=L_0
    return loss,F

loss,F = construct_loss(X,Y,dictU,Num,B_NN_Logistic_regression_Model_prior,D)



# Actual Prediction for test data
Ut=[_Ut["Ut_"+ str(j)] for j in range (D)]
_w_t=[f(u) for f,u in zip(F,Ut)]

Wt=[minus_inf_inf_to_zero_one(__w) for __w in _w_t]
model_output_t=B_NN_Logistic_regression_Model_output(X,Wt)
                 
A=tf.reduce_mean(tf.sigmoid(model_output_t),1)
prediction = tf.round(A)
predictions_correct = tf.cast(tf.equal(prediction[:,None], Y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)


optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs/B_NN_logistic_reg', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = 50#int(N_train/batch_size)
	U_batches = 100#int(Batch_U/minibatch_U)
	for i in range(1): # train the model n_epochs times
		print(i)
		for _ in range(20):
			X_batch, Y_batch = next(data)
			total_loss = 0
			for _ in range(U_batches):
			     feed_U=OrderedDict()
			     feed_U[X]=X_batch
			     feed_U[Y]=Y_batch
			     #for i,j in zip(feed_XY.values(),[X_batch,Y_batch]):
			     #    feed_U[i] = j
			     feedings=feed_to_U_MF(feed_U,dictU,Uniform,D)
			     _, loss_batch = sess.run([optimizer, loss], feed_dict=feedings)
			     total_loss += loss_batch
			     feedings2=feed_to_U_MF(feed_U,_Ut,Uniform_t,D)
			     print(sess.run([accuracy], feed_dict=feedings2))
			     #print(sess.run([tf.gradients(loss,tf.trainable_variables())], feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			     #print(sess.run([w1,w2,b], feed_dict={X: X_batch, Y:Y_batch, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None]}))
			print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))
	#U_batch = next(Uniform)[0]
	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	feed_U=OrderedDict()
	feed_U[X]=X_test
	feed_U[Y]=y_test
	feedings3=feed_to_U_MF(feed_U,_Ut,Uniform_t,D)
	Acc = sess.run([accuracy], feed_dict=feedings3)
	print('Accuracy {0}'.format(Acc))
    
	writer.close()
    