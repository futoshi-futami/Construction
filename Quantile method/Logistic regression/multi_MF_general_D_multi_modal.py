# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:17:55 2018

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
from utils import make_trials
from utils import minus_inf_inf_to_zero_one,minus_inf_inf_to_zero_one_Log_det_J
from utils import feed_to_U_MF,feed_to_U


####


Use_simple=True

# how many parameters are in the model we use
D=1
# how many pieces we split [0,1]
Grid_num=100

# make grid samples

if Use_simple:
    Comb,Batch_U=grid_samples_simple(D,Grid_num)
else:
    Comb,Batch_U=grid_samples(D,Grid_num)

# Define paramaters for the model
learning_rate = 0.01

'''
the minibatch size of data X and Y. This should be as large values as possible for numerical stability 
'''

minibatch_U=1
n_epochs = 60

#hom many placeholders we use for the loss
Num=90

Uniform = generator([Comb], minibatch_U)

#how many samples we use for the predictions
Ut=tf.contrib.distributions.Uniform(low=0.1,high=0.8).sample(sample_shape=(1000,D))

#prepare the placeholders
dictU = make_placeholders_ones(Num,D)

#this is the place holder for the prediction

################## Define your model

def Gauss(loc,scale,x):
    return tf.exp(-tf.square(x-loc)/np.sqrt((2*np.square(scale))))/(np.sqrt((2*np.pi*np.square(scale))))

def MoG(x):
    def _Mixture_Gauss(loc1,loc2,scale1,scale2,x,mix):
        return mix * Gauss(loc1,scale1,x) + (1-mix)* Gauss(loc2,scale2,x)
    return _Mixture_Gauss(0.,3.,1.,1.,x,1.)

        

def construct_loss(dic,Num,Model_prior,D):
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
            minus_Log_det_J+=tf.log(tf.abs(tf.reduce_sum(minus_inf_inf_to_zero_one_Log_det_J(i),1)))
        #likelihood and prior calculation
        #if you want to consider other than logistic regression, change the model and prior
        Log_joint=-0.5*(W[0]**2)

        Posteri_numerator=Log_joint+minus_Log_det_J[:,None]

        dp_du=[tf.gradients(Posteri_numerator,u)[0] for u in U]      
        
        d2w_du2=[tf.gradients(tf.log(tf.abs(tf.gradients(_p,__u)[0])),__u)[0] for _p,__u in zip(raw_params,U)]
        
        _G=[(i+j)**2 for i,j in zip(d2w_du2,dp_du)]
        L_0=tf.reduce_sum(_G)
        
    loss +=L_0
    return loss,F

U_B = tf.placeholder(tf.float32, [None, 1], name='U_B_placeholder')
U_B2 = tf.placeholder(tf.float32, [None, 1], name='U_B2_placeholder')
BC0_data=np.array([[0.]])
BC1_data=np.array([[1.]])

_loss,F = construct_loss(dictU,Num,MoG,D)

_w1_B0=F[0](U_B)
_w1_B1=F[0](U_B2)

loss=_loss + _w1_B0**2 + (_w1_B1-1.)**2#+tf.losses.get_regularization_loss()

_Z=F[0](Ut)
Z=minus_inf_inf_to_zero_one(_Z)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)




with tf.Session() as sess:
	#writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = 50#int(N_train/batch_size)
	U_batches = 50#int(Batch_U/minibatch_U)
	for i in range(1): # train the model n_epochs times
		print(i)
		for _ in range(500):
			total_loss = 0
			for _ in range(U_batches):
			     feed_U=OrderedDict()
			     feed_U[U_B]=BC0_data
			     feed_U[U_B2]=BC1_data
			     #for i,j in zip(feed_XY.values(),[X_batch,Y_batch]):
			     #    feed_U[i] = j
			     feedings=feed_to_U_MF(feed_U,dictU,Uniform,D)
			     
			     _, loss_batch = sess.run([optimizer, loss], feed_dict=feedings)
			     total_loss += loss_batch

			     #print(sess.run([tf.gradients(loss,tf.trainable_variables())], feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			     #print(sess.run([w1,w2,b], feed_dict={X: X_batch, Y:Y_batch, U1:U_batch[:,0][:,None],U2:U_batch[:,1][:,None],U3:U_batch[:,2][:,None]}))
			print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))
	#U_batch = next(Uniform)[0]
	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	h = sess.run([Z])
	#writer.close()
    
####
'''
import matplotlib.pyplot as plt

import matplotlib
#import matplotlib.gridspec as gridspec
'''
##以下の3つのコマンドは呪文として絶対やること。さもないとlatexにはっつけたときにｐｄｆ変換でフォントがtype3を出しやがるので
#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
'''


fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.hist(h[0], bins=50)
ax.set_xlabel('x')
ax.set_ylabel('freq')
fig.show()
'''
###