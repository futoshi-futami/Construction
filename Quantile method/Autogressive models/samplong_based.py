# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:38:06 2018

@author: Futami
"""

import time
import numpy as np
import tensorflow as tf
rng = np.random.RandomState(1234)
from data_load import generator
#from MADE import masked_autoregressive_default_template,MADE_NN

from collections import OrderedDict

from utils import grid_samples_simple,grid_samples
from utils import make_placeholders_ones,make_placeholders_ones_test_data,make_placeholders
from utils import make_trials,make_MADE_trial,feed_to_U,MaskedAutoregressiveFlow
from utils import minus_inf_inf_to_zero_one,minus_inf_inf_to_zero_one_Log_det_J

tfd = tf.contrib.distributions
tfb = tfd.bijectors


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
####


Use_simple=True

# how many parameters are in the model we use
D=3
# how many pieces we split [0,1]
Grid_num=100

# make grid samples

if Use_simple:
    Comb,Batch_U=grid_samples_simple(D,Grid_num)
else:
    Comb,Batch_U=grid_samples(D,Grid_num)



# Define paramaters for the model
learning_rate = 0.05
batch_size = 1000
minibatch_U=1
minibatch_Ut=100
n_epochs = 60

data = generator([X_train, y_train], batch_size)
data2 = generator([X_test, y_test], batch_size)

Uniform = generator([Comb], minibatch_U)

Num=10


#prepare the placeholders
dictU = make_placeholders(Num,D)
X = tf.placeholder(tf.float32, [None, 2], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [None, 1], name='Y_placeholder')


#this is the place holder for the prediction
iaf = tfd.TransformedDistribution(
    distribution=tfd.Uniform(),
    bijector=tfb.Invert(MaskedAutoregressiveFlow(
        shift_and_log_scale_fn=masked_autoregressive_default_template(
            hidden_layers=[30]))),
    event_shape=[D])


################## Define your model

def Logistic_regression_Model_prior(X,Y,params,minibatch_U=1):
    '''
    This is the function which calculating the minus-log-likelihood and minus-log-prior of logistic-regression.
    
    X,Y : data, which is needed for calculating the likelihood
    params : the parameter of the model, which is already transformed.
    minibatch_U : number of minibatch of U. this should be set to 1.
    '''
    
    w = params[:,:2]
    b = params[:,2]

    model_output = tf.matmul(X, tf.transpose(w)) + b[None,:]
    Log_likelihood = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,minibatch_U))),0)
    #Minus_Log_likelihood = tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=tf.tile(Y,(1,None*にしたかったがエラーに・・・)))[:,0]
    Prior=-0.5*tf.reduce_sum(tf.multiply(params, params),1)     
    return Log_likelihood,Prior

def Logistic_regression_Model_output(X,params):
    '''
    This is the function which calculating the output of logistic regression
    almost same as the function "Logistic_regression_Model_prior", but this is needed for prediction
    '''    
    w = params[:,:2]
    b = params[:,2]

    model_output = tf.matmul(X, tf.transpose(w)) + b[None,:]   
    return model_output
        

def construct_loss(X,Y,F,Num,Model_prior,D):
    '''
    This is the function calculating the loss function.
    
    X,Y : data, they are place holders
    dic : this is the ordered dictionally which contains placeholders of U
    Num : how many placeholders are included in dic
    Model_prior : what kind of model we use
    D : how many parameters are in the model
    '''

    loss = 0
    for  i in range(0,Num):
        
        U=dic["U_"+ str(i)]

        _W,sigma_list=F(U)
        _W=tf.nn.sigmoid(_W)
        #variable transformation        
        W=minus_inf_inf_to_zero_one(_W)        
        
        #Variable transoformation's Log_det_Jacobian
        Log_det_J=tf.log(tf.abs(tf.reduce_sum(minus_inf_inf_to_zero_one_Log_det_J(_W),1)))
        #likelihood and prior calculation
        #if you want to consider other than logistic regression, change the model and prior
        Log_likelihood,Prior=Model_prior(X,Y,W)

        Posteri_numerator=Log_likelihood[:,None]+Prior+Log_det_J[:,None]

        dp_du=tf.gradients(Posteri_numerator,U)[0]
        
        
        log_det_J=tf.reduce_sum(tf.log(sigma_list))
        
        d2w_du2=tf.gradients(log_det_J,U)[0]
        
        L_0=tf.reduce_sum((d2w_du2+dp_du)**2)
        
        loss +=L_0
    return loss

F=MADE_NN(3,8)
#F=make_trials(3,MF=False)

loss = construct_loss(X,Y,F,dictU,Num,Logistic_regression_Model_prior,D)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)






Ut=tf.contrib.distributions.Uniform().sample(sample_shape=(300,D))
_Wt,_=F(Ut)

Wt=minus_inf_inf_to_zero_one(_Wt)
model_output_t=Logistic_regression_Model_output(X,Wt)
                 
A=tf.reduce_mean(tf.sigmoid(model_output_t),1)
prediction = tf.round(A)
predictions_correct = tf.cast(tf.equal(prediction[:,None], Y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

####sample
_Us=tf.contrib.distributions.Uniform().sample(sample_shape=(10000,D))
_Ws,_=F(_Us)
Ws=minus_inf_inf_to_zero_one(_Ws)


if True:    
	sess=tf.InteractiveSession()
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
			     feed_U=OrderedDict()
			     feed_U[X]=X_train
			     feed_U[Y]=y_train
			     #for i,j in zip(feed_XY.values(),[X_batch,Y_batch]):
			     #    feed_U[i] = j
			     feedings=feed_to_U(feed_U,dictU,Uniform,D)
			     _, loss_batch = sess.run([optimizer, loss], feed_dict=feedings)
			     total_loss += loss_batch
			     print(sess.run([accuracy], feed_dict={X: X_test, Y:y_test}))
			     #print(sess.run([tf.gradients(loss,tf.trainable_variables())], feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			     #print(sess.run(tf.reduce_mean(tf.trainable_variables()[0]), feed_dict={X: X_batch, Y:Y_batch, U:U_batch}))
			print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))
	
	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	Acc = sess.run([accuracy], feed_dict={X: X_test, Y:y_test})
	print('Accuracy {0}'.format(Acc))
	Z=sess.run(tf.sigmoid(model_output_t), feed_dict={X: Z_0}) 
	s=sess.run(Ws)
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