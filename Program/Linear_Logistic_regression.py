"""
Created on Mon Jan  8 13:27:23 2018

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

# Define paramaters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 60

data = generator([X_train, y_train], batch_size)
data2 = generator([X_test, y_test], batch_size)

#X = tf.placeholder(tf.float32, [batch_size, 2], name='X_placeholder') 
#Y = tf.placeholder(tf.float32, [batch_size, 1], name='Y_placeholder')

X = tf.placeholder(tf.float32, [None, 2], name='X_placeholder') 
Y = tf.placeholder(tf.float32, [None, 1], name='Y_placeholder')

w = tf.Variable(tf.random_normal(shape=[2, 1], stddev=0.01), name='weights')
b = tf.Variable(tf.zeros([1, 1]), name="bias")

model_output = tf.matmul(X, w) + b 

# Loss function
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=Y))

# Actual Prediction
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, Y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

test_gradient=tf.gradients(loss,X)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs/logistic_reg', sess.graph)

	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(N_train/batch_size)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = next(data)
			_, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y:Y_batch}) 
			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs
	tot=0
	#for i in range(n_batches):
	#	X_batch, Y_batch = next(data2)
	#	accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch}) 
	#	tot += accuracy_batch[0]
	#Acc=tot/N_test
	Acc = sess.run([accuracy], feed_dict={X: X_test, Y:y_test})
	_w = sess.run([w], feed_dict={X: X_test, Y:y_test})
	_b = sess.run([b], feed_dict={X: X_test, Y:y_test})
	print('Accuracy {0}'.format(Acc))
	print('Accuracy {0}'.format(_w))
	print('Accuracy {0}'.format(_b))
	Z=sess.run(tf.sigmoid(model_output), feed_dict={X: Z_0})
	#G = sess.run([test_gradient], feed_dict={X: X_test, Y:y_test})
	#print('Gradient {0}'.format(G))
    
	writer.close()
    
    
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

Z=np.reshape(Z, (n_boundary, n_boundary))

plt.contour(X1, X2, Z, levels=[0.5],colors='k')
plt.scatter(plus[:,0],plus[:,1], c='red',s=10.5)
plt.scatter(minus[:,0],minus[:,1], c='blue',s=10.5)
plt.show()
'''