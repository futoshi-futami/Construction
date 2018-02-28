# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 12:55:28 2018

@author: Futami
"""

import matplotlib.pyplot as plt

import matplotlib
#import matplotlib.gridspec as gridspec
'''
##以下の3つのコマンドは呪文として絶対やること。さもないとlatexにはっつけたときにｐｄｆ変換でフォントがtype3を出しやがるので
#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
'''

import numpy as np

def Gauss(loc,scale,x):
    return np.exp(-np.square(x-loc)/np.sqrt((2*np.square(scale))))/(np.sqrt((2*np.pi*np.square(scale))))


def minus_inf_inf_to_zero_one(variable):
    '''
    This is the function converting the support [0,1] to [-∞,∞] by inverse of sigmoid function
    '''
    return np.log(variable/(1.-variable))

def minus_inf_inf_to_zero_one_Log_det_J(variable):
    '''
    This is the function of calculating the jacobian of inverse of sigmoid function
    '''
    return np.log(1./(variable*(1.-variable)))


U=np.random.uniform(low=0.01,high=0.99,size=1000)
det=minus_inf_inf_to_zero_one_Log_det_J(U)


_z= minus_inf_inf_to_zero_one(U)

from scipy.stats import norm

h=norm.ppf(U)

#h=Gauss(0.,1.,_z)*np.abs(det)

#h=np.random.normal(loc=0.,scale=1.,size=1000)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)



ax.hist(h, bins=50)
ax.set_xlabel('x')
ax.set_ylabel('freq')
fig.show()