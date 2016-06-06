from __future__ import division

import numpy as np
import os
import scipy as sp
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
ion()

## FONCTION CALCUL MATRICE

## Calcul Rigidite
def CalculR(N,a,b):
    h = (b-a)/(N-1)
    R1 = 2/h*np.eye(N,k=0,dtype=float)
    R2 = -1/h*np.eye(N,k=1,dtype=float)
    R3 = -1/h*np.eye(N,k=-1,dtype=float)
    return R1+R2+R3


## Calcul Masse
def CalculM(N,a,b):
    h = (b-a)/(N-1)
    M1 = 2*h/3*np.eye(N,k=0,dtype=float)
    M2 = h/6*np.eye(N,k=1,dtype=float)
    M3 = h/6*np.eye(N,k=-1,dtype=float)
    return M1+M2+M3


## Calcul xi*xj*Masse
def CalculA(N,a,b):
    h = (b-a)/(N-1)
    Mat = zeros((N,N),dtype=float)
    
    Mat[0,0] =a*a*2*h/3
    Mat[0,1] = a*(a+h)*h/6
    
    for i in range(1,N-1):
        Mat[i,i-1] = (a+i*h)*(a+(i-1)*h)*h/6
        Mat[i,i] = (a+i*h)**2*2*h/3
        Mat[i,i+1] = (a+i*h)*(a+(i+1)*h)*h/6
    
    Mat[N-1,N-2] =b*(b-h)*h/6
    Mat[N-1,N-1] =b*b*2*h/3
    
    return Mat


