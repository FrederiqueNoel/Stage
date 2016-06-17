from __future__ import division

import numpy as np
import os
import scipy as sp
from scipy import linalg

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
    Mat = CalculM(N,a,b)
    Mat[0,0] =a*a*Mat[0,0]
    Mat[0,1] = a*(a+h)*Mat[0,1]
    
    for i in range(1,N-1):
        Mat[i,i-1] = (a+i*h)*(a+(i-1)*h)*Mat[i,i-1]
        Mat[i,i] = (a+i*h)**2*Mat[i,i]
        Mat[i,i+1] = (a+i*h)*(a+(i+1)*h)*Mat[i,i+1]
    
    Mat[N-1,N-2] =b*(b-h)*Mat[N-1,N-2]
    Mat[N-1,N-1] =b*b*Mat[N-1,N-1]

    return Mat

## Calcul beta phi^4

def CalculNL(N,a,b):
    h = (b-a)/(N-1)
    M1 = (7/5*h)*np.eye(N,k=0,dtype=float)
    M1[0,0] = 11*h/10
    M1[N-1,N-1] = 11*h/10
    M2 = 3*h/10*np.eye(N,k=1,dtype=float)
    M3 = 3*h/10*np.eye(N,k=-1,dtype=float)

    return M1+M2+M3