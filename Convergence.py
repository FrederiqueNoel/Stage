from __future__ import division

import numpy as np
import os
import scipy as sp
import Matrice
import MethodeL
import Argument
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
ion()


b_inf = -2
b_sup = 2
N = 100
eps = 1.0e-11

pre = Argument.mainC(sys.argv[1:])
print(pre)

nbi1 = np.zeros((4,1),dtype = int)
nbi2 = np.zeros((4,1),dtype = int)
nbi3 = np.zeros((4,1),dtype = int)
nbi4 = np.zeros((4,1),dtype = int)
h = np.zeros((4,1),dtype = float)
j = 0
for i in [10,50,100,200] :
    h[j] = (b_sup - b_inf)/(i-1)
    R = Matrice.CalculR(i,b_inf,b_sup)
    M = Matrice.CalculM(i,b_inf,b_sup)
    A = Matrice.CalculA(i,b_inf,b_sup)
    Mat = R+A
    vp = np.linalg.eig(Mat)
    alpha = 2/(min(vp[0]) + max(vp[0]))
    x0 = np.linspace(1,1,i)
    x0 = x0.T
    hi = (b_sup-b_inf)/(i-1)
    gamma = hi*hi
    I = np.eye(i,i)
    P = R+gamma*I
    
    print(" Nombre de points : " , i)
    if pre == '0' :
        (x1,nbi1[j]) = MethodeL.Grad(Mat,M,x0,eps,alpha)
        print("Gradient Pas Fixe : ",nbi1[j][0])
        (x2,nbi2[j]) = MethodeL.Grad(Mat,M,x0,eps)
        print("Gradient Pas Optimal : ",nbi2[j][0])
        (x3,nbi3[j]) = MethodeL.GCRR(Mat,M,x0,eps)
        print("Gradient Conjugue RR3 : ",nbi3[j][0])
        (x4,nbi4[j]) = MethodeL.GCNL(Mat,M,x0,eps)
        print("Gradient Conjugue Non Lineaire : ",nbi4[j][0])
    
    else :
        (x1,nbi1[j]) = MethodeL.Grad(Mat,M,x0,eps,alpha,P = P)
        print("Gradient Pas Fixe : ",nbi1[j][0])
        (x2,nbi2[j]) = MethodeL.Grad(Mat,M,x0,eps,P = P)
        print("Gradient Pas Optimal : ",nbi2[j][0])
        (x3,nbi3[j]) = MethodeL.GCRR(Mat,M,x0,eps,P = P)
        print("Gradient Conjugue RR3 : ",nbi3[j][0])
        (x4,nbi4[j]) = MethodeL.GCNL(Mat,M,x0,eps,P=P)
        print("Gradient Conjugue Non Lineaire : ",nbi4[j][0])

    j = j+1


plt.plot(h,nbi1,label='Gradient Pas Fixe')
plt.plot(h,nbi2, label= 'Gradient Pas Optimal')
plt.plot(h,nbi3, label = 'Gradient Conjugue')
plt.plot(h,nbi4, label = 'Gradient Conjugue non lineaire')
plt.yscale('log')
plt.xscale('log')
legend(loc='upper left')
plt.show()
input("Press enter to continue")

