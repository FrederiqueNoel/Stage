from __future__ import division

import numpy as np
import os
import scipy as sp
import Matrice
import Gradient
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
ion()


b_inf = -2
b_sup = 2
N = 10
eps = 1.0e-11

norm1 = np.zeros((4,1),dtype = float)
norm2 = np.zeros((4,1),dtype = float)
norm3 = np.zeros((4,1),dtype = float)
nbi1 = np.zeros((4,1),dtype = float)
nbi2 = np.zeros((4,1),dtype = float)
nbi3 = np.zeros((4,1),dtype = float)
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
    print(" Nombre de points : " , i)
    (x1,nbi1[j]) = Gradient.gradient(Mat,M,x0,eps,alpha)
    (x2,nbi2[j]) = Gradient.gradientOptimal(Mat,M,x0,eps)
    (x3,nbi3[j]) = Gradient.gradientConjugue(Mat,M,x0,eps)
    (valp,vectp) = sp.linalg.eigh(Mat,M)
    sol = abs(vectp[:,0])
    norm1[j] = np.linalg.norm(abs(x1)-sol,ord=2)
    norm2[j] = np.linalg.norm(abs(x2)-sol,ord=2)
    norm3[j] = np.linalg.norm(abs(x3)-sol,ord=2)
    j = j+1

print(norm1)
print(norm2)
print(norm3)
plt.plot(h,nbi1,label='Gradient Pas Fixe')
plt.plot(h,nbi2, label= 'Gradient Pas Optimal')
plt.plot(h,nbi3, label = 'Gradient Conjugue')
plt.yscale('log')
plt.xscale('log')
legend(loc='upper left')
plt.show()
input("Press enter to continue")

pente1 = np.zeros((3,1),dtype = float)
pente2 = np.zeros((3,1),dtype = float)
pente3 = np.zeros((3,1),dtype = float)
for i in range(3):
    pente1[i] = (log(nbi1[i+1]) - log(nbi1[i]))/(log(h[i+1]) - log(h[i]))
    pente2[i] = (log(nbi2[i+1]) - log(nbi2[i]))/(log(h[i+1]) - log(h[i]))
    pente3[i] = (log(nbi3[i+1]) - log(nbi3[i]))/(log(h[i+1]) - log(h[i]))

print(pente1)
print(pente2)
print(pente3)
