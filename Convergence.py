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
norm4 = np.zeros((4,1),dtype = float)
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
    #(x1,nbi1[j]) = Gradient.gradient(Mat,M,x0,eps,alpha)
    (x1,nbi1[j]) = Gradient.gradientP(Mat,M,x0,eps,alpha,P)
    print("Gradient Pas Fixe : ",nbi1[j][0])
    #(x2,nbi2[j]) = Gradient.gradientOptimal(Mat,M,x0,eps)
    (x2,nbi2[j]) = Gradient.gradientOptimalP(Mat,M,x0,eps,P)
    print("Gradient Pas Optimal : ",nbi2[j][0])
    #(x3,nbi3[j]) = Gradient.gradientConjugue(Mat,M,x0,eps)
    (x3,nbi3[j]) = Gradient.gradientConjugueP(Mat,M,x0,eps,P)
    print("Gradient Conjugue RR3 : ",nbi3[j][0])
    #(x4,nbi4[j]) = Gradient.gradientConjugueNL(Mat,M,x0,eps)
    (x4,nbi4[j]) = Gradient.gradientConjugueNLP(Mat,M,x0,eps,P)
    print("Gradient Conjugue Non Lineaire : ",nbi4[j][0])
    (valp,vectp) = sp.linalg.eigh(Mat,M)
    sol = vectp[:,0]
    norm1[j] = np.dot(x1,x1)-np.dot(sol,sol)
    norm2[j] = np.dot(x2,x2)-np.dot(sol,sol)
    norm3[j] = np.dot(x3,x3)-np.dot(sol,sol)
    norm4[j] = np.dot(x4,x4)-np.dot(sol,sol)
    j = j+1

print(norm1)
print(norm2)
print(norm3)
plt.plot(h,nbi1,label='Gradient Pas Fixe')
plt.plot(h,nbi2, label= 'Gradient Pas Optimal')
plt.plot(h,nbi3, label = 'Gradient Conjugue')
plt.plot(h,nbi4, label = 'Gradient Conjugue non lineaire')
plt.yscale('log')
plt.xscale('log')
legend(loc='upper left')
plt.show()
input("Press enter to continue")

pente1 = np.zeros((3,1),dtype = float)
pente2 = np.zeros((3,1),dtype = float)
pente3 = np.zeros((3,1),dtype = float)
pente4 = np.zeros((3,1),dtype = float)
for i in range(3):
    pente1[i] = (log(nbi1[i+1]) - log(nbi1[i]))/(log(h[i+1]) - log(h[i]))
    pente2[i] = (log(nbi2[i+1]) - log(nbi2[i]))/(log(h[i+1]) - log(h[i]))
    pente3[i] = (log(nbi3[i+1]) - log(nbi3[i]))/(log(h[i+1]) - log(h[i]))
    pente4[i] = (log(nbi4[i+1]) - log(nbi4[i]))/(log(h[i+1]) - log(h[i]))

print(pente1)
print(pente2)
print(pente3)
print(pente4)
