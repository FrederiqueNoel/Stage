
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

## EXECUTION

## DONNEES

b_inf = -2
b_sup = 2
N = 300
eps = 1.0e-10

## CALCUL MATRICE

R = Matrice.CalculR(N,b_inf,b_sup)
M = Matrice.CalculM(N,b_inf,b_sup)
A = Matrice.CalculA(N,b_inf,b_sup)

## Somme de Rigidité et x^2*Masse
Mat = R+A


## Resolution du systeme
(valp,vectp) = sp.linalg.eigh(Mat,M)
sol = vectp[:,0]

## Affichage solution
X = np.linspace(b_inf,b_sup,N)
plt.plot(X,abs(sol))
#plt.show()

#input("Press enter to continue")

## Gradient
vp = np.linalg.eig(Mat)
val = sorted(vp[0])

print("Conditionnement de la matrice : ", cond(Mat))
alpha = 2/(min(vp[0]) + max(vp[0]))
print(alpha)
alpha2 = 2/(val[0]+val[N-1])
print(alpha2)
x0 = np.linspace(1,1,N)
x0 = x0.T

(x,i) = Gradient.gradient(Mat,M,x0,eps,alpha)
(x2,i2) = Gradient.gradient(Mat,M,x0,eps,alpha2)

plt.plot(X,x)
plt.show()

input("Press enter to continue")
