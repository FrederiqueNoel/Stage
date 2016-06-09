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
N = 100
eps = 1.0e-10

## CALCUL MATRICE

R = Matrice.CalculR(N,b_inf,b_sup)
M = Matrice.CalculM(N,b_inf,b_sup)
A = Matrice.CalculA(N,b_inf,b_sup)

## Somme de Rigidite et x^2*Masse
Mat = R+A
Mat = R


## Resolution du systeme
(valp,vectp) = sp.linalg.eigh(Mat,M)
sol = vectp[:,0]

print(min(valp))

## Affichage solution
X = np.linspace(b_inf,b_sup,N)
plt.plot(X,sol)
plt.show()

input("Press enter to continue")

## Gradient
vp = np.linalg.eig(Mat)
print(cond(Mat))
alpha = 2/(min(vp[0]) + max(vp[0]))
x0 = np.linspace(1,1,N)
x0 = x0.T

(x,i) = Gradient.gradientConjugueNL(Mat,M,x0,eps)

plt.plot(X,x)
plt.show()
input("Press enter to continue")
