import numpy as np
import os
import scipy as sp
import Matrice
import MethodeNL
import Argument
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import getopt
from pylab import *
ion()

## EXECUTION PROBLEME NON LINEAIRE


#methode = Argument.mainE(sys.argv[1:])


## DONNEES

b_inf = -2
b_sup = 2
N = 100
eps = 1.0e-10
h = (b_sup-b_inf)/(N-1)
gamma = h*h

## CALCUL MATRICE

R = Matrice.CalculR(N,b_inf,b_sup)
M = Matrice.CalculM(N,b_inf,b_sup)
A = Matrice.CalculA(N,b_inf,b_sup)
NL = Matrice.CalculNL(N,b_inf,b_sup)

## Somme de Rigidite et x^2*Masse
Mat = R+A

## On veut resoudre Mat*x + NL*x^3 = l * M*x

vp = np.linalg.eig(Mat)
alpha = 2/(min(vp[0]) + max(vp[0]))

## Vecteur Initial
x0 = np.linspace(1,1,N)
x0 = x0.T

I = np.eye(N,N)
Pre = R+gamma*I

(x,i) = MethodeNL.GradO(Mat,NL,M,x0,eps)
(x2,i2) = MethodeNL.Grad(Mat,NL,M,x0,eps,alpha,P=Pre)
print(i)
print(i2)
X = np.linspace(b_inf,b_sup,N)
#plt.plot(X,x)
plt.plot(X,x2)
plt.show()
input("Press enter to continue")