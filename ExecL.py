import numpy as np
import os
import scipy as sp
import Matrice
import MethodeL
import Argument
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import getopt
from pylab import *
ion()

## EXECUTION PROBLEME LINEAIRE


methode = Argument.mainE(sys.argv[1:])


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

## Somme de Rigidite et x^2*Masse
Mat = R+A

## Resolution du systeme
(valp,vectp) = sp.linalg.eigh(Mat,M)
sol = vectp[:,0]

## Affichage solution
X = np.linspace(b_inf,b_sup,N)
plt.plot(X,sol)
plt.show()

#input("Press enter to continue")

## Alpha
vp = np.linalg.eig(Mat)
alpha = 2/(min(vp[0]) + max(vp[0]))

## Vecteur Initial
x0 = np.linspace(1,1,N)
x0 = x0.T

## Preconditionner
I = np.eye(N,N)
Pre = R+gamma*I

if methode == '0':
    (x,i) = MethodeL.Grad(Mat,M,x0,eps,alpha)
    (x2,i2) = MethodeL.Grad(Mat,M,x0,eps,alpha,P=Pre)

    print("Nombre d'iterations gradient à pas fixe : ", i)
    print("Nombre d'iterations gradient à pas fixe avec preconditionneur: ",i2)
    plt.plot(X,x)
    plt.plot(X,x2)
    plt.show()
    input("Press enter to continue")

elif methode == '1':
    (x,i) = MethodeL.Grad(Mat,M,x0,eps)
    (x2,i2) = MethodeL.Grad(Mat,M,x0,eps,P=Pre)
    
    print("Nombre d'iterations gradient à pas optimal : ", i)
    print("Nombre d'iterations gradient à pas optimal avec preconditionneur: ",i2)
    plt.plot(X,x)
    plt.plot(X,x2)
    plt.show()
    input("Press enter to continue")

elif methode == '2':
    (x,i) = MethodeL.GCRR(Mat,M,x0,eps)
    (x2,i2) = MethodeL.GCRR(Mat,M,x0,eps, P=Pre)
    
    print("Nombre d'iterations gradient conjugue RR3 : ", i)
    print("Nombre d'iterations gradient conjugue RR3 avec preconditionneur: ",i2)
    plt.plot(X,x)
    plt.plot(X,x2)
    plt.show()
    input("Press enter to continue")

elif methode == '3':
    (x,i) = MethodeL.GCNL(Mat,M,x0,eps)
    (x2,i2) = MethodeL.GCNL(Mat,M,x0,eps, P=Pre)
    
    print("Nombre d'iterations gradient conjugue non lineaire : ", i)
    print("Nombre d'iterations gradient conjugue non lineaire avec preconditionneur: ",i2)
    plt.plot(X,x)
    plt.plot(X,x2)
    plt.show()
    input("Press enter to continue")

else:
    print("L'argument entré n'était pas le numéro d'une méthode")


