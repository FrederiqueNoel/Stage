import numpy as np
import os
import scipy as sp
import Matrice
import MethodeNL
import MethodeL
import Argument
from scipy import linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import getopt
from pylab import *
ion()

## EXECUTION PROBLEME NON LINEAIRE


methode = Argument.mainENL(sys.argv[1:])


## DONNEES

b_inf = -2
b_sup = 2
N = 100
eps = 1.0e-11
h = (b_sup-b_inf)/(N-1)
beta = 10
gamma = beta

## CALCUL MATRICE

R = Matrice.CalculR(N,b_inf,b_sup)
M = Matrice.CalculM(N,b_inf,b_sup)
A = Matrice.CalculA(N,b_inf,b_sup)
NL = Matrice.CalculNL(N,b_inf,b_sup)

RN = Matrice.CalculDFR(N,b_inf,b_sup)
## Somme de Rigidite et x^2*Masse
#Mat = R+A

Mat = Matrice.CalculDF(N,b_inf,b_sup)


## On veut resoudre Mat*x + NL*x^3 = l * M*x

vp = np.linalg.eig(Mat)

alpha1 = 2/(min(vp[0]) + max(vp[0]))
alpha = 1

X = np.linspace(b_inf,b_sup,N)

## Vecteur Initial
x0 = np.linspace(1,1,N)
x0 = x0.T

I = np.eye(N,N)
Pre = RN+gamma*I


if methode == '0':
    (x,i) = MethodeNL.Grad(Mat,2*beta*I,I,x0,eps,alpha)
    (x2,i2) = MethodeNL.Grad(Mat,2*beta*I,I,x0,eps,alpha,P=Pre)
    
    print("Nombre d'iterations gradient à pas fixe : ", i)
    print("Nombre d'iterations gradient à pas fixe avec preconditionneur: ",i2)
    plt.plot(X,x)
    plt.plot(X,x2)
    plt.show()
    input("Press enter to continue")

elif methode == '1':
    (x,i) = MethodeNL.GradO(Mat,2*beta*I,I,x0,eps)
    (x2,i2) = MethodeNL.GradO(Mat,2*beta*I,I,x0,eps,P=Pre)
    
    print("Nombre d'iterations gradient à pas optimal : ", i)
    print("Nombre d'iterations gradient à pas optimal avec preconditionneur: ",i2)
    plt.plot(X,x)
    plt.plot(X,x2)
    plt.show()
    input("Press enter to continue")

elif methode == '2':
    (x,i) = MethodeNL.GradC(Mat,2*beta*I,I,x0,eps)
    #(x2,i2) = MethodeNL.GradC(Mat,2*beta*I,I,x0,eps, P=Pre)
    
    print("Nombre d'iterations gradient conjugue : ", i)
    #print("Nombre d'iterations gradient conjugue avec preconditionneur: ",i2)
    plt.plot(X,x)
    #plt.plot(X,x2)
    plt.show()
    input("Press enter to continue")

else:
    print("L'argument entré n'était pas le numéro d'une méthode")
