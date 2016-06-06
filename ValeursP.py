
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

## DONNEES
b_inf = -2
b_sup = 2
N = 100
eps = 1.0e-10

## CALCUL MATRICE

R = Matrice.CalculR(N,b_inf,b_sup)
M = Matrice.CalculM(N,b_inf,b_sup)
A = Matrice.CalculA(N,b_inf,b_sup)

## Somme de Rigidité et x^2*Masse
Mat = R+A


## Resolution du systeme
(valp,vectp) = sp.linalg.eigh(Mat,M)
print(valp)
abs = range(len(valp))

## Affichage valeurs propres
plt.bar(abs,valp,1.0, color='b')
plt.show()
input("Press enter to continue")
