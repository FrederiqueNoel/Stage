import numpy as np
import os
import math
import scipy as sp
from scipy import linalg
from math import sqrt
from math import atan2

def Grad(A,B,C,x0,eps,alpha,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)
    
    x = x0
    r = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x))) - np.dot(x,np.dot(A,x))*np.dot(C,x)
    #l = np.dot(x,np.dot(A+B,x))
    #r = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x))) - l*np.dot(C,x)
    z = np.dot(Pre,r)
    i=1

    while sqrt(np.dot(r,np.dot(C,r))) > eps:
        x = x - alpha*z
        #l = np.dot(x,np.dot(A+B,x))
        #r = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x))) - l*np.dot(C,x)
        r = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x))) - np.dot(x,np.dot(A,x))*np.dot(C,x)
        z = np.dot(Pre,r)
        i = i+1
    
    x = x/sqrt(np.dot(x,np.dot(C,x)))
    return(x,i)


def GradO(A,B,C,x0,eps,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)
    
    x = np.asmatrix(x0).T
    x = x/sqrt(np.dot(x.T,np.dot(C,x)))
    H = A + np.dot(x, np.dot(B,x).T)
    l = np.dot(x.T,np.dot(H,x))[0,0]
    r = np.dot(H,x) - l*x
    
    gamma = 1/sqrt(np.dot(r.T,np.dot(C,r)))
    i = 1
    while i<5000:
        #while sqrt(np.dot(r.T,np.dot(C,r))) > eps:
        #print(sqrt(np.dot(r.T,np.dot(C,r))))
        #print(np.linalg.norm(r,ord=2))
        theta = 0.5*atan2(2*gamma*np.dot(x.T,np.dot(H,r)),l-gamma*gamma*np.dot(r.T,np.dot(H,r)))
        x = np.cos(theta)*x + np.sin(theta)*gamma*r
        x = x/sqrt(np.dot(x.T,np.dot(C,x)))
        H = A + np.dot(x, np.dot(B,x).T)
        l = np.dot(x.T,np.dot(H,x))[0,0]
        r = np.dot(H,x) - l*x
        gamma = 1/sqrt(np.dot(r.T,np.dot(C,r)))
        i=i+1

    x = x/sqrt(np.dot(x.T,np.dot(C,x)))
    return(x,i)

