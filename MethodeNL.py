import numpy as np
import os
import math
import scipy as sp
from scipy import linalg
from math import sqrt
from math import atan2

def Vnorme(A,x):
    return x/sqrt(np.dot(x,np.dot(A,x)))

def Norme(A,x):
    return sqrt(np.dot(x,np.dot(A,x)))

def Residu(A,B,C,x):
    l = np.dot(A,x) + np.dot(B,x**3)
    r = l - np.dot(x.T,l)*np.dot(C,x)
    return r

def Jacob(A,B,C,x,h):
    j = 2*Residu(A,B,C,x)
    return np.dot(j,h)

def Hessienne(A,B,C,x,h1,h2):
    hes = 2*np.dot(A,h1) + 6*np.dot(np.dot(B,x**2),h1)
    return np.dot(h2.T,hes)

def BackTrack(A,B,C,x,d,r,alpha):
    # Donnees
    c= 0.3
    tho = 0.7
    
    # Residu
    f = Norme(C,r)
    
    #Nouveau calcul
    xp = x + alpha*d
    xp = Vnorme(C,xp)
    rp = Residu(A,B,C,xp)
    fp = Norme(C,rp)
    a = alpha
    while fp > f+c*a*np.dot(r,d):
        a = a*tho
        xp = x + a*d
        xp = Vnorme(C,xp)
        rp = Residu(A,B,C,xp)
        fp = Norme(C,rp)
    
    return (xp,rp,a)

def Res1(A,B,C,x,r):
    den = Hessienne(A,B,C,x,r,r) - Jacob(A,B,C,x,r)
    if den < 0 :
        (xp,rp,a) = BackTrack(A,B,C,x,-r,Residu(A,B,C,x),1)
    else :
        theta = - Jacob(A,B,C,x,r)/den
        xp = np.cos(theta)*x + np.sin(theta)*r
        xp = Vnorme(C,xp)
        rp = Residu(A,B,C,xp)

    return(xp,rp)


def Res2(A,B,C,x,r,p):
    
    M = np.zeros((2,2),dtype=float)
    M[0,0] = Hessienne(A,B,C,x,r,r) - Jacob(A,B,C,x,x)
    M[0,1] = Hessienne(A,B,C,x,p,r)
    M[1,0] = Hessienne(A,B,C,x,r,p)
    M[1,1] = Hessienne(A,B,C,x,p,p) - Jacob(A,B,C,x,x)
    if min(np.linalg.eigh(M)[0]) < 0:
        (xp,rp) = Res1(A,B,C,x,r)
    else:
        b = np.zeros((2,1),dtype=float)
        b[0] = -Jacob(A,B,C,x,r)
        b[1] = -Jacob(A,B,C,x,p)
        theta = np.linalg.solve(M,b)
        xp = (1-(theta[0]**2)/2-(theta[1]**2)/2)*x + theta[0]*r + theta[1]*p
        xp = Vnorme(C,xp)
        rp = Residu(A,B,C,xp)

    return (xp,rp)

def Grad(A,B,C,x0,eps,alpha,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)
    
    x = Vnorme(C,x0)
    r = Residu(A,B,C,x)
    rn = Vnorme(C,r)

    z = np.dot(Pre,rn)
    z = Vnorme(C,z)

    i=1
    a = alpha
    
    while Norme(C,r) > eps:
        
        (xp,rp,a) = BackTrack(A,B,C,x,-z,r,a)
        
        rn = Vnorme(C,rp)
        z = np.dot(Pre,rn)
        
        z = Vnorme(C,z)
        
        x = xp
        r = rp
        i = i+1
    
    x = Vnorme(C,x)
    return(x,i)


def GradO(A,B,C,x0,eps,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)
    
    x = Vnorme(C,x0)
    r = Residu(A,B,C,x)

    z = np.dot(Pre,r)
    d = z - np.dot(z,x)*x
    d = Vnorme(C,d)
    
    i = 1

    while Norme(C,r) > eps:
        (x,r) = Res1(A,B,C,x,d)
        
        z = np.dot(Pre,r)
        d = z - np.dot(z,np.dot(C,x))*x
        d = Vnorme(C,d)

        i=i+1

    x = Vnorme(C,x)
    return(x,i)


def GradC(A,B,C,x0,eps,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)
    
    x = Vnorme(C,x0)
    r = Residu(A,B,C,x)
    m = np.dot(Pre,r)
    d = m - np.dot(m,np.dot(C,x))*x
    d = Vnorme(C,d)
    (xp,r) = Res1(A,B,C,x,d)

    i = 1

    while Norme(C,r) > eps:
        m = np.dot(Pre,r)
        d = m - np.dot(m,xp)*xp
        d = Vnorme(C,d)
        
        p = xp-x
        p = Vnorme(C,p)
        
        z = p - np.dot(p,xp)*xp - np.dot(p,d)*d
        z = Vnorme(C,z)
        
        x = xp
        
        (xp,r) = Res2(A,B,C,x,d,z)
        
        i = i+1
    
    x = Vnorme(C,xp)
    return(x,i)

