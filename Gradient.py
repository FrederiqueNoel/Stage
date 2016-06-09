import numpy as np
import os
import scipy as sp
from scipy import linalg
from math import sqrt

## GRADIENT A PAS FIXE
 
def gradient(A,B,x0,eps, alpha):
    x = x0
    x_aux = x - alpha*np.dot(A,x)
    i=1
    
    lb = np.vdot(x_aux,np.dot(A,x_aux))
    r = np.dot(A,x_aux) - lb*x_aux
    while np.linalg.norm(r,ord=2) > eps:
        x = x_aux
        x_aux = x - alpha*np.dot(A,x)
        lb = np.vdot(x_aux,np.dot(A,x_aux))
        r = np.dot(A,x_aux) - lb*x_aux
        i = i+1
    print(i)
    proj = sqrt(np.vdot(x_aux,np.dot(B,x_aux)))
    xp = x_aux/proj
    return (xp,i)


## GRADIENT A PAS OPTIMAL

def gradientOptimal(A,B,x0,eps):
    x = x0
    r = np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x)
    rn = r / sqrt(np.vdot(r, B.dot(r)))
    vp = CalculCoeff2(A,B,x,r)
    xp = vp[0]*x + vp[1]*r
    i=1
    while sqrt(np.dot(r,np.dot(B,r))) > eps:
        x = xp
        r = np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x)
        rn = r / sqrt(np.vdot(r, B.dot(r)))
        vp = CalculCoeff2(A,B,x,rn)
        xp = vp[0]*x + vp[1]*rn
        i = i+1
    print(i)

    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)

## RESOLUTION MATRICE 2x2
def CalculCoeff2(A,B,x,r):
    
    v = np.array([x,r])
    M1 = np.dot(np.dot(v,A),v.transpose())
    M2 = np.dot(np.dot(v,B),v.transpose())
    (valp,vectp) = sp.linalg.eigh(M1,M2)
    return vectp[:,0]


## GRADIENT CONJUGUE

def gradientConjugue(A,B,x0,eps):
    x = x0
    r = np.dot(A,x)
    vp = CalculCoeff2(A,B,x,r)
    xp = vp[0]*x + vp[1]*r
    i=1
    while sqrt(np.dot(r,np.dot(B,r))) > eps:
        p = (xp-x)/np.linalg.norm(xp-x,ord=2)
        p = p / sqrt(np.vdot(p, B.dot(p)))
        x = xp
        r = np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x)
        rn = 1/np.linalg.norm(r,ord=2)*r
        rn = rn / sqrt(np.vdot(rn, B.dot(rn)))
        vp = CalculCoeff3(A,B,x,rn,p)
        xp = vp[0]*x + vp[1]*rn + vp[2]*p
        i = i+1
    print(i)

    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)


## RESOLUTION MATRICE 3x3

def CalculCoeff3(A,B,x,r,p):
    v = np.array([x,r,p])
    M1 = np.dot(np.dot(v,A),v.transpose())
    M2 = np.dot(np.dot(v,B),v.transpose())
    
    (valp,vectp) = sp.linalg.eigh(M1,M2)
    
    return vectp[:,0]


## GRADIENT CONJUGUE NON LINEAIRE

def gradientConjugueNL(A,B,x0,eps):
    x = x0
    r = np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x)
    rn = r / sqrt(np.vdot(r, B.dot(r)))
    vp = CalculCoeff2(A,B,x,rn)
    print(vp[1])
    xp = vp[0]*x + vp[1]*rn
    s = rn
    i=1
    while i<100:
        #while sqrt(np.dot(r,np.dot(B,r))) > eps:
        #print(sqrt(np.dot(r,np.dot(B,r))))
        x = xp
        r = np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x)
        rnp = r / sqrt(np.vdot(r, B.dot(r)))
        beta = np.dot(rnp,rnp)/np.dot(rn,rn)
        #print(beta)
        rn = rnp
        sn = beta*s + rn
        vp = CalculCoeff2(A,B,x,sn)
        xp = vp[0]*x + vp[1]*sn
        print(vp[1])
        s=sn
        i = i+1
    #print(i)
    print(i)
    
    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)