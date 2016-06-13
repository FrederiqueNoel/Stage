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
    proj = sqrt(np.vdot(x_aux,np.dot(B,x_aux)))
    xp = x_aux/proj
    return (xp,i)

def gradient2(A,B,x0,eps, alpha):
    print("DEBUT")
    x = x0
    r = np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x)
    print(r)
    xp = x - alpha*r
    print(sqrt(np.dot(r,np.dot(B,r))))
    i=1
    while i < 1000:
        x = xp
        r = np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x)
        xp = x - alpha*r
        rp = np.dot(A,xp) - np.vdot(xp, A.dot(xp))*B.dot(xp)
        print(sqrt(np.dot(r,np.dot(B,r))))
        i = i+1
    #print(i)

    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)

def gradientP(A,B,x0,eps, alpha,P):
    Pre = np.linalg.inv(P)
    x = x0
    x_aux = x - alpha*np.dot(Pre,np.dot(A,x))
    i=1
    lb = np.vdot(x_aux,np.dot(A,x_aux))
    r = np.dot(A,x_aux) - lb*x_aux
    while np.linalg.norm(r,ord=2) > eps:
        x = x_aux
        x_aux = x - alpha*np.dot(A,x)
        lb = np.vdot(x_aux,np.dot(A,x_aux))
        r = np.dot(A,x_aux) - lb*x_aux
        i = i+1
    #print(i)
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
    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)

def gradientOptimalP(A,B,x0,eps,P):
    Pre = np.linalg.inv(P)
    x = x0
    r = np.dot(Pre,np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x))
    rn = r / sqrt(np.vdot(r, B.dot(r)))
    vp = CalculCoeff2(A,B,x,r)
    xp = vp[0]*x + vp[1]*r
    i=1
    while sqrt(np.dot(r,np.dot(B,r))) > eps:
        x = xp
        r =  np.dot(Pre,np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x))
        rn = r / sqrt(np.vdot(r, B.dot(r)))
        vp = CalculCoeff2(A,B,x,rn)
        xp = vp[0]*x + vp[1]*rn
        i = i+1


    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)


## RESOLUTION MATRICE 2x2
def CalculCoeff2(A,B,x,r):
    v = np.array([x,r]).T
    M1 = np.dot(v.T,np.dot(A,v))
    M2 = np.dot(v.T,np.dot(B,v))
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
    #print(i)

    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)

def gradientConjugueP(A,B,x0,eps,P):
    Pre = np.linalg.inv(P)
    x = x0
    r = np.dot(Pre,np.dot(A,x))
    vp = CalculCoeff2(A,B,x,r)
    xp = vp[0]*x + vp[1]*r
    i=1
    while sqrt(np.dot(r,np.dot(B,r))) > eps:
        p = (xp-x)/np.linalg.norm(xp-x,ord=2)
        p = p / sqrt(np.vdot(p, B.dot(p)))
        x = xp
        r = np.dot(Pre,np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x))
        rn = 1/np.linalg.norm(r,ord=2)*r
        rn = rn / sqrt(np.vdot(rn, B.dot(rn)))
        vp = CalculCoeff3(A,B,x,rn,p)
        xp = vp[0]*x + vp[1]*rn + vp[2]*p
        i = i+1
    #print(i)

    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)

## RESOLUTION MATRICE 3x3

def CalculCoeff3(A,B,x,r,p):
    v = np.array([x,r,p]).T
    M1 = np.dot(v.T,np.dot(A,v))
    M2 = np.dot(v.T,np.dot(B,v))
    
    (valp,vectp) = sp.linalg.eigh(M1,M2)
    
    return vectp[:,0]


## GRADIENT CONJUGUE NON LINEAIRE

def gradientConjugueNL(A,B,x0,eps):
    x = x0
    r = -np.dot(A,x) + np.vdot(x, A.dot(x))*B.dot(x)
    rn = r / sqrt(np.vdot(r, B.dot(r)))
    vp = CalculCoeff2(A,B,x,rn)
    xp = vp[0]*x + vp[1]*rn
    sn = rn
    i=1
    #while i<100:
    while sqrt(np.dot(r,np.dot(B,r))) > eps:
        x = xp
        rp = - np.dot(A,x) + np.vdot(x, A.dot(x))*B.dot(x)
        rnp = rp / sqrt(np.vdot(rp, B.dot(rp)))
        beta = np.dot(rnp,rnp)/np.dot(rn,rn)
        rn = rnp
        r = rp
        sn = beta*sn + rn
        sn = sn/sqrt(np.vdot(sn, B.dot(sn)))
        vp = CalculCoeff2(A,B,x,sn)
        xp = vp[0]*x + vp[1]*sn
        i = i+1
    #print(i)
    
    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)

def gradientConjugueNLP(A,B,x0,eps,P):
    Pre = np.linalg.inv(P)
    x = x0
    r = np.dot(Pre,-np.dot(A,x) + np.vdot(x, A.dot(x))*B.dot(x))
    rn = r / sqrt(np.vdot(r, B.dot(r)))
    vp = CalculCoeff2(A,B,x,rn)
    xp = vp[0]*x + vp[1]*rn
    sn = rn
    i=1
    #while i<100:
    while sqrt(np.dot(r,np.dot(B,r))) > eps:
        x = xp
        rp = np.dot(Pre,- np.dot(A,x) + np.vdot(x, A.dot(x))*B.dot(x))
        rnp = rp / sqrt(np.vdot(rp, B.dot(rp)))
        beta = np.dot(rnp,rnp)/np.dot(rn,rn)
        rn = rnp
        r = rp
        sn = beta*sn + rn
        sn = sn/sqrt(np.vdot(sn, B.dot(sn)))
        vp = CalculCoeff2(A,B,x,sn)
        xp = vp[0]*x + vp[1]*sn
        i = i+1
    #print(i)
    
    proj = sqrt(np.vdot(xp,np.dot(B,xp)))
    xf = xp/proj
    return (xf,i)