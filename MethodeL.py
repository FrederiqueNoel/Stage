import numpy as np
import os
import scipy as sp
from scipy import linalg
from math import sqrt

## RAYLEIGH RITZ

def RR(A,B,v):
    M1 = np.dot(v.T,np.dot(A,v))
    M2 = np.dot(v.T,np.dot(B,v))
    (valp,vectp) = sp.linalg.eigh(M1,M2)
    return vectp[:,0]

## GRADIENT

def Grad(A,B,x0,eps,alpha=None,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)

    x = x0/sqrt(np.dot(x0,np.dot(B,x0)))
    r = np.dot(A,x) - np.dot(x,np.dot(A,x))*np.dot(B,x)
    z = np.dot(Pre,r)
    i=1

    while sqrt(np.dot(r,np.dot(B,r))) > eps:
        if alpha is None :
            v = np.array([x,z]).T
            vp = RR(A,B,v)
            x = vp[0]*x + vp[1]*z
        else :
            x = x - alpha*z

        x = x/sqrt(np.dot(x,np.dot(B,x)))
        r = np.dot(A,x) - np.dot(x,np.dot(A,x))*np.dot(B,x)
        z = np.dot(Pre,r)
        i = i+1

    x = x/sqrt(np.dot(x,np.dot(B,x)))
    return(x,i)

## GRADIENT CONJUGUE


def GCRR(A,B,x0,eps,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)


    x = x0
    r = np.dot(A,x)-np.dot(x,np.dot(A,x))*np.dot(B,x)
    z = np.dot(Pre,r)
    v = np.array([x,z]).T
    vp = RR(A,B,v)
    xp = vp[0]*x + vp[1]*z
    i=1

    while sqrt(np.dot(r,np.dot(B,r))) > eps:
        p = (xp-x)/ np.linalg.norm(xp-x,ord=2)
        p = p/sqrt(np.dot(p,np.dot(B,p)))
        x = xp
        r = np.dot(A,x) - np.vdot(x, A.dot(x))*B.dot(x)
        z = np.dot(Pre,r)
        z = z/sqrt(np.vdot(z, B.dot(z)))
        v = np.array([x,z,p]).T
        vp = RR(A,B,v)
        xp = vp[0]*x + vp[1]*z + vp[2]*p
        i = i+1
    
    
    xp = xp/sqrt(np.dot(xp,np.dot(B,xp)))
    return (xp,i)

def GCNL(A,B,x0,eps,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)

    ## INITIALISATION
    res = []
    N = np.size(x0)
    k = N%500+100
    x = x0/sqrt(np.dot(x0,np.dot(B,x0)))
    w = np.dot(A,x)
    l = np.dot(w.T,x)
    g = w - l*np.dot(B,x)
    res.append(sqrt(np.dot(g,np.dot(B,g))))
    p = np.dot(Pre,g)
    z = np.dot(A,p)
    beta = 0
    i = 1
    
    ## BOUCLE
    while sqrt(np.dot(g,np.dot(B,g))) > eps:
        #print(sqrt(np.dot(g,np.dot(B,g))))
        a = np.dot(z.T,x)
        b = np.dot(z.T,p)
        c = np.dot(x.T,np.dot(B,p))
        d = np.dot(p.T,np.dot(B,p))
        
        delta = (l*d-b)*(l*d-b) - 4*(b*c-a*d)*(a-l*c)
        alpha = (l*d-b+sqrt(delta))/(2*(b*c-a*d))
        l = (l+a*alpha)/(1+c*alpha)
        gamma = sqrt(1+2*c*alpha+d*alpha*alpha)
        
        xp = (x+alpha*p)/gamma
        w = (w+alpha*z)/gamma
        gp = w-l*np.dot(B,xp)
        
        beta = np.dot(gp,np.dot(Pre,gp))/np.dot(g,np.dot(Pre,g))
        if i%k== 0:
            beta = 0
        p = np.dot(Pre,gp) + beta*p
        z = np.dot(A,p)
        x = xp
        g = gp
        i = i+1
        res.append(sqrt(np.dot(g,np.dot(B,g))))

    
    proj = sqrt(np.vdot(x,np.dot(B,x)))
    xf = xp/proj
    return (xf,i,res)
