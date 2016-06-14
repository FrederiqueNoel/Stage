import numpy as np
import os
import scipy as sp
from scipy import linalg
from math import sqrt

## GRADIENT A PAS FIXE

def gradient(A,B,x0,eps,alpha):
    x = x0/sqrt(np.dot(x0,np.dot(B,x0)))
    r = np.dot(A,x) - np.dot(x,np.dot(A,x))*np.dot(B,x)
    i=1

    while  sqrt(np.dot(r,np.dot(B,r)))> eps:
        print(sqrt(np.dot(r,np.dot(B,r))))
        x = x - alpha*r
        x = x/sqrt(np.dot(x,np.dot(B,x)))
        r = np.dot(A,x) - np.dot(x,np.dot(A,x))*np.dot(B,x)
        i = i+1

    x = x/sqrt(np.dot(x,np.dot(B,x)))
    return(x,i)

def gradientP(A,B,x0,eps,alpha,P):
    Pre = np.linalg.inv(P)
    x = x0/sqrt(np.dot(x0,np.dot(B,x0)))
    r = np.dot(A,x) - np.dot(x,np.dot(A,x))*np.dot(B,x)
    z = np.dot(Pre,r)
    i=1
    
    while  sqrt(np.dot(r,np.dot(B,r)))> eps:
        #print(sqrt(np.dot(r,np.dot(B,r))))
        x = x - alpha*z
        x = x/sqrt(np.dot(x,np.dot(B,x)))
        r = np.dot(A,x) - np.dot(x,np.dot(A,x))*np.dot(B,x)
        z = np.dot(Pre,r)
        i = i+1
    
    x = x/sqrt(np.dot(x,np.dot(B,x)))
    return(x,i)


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

def GCNL(A,B,x0,eps):
## INITIALISATION
    x = x0/sqrt(np.dot(x0,np.dot(B,x0)))
    w = np.dot(A,x)
    l = np.dot(w.T,x)
    g = w - l*np.dot(B,x)
    p = g
    z = np.dot(A,p)
    i = 1

## BOUCLE
    while sqrt(np.dot(g,np.dot(B,g))) > eps:
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

        beta = np.dot(gp,gp)/np.dot(g,g)
        p = gp + beta*p
        z = np.dot(A,p)

        x = xp
        g = gp
        i = i+1

    proj = sqrt(np.vdot(x,np.dot(B,x)))
    xf = xp/proj
    return (xf,i)

def GCNLP(A,B,x0,eps,P):
    ## INITIALISATION
    Pre = np.linalg.inv(P)
    x = x0/sqrt(np.dot(x0,np.dot(B,x0)))
    w = np.dot(A,x)
    l = np.dot(w.T,x)
    g = w - l*np.dot(B,x)
    p = np.dot(Pre,g)
    z = np.dot(A,p)
    i = 1
    
    ## BOUCLE
    while sqrt(np.dot(g,np.dot(B,g))) > eps:
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
        p = np.dot(Pre,gp) + beta*p
        z = np.dot(A,p)
        
        x = xp
        g = gp
        i = i+1
    
    proj = sqrt(np.vdot(x,np.dot(B,x)))
    xf = xp/proj
    return (xf,i)
