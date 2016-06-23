import numpy as np
import os
import math
import scipy as sp
from scipy import linalg
from math import sqrt
from math import atan2

def BackTrack(A,B,C,x,d,r,alpha):
    c= 0.5
    tho = 0.5
    m = np.dot(d,r)
    t = -c*m
    f = sqrt(np.dot(r,np.dot(C,r)))
    xp = x + alpha*d
    l = np.dot(A,xp) + np.dot(B,np.dot(xp,np.dot(xp,xp)))
    rp = l - np.dot(xp.T,l)*np.dot(C,xp)
    fp = sqrt(np.dot(rp,np.dot(C,rp)))
    a = alpha
    
    while f - fp < 0:
        a = a*0.80
        xp = x + a*d
        xp = xp/sqrt(np.dot(xp,np.dot(C,xp)))
        l = np.dot(A,xp) + np.dot(B,np.dot(xp,np.dot(xp,xp)))
        rp = l - np.dot(xp.T,l)*np.dot(C,xp)
        fp = sqrt(np.dot(rp,np.dot(C,rp)))
    
    return (xp,rp,a)

def Res2(A,B,x,r,p):
    M = np.zeros((2,2),dtype=float)
    grad = 2*(np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x))))
    lapr = 2*np.dot(A,r) + 6*np.dot(B,np.dot(x,np.dot(x,r)))
    lapp = 2*np.dot(A,p) + 6*np.dot(B,np.dot(x,np.dot(x,p)))
    M[0,0] = -np.dot(grad,x) + np.dot(lapr,r)
    M[0,1] = np.dot(lapr,p)
    M[1,0] = np.dot(lapp,r)
    M[1,1] = -np.dot(grad,x) + np.dot(lapp,p)
    b = np.zeros((2,1),dtype=float)
    b[0] = -np.dot(grad,r)
    b[1] = -np.dot(grad,p)
    theta = np.linalg.solve(M,b)
    return theta


def Grad(A,B,C,x0,eps,alpha,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)

    x = x0/sqrt(np.dot(x0.T,np.dot(C,x0)))
    l = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x)))
    r = l - np.dot(x.T,l)*np.dot(C,x)
    rn = r/sqrt(np.dot(r,np.dot(C,r)))
    z = np.dot(Pre,rn)
    z = z/sqrt(np.dot(z,np.dot(C,z)))
    i=1
    a = alpha
    
    while sqrt(np.dot(r,np.dot(C,r))) > eps:
        #print(sqrt(np.dot(r,np.dot(C,r))))
        
        (xp,rp,a) = BackTrack(A,B,C,x,-z,r,a)
        rn = rp/sqrt(np.dot(rp,np.dot(C,rp)))
        z = np.dot(Pre,rn)
        z = z/sqrt(np.dot(z,np.dot(C,z)))
        x = xp
        r = rp
        i = i+1

    x = x/sqrt(np.dot(x,np.dot(C,x)))
    return(x,i)


def GradO(A,B,C,x0,eps,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)
    
    x = x0/sqrt(np.dot(x0.T,np.dot(C,x0)))
    l = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x)))
    r = l - np.dot(x.T,l)*np.dot(C,x)
    z = np.dot(Pre,r)
    d = z - np.dot(z,np.dot(C,x))*x
    gamma = 1/sqrt(np.dot(d.T,np.dot(C,d)))
    d = d*gamma
    i = 1

    while sqrt(np.dot(r.T,np.dot(C,r))) > eps:
        grad = 2*l
        lap = 2*np.dot(A,d) + 6*np.dot(B,np.dot(x,np.dot(x,d)))
        theta = -np.dot(grad,d)/(np.dot(lap,d)-np.dot(grad,d))
        x = np.cos(theta)*x + np.sin(theta)*d
        x = x/sqrt(np.dot(x.T,np.dot(C,x)))
        l = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x)))
        r = l - np.dot(x.T,l)*np.dot(C,x)
        z = np.dot(Pre,r)
        d = z - np.dot(z,np.dot(C,x))*x
        gamma = 1/sqrt(np.dot(d.T,np.dot(C,d)))
        d = d*gamma

        i=i+1

    x = x/sqrt(np.dot(x.T,np.dot(C,x)))
    return(x,i)

def GradOP(A,B,C,x0,eps,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)
    
    x = x0/sqrt(np.dot(x0.T,np.dot(C,x0)))
    l = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x)))
    r = l - np.dot(x.T,l)*np.dot(C,x)
    z = np.dot(Pre,r)
    d = z - np.dot(z,np.dot(C,x))*x
    gamma = 1/sqrt(np.dot(d.T,np.dot(C,d)))
    d = d*gamma
    i = 1
    
    while sqrt(np.dot(r.T,np.dot(C,r))) > eps:
        grad = 2*l
        lap = 2*np.dot(A,d) + 6*np.dot(B,np.dot(x,np.dot(x,d)))
        theta = -np.dot(grad,d)/(np.dot(lap,d)-np.dot(grad,d))
        x = np.cos(theta)*x + np.sin(theta)*d
        x = x/sqrt(np.dot(x.T,np.dot(C,x)))
        l = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x)))
        r = l - np.dot(x.T,l)*np.dot(C,x)
        z = np.dot(Pre,r)
        d = z - np.dot(z,np.dot(C,x))*x
        gamma = 1/sqrt(np.dot(d.T,np.dot(C,d)))
        d = d*gamma
        
        i=i+1
    
    x = x/sqrt(np.dot(x.T,np.dot(C,x)))
    return(x,i)


def GradC(A,B,C,x0,eps,P=None):
    if P is None:
        Pre = np.eye(x0.size)
    else:
        Pre = np.linalg.inv(P)
    
    x = x0/sqrt(np.dot(x0.T,np.dot(C,x0)))
    l = np.dot(A,x) + np.dot(B,np.dot(x,np.dot(x,x)))
    r = l - np.dot(x.T,l)*np.dot(C,x)
    m = np.dot(Pre,r)
    d = m - np.dot(m,np.dot(C,x))*x
    gamma = 1/sqrt(np.dot(d.T,np.dot(C,d)))
    d = d*gamma
    grad = 2*l
    lap = 2*np.dot(A,d) + 6*np.dot(B,np.dot(x,np.dot(x,d)))
    theta = -np.dot(grad,d)/(np.dot(lap,d)-np.dot(grad,d))
    xp = np.cos(theta)*x + np.sin(theta)*d
    xp = xp/sqrt(np.dot(xp,np.dot(C,xp)))
    
    i = 1
    
    while sqrt(np.dot(r.T,np.dot(C,r))) > eps:
        l = np.dot(A,xp) + np.dot(B,np.dot(xp,np.dot(xp,xp)))
        r = l - np.dot(xp.T,l)*np.dot(C,xp)
        m = np.dot(Pre,r)
        d = m - np.dot(m,np.dot(C,x))*x
        gamma = 1/sqrt(np.dot(d.T,np.dot(C,d)))
        d = d*gamma

        p = (xp-x)
        p = p/sqrt(np.dot(p,np.dot(C,p)))
        z = p - np.dot(p,np.dot(C,xp))*xp - np.dot(p,np.dot(C,d))*d
        z = z /sqrt(np.dot(z.T,np.dot(C,z)))
        x = xp
        theta = Res2(A,B,x,d,z)
        xp = (1-theta[0]**2/2-theta[1]**2/2)*x + theta[0]*d + theta[1]*z
        xp = xp/sqrt(np.dot(xp,np.dot(C,xp)))

        i = i+1

    x = xp/sqrt(np.dot(xp.T,np.dot(C,xp)))
    return(x,i)


