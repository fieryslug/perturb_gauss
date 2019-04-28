import numpy as np
import mpmath as mp
from scipy import integrate
from matplotlib import pyplot as plt

#set mpmath precision
mp.mp.dps = 30

#mpmath numerical integration
def mpintegr(l=0):
    I = mp.quad(lambda x : mp.exp(-(x**2 + l*x**4)), [-mp.inf, mp.inf])
    return I

#taylor polynomial of degree D
def taylor(l, D):
    s = 0
    for n in range(0, D+1):
        s += taylor_coeff(n) * mp.power(l, n)
    return s

#taylor polynomial coefficient
def taylor_coeff(n):
    return mp.power(-1, n) * mp.gamma(2*n+1/2) / mp.factorial(n)

#calculate error of Dth taylor approximation at lamda=l
def err(l, D):
    print(l)
    return taylor(l, D) - mpintegr(l)

#calculate relative error
def relerr(l, D):
    return abs(err(l, D)) / mpintegr(l)

#graph exact value vs approximation comparison
def draw(MAXD=16, MAXL=1):
    #sample 1000 lambda values from 0 to MAXL
    L = mp.linspace(0, MAXL, 1000)

    #exact and approximation values are stored as arrays.
    #tests taylor polynomials of degree 0~(MAXD-1)
    exact = []
    P = [[] for D in range(MAXD)]

    for l in L:
        I = mpintegr(l)
        print(l)
        exact.append(I)
        for D in range(MAXD):
            P[D].append(taylor(l, D))

    #plotting
    plt.xlim(-0.01, 1.01)
    plt.ylim(1.2, 2.5)
    plt.plot(L, exact)
    for D in range(MAXD):
        plt.plot(L, P[D])
    plt.show()

draw(MAXD=20)
