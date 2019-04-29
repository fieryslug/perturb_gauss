import numpy as np
import mpmath as mp
from scipy import integrate
from matplotlib import pyplot as plt
import coeff

#set mpmath precision
mp.mp.dps = 50

#mpmath numerical integration
def mpintegr(l=0):
    I = mp.quad(lambda x : mp.exp(-(x**2 + l*x**4)), [-mp.inf, mp.inf])
    return I

#taylor polynomial of degree D
def taylor(l, D):
    s = 0
    for n in range(0, D+1):
        s += coeff.GAUSS_4_COEFF[n] * mp.power(l, n)
    return s

#taylor polynomial coefficient
def taylor_coeff(n):
    return mp.power(-1, n) * mp.gamma(2*n+1/2) / mp.factorial(n)

#calculate relative error
def relerr(l, D):
    I = mpintegr(l)
    return abs(taylor(l, D)-I) / I

#graph exact value vs approximation comparison
def graph(MAXD=16, MAXL=1):
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


def graph_mejor(MAXD=16, MAXL=1):
    

#generates relative errors for taylor polynomials of degree 1~(MAXD-1) at lambda=l
def errarray(l, MAXD=1000):
    print(l)
    s = 0
    ERR_ARRAY = []
    I = mpintegr(l)
    for i in range(MAXD):
        s = s + taylor_coeff(i) * mp.power(l, i)
        ERR_ARRAY.append(abs(s-I)/I)
    return ERR_ARRAY

graph(MAXD=100)
