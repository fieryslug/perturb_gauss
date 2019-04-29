import numpy as np
import mpmath as mp
from mpmath import mpf
from scipy import integrate
from matplotlib import pyplot as plt
import coeff
import exact_value

#set mpmath precision
mp.mp.dps = 50

#mpmath numerical integration
def mpintegr(l=0):
    print(l)
    I = mp.quad(lambda x : mp.exp(-(x**2 + l*x**4)), [-mp.inf, mp.inf])
    return I

#taylor polynomial of degree D
def taylor(l, D):
    s = mpf('0')
    for n in range(0, D+1):
        s += coeff.COEFF_0_1000_100[n] * mp.power(l, n)
    return s

#taylor polynomials up to degree D
def taylors(l, D):
    s = 0
    p = []
    for n in range(0, D+1):
        s += coeff.COEFF_0_1000_100[n] * mp.power(l, n)
        p.append(s)
    return np.asarray(p)

#taylor polynomials for l in L up to degree D
def taylors_l(L, D):
    taylors_v = np.vectorize(taylors)
    tp = taylors_v(L, D)
    a = []
    for pl in tp:
        for x in pl:
            a.append(x)
    a = np.asarray(a).reshape(len(L), D+1)
    return a


#taylor polynomial coefficient
def taylor_coeff(n):
    return mp.power(-1, n) * mp.gamma(2*n+1/2) / mp.factorial(n)



#calculate relative error
def relerr(l, D):
    I = mpintegr(l)
    return abs(taylor(l, D)-I) / I

#graphs exact value vs approximation comparison
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

#same as graph() but faster
def graph2(MAXD=16):
    taylors_v = np.vectorize(taylors)

    L = np.asarray(exact_value.LIN_0_1_1001_100)
    exact = np.asarray(exact_value.G_0_1_1001_100)

    tp = taylors_v(L, MAXD)
    a = []
    for pl in tp:
        for x in pl:
            a.append(x)
    a = np.asarray(a).reshape(len(L), MAXD+1)

    plt.xlim(-0.01, 1.01)
    plt.ylim(1.2, 2.5)
    plt.plot(L, exact)
    for i in range(MAXD+1):
        plt.plot(L, a[:, i])
    plt.show()

#graphs log of relative error; l = 0, 0.01, 0.02, ..., 1
def graph_error(MAXD=16):
    L = exact_value.LIN_0_1_1001_100
    a = taylors_l(L, MAXD)
    exact = np.asarray(exact_value.G_0_1_1001_100)

    b = np.asarray([[exact[i] for j in range(MAXD+1)] for i in range(1001)])
    
    log_v = np.vectorize(mp.log)
    plt.plot(L, log_v(abs((a-b)/b)))
    plt.show()


graph_error()
