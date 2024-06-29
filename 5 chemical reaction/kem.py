import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from io import StringIO
from scipy.optimize import linprog
from decimal import Decimal
from numpy import linalg as LA
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
import cdd as pcdd
import time
from scipy.optimize import fsolve
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from scipy.integrate import solve_ivp
from itertools import cycle
import matplotlib.cm as cm
from ddeint import ddeint
from scipy.signal import find_peaks
import matplotlib.colors as colors
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
import matplotlib as mpl
from scipy.optimize import curve_fit





############## 1.NALOGA #################
t_konc = 50
t = np.linspace(0, t_konc, 4000)
def eksaktno():
    def sistem_enacb(t,y, q, z):
        A, A_zv,B=y
        return([-A**2 + q*A*A_zv, A**2 - q*A*A_zv - q*z*A_zv, z*q*A_zv])

    Q = 1000
    z_ar = np.array([0.1,1,10])
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    ax2.ticklabel_format(axis='both', scilimits=(-3,3))

    sol = solve_ivp(sistem_enacb, (0,t_konc), [0.9,0.1,0],t_eval=t, args=(Q,z_ar[0]))
    line, = ax1.plot(t,sol.y[0], label='A, z=0.1')
    ax1.plot(t,sol.y[2],color = line.get_color(),linestyle='--', label='B=C')
    line3, = ax2.plot(t[:50],sol.y[1][0:50], label=r'$A^{*}, z=0.1$')


    sol = solve_ivp(sistem_enacb, (0,t_konc), [0.9,0.1,0],t_eval=t, args=(Q,z_ar[1]))
    line,  =ax1.plot(t,sol.y[0], label='A, z=1')
    ax1.plot(t,sol.y[2], label='B=C', color=line.get_color(), linestyle='--')
    ax2.plot(t[:50],sol.y[1][0:50], label=r'$A^{*}, z=1$')

    sol = solve_ivp(sistem_enacb, (0,t_konc), [0.9,0.1,0],t_eval=t, args=(Q,z_ar[2]))
    line, = ax1.plot(t,sol.y[0], label='A, z=10')
    ax1.plot(t,sol.y[2], label='B=C', color=line.get_color(), linestyle='--')
    ax2.plot(t[:50],sol.y[1][0:50], label=r'$A^*, z=10$')


    ax1.legend()
    ax2.legend()
    fig.suptitle(r'$A(0) = 9, A^*(0) = 0.1, B(0)=C(0) = 0, Q = 1000 $')
    ax1.set_xlabel('t')
    ax1.set_ylabel('c')
    ax2.set_xlabel('t')
    ax2.set_ylabel('c')
    plt.show()
    return None

def stacionarno():
    def sistem_enacb(t,y, q, z):
        A,B=y
        return([-A**2*z/(A+z),A**2*z/(A+z) ])

    Q = 0.1
    z_ar = np.array([0.1,1,10])
    fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)
    ax2.ticklabel_format(axis='both', scilimits=(-3,3))

    sol = solve_ivp(sistem_enacb, (0,t_konc), [1,0],t_eval=t, args=(Q,z_ar[0]))
    A_zv = (1/Q)*(sol.y[0]**2/(sol.y[0] + z_ar[0]))
    line, = ax1.plot(t,sol.y[0], label='A, z=0.1')
    ax1.plot(t,sol.y[1],color = line.get_color(),linestyle='--', label='B=C')
    line3, = ax2.plot(t,A_zv, label=r'$A^{*}, z=0.1$')


    sol = solve_ivp(sistem_enacb, (0,t_konc), [1,0],t_eval=t, args=(Q,z_ar[1]))
    A_zv = (1/Q)*(sol.y[0]**2/(sol.y[0] + z_ar[1]))
    line,  =ax1.plot(t,sol.y[0], label='A, z=1')
    ax1.plot(t,sol.y[1], label='B=C', color=line.get_color(), linestyle='--')
    ax2.plot(t,A_zv, label=r'$A^{*}, z=1$')

    sol = solve_ivp(sistem_enacb, (0,t_konc), [1,0],t_eval=t, args=(Q,z_ar[2]))
    A_zv = (1/Q)*(sol.y[0]**2/(sol.y[0] + z_ar[2]))
    line, = ax1.plot(t,sol.y[0], label='A, z=10')
    ax1.plot(t,sol.y[1], label='B=C', color=line.get_color(), linestyle='--')
    ax2.plot(t,A_zv, label=r'$A^*, z=10$')


    ax1.legend()
    ax2.legend()
    fig.suptitle(r'$A(0) = 1, B(0)=C(0) = 0, Q = 1 $')
    ax1.set_xlabel('t')
    ax1.set_ylabel('c')
    ax2.set_xlabel('t')
    ax2.set_ylabel('c')

    plt.show()
    return None
#stacionarno()

def razlika_B():
    z = np.linspace(0.1, 10, 10)
    q = np.linspace(1, 1000, 10)
    res = np.zeros((len(z), len(q)))

    def sistem_enacb_stac(t,y, q, z):
        A,B=y
        return([-A**2*z/(A+z),A**2*z/(A+z) ])
    def sistem_enacb(t,y, q, z):
        A, A_zv,B=y
        return([-A**2 + q*A*A_zv, A**2 - q*A*A_zv - q*z*A_zv, z*q*A_zv])

    for i in range(len(z)):
        for j in range(len(q)):
            sol_stac = solve_ivp(sistem_enacb_stac, (0,t_konc), [1,0],t_eval=t, args=(q[j],z[i]))
            sol = solve_ivp(sistem_enacb, (0,t_konc), [1.0,0,0],t_eval=t, args=(q[j],z[i]))

            res[i,j] = LA.norm(sol_stac.y[1]-sol.y[2])

    print(res.min)
    fig, ax = plt.subplots()
    cp = ax.contourf(z, q, res, levels=[1e-4,1e-3, 1e-2, 1e-1, 1e0, 1e1],cmap=plt.cm.viridis,norm=colors.LogNorm())
    cbar = fig.colorbar(cp,label=r'$||\bf{B}-\bf{B_{stac}}||$', format='%.0e')


    ax.set_xlabel('z')
    ax.set_ylabel('q')
    plt.title(r'$A(0)=1, A^{*}(0)=0,B(0)=C(0)= 0, $')
    plt.show()



    return None
#razlika_B()


############## 2.naloga ###########


def resitev():
    t_konc = 20
    t = np.linspace(0, t_konc, 1000)
    p=1.0
    s = 1.0
    q = 1.0
    r = 0.5
    t_par = 2.5
    m = t_par/s

    def eksaktno(t,y,p,q,r,s,t_par):
        v,z,u,x,y = y
        return([-p*v + q*z**2 - t_par*v*y, -2*q*z**2 - r*z*u + 2*p*v + t_par*y*v + s*x*y,-r*u*z + s*x*y, r*z*u +t_par*v*y - s*x*y, r*z*u - s*x*y - t_par*y*v])

    def priblizno(t,y,m,k):
        x,u,v = y
        return([k*np.sqrt(v)*u/(m+x/v), -k*u*np.sqrt(v)/(2*(m+x/v)),-k*u*np.sqrt(v)/(2*(m+x/v))])

    sol = solve_ivp(eksaktno, (0,t_konc), [1,0,10,0,0],t_eval=t, args=(p,q,r,s,t_par,))
    #sol1 = solve_ivp(priblizno, (0,t_konc), [0,10,1],t_eval=t, args=(m,))

    fig, ax = plt.subplots()
    """
    prvi, = ax.plot(t,sol.y[0], label=r'$[Br_{2}]$')
    ax.plot(t,sol1.y[2], '--',color = prvi.get_color())

    drugi, = ax.plot(t, sol.y[1], label=r'$[Br]$')
    ax.plot(t, np.sqrt(sol1.y[2])*np.sqrt(p/q),'--', color = drugi.get_color())

    tre, = ax.plot(t, sol.y[2], label=r'$[H_2]$')
    ax.plot(t, sol1.y[1],'--', color=tre.get_color())

    cet, = ax.plot(t, sol.y[3], label=r'$[HBr]$')
    ax.plot(t, sol1.y[0],'--', color =cet.get_color())

    pet, = ax.plot(t, sol.y[4], label=r'$[H]$')
    ax.plot(t, np.sqrt(p/q)*r*sol1.y[1]*np.sqrt(sol1.y[2])/(s*sol1.y[0] + t_par*sol1.y[2]), '--', color=pet.get_color())

    hbrar = np.linspace(0,50,7)
    for i in range(len(hbrar)):
        sol = solve_ivp(eksaktno, (0,t_konc), [1,0,1,hbrar[i],0],t_eval=t, args=(p,q,r,s,t_par,))
        plt.plot(t, sol.y[3]-hbrar[i], label='[Hbr(0)]={}'.format(hbrar[i].round(2)))

    ax.set_xlabel(r'$\tilde{t}$')
    ax.set_ylabel(r'[Hbr]-[Hbr(0)]')

    sol = solve_ivp(eksaktno, (0,t_konc), [1,0,1,0,0],t_eval=t, args=(p,q,r,5,2,))
    sol1 = solve_ivp(priblizno, (0,t_konc), [0,1,1],t_eval=t, args=(2/5,))
    print(LA.norm(sol.y[3]-sol1.y[0]))

    t_ar = np.linspace(1, 10, 10)
    s_ar = np.linspace(1, 10, 10)
    res = np.zeros((len(t_ar), len(s_ar)))
    for i in range(len(t_ar)):
        for j in range(len(s_ar)):
            sol = solve_ivp(eksaktno, (0,t_konc), [1,0,1,0,0],t_eval=t, args=(p,q,r,s_ar[j],t_ar[i],))
            sol1 = solve_ivp(priblizno, (0,t_konc), [0,1,1],t_eval=t, args=(t_ar[i]/s_ar[j],))

            res[i,j] = LA.norm(sol.y[3]-sol1.y[0])


    fig, ax = plt.subplots()
    cp = ax.contourf(t_ar, s_ar,res,100,cmap=plt.cm.viridis)
    cbar = fig.colorbar(cp,label=r'$||\bf{[Hbr]}-\bf{[Hbr]}_{stac}||$')
    """
    k=2*np.sqrt(p/q)*r*t_par/s
    sol = solve_ivp(eksaktno, (0,t_konc), [1,0,1,0,0],t_eval=t, args=(p,q,r,s,t_par,))
    sol1 = solve_ivp(priblizno, (0,t_konc), [0,1,1],t_eval=t, args=(2.5,k,))
    dt = t_konc/(999)

    print(k)
    X = (dt*np.sqrt(sol.y[0][100:-1])*sol.y[2][100:-1])/(np.diff(sol.y[3][100:]))
    X_prib = (dt*np.sqrt(sol1.y[2][:-1])*sol1.y[1][:-1])/(np.diff(sol1.y[0][:]))
    def func(x, a,b):
        return a*x+ b

    plt.plot(X[:-1:10], sol.y[3][100:-1:10]/sol.y[0][100:-1:10],'ro',markersize = 2,   label='Izmerki(numerična rešitev)')
    popt, pcov = curve_fit(func, X_prib[::],  sol.y[0][:-1]/sol.y[1][:-1])
    print(popt)
    plt.plot(X[::10], popt[0]*X[::10] + popt[1], label='fit, m=2.519, k = 2.494')

    ax.set_xlabel(r'$\sqrt{[Br_{2}]}[H]/\dot{[Hbr]}$')
    ax.set_ylabel(r'$[Hbr]/[Br_{2}]$')
    plt.title(r'$p=q=s=1, r=0.5, m=k=2.5$')

    #ax.set_yscale('log')
    plt.legend()
    plt.show()

resitev()
############ 3 naloga ##########
t_konc = 20
t = np.linspace(0, t_konc, 1000)
def sistem(t,y, L):
    I,I2,SO3 = y
    return([-2*I**2 + L*2*SO3**2 * I2, I**2 - L*SO3**2 * I2, -2*L*SO3**2 * I2])


sol = solve_ivp(sistem, (0,t_konc), [1,0,2],t_eval=t, args=(1,))
sol1 = solve_ivp(sistem, (0,t_konc), [1,0,2],t_eval=t, args=(10,))
sol2 = solve_ivp(sistem, (0,t_konc), [1,0,2],t_eval=t, args=(100,))

fig, ax = plt.subplots()
"""
prvi, = ax.plot(t,sol.y[0], label=r'$[I^{-}], \lambda = 1$')
ax.plot(t,sol1.y[0], '--',color = prvi.get_color(), label=r'$\lambda = 10$')
ax.plot(t,sol2.y[0], ':' ,color = prvi.get_color(), label=r'$\lambda = 100$')


drugi, = ax.plot(t, sol.y[1], label=r'$[I_{2}]$')
ax.plot(t, sol1.y[1],'--', color = drugi.get_color())
ax.plot(t, sol2.y[1],':', color = drugi.get_color())

tre, = ax.plot(t, sol.y[2], label=r'$[S_{2}O_{3}^{2-}]$')
ax.plot(t, sol1.y[2],'--', color=tre.get_color())
ax.plot(t, sol2.y[2],':', color=tre.get_color())
so3ar = np.linspace(1,20,10)
lamda =np.linspace(1,100, 10)
res = np.zeros((len(so3ar), len(lamda)))
for j in range(len(lamda)):
    for i in range(len(so3ar)):
        sol = solve_ivp(sistem, (0,t_konc), [1,0,so3ar[i]],t_eval=t, args=(lamda[j],))
        for k in range(len(sol.y[1])):
            if sol.y[1][k] > np.max(sol.y[1])*0.5:
                res [i,j] = k*t_konc/1000
                break
con = plt.contourf(lamda, so3ar, res, 100)
plt.colorbar(con, label=r'$t_{prehod}$')


plt.legend()
plt.xlabel(r'$\lambda$')
plt.ylabel(r'$[S_{2}O_{3}^{2-}](0)$')
"""
