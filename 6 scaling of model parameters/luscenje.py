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
from matplotlib.path import Path
import matplotlib.patches as patches



####### 1 NALOGA #########

podatki = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/6_luscenje_modelskih_parametrov/farmakoloski.dat")
x = podatki[:,0]
y = podatki[:,1]
dy = podatki[:,2]

##analiticna resitev##

A = np.array([[0,1],[0,0]])

vec_b = np.array([0,0])


for i in range(len(x)):
    A[0,0] += y[i]**4 /dy[i]**2
    A[0,1] += y[i]**4 /(dy[i]**2 * x[i])
    A[1,1] += y[i]**4 /(dy[i]**2 * x[i]**2)
    vec_b[0] += y[i]**3 / dy[i]**2
    vec_b[1] += y[i]**3 / (x[i]*dy[i]**2)

A[1,0] = A[0,1]
A_inv = np.linalg.inv(A)
a1,a2 = np.linalg.solve(A, vec_b)

y0 = 1/a1



sigma_y0 = np.sqrt(A_inv[0,0])/a1**2
a = a2*y0
sigma_a = (np.sqrt(A_inv[1,1])-sigma_y0*a2*a1)/(a1*(1+sigma_y0/y0))
korelacija = A_inv[1,0]/np.sqrt(A_inv[1,1]*A_inv[0,0])


def fja(X, y0,a):
    return(y0*X /(x+a))

popt1, pcov1 = curve_fit(fja, x, y, p0=[1, 1], sigma=dy, absolute_sigma=True)




Y0, A = popt1


sigma_Y0 = np.sqrt(pcov1[0,0])
sigma_A = np.sqrt(pcov1[1,1])
#print(popt, sigma_Y0, sigma_A, np.sqrt(pcov[1,0])/np.sqrt(pcov[0,0]*pcov[1,1]))

x_fit = np.linspace(0, 0.2, 1000)

#plt.plot(x_fit,np.abs(y0*x_fit/(x_fit+a) - Y0*x_fit/(x_fit + A))/(Y0*x_fit/(x_fit + A)))

napake=dy/y**2


def narisi():

    plt.errorbar(1/x[2:],1/y[2:], napake[2:], marker="o",label='izmerki',color="red", linestyle='None', markersize = 3, capsize=2)
    plt.plot(x_fit, Y0 + A*x_fit, label='fit',color='green')
    plt.fill_between(x_fit, (Y0-sigma_Y0 )+ (A-sigma_A)*x_fit, (Y0+sigma_Y0 )+ (A+sigma_A)*x_fit, color="green",
    alpha = 0.2, label=r'$fit \pm f(\sigma_{a1},\sigma_{a2})$')


    plt.errorbar(x,y,dy, marker="o",label='izmerki',color="red", linestyle='None', markersize = 3, capsize=2)
    plt.plot(x_fit, y0*x_fit/(x_fit+a),label='fit', color='green')
    plt.fill_between(x_fit, (y0-sigma_y0)*x_fit/(x_fit+(a-sigma_a)), (y0+sigma_y0)*x_fit/(x_fit+(a+sigma_a)), color="green",
    alpha = 0.2, label=r'$fit \pm f(\sigma_{a},\sigma_{y0})$')

    plt.xlabel('1/x')
    plt.ticklabel_format(axis="y", style="sci")
    #plt.ylabel(r'$\frac{|y-y_{scipy}|}{y_{scipy}}$', fontsize=13)
    plt.ylabel('1/y')
    plt.text(0.2, 0.05, r'$y0 = 104.6 \pm 2.5$ ' + '\n' r'$a = 21.1 \pm 1.2$'
    + '\n' + r'$\rho(y0,a) = -0.52$'
    )

    plt.legend()

    return None

###plt.show()

##### b) nelinearna minimizacija:
def nelin_min():
    def f_nelin(x,y0, a, p):
        return(y0*x**p / (x**p + a**p))

    def jacobi(x1 ,y0,a,p):

        dy0 = x1**p / (x1**p + a**p)
        da = -(y0*x1**p * p* a**(p-1)) / (x1**p + a**p)**2
        dp = (a**p * x**p * (np.log(x)-np.log(a))*y0
        / (x**p + a**p)**2)
        return(np.array([dy0, da,dp]).transpose())

    start = time.time()
    popt, pcov = curve_fit(f_nelin, x, y, p0=[1,1,1], sigma=dy, absolute_sigma=True, method='lm', jac=jacobi)
    end =time.time()
    print(end-start)
    zac = time.time()
    popt1, pcov1 = curve_fit(f_nelin,  x, y, p0=[1,1,1], sigma=dy, absolute_sigma=True, method='lm')
    kon = time.time()
    print(kon-zac)

    x_fit = np.linspace(0, np.max(x), 1000)
    y0,a,p = popt
    chi = np.sum((y - y0*x**p / (x**p + a**p))**2/dy**2)

    perr = np.sqrt(np.diag(pcov))
    sigma_y0,sigma_a, sigma_p = perr
    rhoya = pcov[2,1]/(sigma_a*sigma_p)


    y01,a1,p1 = popt1
    chi = np.sum((y - y01*x**p1 / (x**p1 + a1**p1))**2/dy**2)

    perr1 = np.sqrt(np.diag(pcov1))
    sigma_y01,sigma_a1, sigma_p1 = perr1
    plt.plot(x_fit,np.abs(y01*x_fit**p1 / (x_fit**p1 + a1**p1) -y0*x_fit**p / (x_fit**p + a**p)),  color="green")
    """
    plt.errorbar(x,y,dy, marker="o",label='izmerki',color="red", linestyle='None', markersize = 3, capsize=2)
    plt.plot(x_fit,y0*x_fit**p / (x_fit**p + a**p), label='razširjen model', color="green")
    plt.fill_between(x_fit, (Y0-sigma_Y0 )*x_fit/((A-sigma_A)+x_fit), (Y0+sigma_Y0 )*x_fit/ ((A+sigma_A)+x_fit), color="orange",
    alpha = 0.5, label=r'$fit \pm f(\sigma_{a},\sigma_{y0})$')
    plt.plot(x_fit, Y0*x_fit /(x_fit+A), label='osnovni model', color='orange')
    plt.fill_between(x_fit, (y0-sigma_y0)*x_fit**p/(x_fit**p+(a-sigma_a)**p), (y0+sigma_y0)*x_fit**p/(x_fit**p+(a+sigma_a)**p), color="green",
    alpha = 0.2, label=r'$fit \pm f(\sigma_{a},\sigma_{y0})$')
    """
    plt.xlabel('x')

    plt.ticklabel_format(axis="y", style="sci")
    #plt.ylabel(r'$\frac{|y-y_{scipy}|}{y_{scipy}}$', fontsize=13)
    plt.ylabel(r'$|y_{jac}-y|$')
    plt.text(700,6*10**((-8)), r'$t_{J} = 0.83 ms$' + '\n' + r'$t_{brez \  J} = 1.2 ms$')
    #plt.text(700, 50, r'$y0 = 99.6 \pm 2.6$ ' + '\n' r'$a = 19.8 \pm 1.7$'
    #+ '\n' + r'$p=1.36 \pm 0.13$' + '\n'+r'$\rho(y0,a) = 0.706$' + '\n' +
    #r'$\rho(y0,p) = -0.596$' +'\n' + r'$\rho(a,p) = -0.582$'
    #)

    plt.legend()
    plt.show()

    return None
#nelin_min()
############# DRUGA NALOGA -LEDVICE ###################
def druga():

    podatki = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/6_luscenje_modelskih_parametrov/ledvice.dat")

    cas= podatki[:, 0]
    N = podatki[:,1]


    def f1(t, C0,L):
        return(C0*np.exp(-L*t))

    def f2(t, C0, L,O):
        return(C0*np.exp(-L*t) + O)

    def f3(t, C0, L,O):
        return(C0*np.exp(-L*np.sqrt(t)) + O)

    def f4(t, C0, L, O, k):
        return(C0*np.exp(-L*t) + O + k*t)

    def f5(t, C0, L, O, k):
        return(C0*np.exp(-L*np.sqrt(t)) + O + k*t)

    def f6(t, C0, L):
        return(C0*np.exp(-L*np.sqrt(t)))

    def jac1(t, C0, L):
        return(np.array([np.exp(-L*t),-t*C0*np.exp(-L*t)]).transpose())

    def jac6(t, C0, L):
        return(np.array([np.exp(-L*np.sqrt(t)),-np.sqrt(t)*C0*np.exp(-L*np.sqrt(t))]).transpose())

    def jac2(t, C0, L, O):
        return(np.array([np.exp(-L*t),-t*C0*np.exp(-L*t),1]).transpose())

    def jac3(t, C0,L,O3):
        return(np.array([np.exp(-L*np.sqrt(t)),-(np.sqrt(t))*C0*np.exp(-L*np.sqrt(t)),O3 ]).transpose())


    popt, pcov = curve_fit(f1, cas, N, p0=[10000,0.1], method='lm', jac=jac1 )
    C,L = popt
    popt6, pcov6 = curve_fit(f6, cas, N, p0=[14000,0.1], method='lm', jac=  jac6)
    C6,L6 = popt6
    popt1, pcov1 = curve_fit(f2 ,cas, N, p0=[10000,0.1, 2000], method='lm' )
    C1,L1 ,O1= popt1
    popt2, pcov2 = curve_fit(f3 ,cas, N, p0=[10000,0.1, 2000], method='lm')
    C2,L2, O2 =popt2
    popt3, pcov3 = curve_fit(f4 ,cas, N, p0=[10000,0.1, 2000, 0.0001], method='lm')
    C3,L3,O3, k3 = popt3
    popt4, pcov4 = curve_fit(f5 ,cas, N, p0=[10000,0.1, 2000, 0.0001], method='lm')
    C4,L4,O4, k4 = popt4



    def dvo1(t, A,B,L1,L2):
        return(A*np.exp(-L1*t) + B*np.exp(-L2*t))

    def dvo2(t, A,B,L1,L2):
        return(A*np.exp(-L1*np.sqrt(t)) + B*np.exp(-L2*np.sqrt(t)))

    def dvo3(t, A,B,L1,L2, C):
        return(A*np.exp(-L1*t) + B*np.exp(-L2*t)+ C)
    def dvo4(t, A,B,L1,L2, C):
        return(A*np.exp(-L1*np.sqrt(t)) + B*np.exp(-L2*np.sqrt(t))+ C)

    popt21, pcov21 = curve_fit(dvo1, cas, N, p0=[10000,5000, 0.1, 0.01], method='lm' )
    A21,B21,L211,L212 = popt21
    popt22, pcov22 = curve_fit(dvo2, cas, N, p0=[10000,5000, 0.1, 0.01], method='lm' )
    A22,B22,L221,L222 = popt22
    popt23, pcov23 = curve_fit(dvo3, cas, N, p0=[14000,8000, 0.1, 0.01,1000], method='lm' )
    A23,B23,L231,L232, O23  = popt23

    popt24, pcov24 = curve_fit(dvo4, cas, N, p0=[14000,5000, 0.1, 0.01,1000], method='lm' )
    A24,B24,L241,L242, O24 = popt24

    chi = np.sum((N - dvo4(cas, A24,B24,L241,L242,O24) )**2)
    print(chi)


    cas_int = np.linspace(np.min(cas), np.max(cas), 2000)

    plt.plot(cas, N, 'r+', label='izmerki')
    #plt.plot(cas_int, C*np.exp(-L*cas_int), label=r'$Ce^{-\lambda t}$' )
    #plt.plot(cas_int, C6*np.exp(-L6*np.sqrt(cas_int)), label=r'$Ce^{-\lambda \sqrt{t}}$' )
    #plt.plot(cas_int, C1*np.exp(-L1*cas_int) + O1, label=r'$Ce^{-\lambda t} + O$' )
    #plt.plot(cas_int, C2*np.exp(-L2*np.sqrt(cas_int)) + O2, label=r'$Ce^{-\lambda \sqrt{t}} + O$' )
    #plt.plot(cas_int, C3*np.exp(-L3*cas_int) + O3 + k3*cas_int, label=r'$Ce^{-\lambda t} + O + kt$' )
    #plt.plot(cas_int, C4*np.exp(-L4*np.sqrt(cas_int)) + O4 + k4*cas_int, label=r'$Ce^{-\lambda \sqrt{t}} + O + kt$' )
    plt.plot(cas_int, dvo1(cas_int, A21,B21,L211,L212), label=r'$Ae^{-\lambda_1 t} + Be^{-\lambda_2t}$')
    plt.plot(cas_int, dvo2(cas_int,A22,B22,L221,L222), label=r'$Ae^{-\lambda_1 \sqrt{t}} + Be^{-\lambda_2 \sqrt{t}}$')
    #plt.plot(cas_int, dvo3(cas_int, A23,B23,L231,L232, O23), label=r'$Ae^{-\lambda_1 t} + Be^{-\lambda_2t} + O$')
    #plt.plot(cas_int, dvo4(cas_int, A24,B24,L241,L242, O24),color='green', label=r'$Ae^{-\lambda_1 \sqrt{t}} + Be^{-\lambda_2 \sqrt{t}} + O$')

    plt.xlabel('t')
    #plt.xlim([-50,1000])
    plt.ylabel('N')

    """
    fig, (ax1,ax2) = plt.subplots(1, 2)
    arr = np.array([1,2,3])
    X,Y = np.meshgrid(arr,arr)
    oznake = ['A','B',r'$\lambda_1$',r'$\lambda_2$', 'O' ]
    for i in range(5):
        for j in range(5):
            if i != j:
                pcov23[i,j] = pcov23[i,j]/np.sqrt(pcov23[i,i]*pcov23[j,j])
                pcov24[i,j] = pcov24[i,j]/np.sqrt(pcov24[i,i]*pcov24[j,j])
    np.fill_diagonal(pcov23, 1)
    np.fill_diagonal(pcov24, 1)


    #plt.contourf(X,Y,pcov2, levels=0+levels,cmap=plt.get_cmap('rainbow'))
    im1 = ax1.imshow(pcov23,vmin=-1, vmax=1, interpolation='None', aspect='auto')
    im2 = ax2.imshow(pcov24, vmin = -1, vmax = 1, interpolation='None', aspect='auto')

    ax1.set_xticks(np.arange(len(oznake)), labels=oznake)
    ax1.set_yticks(np.arange(len(oznake)), labels=oznake)
    ax2.set_xticks(np.arange(len(oznake)), labels=oznake)
    ax2.set_yticks(np.arange(len(oznake)), labels=oznake)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    col = fig.colorbar(im2,cax=cbar_ax, cmap='viridis')
    col.ax.set_title(r'$\rho$')
    ax1.set_title(r'$Ae^{\lambda_1 t} + Be^{\lambda_2t} + O$')
    ax2.set_title(r'$Ae^{\lambda_1 \sqrt{t}} + Be^{\lambda_2 \sqrt{t}} + O$')


    """
    plt.legend()
    plt.show()

    return None

druga()

############# 3 NALOGA ################


podatki = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/6_luscenje_modelskih_parametrov/thtg-xfp-thfp.dat")

th_tg = podatki[:,0]*np.pi/180
x_fp = podatki[:,1]
th_fp = podatki[:, 2]*np.pi/180

def model1(vec_x_i, vec_a):
    return(vec_x_i[0]*vec_a[0] +vec_x_i[1]*vec_a[1] + vec_x_i[2]*vec_a[2]
    + vec_x_i[3]*vec_a[3]+vec_x_i[4]*vec_a[4]+vec_x_i[5]*vec_a[5]+
    vec_x_i[6]*vec_a[6]+ vec_x_i[7]*vec_a[7]+vec_x_i[8]*vec_a[8] )

dth_tg = dth_fp=0.06
dx_fp= 0.1


A = np.zeros((len(x_fp),9 ))
"""
A[:,0] = 1
A[:,1] = x_fp
A[:,2] = x_fp**2
A[:,3] = th_fp
A[:,4] = th_fp*x_fp
A[:,5] = x_fp**2 * th_fp
"""
A[:,0] = 1
A[:,1] = x_fp
A[:,2] = x_fp**2
A[:,3] = th_fp
A[:,4] = th_fp*x_fp
A[:,5] = x_fp**2 * th_fp
A[:,6] = th_fp**2
A[:,7] = x_fp*th_fp**2
A[:,8] = x_fp**2 *th_fp**2
"""
A[:,0] = 1
A[:,1] = x_fp
A[:,2] = th_fp
A[:,3] = th_fp*x_fp
"""
U, s, Vt = LA.svd(A)

V =Vt.transpose()
par = np.zeros(9).transpose()

for i in range(9):
    par +=((U[:,i].dot(th_tg))/s[i])*V[:,i]


chi =0
modelska = np.zeros(len(th_tg))

for i in range(len(th_tg)):
    chi += ((th_tg[i]- model1(A[i,:], par))**2 / 0.06**2)
    modelska[i] = model1(A[i,:], par)






"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x_fp, th_fp, np.abs(th_tg-modelska), marker=2)
ax.set_xlabel(r'$x_{fp}$')
ax.set_ylabel(r'$\theta_{fp}$')
ax.set_zlabel(r'$|\theta_{tg} - \theta_{tg, model}|$')
ax.set_title(r'$\chi^2 = 203$')
plt.show()


fig, (ax1) = plt.subplots(1)

oznake = ['a1','a2','a3','a4', 'a5', 'a6', 'a7', 'a8', 'a9' ]
rho=  np.zeros((9,9))
variance = np.zeros((9,9))
for j in range(9):
    for i in range(9):
        variance[j,j] += V[i,j]**2/s[i]**2

for j in range(9):
    for k in range(9):

        if k != j:
            for i in range(9):
                rho[j,k] += V[i,j]*V[i,k]/s[i]**2/np.sqrt(variance[j,j]*variance[k,k])


np.fill_diagonal(rho, 1)



#plt.contourf(X,Y,pcov2, levels=0+levels,cmap=plt.get_cmap('rainbow'))
im1 = ax1.imshow(rho,vmin=-1, vmax=1, interpolation='None', aspect='auto')

ax1.set_xticks(np.arange(len(oznake)), labels=oznake)
ax1.set_yticks(np.arange(len(oznake)), labels=oznake)


# Rotate the tick labels and set their alignment.
plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
fig.subplots_adjust(right=0.85)

col = fig.colorbar(im1, cmap='viridis')
col.ax.set_title(r'$\rho$')
#ax1.set_title()



plt.legend()
plt.show()
"""
