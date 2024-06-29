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
import math
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
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from scipy.stats import norm
from scipy import signal
from scipy.fft import fft, fftfreq,ifft
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import copy
from matplotlib.ticker import LogFormatter

####Podatki#######
noise_data = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/11_kalmanov_filter/kalman_cartesian_data.dat", delimiter=' ')
t=noise_data[:,0]



cartesian_kontrola = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/11_kalmanov_filter/kalman_cartesian_kontrola.dat", delimiter=' ')
tr,xnr,ynr,vxnr,vynr=cartesian_kontrola[:,0], cartesian_kontrola[:,1], cartesian_kontrola[:,2] \
,cartesian_kontrola[:,3], cartesian_kontrola[:,4]


relative_data = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/11_kalmanov_filter/kalman_relative_data.dat", delimiter=' ')
trel,xnrel,ynrel,axnrel,aynrel=relative_data[:,0], relative_data[:,1], relative_data[:,2] \
,relative_data[:,3], relative_data[:,4]


##### 1 naloga ####

def napoved_modela(stanje, var_mat, prehodmat, cn, wn, Q):
    new_state = prehodmat @ stanje + cn + wn
    new_var =  prehodmat @ var_mat @ prehodmat.transpose() + Q

    return new_state, new_var


########### konstrukcija potrebnih matrik #########
def prehodmat(dt):
    A = np.eye(2)
    mat = np.block([[A,A*dt],[A*0, A]])

    return mat


def Q(dt, sigma_a):
    A = np.eye(2)*0.25*dt**4
    B = np.eye(2)*0.5*dt**3
    C = dt**2 * np.eye(2)
    mat = np.block([[A, B],[B,C]])
    mat = mat*sigma_a**2
    return mat



def Rn(vx, vy):
    Rn = np.zeros((4,4))
    vn = np.sqrt(vy**2 + vx**2)
    sigma_vn = 0.01*vn ################# kaj je napaka hitrosti !!!!!???
    if sigma_vn < 0.27:
        sigma_vn=0.27
    np.fill_diagonal(Rn,[25**2,25**2,sigma_vn**2 , sigma_vn**2])
    return Rn

def cn_3_nal(vx,vy,u):
    Bnvv = 1/(LA.norm([vx,vy]))*np.array([[vx,-vy],[vy,vx]])
    Bn = np.block([[np.eye(2), np.eye(2)*0], [np.eye(2)*0, Bnvv]])

    cn = Bn@u
    return cn


def Qn_3_nal(vx,vy,at,ar,dt,Pnvv,sigma_a):
    vec_v_perp = np.array([-vy,vx])
    vec_a_perp = np.array([-at,ar])
    Bnvv = 1/(LA.norm([vx,vy]))*np.array([[vx,-vy],[vy,vx]])
    Qmat = Q(dt, sigma_a)
    Qnvv = dt**2*(sigma_a**2 * np.eye(2) +
    ((vec_v_perp @ Pnvv @ vec_v_perp)/(vx**2 + vy**2)**2)  *
    np.tensordot((Bnvv @ vec_a_perp),(Bnvv @ vec_a_perp), axes=0))
    Qmat[2:,2:] = Qnvv

    return(Qmat)



######## kalman for dynamics xn+1 = F xn + c + wn , x je 4d vektor
##### x=(x,y,vx,xy)

def kalman_filter(noise_data, kov_mat_start ,sigma_a=0.05):
    t,xn,yn,vxn,vyn,axn,ayn=noise_data[:,0], noise_data[:,1], noise_data[:,2] \
    ,noise_data[:,3], noise_data[:,4], noise_data[:,5], noise_data[:,6]

    zac_pogoj = np.array([xn[0], yn[0], vxn[0], vyn[0]])
    dt = t[1]-t[0]

    Qmat = Q(dt, sigma_a)
    F = prehodmat(dt)

    X_states = zac_pogoj
    P_states = kov_mat_start
    residuali = np.array([])

    Xplus = zac_pogoj
    Pplus = kov_mat_start

    for i in range(len(t)-1):


        wn = np.random.multivariate_normal([0,0,0,0], Qmat)
        cn  = np.array([0.5*axn[i]*dt**2, 0.5*ayn[i]*dt**2 \
         ,axn[i]*dt, ayn[i]*dt])

        Xmin_next, Pmin_next = napoved_modela(Xplus, Pplus, F, cn,wn,Qmat)



        Hn_next = np.eye(4)
        np.fill_diagonal(Hn_next,[1,1,1,1])
        Rnmat_next = Rn(vxn[i+1], vyn[i+1])

        #rn_next = np.random.multivariate_normal([0,0,0,0], Rnmat_next)
        zn_next = np.array([xn[i+1], yn[i+1], vxn[i+1], vyn[i+1]])
        Kn_next = Pmin_next @ Hn_next.transpose() @ \
        LA.inv(Hn_next @Pmin_next@ Hn_next.transpose() + Rnmat_next)


        Xplus = Xmin_next + Kn_next @ (zn_next - Hn_next @ Xmin_next)
        Pplus = (np.eye(4) - Kn_next @ Hn_next ) @ Pmin_next


        X_states = np.vstack((X_states, Xplus))
        P_states = np.dstack((P_states, Pplus))
        residuali = np.append(residuali, LA.norm(zn_next - Hn_next@Xmin_next))



    return X_states, P_states, residuali

def kalman_filter_time_delay(noise_data, kov_mat_start ,t_zac, t_kon,sigma_a=0.05):
    t,xn,yn,vxn,vyn,axn,ayn=noise_data[:,0], noise_data[:,1], noise_data[:,2] \
    ,noise_data[:,3], noise_data[:,4], noise_data[:,5], noise_data[:,6]

    zac_pogoj = np.array([xn[0], yn[0], vxn[0], vyn[0]])
    dt = t[1]-t[0]

    Qmat = Q(dt, sigma_a)
    F = prehodmat(dt)

    X_states = zac_pogoj
    P_states = kov_mat_start
    residuali = np.array([])

    Xplus = zac_pogoj
    Pplus = kov_mat_start

    for i in range(len(t)-1):


        wn = np.random.multivariate_normal([0,0,0,0], Qmat)
        cn  = np.array([0.5*axn[i]*dt**2, 0.5*ayn[i]*dt**2 \
         ,axn[i]*dt, ayn[i]*dt])

        Xmin_next, Pmin_next = napoved_modela(Xplus, Pplus, F, cn,wn,Qmat)


        if t[i]>=t_zac and t[i] < t_kon:
            Hn_next = np.eye(4)*0
        else:
            Hn_next = np.eye(4)
            np.fill_diagonal(Hn_next,[1,1,1,1])

        Rnmat_next = Rn(vxn[i+1], vyn[i+1])

        #rn_next = np.random.multivariate_normal([0,0,0,0], Rnmat_next)

        zn_next = np.array([xn[i+1], yn[i+1], vxn[i+1], vyn[i+1]])
        Kn_next = Pmin_next @ Hn_next.transpose() @ \
        LA.inv(Hn_next @Pmin_next@ Hn_next.transpose() + Rnmat_next)


        Xplus = Xmin_next + Kn_next @ (zn_next - Hn_next @ Xmin_next)
        Pplus = (np.eye(4) - Kn_next @ Hn_next ) @ Pmin_next


        X_states = np.vstack((X_states, Xplus))
        P_states = np.dstack((P_states, Pplus))
        residuali = np.append(residuali, LA.norm(zn_next - Hn_next@Xmin_next))



    return X_states, P_states, residuali

def kalman_filter_mod(noise_data, kov_mat_start ,mode,sigma_a=0.05):
    t,xn,yn,vxn,vyn,axn,ayn=noise_data[:,0], noise_data[:,1], noise_data[:,2] \
    ,noise_data[:,3], noise_data[:,4], noise_data[:,5], noise_data[:,6]

    zac_pogoj = np.array([xn[0], yn[0], vxn[0], vyn[0]])
    dt = t[1]-t[0]

    Qmat = Q(dt, sigma_a)
    F = prehodmat(dt)

    X_states = zac_pogoj
    P_states = kov_mat_start
    residuali = np.array([])

    Xplus = zac_pogoj
    Pplus = kov_mat_start
    k=0

    for i in range(len(t)-1):


        wn = np.random.multivariate_normal([0,0,0,0], Qmat)
        cn  = np.array([0.5*axn[i]*dt**2, 0.5*ayn[i]*dt**2 \
         ,axn[i]*dt, ayn[i]*dt])

        Xmin_next, Pmin_next = napoved_modela(Xplus, Pplus, F, cn,wn,Qmat)


        ########vsaka nta in mta meritev
        if mode=='lessmes':
            m_pot=20
            n_hitrost=20
            if (i+1)%m_pot ==0 and (i+1)%n_hitrost ==0:
                #print(xn[i+1])
                Hn_next = np.eye(4)
                np.fill_diagonal(Hn_next,[0,0,1,1])
            elif  (i+1)%m_pot ==0:
                Hn_next = np.eye(4)
                np.fill_diagonal(Hn_next,[1,1,0,0])*0
            elif (i+1)%n_hitrost ==0:
                Hn_next = np.eye(4)
                np.fill_diagonal(Hn_next,[0,0,1,1])*0
            else:
                Hn_next = np.eye(4)*0

        ###Periodično vzorčenje:
        if mode=='periodic':

            Hn_next = np.eye(4)
            if k==0:
                np.fill_diagonal(Hn_next,[1,0,0,0])
                k+=1
            elif k==1:
                np.fill_diagonal(Hn_next,[0,1,0,0])
                k+=1
            elif k==2:
                np.fill_diagonal(Hn_next,[0,0,1,0])
                k+=1
            elif k==3:
                np.fill_diagonal(Hn_next,[0,0,0,1])
                k=0

            print(k)
        #Hn_next = np.eye(4)
        #np.fill_diagonal(Hn_next,[1,1,1,1])
        Rnmat_next = Rn(vxn[i+1], vyn[i+1])

        #rn_next = np.random.multivariate_normal([0,0,0,0], Rnmat_next)
        zn_next = np.array([xn[i+1], yn[i+1], vxn[i+1], vyn[i+1]])
        Kn_next = Pmin_next @ Hn_next.transpose() @ \
        LA.inv(Hn_next @Pmin_next@ Hn_next.transpose() + Rnmat_next)

        Xplus = Xmin_next + Kn_next @ (zn_next - Hn_next @ Xmin_next)
        Pplus = (np.eye(4) - Kn_next @ Hn_next ) @ Pmin_next


        X_states = np.vstack((X_states, Xplus))
        P_states = np.dstack((P_states, Pplus))
        residuali = np.append(residuali, LA.norm(zn_next - Hn_next@Xmin_next))



    return X_states, P_states, residuali

kov_mat_start = np.eye(4)
np.fill_diagonal(kov_mat_start, [100, 100, 5,5])

X, P, residuali = kalman_filter(noise_data, kov_mat_start)
#X1,P1,residuali1 = kalman_filter_time_delay(noise_data, kov_mat_start,2000,2400)
X1,P1,residuali2 = kalman_filter_mod(noise_data, kov_mat_start,'periodic')

def narisi_stat_opt_filter(X,P, kartesian_kontrola, noise_data, residual):
    t,xn,yn,vxn,vyn,axn,ayn=noise_data[:,0], noise_data[:,1], noise_data[:,2] \
    ,noise_data[:,3], noise_data[:,4], noise_data[:,5], noise_data[:,6]
    tr,xnr,ynr,vxnr,vynr=cartesian_kontrola[:,0], cartesian_kontrola[:,1], cartesian_kontrola[:,2] \
    ,cartesian_kontrola[:,3], cartesian_kontrola[:,4]

    fig,ax =  plt.subplots(3,2)

    ax[0,0].plot(X[:,0], X[:,1], label='filtered')
    ax[0,0].plot(xnr, ynr, label='exact')

    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    ax[0,0].legend(prop={'size': 6})
    ax[0,0].ticklabel_format(scilimits=(-2,2))





    ax[0,1].set_xlabel(r'$x_{exact}-x_{i}$')
    ax[0,1].set_ylabel(r'$y_{exact}-y_{i}$')

    cm = plt.cm.get_cmap('winter') # 'plasma' or 'viridis'
    sc = ax[0,1].scatter(xnr-X[:,0], ynr-X[:,1],s=2, c=t,
    vmin = t[0], vmax=t[-1], cmap=cm, alpha= 1 , label='filtered')
    plt.colorbar(sc, ax = ax[0,1], label='t', pad=-0.09)
    cm = plt.cm.get_cmap('autumn') # 'plasma' or 'viridis'
    sc = ax[0,1].scatter(xnr-xn, ynr-yn,s=2, c=t,
    vmin = t[0], vmax=t[-1], cmap=cm, alpha = 0.2, label='w/noise')
    a = plt.colorbar(sc, ax = ax[0,1])
    a.set_ticks([])
    ax[0,1].legend(prop={'size': 6})


    ax[1,0].set_xlabel(r'$v_{x,exact}-v_{x,i}$')
    ax[1,0].set_ylabel(r'$v_{y,exact}-v_{y,i}$')
    cm = plt.cm.get_cmap('winter') # 'plasma' or 'viridis'
    sc = ax[1,0].scatter(vxnr-X[:,2], vynr-X[:,3],s=2, c=t,
    vmin = t[0], vmax=t[-1], cmap=cm, alpha= 1 , label='filtered')
    plt.colorbar(sc, ax = ax[1,0], label='t', pad=-0.09)
    cm = plt.cm.get_cmap('autumn') # 'plasma' or 'viridis'
    sc = ax[1,0].scatter(vxnr-vxn, vynr-vyn,s=2, c=t,
    vmin = t[0], vmax=t[-1], cmap=cm, alpha = 0.2, label='w/noise')
    a = plt.colorbar(sc, ax = ax[1,0])
    a.set_ticks([])
    ax[1,0].legend(prop={'size': 6})

    ax[1,1].plot(t[1:], residual)
    ax[1,1].set_xlabel('t')
    ax[1,1].set_ylabel(r'$||{\bf{z}}_{n+1}-H_{n+1} {\bf{x}}_{n+1}^{-}||$')
    ax[1,1].set_yscale('log')

    AVD = np.sum(np.sqrt((xnr-X[:,0])**2+( ynr-X[:,1])**2))/len(np.sqrt((xnr-X[:,0])**2+( ynr-X[:,1])**2))
    AVD1=np.sum(np.sqrt((xnr-xn)**2+( ynr-yn)**2))/len(np.sqrt((xnr-xn)**2+( ynr-yn)**2))
    ax[2,0].hist(np.sqrt((xnr-X[:,0])**2+( ynr-X[:,1])**2), bins = 50,density=True,
     label='filtered, AVG:{}'.format(round(AVD,2)), alpha=1)
    ax[2,0].hist(np.sqrt((xnr-xn)**2+( ynr-yn)**2),  bins = 50,density=True,
     label='w/noise, AVG:{}'.format(round(AVD1,2)), alpha=0.6)
    ax[2,0].set_xlim((0,250))
    ax[2,0].set_xlabel(r'$|{\bf{r}}_{exact}-{\bf{r}}_i|$')
    ax[2,0].legend(prop={'size': 6})
    ax[2,0].set_ylabel(r'PD')

    AVD = np.sum(np.sqrt((vxnr-X[:,2])**2+( vynr-X[:,3])**2))/len(np.sqrt((vxnr-X[:,2])**2+( vynr-X[:,3])**2))
    AVD1=np.sum(np.sqrt((vxnr-vxn)**2+( vynr-vyn)**2))/len(np.sqrt((vxnr-vxn)**2+( vynr-vyn)**2))
    ax[2,1].hist(np.sqrt((vxnr-X[:,2])**2+( vynr-X[:,3])**2), bins = 50,density=True,
     label='filtered, AVG:{}'.format(round(AVD,2)), alpha=1)
    ax[2,1].hist(np.sqrt((vxnr-vxn)**2+( vynr-vyn)**2),  bins = 50,density=True,
     label='w/noise, AVG:{}'.format(round(AVD1,2)), alpha=0.6)
    ax[2,1].set_xlim((0,2.5))
    ax[2,1].set_xlabel(r'$|{\bf{v}}_{exact}-{\bf{v}}_i|$')
    ax[2,1].legend(prop={'size': 6})
    ax[2,1].set_ylabel(r'PD')
    #plt.suptitle('x(0) = y(0) =1000')
    fig.tight_layout(pad=0.1)
    plt.show()

narisi_stat_opt_filter(X1,P1, cartesian_kontrola, noise_data, residuali2)

def narisi_koncne_kovar_matrike(P1,P2,P3):


    fig,ax =  plt.subplots(1,3)

    alpha = ['x', 'y', r'$v_{x}$', r'$v_{y}$']

    my_cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    my_cmap.set_bad(my_cmap.colors[0])
    a = ax[0].matshow(P1[:,:,-1], cmap=my_cmap,
    norm = LogNorm(0.001,15))
    plt.colorbar(a,ax=ax[0],fraction=0.046, pad=0.04)
    for i in range(4):
        for j in range(4):
            c = round(P1[i,j,-1], 3)
            ax[0].text(i, j, str(c), va='center', ha='center')

    ax[0].tick_params(axis="x", bottom=False)
    ax[0].set_xticklabels(['']+alpha)
    ax[0].set_yticklabels(['']+alpha)
    ax[0].set_title('vse meritve')

    my_cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    my_cmap.set_bad(my_cmap.colors[0])
    a = ax[1].matshow(P2[:,:,-1], cmap=my_cmap,
    norm = LogNorm(0.001,15))
    plt.colorbar(a,ax=ax[1],fraction=0.046, pad=0.04)
    for i in range(4):
        for j in range(4):
            c = round(P2[i,j,-1],3)
            ax[1].text(i, j, str(c), va='center', ha='center')
    ax[1].tick_params(axis="x", bottom=False)
    ax[1].set_xticklabels(['']+alpha)
    ax[1].set_yticklabels(['']+alpha)
    ax[1].set_title('periodično vzorčenje')


    my_cmap = copy.copy(mpl.cm.get_cmap('viridis'))
    my_cmap.set_bad(my_cmap.colors[0])
    a = ax[2].matshow(P3[:,:,-1], cmap=my_cmap,
    norm = LogNorm(0.001,15))
    plt.colorbar(a,ax=ax[2],fraction=0.046, pad=0.04)

    for i in range(4):
        for j in range(4):
            c = round(P3[i,j,-1],3)
            ax[2].text(i, j, str(c), va='center', ha='center')
    ax[2].tick_params(axis="x", bottom=False)
    ax[2].set_xticklabels(['']+alpha)
    ax[2].set_yticklabels(['']+alpha)
    ax[2].set_title('10-p, 5-v')

    fig.tight_layout(pad=0.1)
    plt.show()

#narisi_koncne_kovar_matrike(P, P1, P2)

def narisi_napake(Porig,P2):



    fig,ax =  plt.subplots(2,2)

    ax[0,0].plot(t, Porig[0,0,:], label='kalman')
    ax[0,0].plot(t, P2[0,0,:], label='kalman-akcelerometer')

    ax[0,0].set_ylabel(r'$\sigma_r^2$')
    ax[0,0].set_yscale('log')

    ax[0,1].plot(t, Porig[2,2,:])
    ax[0,1].plot(t, P2[2,2,:])

    ax[0,1].set_ylabel(r'$\sigma_v^2$')
    ax[0,1].set_yscale('log')

    ax[1,0].plot(t, Porig[0,2,:])
    ax[1,0].plot(t, P2[0,2,:])
    ax[1,0].set_xlabel('t')
    ax[1,0].set_ylabel(r'$\sigma_{xv_{x}}^2$')
    ax[1,0].set_yscale('log')

    ax[1,1].plot(t, Porig[1,3,:])
    ax[1,1].plot(t, P2[1,3,:])
    ax[1,1].set_xlabel('t')
    ax[1,1].set_ylabel(r'$\sigma_{yv_{y}}^2$')
    ax[1,1].set_yscale('log')

    handles, labels = ax[0,0].get_legend_handles_labels()

    fig.tight_layout(pad=1)
    plt.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0, 0, 1, 1),
           bbox_transform = plt.gcf().transFigure)
    plt.show()

#narisi_napake(P,P1)

def narisi_var_razl(cartesian_kontrola, X1,P1):
    tr,xnr,ynr,vxnr,vynr=cartesian_kontrola[:,0], cartesian_kontrola[:,1], cartesian_kontrola[:,2] \
    ,cartesian_kontrola[:,3], cartesian_kontrola[:,4]

    delez_x = 0
    delez_vx = 0
    for i in range(len(t)):
        if np.abs(X1[i,0]-xnr[i]) <np.sqrt(P1[0,0,i]):
            delez_x+=1
        if np.abs(X1[i,2]-vxnr[i]) <np.sqrt(P1[2,2,i]):
            delez_vx+=1


    delez_x /=1400
    delez_vx /=1400


    fig,ax =  plt.subplots(1,2)


    ax[0].plot(tr, np.abs(X1[:,0]-xnr), label=r'$|x_{exact}-x|$')
    ax[0].plot(tr, np.sqrt(P1[0,0,:]), label=r'$\sigma_x$')
    ax[0].text(515,21,r'$P(|x_{exact}-x|<\sigma_x)=$'+'{}'.format(round(delez_x,2)))
    ax[0].set_xlabel('t')
    ax[0].legend()

    ax[1].plot(tr, np.abs(X1[:,2]-vxnr), label=r'$|v_{x,exact}-v_x|$')
    ax[1].plot(tr, np.sqrt(P1[2,2,:]), label=r'$\sigma_{v_{x}}$')
    ax[1].text(300,3,r'$P(|v_{x,exact}-v_x|<\sigma_{v_{x}})=$'+'{}'.format(round(delez_vx,2)))
    ax[1].set_xlabel('t')
    ax[1].legend()

    plt.show()

#narisi_var_razl(cartesian_kontrola, X1,P1)

def odstopanje_od_izg_signala(cartesian_kontrola):
    tr,xnr,ynr,vxnr,vynr=cartesian_kontrola[:,0], cartesian_kontrola[:,1], cartesian_kontrola[:,2] \
    ,cartesian_kontrola[:,3], cartesian_kontrola[:,4]


    t_zac = np.linspace(1500,2700,10)
    dolzine = np.linspace(100,1000,10)
    odstop = np.zeros((len(t_zac), len(dolzine)))
    for i in range(len(t_zac)):
        for j in range(len(dolzine)):
            X1,P1,residuali1 = kalman_filter_time_delay(noise_data, kov_mat_start,t_zac[i],t_zac[i]+dolzine[j])
            odstop[i,j] = np.abs(np.sqrt(X1[-1,0]**2 + X1[-1,1]**2)
            -np.sqrt(xnr[-1]**2+ ynr[-1]**2))


    a = plt.contourf(t_zac, dolzine,odstop.transpose(),levels=np.logspace(0,4,10), norm=LogNorm())
    plt.colorbar(a, label=r'$|\bf{r}-\bf{r}_{exact}|$')
    plt.xlabel(r'$t_{zac}$')
    plt.ylabel(r'$t_{off}$')
    plt.show()

#odstopanje_od_izg_signala(cartesian_kontrola)


########## 3 naloga ##############

def kalman_filter_rel(relative_data, kov_mat_start ,sigma_a=0.05):
    trel,xnrel,ynrel,atnrel,arnrel=relative_data[:,0], relative_data[:,1], relative_data[:,2] \
    ,relative_data[:,3], relative_data[:,4]

    zac_pogoj = np.array([xnrel[0], ynrel[0], 1, 1])
    dt = t[1]-t[0]


    F = prehodmat(dt)

    X_states = zac_pogoj
    P_states = kov_mat_start
    Rnmat_next = np.zeros((4,4))
    np.fill_diagonal(Rnmat_next,[25**2,25**2,1, 1])
    residuali = np.array([])


    Xplus = zac_pogoj
    Pplus = kov_mat_start

    for i in range(len(t)-1):
        Qnmat=Qn_3_nal(Xplus[2],Xplus[3],atnrel[i],arnrel[i],dt,Pplus[2:,2:],sigma_a)

        wn = np.random.multivariate_normal([0,0,0,0], Qnmat)
        un  = np.array([0.5*atnrel[i]*dt**2, 0.5*arnrel[i]*dt**2 \
         ,atnrel[i]*dt, arnrel[i]*dt])
        cn = cn_3_nal(Xplus[2],Xplus[3],un)

        Xmin_next, Pmin_next = napoved_modela(Xplus, Pplus, F, cn,wn,Qnmat)



        Hn_next = np.eye(4)
        np.fill_diagonal(Hn_next,[1,1,0,0])



        #rn_next = np.random.multivariate_normal([0,0,0,0], Rnmat_next)
        zn_next = np.array([xnrel[i+1], ynrel[i+1], 0,0])
        Kn_next = Pmin_next @ Hn_next.transpose() @ \
        LA.inv(Hn_next @Pmin_next@ Hn_next.transpose() + Rnmat_next)


        Xplus = Xmin_next + (1+np.exp(np.abs(arnrel[i])/np.max(arnrel)))*Kn_next @ (zn_next - Hn_next @ Xmin_next)
        Pplus = (np.eye(4) - Kn_next @ Hn_next ) @ Pmin_next


        X_states = np.vstack((X_states, Xplus))
        P_states = np.dstack((P_states, Pplus))
        residuali = np.append(residuali, LA.norm(zn_next - Hn_next@Xmin_next))



    return X_states, P_states, residuali



kov_mat_start = np.eye(4)
np.fill_diagonal(kov_mat_start, [100, 100, 100,100])

X_rel, P_rel, residuali_rel = kalman_filter_rel(relative_data, kov_mat_start)

def narisi_stat_opt_filter_rel(X,P, kartesian_kontrola, relative_data, residual):
    trel,xnrel,ynrel,axnrel,aynrel=relative_data[:,0], relative_data[:,1], relative_data[:,2] \
    ,relative_data[:,3], relative_data[:,4]
    tr,xnr,ynr,vxnr,vynr=cartesian_kontrola[:,0], cartesian_kontrola[:,1], cartesian_kontrola[:,2] \
    ,cartesian_kontrola[:,3], cartesian_kontrola[:,4]

    fig,ax =  plt.subplots(3,2)

    ax[0,0].plot(X[:,0], X[:,1], label='filtered')
    ax[0,0].plot(xnr, ynr, label='exact')


    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    ax[0,0].legend(prop={'size': 6})
    ax[0,0].ticklabel_format(scilimits=(-2,2))

    ax[0,1].plot(X[:,0], X[:,1], label='filtered')
    ax[0,1].plot(xnr, ynr, label='exact')


    ax[0,1].set_xlabel('x')
    ax[0,1].set_ylabel('y')
    ax[0,1].legend(prop={'size': 6})
    ax[0,1].ticklabel_format(scilimits=(-2,2))
    ax[0,1].set_xlim((9000,12000))
    ax[0,1].set_ylim((5200,5400))

    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    ax[1,0].plot(X[:,0], X[:,1], label='filtered')
    ax[1,0].plot(xnr, ynr, label='exact')
    ax[1,0].plot(xnrel, ynrel, '--',label='w/noise')
    ax[1,0].legend(prop={'size': 6})
    ax[1,0].ticklabel_format(scilimits=(-2,2))
    ax[1,0].set_xlim((800,1700))
    ax[1,0].set_ylim((-2100,-1500))






    ax[2,0].set_xlabel(r'$x_{exact}-x_{i}$')
    ax[2,0].set_ylabel(r'$y_{exact}-y_{i}$')

    cm = plt.cm.get_cmap('winter') # 'plasma' or 'viridis'
    sc = ax[2,0].scatter(xnr-X[:,0], ynr-X[:,1],s=2, c=t,
    vmin = t[0], vmax=t[-1], cmap=cm, alpha= 1 , label='filtered')
    plt.colorbar(sc, ax = ax[2,0], label='t', pad=-0.09)
    cm = plt.cm.get_cmap('autumn') # 'plasma' or 'viridis'
    sc = ax[2,0].scatter(xnr-xnrel, ynr-ynrel,s=2, c=t,
    vmin = t[0], vmax=t[-1], cmap=cm, alpha = 0.2, label='w/noise')
    a = plt.colorbar(sc, ax = ax[2,0])
    a.set_ticks([])
    ax[2,0].legend(prop={'size': 6})




    ax[1,1].plot(t[1:], residual)
    ax[1,1].set_xlabel('t')
    ax[1,1].set_ylabel(r'$||{\bf{z}}_{n+1}-H_{n+1} {\bf{x}}_{n+1}^{-}||$')
    ax[1,1].set_yscale('log')

    AVD = np.sum(np.sqrt((xnr-X[:,0])**2+( ynr-X[:,1])**2))/len(np.sqrt((xnr-X[:,0])**2+( ynr-X[:,1])**2))
    AVD1=np.sum(np.sqrt((xnr-xnrel)**2+( ynr-ynrel)**2))/len(np.sqrt((xnr-xnrel)**2+( ynr-ynrel)**2))
    ax[2,1].hist(np.sqrt((xnr-X[:,0])**2+( ynr-X[:,1])**2), bins = 50,density=True,
     label='filtered, AVG:{}'.format(round(AVD,2)), alpha=1)
    ax[2,1].hist(np.sqrt((xnr-xnrel)**2+( ynr-ynrel)**2),  bins = 50,density=True,
     label='w/noise, AVG:{}'.format(round(AVD1,2)), alpha=0.6)
    ax[2,1].set_xlim((0,250))
    ax[2,1].set_xlabel(r'$|{\bf{r}}_{exact}-{\bf{r}}_i|$')
    ax[2,1].legend(prop={'size': 6})
    ax[2,1].set_ylabel(r'PD')


    #plt.suptitle('x(0) = y(0) =1000')
    fig.tight_layout(pad=0.1)
    plt.show()

#narisi_stat_opt_filter_rel(X_rel,P_rel, cartesian_kontrola, relative_data, residuali_rel)
narisi_napake(P, P_rel)
