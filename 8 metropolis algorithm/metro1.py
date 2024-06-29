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
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence







def energija_stanja(alpha, stanje):
    E=0
    for i in range(0, len(stanje)-1):
        E += alpha*stanje[i] + 0.5*(stanje[i+1]- stanje[i])**2

    return E

def energija_stanjauk(alpha, stanje):
    E=0
    for i in range(0, len(stanje)-1):
        E += alpha*stanje[i] + 0.25*(stanje[i-1]- 2*stanje[i] + stanje[i+1])**2
    return(E)


def izracunaj(stanje, alpha, T,st_iteracij):
    dE = 0
    energije = np.array([energija_stanja(alpha, stanje)])
    ind_en = np.array([0])
    st_potez = 0

    for i in range(st_iteracij):
        ind = np.random.randint(1,16)
        sign =(-1)**np.random.randint(1,3)

        dh = stanje[ind] + sign

        if (dh <=0):

            dE = sign**2 - sign*(stanje[ind+1]-2*stanje[ind]+stanje[ind-1] - alpha)
            if dE<0:
                stanje[ind]= dh
                energije=np.append(energije, energije[-1]+ dE)
                ind_en = np.append(ind_en, i)
                st_potez+=1
            elif dE>=0:
                ksi =  np.random.uniform()
                if (ksi <= np.exp(-dE/T)):
                    energije=np.append(energije, energije[-1]+ dE)
                    ind_en = np.append(ind_en, i)
                    stanje[ind] = dh
                    st_potez+=1
                else:
                    None
        dE=0

    return(stanje, energije, ind_en, st_potez)

def izracunaj_uk(stanje, alpha, T,st_iteracij):
    dE = 0
    energije = np.array([energija_stanjauk(alpha, stanje)])
    ind_en = np.array([0])
    st_potez = 0

    for i in range(st_iteracij):
        ind = np.random.randint(1,16)
        sign =(-1)**np.random.randint(1,3)
        En_last = energija_stanjauk(alpha, stanje)
        dh = stanje[ind] + sign

        if (dh <=0):
            stanje[ind] = stanje[ind] + sign
            Ennova = energija_stanjauk(alpha, stanje)

            dE = Ennova - En_last
            if dE<0:
                energije=np.append(energije, Ennova)
                ind_en = np.append(ind_en, i)
                st_potez+=1
            elif dE>=0:
                ksi =  np.random.uniform()
                if (ksi <= np.exp(-dE/T)):
                    energije=np.append(energije, Ennova)
                    ind_en = np.append(ind_en, i)

                    st_potez+=1
                else:
                    stanje[ind] = stanje[ind] - sign

        dE=0

    return(stanje, energije, ind_en, st_potez)



zac_st = (-1)*np.random.randint(0,19,17)

zac_st[0]=zac_st[-1]=0

#stan = izracunaj(zac_st, 1, 0.1, 4*10**5)

def narisi_veriz():
    zac_st = (-1)*np.random.randint(0,19,17)

    zac_st[0]=zac_st[-1]=0
    T = np.logspace(-2,2,5)
    alpha = np.linspace(0.1, 1, 10)
    cmap = plt.get_cmap('viridis',5)




    for i in range(len(T)):
        stanuk = izracunaj_uk(zac_st, 0.1, T[i], 1*10**5)
        plt.plot([i for i in range(0,17)], stanuk[0], 'o--', color=cmap(i))

    norm = LogNorm(vmin = T.min(),
                              vmax = T.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ticks=np.logspace(-2,2,5),
                label='T')
    plt.title(r'$\alpha = 0.1, N = 10^5$')
    plt.legend()
    plt.xlabel('N-ti člen')
    plt.ylabel('h')
    plt.show()

    return None
#narisi_veriz()

def en_fja_alp_t():

    zac_st = (-1)*np.random.randint(0,50,17)

    zac_st[0]=zac_st[-1]=0
    T = np.logspace(-2,3,10)
    alpha = np.logspace(-1, 1, 10)
    En = np.array([])
    cmap = plt.get_cmap('jet',10)


    for k in range(len(alpha)):
        for i in range(len(T)):
            stan = izracunaj(zac_st, alpha[k], T[i], 1*10**5)


            En = np.append(En, stan[1][-1])


        plt.plot(T, En, 'o--', color=cmap(k))
        En = np.array([])

    norm = LogNorm(vmin = alpha.min(),
                              vmax = alpha.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm,
                label=r'$\alpha$')

    plt.title(r'$ N = 10^5$')
    plt.xscale('log')
    plt.xlabel('T')
    plt.ylabel(r'$E_{kon}$')
    #plt.yscale('log')
    plt.show()
#en_fja_alp_t()

def delez_potez():
    zac_st = (-1)*np.random.randint(0,50,17)

    zac_st[0]=zac_st[-1]=0
    T = np.logspace(-2,3,10)
    alpha = np.logspace(-1, 1, 10)
    En = np.array([])
    cmap = plt.get_cmap('jet',10)


    for k in range(len(alpha)):
        for i in range(len(T)):
            stan = izracunaj(zac_st, alpha[k], T[i], 1*10**5)


            En = np.append(En, stan[3])



        plt.plot(T, En/(1*10**5), 'o--', color=cmap(k))
        En = np.array([])

    norm = LogNorm(vmin = alpha.min(),
                              vmax = alpha.max())
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm,
                label=r'$\alpha$')

    plt.title(r'$ N = 10^5$')
    plt.xscale('log')
    plt.xlabel('T')
    plt.ylabel(r'delež sprejetih potez')
    #plt.yscale('log')
    plt.show()
#delez_potez()
def energija_fja_potez():
    zac_st = (-1)*np.random.randint(0,19,17)
    zac_st = np.zeros(17)
    zac_st[0]=zac_st[-1]=0
    konf = zac_st
    T =  [0.1,1 , 10]

    cmap = plt.get_cmap('jet',10)
    N = 1*10**6
    for i in range(len(T)):
        stan = izracunaj(zac_st, 1, T[i], N)


        plt.plot(stan[2], stan[1], label=r'T={}'.format(round(T[i],3)))



    plt.legend()
    plt.title(r'$\alpha= 1$')
    #plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('N')
    plt.ylabel('E')

    plt.show()
#energija_fja_potez()



############## 2.NALOGA #########


def energija(M, J, H):
    E = 0
    dimv = len(M)
    dims = len(M[0])
    for i in range(dimv):
        for j in range(dims):
                E += -J*M[i,j]*(M[(i+1)%dimv, j] + M[(i-1)%dimv,j] + M[i, (j+1)%dimv] + M[i,(j-1)%dimv])- H*M[i,j]

    return E/2

def nakljucna_matrika(N,M):
    M = np.random.randint(0,2,(N,M))
    M = np.where(M==0, -1, 1)
    return M
def lilmc(M,J,H,T):
    dimv = len(M)
    dims = len(M[0])
    En = energija(M, J,H)
    mag = np.sum(M)
    for i in range(500):

        a = np.random.randint(0, dimv)
        b = np.random.randint(0, dims)
        s =  M[a, b]
        nb = M[(a+1)%dimv,b] + M[a,(b+1)%dimv] + M[(a-1)%dims,b] + M[a,(b-1)%dims]
        cost = 2*s*nb

        if cost < 0:

            s *= -1
            En += cost
            mag +=2*s
        elif (np.random.uniform() < np.exp(-cost/T)):
            s *= -1
            mag +=2*s
            En+=cost
        M[a, b] = s
    return mag



def resitev(M,N_obratov, J, H, T, st_tock):
    dimv = len(M)
    dims = len(M[0])
    E_kon = np.array([energija(M, J, H)])


    mag = np.sum(M)
    dE = 0
    for k in range(N_obratov):
        i = np.random.randint(0, len(M))
        j =np.random.randint(0, len(M[0]))

        dE = 2*J*M[i,j]*(M[(i+1)%dimv, j] + M[(i-1)%dimv,j] + M[i, (j+1)%dimv] + M[i,(j-1)%dimv])+ 2*H*M[i,j]
        if dE <0:
            M[i,j] = (-1)*M[i,j]
            E_kon +=dE
        elif(np.random.uniform()< np.exp(-dE/T)):
            M[i,j] = M[i,j]*(-1)
            E_kon +=dE


    for k in range(st_tock):
        mag += abs(lilmc(M,J,H,T))

        #if k > indeksi[-1]:
        #    if abs(E_povp-E[-1])<100:
        #        print('resitev_najdena')
        #        break


    #c = (E_sum/n - E_av**2)/(dimv**2 * T**2)
    return M

#M_zac = nakljucna_matrika(120,120)
#Mres = resitev(M_zac, 2*10**5, 1,0,1)[0]
#vsota = 0
#for i in range(len(Mres)):
#    for j in range(len(Mres[0])):
#        vsota += Mres[i,j]
#print(vsota/120)

#plt.matshow(Mres[0])
#plt.show()
Nobr=2*10**5
N=50
T = 2.27
J=1
H=0.1

def narisi_konf(N,Nobr,J,H,T):
    M_zac = nakljucna_matrika(N,N)
    Mres = resitev(M_zac, Nobr, J,H,T,1)

    plt.matshow(Mres)
    plt.title('H=0.1, T ={}'.format(T))
    plt.show()

    return None

narisi_konf(N, Nobr, J,H,T)
#M_zac = nakljucna_matrika(N,N)
#Mres = resitev(M_zac, 2*10**5, 1,0,0.1,20)
#print(Mres[3]/(N**2*2))


def povp_en(N,Nobr,J,H):

    energije = np.array([])
    temp = np.linspace(1, 10, 50)

    for i in range(len(temp)):
        M_zac = nakljucna_matrika(N,N)
        en = resitev(M_zac, Nobr, J,H,temp[i], 50)
        energije = np.append(energije, en)



    return(energije)

#A = resitev(M_zac, Nobr, J,H,1)
#print(A[3])
#ind = A[2]
#E = A[1]
#plt.plot(np.linspace(1, 10, 50), povp_en(N, Nobr, J, H))
#plt.show()

def narisi_load():
    mat = np.loadtxt('/home/ziga/Desktop/FMF/magisterij/modelska_1/8_metropolisov_algoritem/podatki.txt')
    print(mat[:,5])
    H = np.linspace(0,1,5)
    for i in range(1,6):
        plt.plot(mat[:,0], mat[:,i], label='H={}'.format(round(H[i-1],2)))

    plt.xlabel('T')
    plt.legend()
    plt.grid()
    plt.title(r'$N=120, N_{iter} = 2 \  10^6 $')
    plt.ylabel(r'$\langle E \rangle$')
    plt.show()

#narisi_load()

#print(povp_en(N, N_obr, J,H))
