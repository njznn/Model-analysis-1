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
from scipy.stats import norm



############### 1 naloga ##############



def one_sim(beta,dt, N0):

    i=0
    while(N0>0):
        N0 -= np.random.poisson(beta*dt*N0)
        i+=1

    return(i*dt)

def onesimdata(beta, dt, N0):
    N = np.array([N0])
    T = np.array([0])
    i=0
    while(N0>0):
        N0 -= np.random.poisson(beta*dt*N0)

        i+=1
        N = np.append(N, N0)
        T = np.append(T, i*dt)
    return(T, N)

def onesimimproveddata(beta, dt, N0):
    N = np.array([N0])
    T = np.array([0])
    i=0
    while(N0>0):
        N0 = N0 -np.random.poisson(5*beta*dt*N0) + np.random.poisson(4*beta*dt*N0)

        i+=1
        N = np.append(N, N0)
        T = np.append(T, i*dt)
    return(T, N)

def onesimimproved(beta,dt, N0):
    i=0
    while(N0>0):
        N0 = N0 -np.random.poisson(5*beta*dt*N0) + np.random.poisson(4*beta*dt*N0)
        i+=1

    return(i*dt)


def narisipotek(st_iteracij, beta, dt, N0, model):

    t_av = 0
    for i in range(st_iteracij):
        if model==0:
            T,N = onesimdata(beta,dt,N0)
        else:
            T,N = onesimimproveddata(beta,dt,N0)
            plt.plot(T,N)
        t_av += T[-1]
        plt.plot(T,N)

    t_av = t_av/(st_iteracij)

    t = np.linspace(0,2*t_av, 500)

    plt.plot(t, N0*np.exp(-beta*t), 'black', linewidth=2, label='analiticna')
    #plt.plot(t, N0*np.exp(-beta*t), 'black', linewidth=2, label='analiticna')



    plt.title(r'$\beta={},$'.format(beta)+'dt={},'.format(dt)+'$N_{0}=$'+'{}'.format(N0))
    plt.xlabel('t')
    plt.ylim(-1)
    plt.ylabel('N')
    plt.legend()
    plt.show()



def narisihistogram(st_iteracij, beta, dt, N0):
    casi = np.array([])
    casi2 = np.array([])
    t_av = 0
    for i in range(st_iteracij):
        casi = np.append(casi,one_sim(beta, dt, N0))
        casi2 = np.append(casi2,onesimimproved(beta, dt, N0))


    t_av = np.sum(casi)/(st_iteracij)
    t_av2 = np.sum(casi2)/(st_iteracij)

    _,bins,_ =plt.hist(casi, 30,color='b',histtype = 'step', rwidth=0.5)
    _,bins,_ =plt.hist(casi2, 30,color='r',histtype = 'step', rwidth=0.5)
    #mu, sigma = norm.fit(casi)
    #best_fit_line = norm.pdf(bins, mu, sigma)
    #plt.plot(bins, best_fit_line)
    plt.xlabel('t')
    plt.ylabel('N')

    plt.axvline(x=t_av,color='b', linewidth=2, label=r'$\overline{t}_{osn}=$'+'{}'.format(round(t_av,2)))
    plt.axvline(x=t_av2,color='r', linewidth=2, label=r'$\overline{t}_{imp}=$'+'{}'.format(round(t_av2,2)))
    plt.title(r'$\beta={},$'.format(beta)+'dt={},'.format(dt)+'$N_{0}=$'+'{}'.format(N0)+ '$ \ N_{iter}$='+'{}'.format(st_iteracij))
    plt.legend()
    plt.show()
narisipotek( 100,1,1,25, 0)
#narisipotek( 100,1,0.01,250, 1)
#narisihistogram(10000, 1,0.7,250)

def av_sim(st_iteracij, beta, dt, N0):
    casi = np.array([])
    t_av = 0
    for i in range(st_iteracij):
        casi = np.append(casi,onesimimproved(beta, dt, N0))


    t_av = np.sum(casi)/(st_iteracij)
    return(t_av)

def odvisnostbetadt():

    dtar = np.logspace(-2,1,3)
    betaar= np.linspace(0.1,0.2,3)
    tar = np.zeros((len(dtar), len(betaar)))

    for i in range(len(dtar)):
        for j in range(len(betaar)):
            tar[i,j] = av_sim(1000, betaar[j],dtar[i],250)


    plt.contourf(dtar, betaar, np.transpose(tar))
    plt.ylabel(r'$\beta$')
    plt.xlabel('dt')
    plt.xscale('log')
    plt.colorbar(label=r'$\langle t \rangle$')
    plt.title(r'$N_{0} =  250$')
    plt.show()
#odvisnostbetadt()

def odvisnostNdt():
    dtar = np.logspace(-2,1,10)
    Nar= np.linspace(25,250,10)
    tar = np.zeros((len(dtar), len(Nar)))
    for i in range(len(dtar)):
        for j in range(len(Nar)):
            tar[i,j] = av_sim(1000, 0.1,dtar[i],Nar[j])


    plt.contourf(dtar, Nar, np.transpose(tar))
    plt.ylabel(r'$N_{0}$')
    plt.xlabel('dt')
    plt.xscale('log')
    plt.colorbar(label=r'$\langle t \rangle$')
    plt.title(r'$\beta =  0.1$')
    plt.show()


    return None
#odvisnostNdt()
def napake(st_iteracij, beta, dt, N0):

    odstop = np.zeros(120)
    st_krat = np.zeros(120)
    odstop2 = np.zeros(120)
    st_krat2 = np.zeros(120)
    Tar = np.array([i*dt for i in range(0, len(odstop))])
    for j in range(st_iteracij):
        T,res = onesimdata(beta, dt, N0)
        T2,res2 = onesimimproveddata(beta, dt, N0)
        for i in range(len(res)):
            st_krat[i]+=1
        for i in range(len(res2)):
            st_krat2[i]+=1
        analit = N0*np.exp(-beta*T)
        analit2 = N0*np.exp(-beta*T2)
        Nap=np.abs(analit-res)/analit
        Nap2=np.abs(analit2-res2)/analit2

        odstop +=np.pad(Nap, (0, np.abs(len(odstop)-len(Nap))))
        odstop2 +=np.pad(Nap2, (0, np.abs(len(odstop2)-len(Nap2))))


    prava = np.array([])
    prava2 = np.array([])
    for k in range(len(odstop)):
        if st_krat[k] !=0:
            prava = np.append(prava,odstop[k]/st_krat[k])
    for k in range(len(odstop2)):
        if st_krat2[k] !=0:
            prava2= np.append(prava2,odstop2[k]/st_krat2[k])


    plt.yscale('log')
    plt.plot(Tar[:len(prava)-1],prava[:-1], label='osnovni model')
    plt.plot(Tar[:len(prava2)-1],prava2[:-1], label='izboljšan model')
    plt.xlabel('t')
    plt.ylabel(r'$(N-N_{prava})/N_{prava}$')
    plt.title(r'$N_{0} = 25, \beta = 1, dt = 0.1$')
    plt.legend()
    #2plt.plot(T, N)
    plt.show()
#napake(100, 1,0.1,25)

############ 2.naloga ##################

def mat(betar, betas, dt, size):
    mat = np.zeros((size, size))
    for i in range(len(mat)):
        for j in range(len(mat)):
            if i==j:
                mat[i,j]=1-j*(betar+betas)*dt
            elif j== i+1:
                mat[i,j] = j*betas*dt
            elif i == j+1:
                mat[i,j] = betar*dt*j
    return mat
def det():
    velikm = np.linspace(20,600,10)
    velikm =[int(i) for i in velikm]
    dtar = np.logspace(-3,-7,10)

    matdet = np.zeros((len(dtar),len(velikm)))
    for i in range(len(dtar)):
        for j in range(len(velikm)):
            matdet[i,j] = LA.det(mat(4,5,dtar[i],velikm[j]))


    matdet = np.where(abs(matdet)<0.00001,0.00001, matdet )
    matdet = np.where(abs(matdet)>1,0.00001, matdet )
    print(matdet)
    plt.contourf(dtar, velikm, np.transpose(abs(matdet)), levels=np.logspace(-6,0,100),norm=LogNorm())
    cbar=plt.colorbar(format='%.0e', label=r'$det(M^{N\times N})$')
    cbar.set_ticks(np.logspace(-5,0,6))

    plt.ylabel('N')
    plt.xlabel('dt')
    plt.title(r'$\beta_s/\beta_r = 5/4$')

    plt.xscale('log')
    plt.show()

#det()
#print(LA.det(mat(4,5,0.00000001,50)))


def izracunaj_evolucijo_smrti(st_korakov, dt, betas, zac_stanje):

    matres = np.zeros((st_korakov, len(zac_stanje)))
    matres[0,:] =zac_stanje
    matrop = mat(0, betas, dt, len(zac_stanje))
    for i in range(1, st_korakov):
        matres[i,:] = LA.matrix_power(matrop, i).dot(zac_stanje)
        if matres[i,0] >0.999:
            matres = matres[:i+1]
            break;

    return matres

def izracunaj_evolucijo_rs(st_korakov, dt, betar,betas, zac_stanje):

    matres = np.zeros((st_korakov, len(zac_stanje)))
    matres[0,:] =zac_stanje
    matrop = mat(betar, betas, dt, len(zac_stanje))
    for i in range(1, st_korakov):
        matres[i,:] = matrop.dot(matres[i-1,:])
        if matres[i,0] >0.995:

            matres = matres[:i+1]
            break;

    return matres

def Snarisi(st_korakov, dt,betar, betas, zac_stanje):
    st_tock = len(zac_stanje)

    stanja = izracunaj_evolucijo_rs(st_korakov,dt,betar, betas, zac_st )
    stanja = np.where(stanja<10**(-4),10**(-4), stanja )


    t = np.array([i*dt for i in range(len(stanja))])
    n = np.linspace(0,st_tock-1,st_tock)
    fs=plt.contourf(t[::1000],n,np.transpose(stanja[::1000]), levels=np.logspace(-4,0,100),norm=LogNorm())
    cbar=plt.colorbar(format='%.0e')
    cbar.set_ticks(np.logspace(-4,0,5))
    plt.title(r'$N_{0} =$'+'{} '.format((st_tock-1)/2)+r'$\ \beta_s/\beta_r =$'+'{}'.format(betas/betar) + '$\ dt=$'+'{}'.format(dt) + r'$\ N_{iter}=$'+'{}'.format(st_korakov))
    plt.xlabel('t')
    plt.ylabel('N')
    plt.ylim((0, 50))
    #plt.xlim(left=6)


    plt.show()

zac_st = np.zeros(51)
zac_st[25] = 1
#Snarisi(1000000, 0.00001, 4,5, zac_st)
def prvi_mom(mat):
    firstmom = np.array([])

    for i in range(0,len(mat),1000):
        vsota = 0
        for j in range(len(mat[0])):
            vsota += j*mat[i,j]
        firstmom = np.append(firstmom, vsota)
    return firstmom

def drugi_mom(mat):
    firstmom = np.array([])

    for i in range(0,len(mat),1000):
        vsota = 0
        for j in range(len(mat[0])):
            vsota += (j**2)*mat[i,j]
        firstmom = np.append(firstmom, vsota)
    return firstmom

def momenti():
    betaeff =1
    dt = 0.00001
    zac_st = np.zeros(51)
    zac_st[25] = 1
    mat = izracunaj_evolucijo_rs(1000000, dt, 4,5, zac_st)
    zac_st = np.zeros(301)
    zac_st[250] = 1
    mat2 = izracunaj_evolucijo_rs(1000000, dt, 4,5, zac_st)
    firstmom = prvi_mom(mat)
    drugimom = drugi_mom(mat)
    sigmanum = np.sqrt(drugimom-firstmom**2)

    firstmom2 = prvi_mom(mat2)
    drugimom2 = drugi_mom(mat2)
    sigmanum2 = np.sqrt(drugimom2-firstmom2**2)




    t = np.array([i*dt for i in range(0,len(mat),1000)])
    ana = 25*np.exp(-betaeff*t)
    anavar = np.sqrt(-25*9*np.exp(-1*t)*(np.exp(-1*t)-1))

    plt.plot(t, np.abs(anavar-sigmanum), label=r'$N_0 = 25$')

    #plt.plot(t, np.abs(ana-firstmom), label=r'$N_0 = 25$')
    t = np.array([i*dt for i in range(0,len(mat2),1000)])
    anavar = np.sqrt(-250*9*np.exp(-1*t)*(np.exp(-1*t)-1))
    #ana = 250*np.exp(-betaeff*t)
    #plt.plot(t, np.abs(ana-firstmom2), label=r'$N_0 = 250$')
    plt.plot(t, np.abs(anavar-sigmanum2), label=r'$N_0 = 250$')
    plt.xlabel('t')
    plt.ylabel(r'$|\mu_2-\mu_{2 an}|$')
    plt.title(r'$RS, dt=0.00001, N_{iter}=1000000$')
    plt.legend()
    plt.yscale('log')
    plt.show()

#momenti()


################## 3 NALOGA #################
# izumrli: 0 so lisice, 1 zajci
def onesimfast(L0,Z0, bet, alp, dt):
    i = 0
    Zt = Z0
    Lt = L0
    x=0
    while(Zt>0 and Lt>0):
        Z = np.random.poisson(5*alp*Zt*dt) - \
        np.random.poisson(4*alp*Zt*dt) - np.random.poisson((alp/L0)*Zt*Lt*dt)
        L =  np.random.poisson(4*bet*Lt*dt) - \
        np.random.poisson(5*bet*Lt*dt) + np.random.poisson((alp/Z0)*Zt*Lt*dt)
        Zt+=Z
        Lt+=L
        i+=1
    if Zt > Lt:
        x = 0
    elif Lt>Zt:
        x = 1
    else:
        print('oba umrla hkrati')
    return(i*dt, x)

#print(onesimfast(50,200,1,1,0.01))

def onesimarr(L0,Z0, bet, alp, dt):
    i = 0
    Zt = Z0
    Lt = L0
    Zar = np.array([Z0])
    Lar = np.array([L0])
    while(Zt>0 and Lt>0):
        Z = np.random.poisson(5*alp*Zt*dt) - \
        np.random.poisson(4*alp*Zt*dt) - np.random.poisson((alp/L0)*Zt*Lt*dt)
        L =  np.random.poisson(4*bet*Lt*dt) - \
        np.random.poisson(5*bet*Lt*dt) + np.random.poisson((alp/Z0)*Zt*Lt*dt)
        Zt+=Z
        Lt+=L
        Zar = np.append(Zar, Zt)
        Lar = np.append(Lar, Lt)
        i+=1
    if Zt > Lt:
        x = 0
    elif Lt>Zt:
        x = 1
    else:
        print('oba umrla hkrati')
    return(Zar, Lar, i*dt, x)


#ar = onesimarr(50,200,1,1,0.001)

def narisifazni(st_slik, L0,Z0, bet, alp, dt):
    for i in range(st_slik):
        ar = onesimarr(L0, Z0, bet, alp, dt)
        plt.plot(ar[0], ar[1])
        plt.plot(Z0, L0, 'o', color='black')

    plt.xlabel('zajci')
    plt.ylabel('lisice')
    plt.title(r'$\beta/\alpha = $'+'{0}/{1}'.format(bet,alp) +'$\ dt=$'+'{}'.format(dt) )
    plt.show()

def simuliraj(st_iteracij,L0,Z0, bet, alp, dt ):
    koncn_cas = 0
    verj = 0
    histcas = np.array([])
    for i in range(st_iteracij):
        res = onesimfast(L0,Z0, bet, alp, dt)
        histcas = np.append(histcas, res[0])
        koncn_cas +=res[0]
        if res[1]==0:
            verj +=1

    verj = verj/st_iteracij
    koncn_cas = koncn_cas/st_iteracij
    return koncn_cas, verj, histcas

#res = simuliraj(1000, 50, 200, 0.7203,1,0.01)[1]
#print(res)
#plt.show()
#narisifazni(100, 50, 200, 1,3,0.01)
def odvisnoddt():
    arver = np.array([])

    dtar = np.logspace(-3,0,50)
    beta=np.array([0.5,1,2])
    for k in range(len(beta)):
        arver = np.array([])
        for i in range(len(dtar)):
            ver = simuliraj(500, 50, 200, beta[k],1,dtar[i])[0]

            arver = np.append(arver,ver)


        plt.plot(dtar, arver, label=r'$\beta/\alpha = $'+ \
         '{}'.format(beta[k]))

    #plt.fill_between(dtar, arver,0, color='b', alpha = 0.5, label='lisice')
    #plt.fill_between(dtar, arver,1, color='r', alpha=0.5, label='zajci')

    plt.xlabel('dt')
    plt.ylabel(r'$\overline{t}$')
    plt.xscale('log')
    plt.title(r'$\ Z_{0}=$'+\
    '{}'.format(200)+'$\ L_{0}=$'+'{}'.format(50)+'$\ N_{iter}=$' + '{}'.format(500))
    plt.legend()

    plt.show()

def odvisnostrazm():
    beta = np.logspace(-1,1,10)
    pl = np.array([])
    for i in range(len(beta)):
        ver = simuliraj(500, 50, 200, beta[i],1,0.01)[1]
        pl = np.append(pl, ver)
    plt.plot(beta, pl, label='lisice')
    plt.plot(beta, 1-pl, label='zajci')
    plt.xscale('log')
    plt.ylabel('P')
    plt.xlabel(r'$\beta/\alpha$')
    plt.title(r'$\ Z_{0}=$'+\
    '{}'.format(200)+'$\ L_{0}=$'+'{}'.format(50)+'$\ N_{iter}=$' + '{}'.format(1000))
    plt.legend()
    plt.show()
#odvisnostrazm()

def hist(st_iteracij,L0,Z0, bet, alp, dt):
    tz = np.array([])
    tl = np.array([])
    for i in range(st_iteracij):
        res = onesimfast(L0,Z0, bet, alp, dt)
        if res[1]==0:
            tl = np.append(tl, res[0])
        elif res[1]==1:
            tz = np.append(tz, res[0])

    t_avz = np.sum(tz)/(len(tz))
    t_avl = np.sum(tl)/(len(tl))
    print(len(tz), len(tl))

    _,bins,_ =plt.hist(tl, 100,color='b', alpha=0.5,rwidth=1)
    _,bins,_ =plt.hist(tz, 100,color='r', alpha=0.3,rwidth=1)
    #mu, sigma = norm.fit(casi)
    #best_fit_line = norm.pdf(bins, mu, sigma)
    #plt.plot(bins, best_fit_line)
    plt.xlabel('t')
    plt.ylabel('N')
    plt.xlim(right=50)
    pl=len(tl)/st_iteracij
    plt.axvline(x=t_avl,color='b', linewidth=2, label=r'$\overline{t}_{lisice}=$'+ \
    '{}'.format(round(t_avl,2)) + '$\ P_{lisice-death} =$'+'{}'.format(round(pl,3)))
    plt.axvline(x=t_avz,color='r', linewidth=2, label=r'$\overline{t}_{zajci}=$'+'{}'.format(round(t_avz,2)) +  \
    '$\ P_{zajci-death} =$'+'{}'.format(round(1-pl,3)))
    plt.title(r'$\beta/\alpha = $'+'{0}/{1}'.format(bet,alp)+'\ dt={}'.format(dt)+\
    '$\ Z_{0}=$'+'{}'.format(Z0)+'$\ L_{0}=$'+'{}'.format(L0)+ \
    '$ \ N_{iter}$='+'{}'.format(st_iteracij))
    plt.legend()
    plt.show()

#hist(5000, 50,200,0.715,1,0.01)

def todvelikosti():
    Z = np.linspace(50,250, 20)
    L = np.linspace(50,250,20)
    tar = np.zeros((len(Z), len(L)))
    for i in range(len(Z)):
        for j in range(len(L)):
            tav = simuliraj(1000, L[j], Z[i],0.7203,1,0.01)[0]
            tar[i,j] =tav



    plt.contourf(Z, L, np.transpose(tar))
    plt.colorbar(label=r'$\overline{t}$')
    plt.title(r'$\beta/\alpha = $'+'{0}'.format(0.7203)+'$\ dt={}$'.format(0.01)+\
    '$ \ N_{iter}$='+'{}'.format(1000))
    plt.xlabel('Z')
    plt.ylabel('L')

    plt.show()

#todvelikosti()



############# 4 naloga #############

def epidemfast(al, bet, gam, D0, B0, I0, dt):
    i = 0
    N = D0+B0+I0
    Dt = D0
    Bt = B0
    It = I0
    while(Bt>0 and Dt>=0 and It>=0):
        B = +np.random.poisson(al*Dt*Bt*dt) - \
        np.random.poisson(bet*Bt*dt)
        D =  -np.random.poisson(al*Dt*Bt*dt)+ \
        np.random.poisson(gam*It*dt)

        Dt+=D
        Bt+=B
        It =N-Dt-Bt
        i+=1


    return(i*dt)

def oneep(al, bet, gam, D0, B0, I0, dt):
    i = 0
    N = D0+B0+I0
    Dt = D0
    Bt = B0
    It = I0
    Dar = np.array([D0])
    Bar = np.array([B0])
    Iar = np.array([I0])
    while(Bt>0 and Dt>=0 and It>=0) :
        D =  -np.random.poisson(al*Dt*Bt*dt)+ \
        np.random.poisson(gam*It*dt)
        B = +np.random.poisson(al*Dt*Bt*dt) - \
        np.random.poisson(bet*Bt*dt)
        I =  np.random.poisson(bet*Bt*dt) - \
        np.random.poisson(gam*It*dt)
        Dt+=D
        Bt+=B
        It +=I
        It =N-Dt-Bt
        i+=1
        Dar = np.append(Dar, Dt)
        Bar = np.append(Bar, Bt)
        Iar = np.append(Iar, It)
        if i == 1000000:
            break;

    return(i*dt, Dar, Bar, Iar)

def simulirajep(st_iteracij,al, bet, gam, D0, B0, I0, dt ):
    koncn_cas = 0
    for i in range(st_iteracij):
        res = epidemfast(al, bet, gam, D0, B0, I0, dt )
        koncn_cas +=res

    koncn_cas = koncn_cas/st_iteracij
    return koncn_cas

def epiddt():
    arver = np.array([])

    dtar = np.logspace(-3,0,10)
    gamma=np.array([0.001,0.002, 0.003])
    for k in range(len(gamma)):
        arver = np.array([])
        for i in range(len(dtar)):
            ver = simulirajep(1000,0.005,0.05,gamma[k], 90, 3, 7, dtar[i])

            arver = np.append(arver,ver)


        plt.plot(dtar, arver, label=r'$\gamma ={}$'.format(gamma[k]))

    #plt.fill_between(dtar, arver,0, color='b', alpha = 0.5, label='lisice')
    #plt.fill_between(dtar, arver,1, color='r', alpha=0.5, label='zajci')

    plt.xlabel('dt')
    plt.ylabel(r'$\overline{t}$')
    plt.xscale('log')
    plt.title(r'$\alpha=$'+'{}'.format(0.005)+'$\ \beta=$'+'{}'.format(0.05)+'$\ N_{iter}=$' + '{}'.format(1000))
    plt.legend()

    plt.show()


#epiddt()

def narisipotek():
    dt = 1
    N = 300
    res = oneep(0.001,0.05,0.02,0.98*N,0.01*N,0.01*N, dt)
    #res2 = oneep(0.005,0.05,0.005,90,3,7, dt)
    t = np.array([i*dt for i in range(0, len(res[1]))])
    #t2 = np.array([i*dt for i in range(0, len(res2[1]))])
    plt.plot(t, res[1], 'b', label=r'Dovzetni')
    plt.plot(t, res[2], 'r', label='Bolni')
    plt.plot(t, res[3], 'g', label='Imuni')
    #plt.plot(t2, res2[1], 'b--', label=r'$Dovzetni, \gamma$')
    #plt.plot(t2, res2[2], 'r--', label='Bolni')
    #plt.plot(t2, res2[3], 'g--', label='Imuni')
    plt.title(r'$\alpha=$'+'{}'.format(0.005)+r'$\ \beta=$'+'{}'.format(0.05)+\
    '$\ \gamma=0.005$')
    plt.ylabel('N')
    plt.xlabel('t')
    plt.legend()
    plt.show()

narisipotek()

def narisihistogramep(st_iteracij,al, bet, D0, B0, I0, dt ):
    casi = np.array([])
    casi2 = np.array([])
    casi3 = np.array([])

    for i in range(st_iteracij):
        casi = np.append(casi,epidemfast(al, bet, 0.001, 90, 3, 7, dt ))
        casi2 = np.append(casi2,epidemfast(al, bet, 0.001, 180, 6, 14, dt ))
        casi3 = np.append(casi3,epidemfast(al, bet, 0.001, 270, 9, 21, dt ))


    t_av = np.sum(casi)/(st_iteracij)
    t_av2 = np.sum(casi2)/(st_iteracij)
    t_av3 = np.sum(casi3)/(st_iteracij)


    plt.hist(casi2,30,color='r', alpha=0.4)
    plt.hist(casi,30,color='b', alpha = 0.5)
    plt.hist(casi3,30,color='g', alpha = 0.6)
    #mu, sigma = norm.fit(casi)
    #best_fit_line = norm.pdf(bins, mu, sigma)
    #plt.plot(bins, best_fit_line)
    plt.xlabel(r'$t_{end}$')

    plt.ylabel('N')

    plt.axvline(x=t_av,color='b', linewidth=2, label=r'$N=100, \overline{t_{end}}=$'+'{}'.format(round(t_av,2)))
    plt.axvline(x=t_av2,color='r', linewidth=2, label=r'$N=200, \overline{t_{end}}=$'+'{}'.format(round(t_av2,2)))
    plt.axvline(x=t_av3,color='g', linewidth=2, label=r'$N=300, \overline{t_{end}}=$'+'{}'.format(round(t_av3,2)))
    plt.title(r'$\beta={},$'.format(bet)+'$\ \alpha={},$'.format(al)+'dt={},'.format(dt)+'$\gamma = 0.001$')
    plt.legend()
    plt.show()


#narisihistogramep(1000,0.005, 0.05, 90, 3, 7, 0.01)


def odvisnostodN(st_tock):
    td = np.array([])
    N = np.linspace(100, 10000, st_tock)
    for i in range(len(N)):
        t = simulirajep(1, 0.001,0.05,0.02,0.9*N[i],0.03*N[i],0.07*N[i], 1)
        td = np.append(td, t)

    plt.plot(N, td, 'o--')
    plt.xlabel('N')
    #plt.yscale('log')

    plt.ylabel(r'$\overline{t_{end}}$')
    plt.title(r'$\alpha=0.01/N \ \beta={}$'.format(0.05)+'$\ \gamma=0.1/N$'+'$\ dt={}$'.format(0.1))
    plt.show()

#odvisnostodN(50)
