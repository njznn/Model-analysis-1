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
from spectrum import *
from scipy.signal import chirp, find_peaks, peak_widths
from spectrum import CORRELATION
from scipy.linalg import toeplitz, solve_toeplitz


val2 = np.loadtxt(
"/home/ziga/Desktop/FMF/magisterij/modelska_1/12_linearna_napoved_in_max_entropija/val2.dat")
val3=np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/12_linearna_napoved_in_max_entropija/val3.dat")
co2 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/12_linearna_napoved_in_max_entropija/co2.dat")
luna = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/12_linearna_napoved_in_max_entropija/luna.dat", delimiter=' ', usecols=(1,2), skiprows=2)
wolf = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/12_linearna_napoved_in_max_entropija/Wolf_number.dat", usecols=(0,2))
borza = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/12_linearna_napoved_in_max_entropija/borza.dat")


wyear = wolf[:,0]
wolfnum = wolf[:,1]
letnica = np.linspace(1995,2001,len(luna[:,0]))
ra = luna[:,0]
dec = luna[:,1]
letnice = co2[:,0]
co2c = co2[:,1]


def PSD(podatki, veldat, okno):

    n = len(podatki)


    dx = 1/veldat
    freq = np.fft.rfftfreq(n-1, dx)
    Pk = np.array([])

    if okno == "none":
        DFT= np.fft.fft(podatki)
        w=n
    elif okno=="hamming":
        koef = np.hamming(n)
        DFT= np.fft.fft(podatki*koef)
        w = np.sum(koef**2)
    elif okno == "bartlett":
        koef = np.bartlett(n)
        DFT= np.fft.fft(podatki*koef)
        w = np.sum(koef**2)
    elif okno == "blackman":
        koef = np.blackman(n)
        DFT= np.fft.fft(podatki*koef)
        w = np.sum(koef**2)
    elif okno=='kaiser':
        koef = np.kaiser(n,0)
        DFT= np.fft.fft(podatki*koef)
        w = np.sum(koef**2)
    elif okno== "hanning":
        koef = np.hanning(n)
        DFT= np.fft.fft(podatki*koef)
        w = np.sum(koef**2)
    else:
        print("ni pravi vnos!")

    for i in range(1,int(n/2)-1, ):
        pk = (1/w)*(np.abs(DFT[i])**2 + np.abs(DFT[n-i])**2)/2
        Pk = np.append(Pk, pk)

    Pk = np.insert(Pk, 0, (1/w)*np.abs(DFT[0])**2)
    Pk = np.append(Pk, (1/w)*np.abs(DFT[int(n/2)])**2)

    return freq, Pk

def acf(sn, k):
    res=0
    N=len(sn)
    for i in range(N-k):
        res += sn[i]*sn[i+k]
    return res/(N-k)


def e_min(cor, ak):
    res = cor[0]
    for i in range(0,len(ak)):
        res += ak[i]*cor[i+1]

    return np.abs(res)

#VEDNO NAJPREJ ODŠTEJ TREND PRED RAČUNANJEM KORELACIJ IN YW SISTEMA
#TREND- RECIMO NIHANJE OKROG LINEARNE FUNKCIJE-ODŠTEJ LIN FUNKCIJO
def autocol_signal(sn,p):
    autocol = np.array([])

    for k in range(p+1):
        autocol = np.append(autocol, acf(sn,k))

    return(autocol)



def solve_youlewalker(correlations, p):
    mat = np.zeros((p,p))
    for diag in range(p):
        mat += np.diag(correlations[diag]*np.ones(p-diag), k=diag)
        if diag>0:
            mat += np.diag(correlations[diag]*np.ones(p-diag), k=-diag)



    ak = LA.solve(mat, -correlations[1:p+1])

    #r = CORRELATION(re, maxlags=p, norm='unbiased')
    #A, P, k = LEVINSON(correlations, allow_singularity=True)
    roots, ak = fixroots(ak, 'None')

    return ak

    #return ak


def mem_psd(podatki,p, st_tock_spektra):
    col = autocol_signal(podatki,p)
    ak = solve_youlewalker(col,p)
    w = np.linspace(0,np.pi,st_tock_spektra)
    epsilon=e_min(col, ak)
    P=np.array([])
    for om in w:
        vsota=0
        for k in range(1,len(ak)+1):
            vsota += ak[k-1]*np.exp(-1j*om*k)

        P = np.append(P,epsilon/(np.abs(1+vsota))**2)
    return w,P

def fixroots(ak, mode):
    koef = np.insert(ak,0,1)
    roots=np.roots(koef)


    for i in range(len(roots)):
        if np.abs(roots[i]) >1.001:
            if mode=='unit_circle':
                roots[i] = roots[i]/np.abs(roots[i])
            elif mode=='mirror':
                roots[i] = 1/roots[i]
                print('POL POPRAVLJEN')
            elif mode=='None':
                None
            else:
                print('vstavi pravi mode!')
                exit(1)

    akfix = np.poly(roots)[1:]
    return roots, akfix

def narisi_nicle(arr_of_arr_of_zero):


    for i in range(len(arr_of_arr_of_zero)):
        x = arr_of_arr_of_zero[i].real
        y = arr_of_arr_of_zero[i].imag

        plt.scatter(x,y,s=70,marker='+', label='p={}'.format(len(arr_of_arr_of_zero[i])))

    t = np.linspace(0,2*np.pi,100)
    plt.plot(np.cos(t),np.sin(t), 'r--')
    plt.legend()
    plt.grid()
    plt.axis("equal")
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.show()

def epsilon_fja_p(podatki):
    col = autocol_signal(podatki)
    p = np.array([i for i in range(5,50)])
    e_minn = np.array([])

    for i in range(len(p)):
        ak=solve_youlewalker(col, p[i])
        epsilon =e_min(col, ak)
        e_minn = np.append(e_minn, epsilon)

    plt.plot(p, e_minn, '-o')
    plt.yscale('log')
    plt.xlabel('p')
    plt.ylabel(r'$\epsilon_{min}$')
    plt.title('val3')
    #plt.yscale('log')
    plt.show()

def narisi_spekter(podatki,p1,p2,p3, samplesize):
    podatki=podatki[:samplesize]
    freq,Pfft = PSD(podatki, 512,'none')
    freq1,Pfft1 = PSD(podatki, 512,'blackman')

    P1 = mem_psd(podatki, p1, 1000)[1]
    P2 = mem_psd(podatki, p2, 1000)[1]
    P3 = mem_psd(podatki, p3, 1000)[1]

    nu = np.linspace(0,1,1000)
    nufft = np.linspace(0,1,256)

    fig, ax1 = plt.subplots()
    l, b, h, w = 0.75,0.7, .25, .2
    ax2 = fig.add_axes([l, b, w, h])
    ax1.plot(nufft, Pfft, label='FFT')
    ax2.plot(nufft, Pfft)
    ax1.plot(nufft,Pfft1, label='FFT+Blackman')
    ax2.plot(nufft,Pfft1)
    ax1.plot(nu,P1, label='MEM,p={}'.format(p1))
    ax2.plot(nu,P1)
    ax1.plot(nu,P2,label='MEM,p={}'.format(p2))
    ax2.plot(nu,P2)
    ax1.plot(nu,P3,label='MEM,p={}'.format(p3))
    ax2.plot(nu,P3)

    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_title('2.vrh')
    ax1.set_title('co2 - kvad. trend')
    ax2.set_xlim((0.14,0.17))
    ax2.set_ylim(bottom=0.1)
    ax1.legend(loc='upper center')
    ax1.set_ylabel(r'$PSD(\nu)$')
    ax1.set_ylim((0.0001))
    ax1.set_xlabel(r'$\nu$')
    plt.show()
narisi_spekter(val3, 5,10,20,512)
def FWHM_vrhov(podatki):
    wfft, Pfft =PSD(podatki, 512,'none')
    vv =2

    dx = 1/255
    peaksfft,_=find_peaks(Pfft,vv)
    fftpeakwidth = peak_widths(Pfft,peaksfft)[0]*dx
    wfft1, Pfft1 =PSD(podatki, 512,'blackman')
    peaksfft1,_=find_peaks(Pfft1,vv)
    fftpeakwidth1 = peak_widths(Pfft1,peaksfft1)[0]*dx
    p = np.linspace(8,20,13)
    matpeak = np.zeros((len(p), len(peaksfft)))
    for i in range(len(p)):
        w1,P1 = mem_psd(podatki, int(p[i]),512)
        peaks, _ = find_peaks(P1,vv)
        dxx=1/255
        sirine = peak_widths(P1,peaks)[0]*dxx
        matpeak[i,:] = sirine[:2]


    st_vrhov=len(fftpeakwidth)
    for i in range(st_vrhov):
        lines = plt.plot(p,matpeak[:,i], label='{}.vrh'.format(i+1))
        linecolor = lines[0].get_color()

        plt.plot(p,fftpeakwidth[i]*np.ones(len(p)),'--', color=linecolor)

    plt.text(19,0.0075,'-- FFT')
    plt.legend()
    #plt.yscale('log')

    plt.ylabel('FWHM')
    plt.xlabel('p')
    plt.title('val2')
    plt.show()
    return None
FWHM_vrhov(val2)
def FWHM_numberdata(podatki):
    datasize = np.array([64,128,256,512])
    st_vrhov=2
    mat = np.zeros((4,st_vrhov))
    mat_fft =  np.zeros((4,st_vrhov))
    for i in range(len(datasize)):

        w1,P1 = mem_psd(podatki[:datasize[i]], 15,512)
        dx=1/255
        peaks, _ = find_peaks(P1, 1)
        sirine = peak_widths(P1,peaks,rel_height=0.5)[0]*dx
        mat[i,:] = sirine[:st_vrhov]

        wfft, Pfft =PSD(podatki[:datasize[i]], 512,'none')

        dx = 1/len(wfft)


        peaksfft,_=find_peaks(Pfft,1)

        fftpeakwidth = peak_widths(Pfft,peaksfft,rel_height=0.5)[0]*dx

        mat_fft[i,:]=fftpeakwidth[:st_vrhov]


    for i in range(len(mat[0])):
        lines = plt.plot(datasize,mat[:,i], '-o',label='{}.vrh'.format(i+1))
        linecolor = lines[0].get_color()
        plt.plot(datasize,mat_fft[:,i],'--o', color=linecolor)
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('FWHM')
    plt.text(470,0.02,'-- FFT')
    plt.yscale('log')
    plt.title('val2')
    plt.show()

FWHM_numberdata(val2)
def locljivost_metode():
    t = np.linspace(0,1,512)
    dnu = np.linspace(10,1,10)
    print(dnu)

    Par = np.array([])


    for i in range(len(dnu)):
        s = sin(2*np.pi*100*t) + sin(2*np.pi*(100+dnu[i])*t)
        w,P=mem_psd(s,10,1000)
        plt.plot(w,P)
        plt.show()
        st_vrhov=1
        p=1
        while(True):
            w,P=mem_psd(s,p,1000)
            st_vrhov = len(find_peaks(P,0.1)[0])


            if st_vrhov==2:
                Par = np.append(Par, p)
                break
            elif st_vrhov>2:
                break

            p+=1
            print(p)
        continue


    plt.plot(dnu,Par, '-o')
    plt.xlabel(r'$\Delta \nu $')
    plt.ylabel(r'$p_{min}$')
    plt.title(r'$sin(2 \pi 100 t) + sin(2 \pi (100+\Delta \nu) t)$')
    plt.show()

#locljivost_metode()

def napovej(data, zacind, order):
    podatki = data[:zacind]
    for i in range(len(data)-(zacind)):
        #col = autocol_signal(podatki, order)
        #ak = solve_youlewalker(col, order)
        ak = aryule(data, order, norm='unbiased', allow_singularity=True)[0]
        sn = -np.sum(ak[::-1]*podatki[-order:len(podatki)])
        podatki = np.append(podatki, sn)

    return podatki

#locljivost_metode()
def narisi_napovedi(xdata,ydata, zacind,p1,p2,p3):
    res1 = napovej(ydata,zacind,p1)
    res2 = napovej(ydata, zacind, p2)
    res3 = napovej(ydata, zacind, p3)
    plt.plot(xdata[zacind-20:],ydata[zacind-20:], label='prava', color='black')
    plt.plot(xdata[zacind-20:],res1[zacind-20:], label='p={}'.format(p1), color='blue')
    plt.plot(xdata[zacind-20:],res2[zacind-20:], label='p={}'.format(p2), color='lime')
    plt.plot(xdata[zacind-20:],res3[zacind-20:], label='p={}'.format(p3), color='red')
    plt.xlabel('leto')
    plt.ylabel(r'$dec[°]$')
    plt.axvline(x=xdata[zacind], color='brown')
    plt.legend()
    #plt.title(r'$ $')
    plt.show()
    return None

def narisi_razlike(xdata,ydata, zacind,p1,p2,p3):
    res1 = napovej(ydata,zacind,p1)
    res2 = napovej(ydata, zacind, p2)
    res3 = napovej(ydata, zacind, p3)
    plt.plot(xdata[zacind:],np.abs(ydata[zacind:]-res1[zacind:]),  label='p={}'.format(p1), color='blue')
    plt.plot(xdata[zacind:],np.abs(ydata[zacind:]-res2[zacind:]),  label='p={}'.format(p2), color='green')
    plt.plot(xdata[zacind:],np.abs(ydata[zacind:]-res3[zacind:]),  label='p={}'.format(p3), color='red')
    plt.xlabel('leto')
    plt.ylabel('absolutna napaka')
    plt.yscale('log')
    plt.legend()
    #plt.title('')
    plt.show()

def napaka_od_p(xdata,ydata, zacind):
    #co2lin = odstej_trend(xdata, ydata, 'linearen')
    #co2kv = odstej_trend(xdata, ydata, 'kvadrat')
    povp_lin = np.array([])
    povp_kvad = np.array([])
    p = np.linspace(100,500,5)
    for i in range(len(p)):
        #reslin = napovej(co2lin,zacind,int(p[i]))
        #reskv = napovej(co2kv,zacind,int(p[i]))
        reslin = napovej(ydata,zacind,int(p[i]))
        povplin = np.sum(np.abs(ydata[zacind:]-reslin[zacind:]))/len(ydata[zacind:])
        #povpkv = np.sum(np.abs(co2kv[zacind:]-reskv[zacind:]))/len(ydata[zacind:])
        povp_lin = np.append(povp_lin, povplin)
        #povp_kvad = np.append(povp_kvad, povpkv)

    #plt.plot(p, povp_lin, '-o', label='linearen trend')
    #plt.plot(p, povp_kvad, '-o', label='kvadratni trend')
    plt.plot(p, povp_lin, '--o')
    plt.legend()
    #plt.title(r'deklinacija')
    #plt.yscale('log')
    plt.xlabel('p')
    plt.ylabel('povprečna abs. napaka')
    plt.show()






def odstej_trend(x,y, trend):
    if trend=='linearen':
        f = lambda x,a,b: a*x+b
        popt, pcov = curve_fit(f, x, y)
        return(y-f(x, *popt))

    elif trend=='kvadrat':
        g = lambda x,a,b,c: a*x**2 + b*x+ c
        popt1, pcov1 = curve_fit(g, x, y)
        return(y-g(x, *popt1))
    elif trend=='povprecje':
        povp = np.sum(y)/len(y)
        return (y-povp)

    return None
#locljivost_metode()
#FWHM_numberdata(val3)

#FWHM_vrhov(val3)
#narisi_spekter(val3[:512],7,9,15,512)

#col = autocol_signal(val3)
#ak=solve_youlewalker(col, 8)
#nicle, ak = fixroots(ak, 'mirror')
#arrnicle=np.array([np.array([0]),nicle], dtype=object)
#narisi_nicle(arrnicle)

#t = np.linspace(0,511,512)
#sig = np.sin(t/2)


#res = napovej(sig,256,5)



###############1 naloga ############
'''
t = np.linspace(0,1,512)
s = sin(2*np.pi*100*t) + sin(2*np.pi*(100+10)*t)
P=mem_psd(s,5,1000)[1]
w = np.linspace(0,256,1000)
plt.plot(w,P, label='p=5')
P=mem_psd(s,10,1000)[1]
plt.plot(w,P,label='p=10')
P=mem_psd(s,20,1000)[1]
plt.plot(w,P,label='p=20')
P=mem_psd(s,40,1000)[1]
plt.plot(w,P,label='p=40')
freq,Pfft = PSD(s, 512,'none')
plt.plot(freq, Pfft, label='FFT')
plt.yscale('log')
plt.xlim((95,115))
plt.ylim(bottom=0.001, top=1000)
plt.xlabel(r'$\nu$')
plt.ylabel('PSD')
plt.title(r'$sin(2 \pi \ 100 t) + sin(2 \pi \ 110 t)$')
plt.legend()
plt.show()
'''

#narisi_spekter(val2, 5,10,20,128)
'''
co2m = odstej_trend(letnice, co2c, 'kvadrat')
col = autocol_signal(co2m)
ak5=solve_youlewalker(col, 10)
ak10=solve_youlewalker(col, 20)
ak20 = solve_youlewalker(col, 30)
nicle5, ak5 = fixroots(ak5, 'mirror')
nicle10, ak10 = fixroots(ak10, 'mirror')
nicle20, ak20 = fixroots(ak20, 'mirror')
arrnicle=np.array([nicle5,nicle10,nicle20], dtype=object)
narisi_nicle(arrnicle)
'''
#epsilon_fja_p(val2[:256])
#FWHM_vrhov(val2[:256])
#FWHM_numberdata(val3)

#co2m = odstej_trend(letnice, co2c, 'linearen')
#narisi_spekter(co2m, 10,20,30,50)




###################### 2. naloga ##################
#t = np.linspace(0, 10, num=300)
#n = np.random.normal(scale=0.5, size=t.size)
#s = np.sin(10*t)
#y = np.sin(10*t) + n
#co2m = odstej_trend(letnice, co2c, 'linearen')
#narisi_napovedi(letnice,co2m,302,5,10,20)
#narisi_napovedi(t, y,150,10,15,30)
#wolfnumber = odstej_trend(wyear, wolfnum, 'povprecje')
#t = [i for i in range(len(borza))]
#rekt = odstej_trend(letnica, ra, 'povprecje')
#narisi_napovedi(wyear, wolfnumber,int(len(wolfnumber)/2) ,100,200,300)

#narisi_razlike(letnice,co2m,302,5,10,20)
#narisi_razlike(wyear, wolfnumber,int(len(wolfnumber)/2) ,100,200,300)
#napaka_od_p(letnice, co2c, 302)
#napaka_od_p(wyear, wolfnumber, int(len(wolfnumber)/2))
