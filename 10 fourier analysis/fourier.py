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




################ 1 NALOGA #############
dat2 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/val2.dat")
dat3 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/val3.dat")

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


def narisispekter(podatki, veldat):
    fig, ax1 = plt.subplots()
    l, b, h, w = 0.15, .13, .25, .2
    ax2 = fig.add_axes([l, b, w, h])
    res = PSD(podatki, veldat, 'hamming')
    ax1.plot(res[0], res[1], label='hamming')
    ax2.plot(res[0], res[1], label='hamming')
    res = PSD(podatki, veldat, 'none')
    ax1.plot(res[0], res[1], color='black', label='no window')
    ax2.plot(res[0], res[1], color='black',label='no window')
    res = PSD(podatki, veldat, 'blackman')
    ax1.plot(res[0], res[1], label='blackman')
    ax2.plot(res[0], res[1], label='blackman')
    res = PSD(podatki, veldat, 'bartlett')
    ax1.plot(res[0], res[1], alpha = 0.5, label='bartlett')
    ax2.plot(res[0], res[1], alpha = 0.5, label='bartlett')
    res = PSD(podatki, veldat, 'hanning')
    ax1.plot(res[0], res[1], alpha = 0.5, label="hanning")
    ax2.plot(res[0], res[1], alpha = 0.5, label="hanning")
    res = PSD(podatki, veldat, 'kaiser')
    ax1.plot(res[0], res[1], 'r',alpha = 0.3, label=r'kaiser,'+r'$\beta=0$')
    ax2.plot(res[0], res[1], 'r',alpha = 0.3, label=r'kaiser,'+r'$\beta=0$')
    #res = PSD(dat2, 512, 'kaiser')
    #plt.plot(res[0], res[1])



    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax2.set_xlim((48,52))
    ax2.tick_params(axis="x",direction="in", pad=-15)

    #plt.xlim((60,75))
    ax1.set_xlabel(r'$\nu$')
    ax1.set_ylabel(r'$PSD(\nu)$')
    ax1.legend()
    #ax2.set_xticks([])
    ax2.set_yticks([])
    ax1.set_title(r'val3.dat, N=256, padding')
    plt.show()
    return None

#narisispekter(np.pad(dat3[:256], 256, constant_values=0),512)



######################## 2 naloga ##################

s0 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/signal0.dat")
s1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/signal1.dat")
s2 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/signal2.dat")
s3 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/signal3.dat")

#plt.plot(signal0, label='signal0')
#plt.plot(signal1, label='signal1')
#plt.plot(signal2, alpha = 0.7, label='signal2')
#plt.plot(signal3, alpha=0.5, label='signal3')
#plt.xlabel('t')
#plt.ylabel('x(t)')
#plt.legend()
#plt.show()

def f(x, a, c):
        return a*x + c

def calc_psd(s0, s1, s2, s3):
    res0 = PSD(s1, 512, 'none')
    povp = np.sum(res0[1][70:])/len(res0[1][70:])
    plt.plot(res0[0],povp*np.ones(256),'--', color='b', label='s0, {:.2e}'.format(povp))
    plt.plot(res0[0], res0[1], 'b')

    #plt.plot(res0[0][:50], f(res0[0][:50], *popt), color='black')
    #print(popt)

    #res = PSD(s1, 512, 'none')
    #plt.plot(res[0], res[1], 'r')
    #povp = np.sum(res[1][70:])/len(res[1][70:])
    #plt.plot(res[0],povp*np.ones(257),'--', color='r', label='s1, {:.2e}'.format(povp))

    #res = PSD(s2, 512, 'none')
    #plt.plot(res[0], res[1], 'g')
    #povp = np.sum(res[1][70:])/len(res[1][70:])
    #plt.plot(res[0],povp*np.ones(257),'--', color='g', label='s2, {:.2e}'.format(povp))

    #res = PSD(s3, 512, 'none')
    #plt.plot(res[0], res[1], color='orange')
    #povp = np.sum(res[1][70:])/len(res[1][70:])
    #plt.plot(res[0],povp*np.ones(257),'--', color='orange',label='s3, {:.2e}'.format(povp))

    plt.yscale('log')
    plt.ylim((10**(-6)) )
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$PSD(\nu)$')
    #plt.title(r'$ocena \  |N(\nu)_{si}|^2 \ (črtkano)$')

    plt.legend()
    plt.show()

#calc_psd(s0, s1, s2, s3)

def getfilter(s1 ,zgmeja_fit):
    res0 = PSD(s1, 512, 'none')
    povp = np.sum(res0[1][70:])/len(res0[1][70:])
    plt.plot(res0[0],povp*np.ones(256),'--', color='green', label=r'$|N(\nu)|^2$')
    plt.plot(res0[0], res0[1], 'b', label=r'$PSD(\nu)$')
    popt, cov = curve_fit(f, res0[0][:zgmeja_fit], np.log(res0[1][:zgmeja_fit]))

    Sfit = popt[1]*np.exp(popt[0]*res0[0])

    #S = res0[1]-povp*np.ones(len(res0[0]))
    S = res0[1][:zgmeja_fit]-povp*np.ones(zgmeja_fit)
    S = np.append(S, np.zeros(len(res0[1])-zgmeja_fit))
    fifit =  Sfit/(povp + Sfit)
    ficut =  S/(povp + S)




    plt.plot(res0[0], Sfit, 'r', label=r'$|S_{fit}(\nu)|^2$')
    plt.plot(res0[0], fifit, label=r'$\Phi_{fit}$')
    plt.plot(res0[0],ficut, label=r'$\Phi_{cut}$')
    plt.yscale('log')
    plt.ylim((10**(-6)) )
    plt.xlabel(r'$\nu$')
    plt.ylabel(r'$PSD(\nu)$')
    plt.title(r'$signal1.dat$')

    plt.legend()
    #plt.show()
    plt.clf()
    return(ficut, fifit)

#getfilter(s1, 50)

def brezwiener(s, tau=16):
    C = np.fft.fft(s)
    t = np.linspace(-255,256,512)
    r = (1/(2*tau)) * np.exp(-np.abs(t)/tau)
    r = np.fft.fftshift(r)
    R = np.fft.fft(r)


    U = C/R


    u = np.real(np.fft.ifft(U))
    return u

def brezwiener_ostalifiltri(podatki, okno,tau=16, kaiserbeta=0):
    n = len(podatki)
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
        koef = np.kaiser(n,kaiserbeta)
        DFT= np.fft.fft(podatki*koef)
        w = np.sum(koef**2)
    elif okno== "hanning":
        koef = np.hanning(n)
        DFT= np.fft.fft(podatki*koef)
        w = np.sum(koef**2)
    else:
        print("ni pravi vnos!")

    t = np.linspace(-255,256,512)
    r = (1/(2*tau)) * np.exp(-np.abs(t)/tau)
    r = np.fft.fftshift(r)
    R = np.fft.fft(r)


    U = DFT/R


    u = np.real(np.fft.ifft(U))
    return u

#res = brezwiener(s0)
#plt.xlabel('t')
#plt.ylabel('u')
#plt.title('signal0.dat')
#plt.plot(res, label='deconvoluted signal')
#plt.plot(s0, label='signal')
#plt.legend()
#plt.show()



def wiener_deconvolution(s, fi):

    C =np.fft.fft(s)


    t = np.linspace(-255,256,512)
    r = (1/(2*16)) * np.exp(-np.abs(t)/16)
    r = np.fft.fftshift(r)
    #plt.plot(r)
    #plt.show()
    R = np.fft.fft(r)
    #plt.plot(R)
    #plt.show()
    fi = np.append(fi, fi[::-1])
    U = C*fi/R
    u = np.fft.ifft(U)
    return u

#ficut, fifit=  getfilter(s3, 15)
#s1cut = wiener_deconvolution(s3, ficut)
#s1fit = wiener_deconvolution(s3, fifit)

def narisi(dekon,dekon2, dek1):
    plt.plot(dek1, label='signal0')
    plt.plot(dekon, label=r'signal3-$S_{cut}$')
    plt.plot(dekon2, label=r'signal3-$S_{fit}$')
    plt.xlabel('t')
    plt.legend()
    plt.title(r'$N_{fit/cut}=15$')
    plt.show()
#narisi(s1cut,s1fit, orig)
orig = brezwiener(s0)
orig2 = brezwiener_ostalifiltri(s0,"bartlett")

def narisi_razliko(s1cut, s1fit, s0):
    t = np.linspace(-255,256,512)
    r = (1/(2*16)) * np.exp(-np.abs(t)/16)
    convoledcut = np.convolve(s1cut,r, mode='same')
    convoledfit = np.convolve(s1fit,r, mode='same')


    #plt.plot(np.abs(convoledcut-s0), label=r'$\tilde{c}_i=\tilde{c}_{cut}$')
    #plt.plot(np.abs(convoledfit-s0), label=r'$\tilde{c}_i=\tilde{c}_{fit}$')
    plt.legend()
    plt.title(r'$signal3, N_{fit/cut}=50$')
    plt.xlabel('t')
    plt.ylabel(r'$|c(t)-\tilde{c}_{i}(t)|$')
    plt.show()



def narisi_razliko_okenske(si,s0):
    t = np.linspace(-255,256,512)
    r = (1/(2*16)) * np.exp(-np.abs(t)/16)
    convoledbartlett = np.convolve(brezwiener_ostalifiltri(si,"bartlett"),r, mode='same')
    convoledhanning = np.convolve(brezwiener_ostalifiltri(si,"hanning"),r, mode='same')
    convoledhamming = np.convolve(brezwiener_ostalifiltri(si,"hamming"),r, mode='same')
    convoledblackman = np.convolve(brezwiener_ostalifiltri(si,"blackman"),r, mode='same')
    convoledkaiser = np.convolve(brezwiener_ostalifiltri(si,"kaiser"),r, mode='same')
    convoledkaiser50 = np.convolve(brezwiener_ostalifiltri(si,"kaiser", 16, 50),r, mode='same')


    plt.plot(np.abs(convoledhamming-s0), label=r'$\tilde{c}_i=\tilde{c}_{hamming}$')
    plt.plot(np.abs(convoledhanning-s0), label=r'$\tilde{c}_i=\tilde{c}_{hanning}$')
    plt.plot(np.abs(convoledbartlett-s0), label=r'$\tilde{c}_i=\tilde{c}_{bartlett}$')
    plt.plot(np.abs(convoledkaiser-s0), label=r'$\tilde{c}_i=\tilde{c}_{kaiser, \beta=0}$')
    plt.plot(np.abs(convoledblackman-s0), label=r'$\tilde{c}_i=\tilde{c}_{blackman}$')
    plt.plot(np.abs(convoledkaiser50-s0), label=r'$\tilde{c}_i=\tilde{c}_{kaiser, \beta=50}$')
    plt.legend()
    plt.title(r'$signal3$')
    plt.xlabel('t')
    plt.yscale('log')
    plt.ylabel(r'$|c(t)-\tilde{c}_{i}(t)|$')
    plt.show()
#narisi_razliko_okenske(s3, s0)

################# 3.NALOGA #################

from PIL import Image

image = Image.open("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/lena_slike/kernel1.pgm")
image1 = Image.open("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/lena_slike/kernel2.pgm")
image2 = Image.open("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/lena_slike/kernel3.pgm")
image3 = Image.open("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/lena_slike/lena_k1_n0.pgm")
image4 = Image.open("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/lena_slike/lena_k2_n4.pgm")
image5 = Image.open("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/lena_slike/lena_k2_n8.pgm")
image6 = Image.open("/home/ziga/Desktop/FMF/magisterij/modelska_1/10_fourier_analiza/lena_slike/lena.ascii.pgm")
k1=np.array(image)
k2 = np.array(image1)
k3 = np.array(image2)
lena_k2_n0 = np.array(image3)
lena_k2_n4 = np.array(image4)
lena_k1_n8 = np.array(image5)
prava = np.array(image6)


def shift_matrix(mat):

    for i in range(len(mat)):
        mat[i] = np.fft.fftshift(mat[i])


    for j in range(len(mat[0])):
        mat[:, j] = np.fft.fftshift(mat[:,j])
    return mat

def psd1d(DFT):
    Pk = np.array([])
    n = len(DFT)
    for i in range(1,int(n/2)-1, ):
        pk = (np.abs(DFT[i])**2 + np.abs(DFT[n-i])**2)/2
        Pk = np.append(Pk, pk)

    Pk = np.insert(Pk, 0, np.abs(DFT[0])**2)
    Pk = np.append(Pk, np.abs(DFT[int(n/2)])**2)

    return Pk

def psd2d(ftmat):
    psd = np.zeros((int(len(ftmat)/2), int(len(ftmat)/2)))

    for k in range(len(psd)):
        psd[k,:] = psd1d(ftmat[k,:])

    return psd


#k2sf = shift_matrix(k2)
#k1sf = shift_matrix(k1)
#k3sf = shift_matrix(k3)

#ftk1 = np.fft.fft2(k1sf)
#ftk2 = np.fft.fft2(k2sf)
#ftk3 = np.fft.fft2(k3sf)


#mat =psd2d(ftlena)
#plt.imshow(psd2d(ftlena), norm=LogNorm())
#plt.colorbar(label=r'$|K(\nu_x, \nu_y)|^2$')
#plt.title('kernel3')
#plt.imshow(np.abs(ftk2), norm=LogNorm())
#plt.show()
#plt.clf()

def get_aver_noise(slika, xmin1, xmax1, ymin1, ymax1, xmin2,xmax2, ymin2,ymax2):
    ftlena = np.fft.fft2(slika)
    ftmat =psd2d(ftlena)
    oneav = np.sum(ftmat[xmin1:xmax1, ymin1:ymax1])/(np.abs(xmax1-xmin1)*np.abs(ymax1-ymin1))
    othav =  np.sum(ftmat[xmin2:xmax2, ymin2:ymax2])/(np.abs(xmax2-xmin2)*np.abs(ymax2-ymin2))
    aver = (oneav+othav)/2


    fig, ax = plt.subplots()

    im = plt.imshow(ftmat, norm = LogNorm())
    rect =patches.Rectangle((xmin1, ymin1), xmax1-xmin1, ymax1-ymin1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    rect =patches.Rectangle((xmin2, ymin2), xmax2-xmin2, ymax2-ymin2, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    fig.colorbar(im)
    plt.show()

    return aver, ftmat


#print(get_aver_noise(lena_k1_n0, 180,250,100,250,50,250,200,250))

def deconvolution_wo_noise(pic, kernel):

    kernelsf = shift_matrix(kernel)
    ftk = np.fft.fft2(kernelsf)
    ftpic = np.fft.fft2(pic)

    U = ftpic/ftk
    u = np.fft.ifft2(U)
    u = np.real(u)
    u = np.where(u <0, 0, u)
    u = np.where(u >255,255 , u)
    return u


def deconvolution_w_noise(pic, kernel, pad):

    kernel = np.pad(kernel, (pad,pad), 'constant', constant_values=0)
    kernelsf = shift_matrix(kernel)
    ftk = np.fft.fft2(kernelsf)

    plt.imshow(np.abs(ftk), norm=LogNorm())
    plt.show()
    avpic = np.sum(pic)/(len(pic)**2)
    pic = np.pad(pic, (pad,pad), 'constant', constant_values=avpic)
    window1d = np.abs(np.blackman(512))
    window2d = np.sqrt(np.outer(window1d,window1d))
    ftpic = np.fft.fft2(pic)
    N,C=get_aver_noise(pic,20,70,20,60,15,50,15,50)



    fioneblock = (C-N)/C
    fioneblock = np.where(fioneblock<0, 0, fioneblock)
    fi_d =fioneblock[:, ::-1]
    fi_ls = fioneblock[::-1, :]
    fi_ds = fioneblock[::-1,::-1]
    fi = np.block([[fioneblock, fi_d],[fi_ls, fi_ds]])

    fig, ax = plt.subplots()

    im = plt.imshow(fi)
    fig.colorbar(im)
    plt.show()


    U = ftpic*fi/ftk

    u = np.fft.ifft2(U)
    u = np.real(u)
    u =  np.where(u<0,  0, u)
    u = np.where(u>255, 255, u)


    u = u/np.max(u)
    u = u[pad:len(u)-pad, pad:len(u[0])-pad]
    return u


imag = deconvolution_w_noise(lena_k2_n0, k1,0)

imag = (imag*255 ).astype(np.uint8)

#plt.imshow(imag, cmap='gray')
#plt.show()
#print(ssim(prava, imag))
img = Image.fromarray(imag,  mode='L')
img.show()

def deconvolution_w_noise_2(pic, kernel, pad, x,y):

    kernel = np.pad(kernel, (pad,pad), 'constant', constant_values=0)
    kernelsf = shift_matrix(kernel)
    ftk = np.fft.fft2(kernelsf)

    avpic = np.sum(pic)/(len(pic)**2)
    pic = np.pad(pic, (pad,pad), 'constant', constant_values=avpic)
    window1d = np.abs(np.blackman(512))
    window2d = np.sqrt(np.outer(window1d,window1d))
    ftpic = np.fft.fft2(pic*window2d)
    N,C=get_aver_noise(pic,x,x+25,y,y+25,x,x+25,y,y+25)



    fioneblock = (C-N)/C
    fioneblock = np.where(fioneblock<0, 0, fioneblock)
    fi_d =fioneblock[:, ::-1]
    fi_ls = fioneblock[::-1, :]
    fi_ds = fioneblock[::-1,::-1]
    fi = np.block([[fioneblock, fi_d],[fi_ls, fi_ds]])



    U = ftpic*fi/ftk

    u = np.fft.ifft2(U)
    u = np.real(u)

    uavg = np.sum(np.abs(u))/(len(u)**2)
    u =  np.where(u<0,  0, u)
    u = np.where(u>255, 255, u)
    u = u/np.max(u)
    u = u[pad:len(u)-pad, pad:len(u[0])-pad]
    return u

def poisci_pravo(prava, pic, pad, kernel):
    razl = 10
    ind = 0
    realssim = 0

    for k in range(0,int(len(pic)/2)-25,1):


        u = deconvolution_w_noise_2(pic, kernel, pad,k, k)*255
        ssim1 = ssim(prava, u)


        if np.abs(1-ssim1) < razl:
            razl = 1-ssim1
            ind = k





    print(1-razl)
    imag = deconvolution_w_noise_2(pic, kernel, pad, ind, ind)
    imag = (imag*255).astype(np.uint8)
    img = Image.fromarray(imag,  mode='L')
    img.show()

#poisci_pravo(prava, lena_k2_n0,0, k2)
