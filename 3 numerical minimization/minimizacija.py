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




#prvi naboj dam na vrh krogle
#p je array iskanih paramtrov, naboji je array nabojev
#print(LA.norm((np.array([1,2,3])-np.array([1,1,0]))))
for i in range(6,2):
    print(i)


def energija_monopol(p, naboji):
    en = 0;
    for i in range(0,int(len(p)/2), 1):
        r0 = np.array([0,0,1])
        ri = np.array([np.cos(p[2*i])*np.sin(p[2*i+1]),np.sin(p[2*i])*np.sin(p[2*i+1]), np.cos(p[2*i+1])])
        en += (naboji[0]*naboji[i])/(LA.norm((ri-r0)))
        for j in range(0,int(len(p)/2), 1):
            if (j>i):
                rj = np.array([np.cos(p[2*j])*np.sin(p[2*j+1]),np.sin(p[2*j])*np.sin(p[2*j+1]), np.cos(p[2*j+1])])
                en += (naboji[i]*naboji[j])/(LA.norm((ri-rj)))
    return(en)



def nakljucni_p0(st_neznanih_tock):
    p0 = np.zeros(int(st_neznanih_tock*2))
    p0[0::2] = np.random.random_sample(st_neznanih_tock)*2*np.pi
    p0[1::4] = np.random.random_sample(int(st_neznanih_tock/2))*np.pi
    p0[3::4] = np.pi*np.tile([1,0], int(st_neznanih_tock/4))
    return p0



naboji = np.ones(8);


def minimiziraj(st_nakljucnih_p0, naboji, metoda):
    nakljucni_p0(2*(len(naboji)-1))
    res_1= minimize(energija_monopol, p0,(naboji), method=metoda, tol=1e-6)
    resitev = res_1.x
    vrednost_E = res_1.fun
    for i in range(st_nakljucnih_p0-1):

        res = minimize(energija_monopol, p0,(naboji), method=metoda, tol=1e-6, options={"maxiter":maxiter})
        if (res.success == True and res.fun< vrednost_E):
            resitev = res.x
            res_1 = res

    return resitev, res_1


#res=minimiziraj(1, naboji, "Nelder-Mead", 10000, nakljucni_p0(len(naboji)-1))

def cas_analiza():
    cas_ameba = np.array([])
    cas_powell = np.array([])
    cas_bfgs = np.array([])
    cas_cg = np.array([])
    cas_newton_cg = np.array([])

    for st_nab in  range(2,30):
        naboji = np.ones(st_nab)
        p0 = nakljucni_p0(len(naboji)-1)
        sta = time.time()
        minimiziraj(1, naboji, "Nelder-Mead", 5000, p0)
        enda=time.time()
        cas_ameba= np.append(cas_ameba,enda-sta)
        sta2 = time.time()
        minimiziraj(1, naboji, "Powell", 5000, p0)
        enda2=time.time()
        cas_powell= np.append(cas_powell,enda2-sta2)
        sta4 = time.time()
        minimiziraj(1, naboji, "BFGS", 5000, p0)
        enda4=time.time()
        cas_bfgs= np.append(cas_bfgs,enda4-sta4)
        if (st_nab<21):
            sta3 = time.time()
            minimiziraj(1, naboji, "CG", 5000, p0)
            enda3=time.time()
            cas_cg= np.append(cas_cg,enda3-sta3)
        #sta5 = time.time()
        #minimiziraj(1, naboji, "Newton-CG", 10000, p0)
        #enda5=time.time()
        #cas_newton_cg= np.append(cas_newton_cg,enda5-sta5)

    plt.plot([i for i in range(2,30)], cas_ameba, 'o--', label='Nelder-Mead')
    plt.plot([i for i in range(2,30)], cas_powell, 'o--',label='Powell')
    plt.plot([i for i in range(2,21)], cas_cg,'o--',label='CG' )
    plt.plot([i for i in range(2,30)], cas_bfgs,'o--', label='BFGS')
    plt.xlabel('število nabojev')
    plt.ylabel('čas [s]')
    plt.legend()
    #plt.plot([i for i in range(2,51)], cas_newton_cg)
    plt.show()
#cas_analiza()

def analiza_natancnost():
    prave_energije = np.array([0.500000000,1.732050808, 3.674234614,6.474691495,9.985281374, 14.452977414,19.675287861,25.759986531,32.716949460,
    40.596450510,49.165253058,58.853230612, 69.306363297, 80.670244114,92.911655302,106.050404829,120.084467447,135.089467557,150.881568334])

    en_ameba = np.array([])
    en_powell = np.array([])
    en_cg = np.array([])
    en_bfgs = np.array([])
    en_bfgs_b= np.array([])
    for i in range(2,21):
        naboji = np.ones(i)
        p0 = nakljucni_p0(len(naboji)-1)
        en_ameba = np.append(en_ameba, minimiziraj(7, naboji, "Nelder-Mead", 10000, p0)[1].fun)
        en_powell = np.append(en_powell, minimiziraj(7, naboji, "Powell", 10000, p0)[1].fun)
        en_cg = np.append(en_cg, minimiziraj(3, naboji, "CG", 5000, p0)[1].fun)
        en_bfgs = np.append(en_bfgs, minimiziraj(7, naboji, "BFGS", 5000, p0)[1].fun)
        en_bfgs_b = np.append(en_bfgs_b, minimiziraj(7, naboji, "L-BFGS-B", 5000, p0)[1].fun)

    en_ameba = np.divide(np.abs(en_ameba-prave_energije), prave_energije)
    en_powell = np.divide(np.abs(en_powell-prave_energije), prave_energije)
    en_cg = np.divide(np.abs(en_cg-prave_energije), prave_energije)
    en_bfgs = np.divide(np.abs(en_bfgs-prave_energije), prave_energije)
    en_bfgs_b = np.divide(np.abs(en_bfgs_b-prave_energije), prave_energije)

    plt.plot([i for i in range(2,21)], en_ameba, 'o--', label='Nelder-Mead')
    plt.plot([i for i in range(2,21)], en_powell, 'o--',label='Powell')
    plt.plot([i for i in range(2,21)], en_cg,'o--',label='CG' )
    plt.plot([i for i in range(2,21)], en_bfgs,'o--', label='BFGS')
    plt.plot([i for i in range(2,21)], en_bfgs_b,'o--', label='L-BFGS-B')
    plt.xlabel('število nabojev')
    plt.ylabel(r'($(E_{method} - E) /E$')
    plt.yscale("log")
    plt.legend()
    plt.show()
#analiza_natancnost()
def tocka(r,phi,theta):
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

def energija_dipola(p, naboj):
    en = 0
    for i in range(0,int(len(p)/4), 1):
        r0 = np.array([0,0,1])
        d0 = np.array([0,0,1])*0.1
        ri = np.array([np.cos(p[4*i])*np.sin(p[4*i+1]),np.sin(p[4*i])*np.sin(p[4*i+1]), np.cos(p[4*i+1])])
        di = 0.1*np.array([np.cos(p[4*i+2])*np.sin(p[4*i+3]),np.sin(p[4*i+2])*np.sin(p[4*i+3]), np.cos(p[4*i+3])])
        en += ((naboj[0]*naboj[i])*np.dot(d0,di))/(LA.norm((r0-ri))**3) -3*(naboj[i]*naboj[0]*np.dot(di, (ri-r0))* np.dot(d0, (ri-r0)))/(LA.norm((r0-ri))**5)
        for j in range(0,int(len(p)/4), 1):
            if (j>i):
                rj = np.array([np.cos(p[4*j])*np.sin(p[4*j+1]),np.sin(p[4*j])*np.sin(p[4*j+1]), np.cos(p[4*j+1])])
                dj = 0.1*np.array([np.cos(p[4*j+2])*np.sin(p[4*j+3]),np.sin(p[4*j+2])*np.sin(p[4*j+3]), np.cos(p[4*j+3])])
                en +=  ((naboj[j]*naboj[i])*np.dot(dj,di))/(LA.norm((rj-ri))**3) -3*(naboj[i]*naboj[j]*np.dot(di, (ri-rj))* np.dot(dj, (ri-rj)))/(LA.norm((rj-ri))**5)
                #en += ((naboj[j]*naboj[i])*np.dot(dj,di) - 3*(naboj[j]*np.dot(dj, (ri-rj))*(naboj[i]*np.dot(di,ri) - naboj[i]*np.dot(di, rj))))/(LA.norm((rj-ri))**3)
    return(en)

 #za dipol je 4 2 naboja


def min_dipol():
    naboji = np.array([1,1,1,1,1])
    p0 = nakljucni_p0(8)
    res_1=minimize(energija_dipola, p0,(naboji), method='BFGS', tol=1e-6)
    resitev = res_1.x

    vrednost_E = res_1.fun
    for i in range(200):
        p0 = nakljucni_p0(8)
        res = minimize(energija_dipola, p0,(naboji), method='BFGS', tol=1e-6)
        if (np.abs(res.fun)< np.abs(vrednost_E)):
            resitev = res.x
            res_1 = res
            vrednost_E = res.fun
            print('ja')

    print(vrednost_E)

    return resitev, res_1







def narisi_monopol(arr_t, res):
    #sfera
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0*pi:100j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)

    xx = np.cos(arr_t[0::4])*np.sin(arr_t[1::4])
    dx = np.cos(arr_t[2::4])*np.sin(arr_t[3::4])
    yy = np.sin(arr_t[0::4])*np.sin(arr_t[1::4])
    dy = np.sin(arr_t[2::4])*np.sin(arr_t[3::4])
    zz = np.cos(arr_t[1::4])
    dz =  np.cos(arr_t[3::4])
    xx = np.append(xx, 0)
    yy = np.append(yy, 0)
    zz = np.append(zz,1)
    dx = np.append(dx, 0)
    dy = np.append(dy, 0)
    dz = np.append(dz,1)


    matrika_tock = np.zeros((len(xx),3))
    matrika_tock[:,0] = xx
    matrika_tock[:,1]=yy
    matrika_tock[: ,2] = zz
    vertices = np.hstack((np.ones((len(xx),1)), matrika_tock))
    mat = pcdd.Matrix(vertices, linear=False, number_type="fraction")
    mat.rep_type = pcdd.RepType.GENERATOR
    poly = pcdd.Polyhedron(mat)

    # get the adjacent vertices of each vertex
    adjacencies = [list(x) for x in poly.get_input_adjacency()]

    # store the edges in a matrix (giving the indices of the points)
    edges = [None]*(len(xx)-1)
    for i,indices in enumerate(adjacencies[:-1]):
        indices = list(filter(lambda x: x>i, indices))
        l = len(indices)
        col1 = np.full((l, 1), i)
        indices = np.reshape(indices, (l, 1))
        edges[i] = np.hstack((col1, indices))
    Edges = np.vstack(tuple(edges))
    start = matrika_tock[Edges[:,0]]
    end = matrika_tock[Edges[:,1]]







    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    for i in range(len(xx)+4):
        ax.plot(
            [start[i,0], end[i,0]],
            [start[i,1], end[i,1]],
            [start[i,2], end[i,2]],
            "blue"
        )
    #ax.plot(xx, yy, zz)ž

    ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='gainsboro', alpha=0.3, linewidth=0)

    ax.scatter(xx,yy,zz,color="r",s=30, label='n ={}'.format(len(xx)))
    #ax.scatter(0,0,1,color="k",s=20)
    ax.quiver(xx,yy,zz, dx,dy,dz, color = 'red', alpha = .8, length=0.3)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_aspect("equal")


    plt.legend()
    plt.tight_layout()
    plt.show()
#rr = min_dipol()
#print(rr[0])
#narisi_monopol(rr[0], rr[1])


#DRUGA NALOGA-SEMAFOR:
#v je vektor hitrosti, vec_t vektor casa
vec_t = np.linspace(0,1,50)


v_zac =np.ones(49)

def F(vec_v, vec_t, l, v_0):
    integral=0
    dt = (np.abs(vec_t[-1]-vec_t[0])/(len(vec_t)-1))
    integral += ((vec_v[0]-v_0)**2)/(dt)
    for i in range(len(vec_v)-1):
        if (i==(len(vec_v)-2)):
            integral +=  0.5*((vec_v[i+1]-vec_v[i])**2)/(dt)
        else:
            integral +=  ((vec_v[i+1]-vec_v[i])**2)/(dt)

    vez = 0;
    vez += 0.5*v_0
    for j in range(len(vec_v)):
        if (j==(len(vec_v)-1)):
            vez += vec_v[j]*0.5
        else:
            vez += vec_v[j]
    vez = vez*dt
    res = integral - l*vez
    return(res)



#print(v0)

def v_od_l(l, v_0, vec_t):
    v_zac =np.ones(len(vec_t)-1)
    res_1= minimize(F2, v_zac,(vec_t,l, v_0), method='BFGS', tol=1e-6)
    resitev = res_1.x
    vrednost_E = res_1.fun
    #for i in range(1):
    #    v_zac =np.ones(49)
    #    res = minimize(F, v_zac,(vec_t,l, v_0), method='Powell', tol=1e-6)
    #    if (res.success == True and res.fun< vrednost_E):
    #        resitev = res.x
    #        res_1 = res
    return resitev
def root_f(l, v_0, vec_t):
    dt = (np.abs(vec_t[-1]-vec_t[0])/(len(vec_t)-1))
    vsota = 0

    vsota += v_0*0.5

    vec_hitr = v_od_l(l, v_0, vec_t)
    for j in range(len(vec_hitr)):
        if (j==(len(vec_hitr)-1)):
            vsota += vec_hitr[j]*0.5
        else:
            vsota += vec_hitr[j]

    vsota = vsota*dt
    rezultat = vsota - 1
    return(rezultat)

v_0 = 1.5

def F2(vec_v, vec_t, kappa, v_0):
    integral=0
    dt = (np.abs(vec_t[-1]-vec_t[0])/(len(vec_t)-1))
    integral += ((vec_v[0]-v_0)**2)/(dt)
    for i in range(len(vec_v)-1):
        if (i==(len(vec_v)-2)):
            integral +=  0.5*((vec_v[i+1]-vec_v[i])**2)/(dt)
        else:
            integral +=  ((vec_v[i+1]-vec_v[i])**2)/(dt)

    F1 = 1;
    delna_vsota = 0.5*v_0
    for j in range(len(vec_v)):
        if (j==(len(vec_v)-1)):
            delna_vsota += vec_v[j]*0.5
        else:
            delna_vsota += vec_v[j]
    F1 += np.exp(kappa*(-delna_vsota + 1/dt))
    res = integral + F1
    return(res)


def narisi_priblizna():
    vec_t = np.linspace(0,1,50)
    dt = (np.abs(vec_t[-1]-vec_t[0])/(len(vec_t)-1))
    v_0_arr = np.array([0,0.5,1.0,1.5,2.0])
    kappa = np.array([0,0.001, 0.01,0.1,1,10,100])

    fig, ax = plt.subplots()
    v = np.zeros(len(vec_t))
    for j in range(len(vec_t)):

        if (j==0):
            v[j] = v_0_arr[1]
        else:
            v[j] = (-3/2)*(1-v_0_arr[1])*vec_t[j]**2 + 3*(1-v_0_arr[1])*vec_t[j] + v_0_arr[1]
    for i in range(len(kappa)):
        r=v_od_l(kappa[i], v_0_arr[1], vec_t)
        integ = r
        integ = np.insert(integ, 0, 0.5*v_0_arr[1])
        integ[-1] = 0.5*integ[-1]
        dolzina = np.sum(integ)*dt
        r = np.insert(r, 0, v_0_arr[1])
        ax.plot(vec_t, r, label=r'$\kappa = $'+ r'{},'.format(round(kappa[i], 3)) + r'L='+ r'{}'.format(round(dolzina,2)))

    ax.plot(vec_t, v, '--',label='analitična')
    ax.set_xlabel(r'$\tilde{t}$')
    ax.set_ylabel(r'$\tilde{v}$')
    plt.legend()
    plt.show()
narisi_priblizna()
#narisi_priblizna()

def contour():
    vec_t = np.linspace(0,1,50)
    dt = (np.abs(vec_t[-1]-vec_t[0])/(len(vec_t)-1))
    kappa = np.logspace(0.1, 3, 50) /10

    v0 = np.linspace(0,2,20)
    (X,Y) = np.meshgrid(v0,kappa)
    l = np.zeros((len(kappa), len(v0)))
    tocke_k = np.array([])
    tocke_v = np.array([])
    for i in range(len(kappa)):
        for k in range(len(v0)):
            r=v_od_l(kappa[i], v0[k], vec_t)
            integ = r
            integ = np.insert(integ, 0, 0.5*v0[k])
            integ[-1] = 0.5*integ[-1]
            dolzina = np.sum(integ)*dt
            l[i,k]=dolzina
            if (dolzina-1)<0.0001:
                tocke_k = np.append(tocke_k, kappa[i])
                tocke_v = np.append(tocke_v, v0[k])




    fig,ax=plt.subplots(1,1)

    cp = ax.contourf(X, Y, l, 200)
    col = plt.colorbar(cp)


    col.ax.set_ylabel('L')
    ax.set_xlabel(r'$\tilde{v}_{0}$')
    ax.set_ylabel(r'$\kappa$')
    plt.yscale('log')
    #plt.plot(tocke_v, tocke_k, 'r')





    #ax.clabel(cp, inline=1, fontsize=10)
    plt.show()
#contour()


def narisi():
    vec_t = np.linspace(0,1,50)
    resitev = fsolve(root_f, 2, (v_0, vec_t))
    print(resitev)
    v_0_arr = np.array([0,0.5,1.0,1.5,2.0])
    napaka = np.array([])

    N = np.linspace(20,160,20)
    lambde = np.array([6.00640218,3.00364541, -3.14440545e-08,-3.00561066, -6.0072825])
    for i in range(0,len(N),1):
        vec_t = np.linspace(0,1, int(N[i]))
        v = np.zeros(int(N[i]))
        for j in range(int(N[i])):

            if (j==0):
                v[j] = v_0_arr[3]
            else:
                v[j] = (-3/2)*(1-v_0_arr[3])*vec_t[j]**2 + 3*(1-v_0_arr[3])*vec_t[j] + v_0_arr[3]



        er = v_od_l(lambde[3], v_0_arr[3], vec_t)

        er = np.insert(er, 0, v_0_arr[3])
        nap = LA.norm(er-v)/LA.norm(v)
        napaka = np.append(napaka, nap)
        #plt.plot(vec_t, er, label=r'$\tilde{v}_{0}$' + r'={}'.format(v_0_arr[i]))
    fig, ax = plt.subplots()
    ax.plot(N,napaka, label=r'$\tilde{v}_{0}=1.5$' )
    ax.legend()
    ax.set_xlabel(r'N')
    ax.set_ylabel(r'$\frac{|\tilde{v}_{num}-\tilde{v}|}{\tilde{v}}$')
    plt.grid()
    ax.ticklabel_format(style='sci')
    plt.show()

#narisi()


#narisi_eks()
