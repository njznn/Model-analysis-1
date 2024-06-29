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
import matplotlib.colors



#   1   NALOGA:
##y je vektor y=(D,B,I)
def sistem_enacb(t,y, alpha, beta):
    D,B,I=y
    return([-alpha*D*B, alpha*D*B - beta*B, beta*B])

st_ljudi = 10**4
t = np.linspace(0, 5, 1000)

alpha = np.linspace(0.0005, 0.001, 20)
beta = np.linspace(0.1, 0.5, 20)
imuni = np.array([0.1,0.2,0.4,0.6,0.8,0.9])

max = np.zeros((int(len(alpha)), int(len(beta))))

#for i in range(len(alpha)):
#    for j in range(len(beta)):
fig = plt.figure()
ax = fig.add_subplot(111)

"""
sol = solve_ivp(sistem_enacb, (0,5), [st_ljudi*(1-0.01-imuni[0]),st_ljudi*(0.01), st_ljudi*imuni[0]],t_eval=t, args=(10**-3,1.0))
line, = ax.plot(t, sol.y[1]/st_ljudi, linestyle = '-', label='bolni')
ax.plot(t, sol.y[2]/st_ljudi,color = line.get_color(), linestyle = '--', label='imuni')
ax.plot(t, sol.y[1]/st_ljudi,color = line.get_color() ,linestyle = '-', label='{} %'.format(imuni[0]*100))


for i in range(1,len(imuni)):
    sol = solve_ivp(sistem_enacb, (0,5), [st_ljudi*(1-0.01-imuni[i]),st_ljudi*(0.01), st_ljudi*imuni[i]],t_eval=t, args=(10**-3,1.0))



    line, = ax.plot(t, sol.y[1]/st_ljudi, linestyle = '-', label='{} %'.format(imuni[i]*100))
    ax.plot(t, sol.y[2]/st_ljudi,color = line.get_color(), linestyle = '--')






    if i==0:
        sol = solve_ivp(sistem_enacb, (0,5), [st_ljudi*(0.99),st_ljudi*(0.01),0],t_eval=t, args=(10**-3,1))
        plt.plot(t, sol.y[0]/st_ljudi, 'b',label=r'$D$')
        plt.plot(t, sol.y[1]/st_ljudi,'r', label=r'$B$')
        plt.plot(t, sol.y[2]/st_ljudi, 'g', label='I')
    else:
        None

    plt.title(r'$\alpha = 10^{-3}, \beta = 1, D(0) = 0.99, B(0) = 0.01$')
    plt.legend()
    """


#cp = plt.contourf(alpha, beta, max, 200)
#col = plt.colorbar(cp, label='dan nastopa maksimuma bolnih')

plt.xlabel(r'$t$')
plt.ylabel(r'$delež$')
plt.title(r'$\alpha = 10^{-3}, \beta = 5$')
plt.legend()


#plt.show()
plt.clf()

##STADIJI OKUZBE:
"""
t = np.linspace(0, 10, 1000)

def sistem_enacb_stadiji(t,y, alpha1,alpha2,alpha3,alpha4, beta1):
    D,Bi,Bk,Bizol,I=y
    return([-alpha1*D*Bk - beta1*D*Bizol, alpha1*D*Bk + beta1*D*Bizol - alpha2*Bi, alpha2*Bi - alpha3*Bk, alpha3*Bk - alpha4*Bizol, +alpha4*Bizol])
sol_std = solve_ivp(sistem_enacb_stadiji, (0,10), [st_ljudi*(0.6),st_ljudi*(0.1), 0,0,0.3*st_ljudi],t_eval=t, args=(7**-3,1.0, 1, 1, 10**(-4)))

plt.plot(t, sol_std.y[0]/st_ljudi, label='dovzetni')
plt.plot(t, sol_std.y[1]/st_ljudi, label='inkubacija')
plt.plot(t, sol_std.y[2]/st_ljudi, label='kužni')
plt.plot(t, sol_std.y[3]/st_ljudi, label='izolacija')
plt.plot(t, sol_std.y[4]/st_ljudi, label='imuni')
plt.title(r'$\alpha_1 = 7^{-3}, \alpha_2 = 1,\alpha_3 = 1, \alpha_4 = 1, \beta_1 = 10^{-4}  $')
plt.ylabel('delež')
plt.xlabel('t')
plt.legend()

plt.show()

#################### Delay model:
st_ljudi=2*10**6
t = np.linspace(0,20,5000)
def sistem_enacb_delay(y,t, Tink, Tdov, alpha, beta1, beta2):
    D,B,I=y(t)
    Dink, Bink, Iink = y(t-Tink)
    Ddov,Bdov,Idov = y(t-Tdov)


    return(np.array([-alpha*D*Bink + beta1*Bdov, alpha*D*Bink - beta2*B, beta2*B - beta1*Bdov ]))

def g(t):
    return(np.array([0.598*st_ljudi,0.002*st_ljudi, 0.4*st_ljudi]))

sol_del = ddeint(sistem_enacb_delay, g,t, fargs=(0.1,10, 12**-5, 1, 1))


plt.plot(t, sol_del[:, 0]/st_ljudi, label='D')
plt.plot(t, sol_del[:,1]/st_ljudi, label='B')
plt.plot(t, sol_del[:,2]/st_ljudi, label='I')
plt.xlabel('t')
plt.ylabel('delež')
plt.title(r'$\alpha = 12^{-5}, \beta_1 = 1,=\beta_2, \frac{\tau_{dov}}{\tau_{ink}} = 100$')
plt.legend()
plt.show()


#################2.naloga, lotka volterra###############
t = np.linspace(0, 20, 1000)
def sistem_lotka(t,y, p):
    z,l=y
    return(np.array([p*z*(1-l), (l/p) *(z-1) ]))


Z0 = np.linspace(0.1,1.0,10)
fig, ax

colors = plt.cm.viridis(np.linspace(0,1,10))
for j in range(len(Z0)):

    sol = solve_ivp(sistem_lotka, (0,20), [0.001,0.001],t_eval=t, args=(2.0,))

    plt.plot(sol.y[0], sol.y[1], color = colors[j])

plt.xlabel('z')
plt.ylabel('l')
plt.title('p = 2.0')
#plt.show()
plt.clf()


fig, (ax1, ax2) = plt.subplots(2, sharex=True)

sol = solve_ivp(sistem_lotka, (0,20), [1.5,1.0],t_eval=t, args=(1.5,))
ax1.plot(t, sol.y[0], label='z')
ax1.plot(t, sol.y[1], label='l')
sol = solve_ivp(sistem_lotka, (0,20), [1.0,1.5],t_eval=t, args=(1.5,))
ax2.plot(t, sol.y[0], label='z')
ax2.plot(t, sol.y[1], label='l')
plt.xlabel(r'$\tau$')
ax1.set_ylabel('z,l')
ax2.set_ylabel('z,l')
fig.suptitle(r'$p = 1.5$')
ax1.legend()
ax2.legend()
plt.clf()
#plt.show()

#casovna odvisnost:
t = np.linspace(0, 50, 1000)
Z0 = np.linspace(0.1,5.0,20)
p_arr = np.linspace(0.5, 2, 10)

cmap = plt.get_cmap("jet", 10)

for i in range(len(p_arr)):
    casi = np.array([])
    for j in range(len(Z0)):

        sol = solve_ivp(sistem_lotka, (0,50), [Z0[j],1.0],t_eval=t, args=(p_arr[i],))
        peaks = find_peaks(sol.y[0])
        indeks1, indeks2 = peaks[0][0], peaks[0][1]
        casi = np.append(casi, np.abs(t[indeks1]-t[indeks2]))
    plt.plot(Z0, casi, 'o--',color = cmap(i), markersize=3)


norm= matplotlib.colors.BoundaryNorm(np.linspace(0.5, 2, 10), 10)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='p')
plt.xlabel('z(0)')
plt.ylabel('t')
plt.title('l(0) = 1.0')

plt.show()
print(casi)

plt.clf()
"""
#########3 naloga########
t = np.linspace(0,50,5000)
def sistem(t,y, r,p):
    A,F = y
    return([r-p*A*(F+1),(F/p)*(A-1)])



#casovna odvisnost:
t = np.linspace(0, 50, 5000)

r_ar = np.linspace(2,10, 20)
A_ar = np.linspace(1.5,3.0, 5)
cmap = plt.get_cmap("jet", 5)


for i in range(len(A_ar)):
    casi = np.array([])
    for j in range(len(r_ar)):

        sol = solve_ivp(sistem, (0,50), [A_ar[i],1.0],t_eval=t, args=(r_ar[j],1))

        peaks = find_peaks(sol.y[1])
        indeks1, indeks2 = peaks[0][0], peaks[0][1]
        casi = np.append(casi, np.abs(t[indeks1]-t[indeks2]))
    plt.plot(r_ar, casi, 'o--', color = cmap(i),markersize=3)
norm= matplotlib.colors.BoundaryNorm(np.linspace(1.5, 3, 6), 6)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, label='A(0)')



plt.xlabel('r')
plt.ylabel('T')
plt.title('F(0)=1.0, p=1')

plt.show()
print(casi)

plt.clf()

"""
for a in range(len(A_ar)):
    for f in range(len(F_ar)):
        cmap = plt.cm.viridis
        cmap1 = plt.cm.jet
        cmap2 = plt.cm.gnuplot
        cmap3 = plt.cm.cool
        color1 = cmap1((a+f)/40)
        color2 = cmap2((a+f)/40)
        color3 = cmap3((a+f)/40)
        color = cmap((a+f)/40)
        sol = solve_ivp(sistem, (0,20), [A_ar[a],F_ar[f]],t_eval=t, args=(2.,1,))

        if (sol.y[1][-1]==0):
            plt.plot(sol.y[0], sol.y[1], color=color)
        else:
            plt.plot(sol.y[0], sol.y[1], color=color)
            plt.plot(sol.y[0][0], sol.y[1][0], 'ro',markersize='3', )


        sol = solve_ivp(sistem, (0,20), [A_ar[a],F_ar[f]],t_eval=t, args=(1.,0.5,))
        plt.plot(sol.y[0], sol.y[1], color=color1)
        sol = solve_ivp(sistem, (0,20), [A_ar[a],F_ar[f]],t_eval=t, args=(3.,1,))
        plt.plot(sol.y[0], sol.y[1], color=color2)
        sol = solve_ivp(sistem, (0,20), [A_ar[a],F_ar[f]],t_eval=t, args=(1.5,0.5,))
        plt.plot(sol.y[0], sol.y[1], c=color3)

plt.xlabel('A')
plt.ylabel('F')
plt.title('r=2, p=1')
plt.show()
"""


""""
sol = solve_ivp(sistem, (0,20), [2,0],t_eval=t, args=(2.,1,))
sol1 = solve_ivp(sistem, (0,20), [1.999,0.001],t_eval=t, args=(2.,1,))
sol2 = solve_ivp(sistem, (0,20), [1.5,1.5],t_eval=t, args=(2.,1,))
sol3 = solve_ivp(sistem, (0,20), [1.7,1.7],t_eval=t, args=(2.,1,))
plt.plot(t, sol.y[1],'b', label='fotoni, F(0)= 0')
plt.plot(t, sol.y[0],'r', label='atomi, A(0)=2')
plt.plot(t, sol2.y[1], 'b', linestyle='dotted', label='F(0) = 1.5')
plt.plot(t, sol2.y[0], 'r', linestyle = 'dotted',label='A(0) = 1.5')
plt.plot(t, sol3.y[1], 'b', linestyle='dashdot',label='F(0) = 1.7')
plt.plot(t, sol3.y[0], 'r', linestyle = 'dashdot',label='A(0) = 1.7')

plt.plot(t, sol1.y[1],'b',linestyle='--', label='F(0) = 0.001' )
plt.plot(t, sol1.y[0],'r',linestyle='--', label= 'A(0) = 1.999')

plt.text(1.0,1.3, 'r=2, \n p=1')
plt.text(1.3,0.3, 'r=1, \n p=0.5')
plt.text(1,3, 'r=1.5, \n p=0.5')
plt.text(1.45,1.4, 'r=3, \n p=1')

plt.xlabel(r'$\tau$')

#plt.title(r'$A(0) \in [1.8,2], F(0) \in [0, 0.2]$')
plt.title('r=2, p=1')
plt.legend()
plt.show()

sol1 = solve_ivp(sistem, (0,10), [1,1],t_eval=t, args=(2.,1,))
sol = solve_ivp(sistem, (0,10), [1.1,1],t_eval=t, args=(2.,1,))
sol2 = solve_ivp(sistem, (0,10), [1.1,1.1],t_eval=t, args=(2.,1,))
sol3 = solve_ivp(sistem, (0,10), [1.4,1.4],t_eval=t, args=(2.,1,))
plt.plot(t, sol1.y[1], 'b', label='fotoni, F(0) = 1')
plt.plot(t, sol1.y[0], 'r', label='atomi, A(0)=1')
plt.plot(t, sol.y[1], 'b', linestyle ='--', label='F(0) = 1')
plt.plot(t, sol.y[0], 'r', linestyle = '--', label='A(0) = 1.1')
plt.plot(t, sol2.y[1], 'b', linestyle='dotted', label='F(0) = 1.1')
plt.plot(t, sol2.y[0], 'r', linestyle = 'dotted',label='A(0) = 1.1')
plt.plot(t, sol3.y[1], 'b', linestyle='dashdot',label='F(0) = 1.4')
plt.plot(t, sol3.y[0], 'r', linestyle = 'dashdot',label='A(0) = 1.4')
plt.legend()
plt.xlabel(r'$\tau$')
plt.title('r=2, p=1')
plt.show()
"""
