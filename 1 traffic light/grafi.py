import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special

v1 = np.linspace(0.0,1.8,10)
p = np.array([1,2,3,5,50])
t = np.linspace(0,1,500)
v = np.zeros(500)
fig, ax = plt.subplots()
v0=2.0
p1=np.linspace(1,4,50)
vec_ukr = np.zeros(4)
def ukr(t, v0, p):
    return(abs(((-1 + 4 *p)* (-1 + (2 *p)/(-1 + 2* p))* (1 - v0)* (1 - t)**(-2 + (
    2*p)/(-1 + 2*p)))/(-1 + 2*p))/(1 + ((-1 + 4*p)**2 *(1 - v0)**2 *(1 - t)**(-2 + (
    4*p)/(-1 + 2*p)))/(-1 + 2*p)**2)**(3/2))

ukr_vec=np.zeros(50)


#for i in range(len(p1)-1):

#    ukr_vec[i]= integrate.quad(ukr, 0, 1, args=(v0, p1[i]))[0]

#print(ukr_vec)
#ax.plot(2*p1[:-2], ukr_vec[:-2])
C = np.linspace(1, 200, 9)
C = np.append(C, [1000])
print(C)


plt.style.use('ggplot')
v1=1
a = np.array([-1,-0.5,0,1,1.5,2])
for a0 in a:
    #v11 = (-3/2)*(1-i)*t**2 + 3*(1-i)*t + i
    #v21= -3*(2 - i - 0.5)*t**2 + 2*(3 - 2*i -0.5)*t + i

    #lamda = 2*c*(np.sqrt(c)*(np.exp(2*np.sqrt(c)) +1) + v0 - np.exp(2*np.sqrt(c))*v0)/(1+ np.sqrt(c) + np.exp(2*np.sqrt(c))*( np.sqrt(c)- 1))
    #A = (2 *c**(3/2)* np.exp(np.sqrt(c)) - lamda + np.exp(np.sqrt(c))* lamda - np.sqrt(c)* np.exp(np.sqrt(c)) *lamda +
    #2* c *v0 - 2* c* np.exp(np.sqrt(c))* v0)/(2* c* (-1 + np.exp(np.sqrt(c)))**2)
    #B = v0-lamda/(2*c) - A
    #A = (v0-lamda/(2*c))/(1+np.exp(2*np.sqrt(c)))
    #B = v0-lamda/(2*c)-A
    #v_4 = lamda/(2*c) + A*np.exp(np.sqrt(c)*t) + B*np.exp(-np.sqrt(c)*t)

    #v0=1.5
    #v =  v0+(4*i-1)/(2*i)*(1-v0)*(1-(1-t)**(2*i/(2*i-1)))
    l=1477.0998123553245* (1. - 0.998645995359778* v1)
    A = -3.2388817519989543* a0 - 0.1779289332192362* l +0.3558578664384724* v1
    B = 3.2388817519989543* a0 - 0.3220710667807638* l +0.6441421335615276* v1
    C1 =  10.132688312370561*a0 - 0.29279184314441853* l +0.5855836862888371* v1
    D1=3.0871271959692153*a0 + 0.04313034430447364* l -0.08626068860894728* v1

    color=next(ax._get_lines.prop_cycler)['color']
    v_x = l/2 + np.exp(np.sqrt(3)*t/2) *A* np.cos(t/2) + np.exp(-np.sqrt(3)*t/2) *B* np.cos(t/2) +np.exp(-np.sqrt(3)*t/2) *C1* np.sin(t/2)+ np.exp(np.sqrt(3)*t/2) *D1* np.sin(t/2)
    ax.plot(t,v_x, label=r'$a$='+'{0:,.1f}'.format(a0), color=color)
    ax.plot(np.linspace(1,2,500),v_x,  color=color)
    #v00=0.5
    #v_ =  v00+(4*i-1)/(2*i)*(1-v00)*(1-(1-t)**(2*i/(2*i-1)))
    #ax.plot(t,v_,linestyle='dashed')
    #v5 = -6*(1-i)*t**2 + 6*(1-i)*t + i
    color=next(ax._get_lines.prop_cycler)['color']

    #ax.plot(np.linspace(0,1,500),v5, label=r'$\tilde{v}$='+'{0:,.1f}'.format(i), color=color)
    #ax.plot(np.linspace(1,2,500),v5, color=color)




#ax.text(0.9,1.0,r'$\tilde{v_{2}}=0.5$')
legend = ax.legend(loc='lower right', fontsize='x-small')
#ax.text(1.0,1.9,r'$\tilde{v_{1}}=2.0$' )
ax.set_xlabel(r'$\tilde{t}$')
ax.set_ylabel(r'$\tilde{v}(\tilde{t})$')
plt.grid()
plt.show()
