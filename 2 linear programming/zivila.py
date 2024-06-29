import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import scipy.special as special
from io import StringIO
from scipy.optimize import linprog
from decimal import Decimal



#na 100g
#1 naloga:

seznam = np.genfromtxt('/home/ziga/Desktop/FMF/magisterij/modelska_1/2_linearno_programiranje/zivila.txt', dtype=None, delimiter="", encoding='utf-8')
lastnosti_zivila = np.array(['energija',	'maščobe',	'ogljikovi \n hidrati',	'proteini',	'Ca',	'Fe',	'Vitamin C',	'Kalij' ,'Natrij','Cena'])
vrednosti = np.zeros((len(seznam), 10))
for i in range(len(seznam)):
    for j in range(1,11):
        vrednosti[i,j-1] = seznam[i][j]
vrednosti[:,4:9] = vrednosti[:,4:9]/1000 #vse v grame
vrednosti_g = vrednosti
vrednosti = vrednosti/100 #delezi


#b = np.array([-2000,-90,-200,-20,-0.5,-18*10**(-3),60*10**(-3),3.5,2.4,-0.5,6, 2000])

b = np.array([-70,-310,-50,-1,-18*10**(-3), 60*10**(-3),3.5,2.4,-0.5,6, 2000])
b = np.append(b,np.ones(49)*500)
#Spreminjanje A:
A = np.transpose(-vrednosti)

#A = A[[0,1,2,3,4,5,6,7,8]]
A = A[[1,2,3,4,5,6,7,8]]
A[6:9] = -A[6:9]

A = np.append(A, [(-1)*A[7]], axis=0) ##zgornja meja za natrij
vektor_soli = np.zeros(49)
vektor_soli[44]=1
A = np.append(A, [vektor_soli], axis=0) #sol
A = np.append(A, [np.ones(49)], axis=0) #masa

A=np.append(A, np.identity(49), axis=0)
C= vrednosti[:,0]

def izracunaj_jedilnik(c,A,b):

    res = linprog(C, A, b)
    return res.x

imena_zivil = np.array([])
for i in range(len(seznam)):
    imena_zivil = np.append(imena_zivil,seznam[i][0])

res = izracunaj_jedilnik(C,A,b)
print(res)


def narisi_grafikon(res, imena_zivil):
    velikosti = np.array([])
    imena = np.array([])
    for i in range(len(res)):
        if res[i] != 0:
            velikosti = np.append(velikosti, res[i])
            imena  = np.append(imena, imena_zivil[i])
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    faktor = np.sum(velikosti)
    def func(velikosti):
        return "{:.2f}%\n({:.2f} g)".format(velikosti, (velikosti/100)*faktor)

    fig1, ax1 = plt.subplots()
    wedges, texts, autotexts =ax1.pie(velikosti, autopct=lambda velikosti: func(velikosti),
            shadow=True, startangle=90)
    ax1.legend(wedges, imena,
          loc="center left",
          bbox_to_anchor=(0.82, 0, 0.5, 1))
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

narisi_grafikon(res, imena_zivil)

def narisi_histogram(res, vrednosti_g):
    velikosti = np.array([])

    imena = np.array([])
    nenicelni_indeksi = np.array([])
    for i in range(len(res)):
        if res[i] != 0:
            velikosti = np.append(velikosti, res[i])
            imena  = np.append(imena, imena_zivil[i])
            nenicelni_indeksi = np.append(nenicelni_indeksi, i)
    normalizacija = np.zeros(8)
    for element in range(1,9):
        for i in range(len(velikosti)):

            ind = nenicelni_indeksi[i]
            normalizacija[element-1] += vrednosti[int(ind), element]

    fig, ax = plt.subplots()
    kumulativne_vred = np.zeros(8)
    for i in  range(len(velikosti)):
        ax.bar(x = lastnosti_zivila[1:9], height=vrednosti[int(nenicelni_indeksi[i]), 1:9]/normalizacija,bottom=kumulativne_vred, width=0.9, label='{}'.format(imena[i]))
        kumulativne_vred += vrednosti[int(nenicelni_indeksi[i]), 1:9]/normalizacija
    plt.ylabel('delež')
    plt.legend()
    plt.show()
narisi_histogram(res, vrednosti_g)
def izracunaj_ceno(res):
    cena = 0
    for i in range(len(res)):
        if (res[i]!=0):
            cena += res[i]*vrednosti[i,9]
    return(cena)
def izracunaj_maso(res):
    mas = 0
    for i in range(len(res)):
        if (res[i]!=0):
            mas += res[i]
    return(mas)

#narisi_grafikon(res, imena_zivil)
#narisi_histogram(res, vrednosti_g)


########3: cena v odvisnosti od max mase zivila
def cena_vs_masa():
    max_masa = np.array([500,400,300,200,150,100,50])
    min_masc = np.array([50,70,100,120,150,170,200])
    min_cena = np.zeros((len(max_masa), len(min_masc)))
    for j in range(len(min_masc)):
        for i in range(len(max_masa)):
            b = np.array([-2000,-min_masc[j],-310,-50,-1,-5*10**(-3),60*10**(-3),3.5,2.4,-0.5,6, 2000])
            b = np.append(b,np.ones(49)*max_masa[i])
            res = izracunaj_jedilnik(C,A,b)

            min_cena[i,j]=izracunaj_ceno(res)
        narisi_grafikon(res, imena_zivil)
    return(max_masa, min_masc, min_cena)
#masa, energija, cena = cena_vs_masa()

#v = np.linspace(np.min(cena), np.max(cena), 1000, endpoint=True)
#plt.contourf(masa, energija, cena,v, cmap=plt.cm.jet )
#x = plt.colorbar(ticks=np.linspace(np.min(cena), np.max(cena), 10), label='cena [EUR]')
#plt.xlabel('Največja dovoljena masa posamezne sestavine [g]')
#plt.ylabel('Minimalen vnos maščob [g]')
#plt.show()

########5 dnevni jedilnik:
def pet_dnevni_jedilnik(c, A, b):
    resitve = np.array([])
    cena = np.array([])
    kalorije = np.array([])
    indeksi = np.array([])
    for i in range(5):
        res = izracunaj_jedilnik(c,A,b)
        #narisi_grafikon(res, imena_zivil)
        resitve = np.append(resitve, res)
        cena = np.append(cena, izracunaj_ceno(res))
        kalorije=  np.append(kalorije, izracunaj_maso(res))
        for j in range(len(res)):
            if (res[j]!=0 and (j not in indeksi)and j != 44):
                indeksi = np.append(indeksi, j)
        A[:,indeksi.astype(int)] =0
    return(cena, kalorije)

def jedilnik():
    cena, kalorije = pet_dnevni_jedilnik(C, A, b)

    color = 'tab:red'
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('dan')
    ax1.plot([i for i in range(1,6)], cena, '--ro')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylabel('cena[EUR]')

    ax2=ax1.twinx()
    color = 'tab:blue'
    ax2.plot([i for i in range(1,6)], kalorije, '--bo')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('masa[g]')





    plt.show()
    return None
