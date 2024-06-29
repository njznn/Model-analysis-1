import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D # noqa: F401 unused import
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

potts_q2_16_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q2_16_J1.txt",
skiprows=1)
potts_q2_32_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q2_32_J1.txt",
skiprows=1)
potts_q2_64_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q2_64_J1.txt",
skiprows=1)
potts_q2_128_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q2_128_J1.txt",
skiprows=1)

potts_q3_16_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q3_16_J1.txt",
skiprows=1)
potts_q3_32_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q3_32_J1.txt",
skiprows=1)
potts_q3_64_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q3_64_J1.txt",
skiprows=1)
potts_q3_128_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q3_128_J1.txt",
skiprows=1)

potts_q4_16_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q4_16_J1.txt",
skiprows=1)
potts_q4_32_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q4_32_J1.txt",
skiprows=1)
potts_q4_64_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q4_64_J1.txt",
skiprows=1)
potts_q4_128_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q4_128_J1.txt",
skiprows=1)

potts_q5_16_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q5_16_J1.txt",
skiprows=1)
potts_q5_32_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q5_32_J1.txt",
skiprows=1)
potts_q5_64_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q5_64_J1.txt",
skiprows=1)
potts_q5_128_J1 = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q5_128_J1.txt",
skiprows=1)



import matplotlib.patches as mpatches
def plot_size(ind_kol, en, dva, tri, stiri):
    beta2 = en[:,0];
    plt.plot(beta2, en[:, ind_kol]/(16**2),'x--', label='16x16');
    plt.plot(beta2, dva[:, ind_kol]/(32**2),'x--', label='32x32');
    plt.plot(beta2, tri[:, ind_kol]/(64**2),'x', label='64x64');
    plt.plot(beta2, stiri[:, ind_kol]/(128**2),'x', label='128x128');
    #plt.axvline(x = 1.4848, color='black', linestyle='--', label=r'$\beta_c$')

    '''
    color = ['blue', 'red', 'green', 'black']
    symbols = ['o', 's', '^', 'd']
    dim = [16,32,64,128]

    for i in range(2,len(color)+2):
        for j in range(len(symbols)):
            y = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts_q"+
            str(i)+"_"+str(dim[j])+"_J1.txt",
            skiprows=1)
            plt.plot(beta2, y[:,1]/(dim[j]**2), marker=symbols[j], color=color[i-2], label=f'{color[i-2]} - Plot {j+1}')



    legend_elements = [plt.Line2D([], [], marker=symbols[j], color='blue', label=f''+str(dim[j])+'x'+str(dim[j])) for j in range(len(symbols))]

    color_handles = [plt.Line2D([], [], color=color[i], linestyle='-', label="q="+str(i+2)) for i in range(4)]
    plt.legend(handles=color_handles+ legend_elements, loc='upper right')
    '''
    plt.legend()
    # Set the title and labels
    plt.title(r'$J=1, N_{relax} \geq 10^8, q=5$')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\langle M \rangle / N^2$')

    # Show the plot
    plt.show()

    return None

#plot_size(2, potts_q5_16_J1, potts_q5_32_J1, potts_q5_64_J1, potts_q5_128_J1)

def mag_diff(en, dva,tri,ind_kol=2, ):
    beta2 = en[:,0];

    #plt.plot(beta2, en[:, ind_kol]/(64**2),'x--', label=r'$64x64, N_{relax}=10^8$');
    #plt.plot(beta2, dva[:, ind_kol]/(64**2),'x', label=r'$64x64,N_{relax}= 10^7$');
    plt.plot(beta2, en[:, ind_kol]/(128**2),'x--', label=r'$ N_{relax}=10^8$');
    plt.plot(beta2, dva[:, ind_kol]/(128**2),'x--', label=r'$ N_{relax}=10^8, \xi = 1.5 \beta \forall \beta \geq 0.8$');
    plt.plot(beta2, tri[:, ind_kol]/(128**2),'x--', label=r'$ N_{relax}=50^9$');
    plt.legend()
    plt.xlabel(r'$\beta$')
    #plt.ylabel(r'$\langle E \rangle / N^2$')
    plt.ylabel(r'$\langle M \rangle / N^2$')
    plt.title(r'$N_{sample}=500, J=1, 128x128$')
    plt.show()

#mag_diff(ising128, ising128_mod, ising128_large)

def suscept(en, dva, tri, stiri):
    beta2 = en[:,0];

    plt.plot(beta2,( (1/np.max((beta2)*(en[:,4]-(en[:,2]**2))))*(beta2)*(en[:,4]-(en[:,2]**2))),'x--', label=r'16x16')
    plt.plot(beta2,( (beta2/(np.max(beta2*(dva[:,4]-dva[:,2]**2))))*(dva[:,4]-dva[:,2]**2)),'x--', label=r'32x32')
    plt.plot(beta2,( (beta2/np.max((beta2)*(tri[:,4]-tri[:,2]**2))))*(tri[:,4]-tri[:,2]**2), 'x--',label=r'64x64')
    plt.plot(beta2,( (beta2/np.max((beta2)*(stiri[:,4]-stiri[:,2]**2))))*(stiri[:,4]-stiri[:,2]**2), 'x--', label=r'128x128')

    plt.axvline(x = 1.4848, color='black', linestyle='--', label=r'$\beta_c$')
    plt.legend()
    plt.title(r'$N_{relax}\geq 10^8, N_{sample}=500,J=1, q=3$')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\chi$')

    plt.show()

#suscept(potts_q3_16_J1, potts_q3_32_J1, potts_q3_64_J1, potts_q3_128_J1)

def cv(en, dva, tri, stiri):
    beta2 = en[:,0];
    plt.plot(beta2,( (1/np.max((beta2)*(en[:,3]-(en[:,1]**2))))*(beta2)*(en[:,3]-(en[:,1]**2))),'x--', label=r'16x16')
    plt.plot(beta2,( (beta2/(np.max(beta2*(dva[:,3]-dva[:,1]**2))))*(dva[:,3]-dva[:,1]**2)),'x--', label=r'32x32')
    plt.plot(beta2,( (beta2/np.max((beta2)*(tri[:,3]-tri[:,1]**2))))*(tri[:,3]-tri[:,1]**2),'x--', label=r'64x64')
    plt.plot(beta2,(beta2/np.max((beta2)*(stiri[:,3]-stiri[:,1]**2)))*abs((stiri[:,3]-stiri[:,1]**2)),'x--', label=r'128x128')
    #plt.axvline(x = 0.65832, color='black', linestyle='--', label=r'$\beta_c$')
    plt.legend()
    plt.title(r'$N_{relax} \geq 10^8, N_{sample}=500, J=1, q=3 $')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$C_{v}$')
    plt.show()

#cv(potts_q3_16_J1, potts_q3_32_J1, potts_q3_64_J1, potts_q3_128_J1)

def susc_q(en, dva, tri):
    beta2 = en[:,0];
    plt.plot(beta2, ( (1/np.max((beta2)*(en[:,3]-(en[:,1]**2))))*(beta2)*(en[:,3]-(en[:,1]**2))), label=r'q=3')
    plt.plot(beta2, ( (beta2/(np.max(beta2*(dva[:,3]-dva[:,1]**2))))*(dva[:,3]-dva[:,1]**2)), label=r'q=4')
    plt.plot(beta2, ( (beta2/np.max((beta2)*(tri[:,3]-tri[:,1]**2))))*(tri[:,3]-tri[:,1]**2), label=r'q=5')
    plt.title(r'$N_{relax} \geq 10^{8}, J=1, lattice:32x32$')
    plt.axvline(x = 1.4848, color='black', linestyle='--', label=r'$\beta_c(q=3)$')
    plt.axvline(x = 1.60944, color='black', linestyle='--', label=r'$\beta_c(q=4)$')
    plt.axvline(x = 1.71015, color='black', linestyle='--', label=r'$\beta_c(q=5)$')

    plt.xlabel(r'$\beta$')
    #plt.ylabel(r'$\chi$')
    plt.ylabel(r'$C_{v}$')
    #plt.ylabel(r'$\langle M \rangle / N^2$')
    plt.legend()
    plt.show()


susc_q(potts_q3_32_J1, potts_q4_32_J1, potts_q5_32_J1)


def qdep(ind_kol):
    beta1 = potts_q5_32_J1[:,0];

    plt.plot(beta1, potts_q3_32_J1[:, ind_kol]/(32**2), label='q=3');
    plt.plot(beta1, potts_q4_32_J1[:, ind_kol]/(32**2), label='q=4');
    plt.plot(beta1, potts_q5_32_J1[:, ind_kol]/(32**2), label='q=5');
    plt.axvline(x = 1.4848, color='black', linestyle='--', label=r'$\beta_c(q=3)$')
    plt.axvline(x = 1.60944, color='black', linestyle='--', label=r'$\beta_c(q=4)$')
    plt.axvline(x = 1.71015, color='black', linestyle='--', label=r'$\beta_c(q=5)$')

    plt.title(r'$N_{relax}  \geq 10^{8}, J=1, lattice:32x32$')
    plt.xlabel(r'$\beta$')
    plt.ylabel(r'$\langle M \rangle / N^2$')
    #plt.ylabel(r'$\langle M \rangle / N^2$')
    plt.legend()
    plt.show()

    return None
qdep(2)

import matplotlib.patches as patches
def plot_ising_configuration(config):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Calculate the size of the configuration
    size = config.shape[0]

    # Create a meshgrid for the coordinates
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    cmap = plt.cm.get_cmap('bwr')

    # Plot the Ising model configuration as small circles
    ax.scatter(x, y, c=config, cmap=cmap, linewidths=0, s=12)
    ax.set_title(r'$\beta = 0.1, H=0.55, N_{relax}= 10^8 $')
    # Set aspect ratio to equal
    ax.set_aspect('equal')



    # Show the plot
    plt.show()

def generate_sequence_sodi(length):
    sequence = []
    num = 1
    increment = 4

    for i in range(length//2):
        sequence.append(num)
        sequence.append(num + 1)
        num = num + increment

    return sequence

def generate_sequence_lihi(length):
    sequence = [0]
    num = 0
    increment = 3

    for i in range(length//2):
        num += increment
        sequence.append(num)
        num +=1
        sequence.append(num)


    return sequence[:-1]


#qdep(2)
mat = np.loadtxt("/home/ziga/Desktop/FMF/magisterij/modelska_1/"
"Zakljucna_potts_na_heksagonalni_mrezi/"
"isingstate_128_3_0.txt",skiprows=1)

def pltstate(mat):
    plt.imshow(mat, cmap='cividis')
    plt.title("Ising Model")
    plt.show()

def preslikaj_na_heks(matrika):
    n = matrika[0,:].size
    sodind = generate_sequence_sodi(n)
    lihind = generate_sequence_lihi(n)


    nova = np.zeros((n,2*n))

    for i in range(n):
        if (i%2==0):
            for j in range(n):
                nova[i, sodind[j]] = matrika[i,j]
        else:
            for j in range(n):
                nova[i, lihind[j]] = matrika[i,j]

    return nova
#preslikana = preslikaj_na_heks(mat)

def plot_ising_configuration(ising_matrix):
    x = []
    y = []

    colors = []
    for i in range(ising_matrix.shape[0]):
        for j in range(ising_matrix.shape[1]):
            if ising_matrix[i, j] != 0:
                x.append(j)
                y.append(i)
                if ising_matrix[i, j] == 1:
                    colors.append('blue')
                elif ising_matrix[i, j]==2:
                    colors.append('green')
                elif ising_matrix[i, j]==3:
                    colors.append('orange')
                elif ising_matrix[i, j]==4:
                    colors.append('pink')
                else:
                    colors.append('red')

    fig, ax = plt.subplots()

    # Connect nearest neighbors with gray lines
    for i in range(ising_matrix.shape[0]):
        for j in range(ising_matrix.shape[1]):
            if ising_matrix[i, j] != 0:
                if i-1 >= 0 and ising_matrix[i-1, j] != 0:
                    ax.plot([j, j], [i, i-1], color='gray', linewidth=1, alpha=0.4)
                if i+1 < ising_matrix.shape[0] and ising_matrix[i+1, j] != 0:
                    ax.plot([j, j], [i, i+1], color='gray', linewidth=1, alpha=0.4)
                if j-1 >= 0 and ising_matrix[i, j-1] != 0:
                    ax.plot([j, j-1], [i, i], color='gray', linewidth=1, alpha=0.4)
                if j+1 < ising_matrix.shape[1] and ising_matrix[i, j+1] != 0:
                    ax.plot([j, j+1], [i, i], color='gray', linewidth=1, alpha=0.4)
                if i-1 >= 0 and j-1 >= 0 and ising_matrix[i-1, j-1] != 0:
                    ax.plot([j, j-1], [i, i-1], color='gray', linewidth=1, alpha=0.4)
                if i-1 >= 0 and j+1 < ising_matrix.shape[1] and ising_matrix[i-1, j+1] != 0:
                    ax.plot([j, j+1], [i, i-1], color='gray', linewidth=1, alpha=0.4)
                if i+1 < ising_matrix.shape[0] and j-1 >= 0 and ising_matrix[i+1, j-1] != 0:
                    ax.plot([j, j-1], [i, i+1], color='gray', linewidth=1, alpha=0.4)
                if i+1 < ising_matrix.shape[0] and j+1 < ising_matrix.shape[1] and ising_matrix[i+1, j+1] != 0:
                    ax.plot([j, j+1], [i, i+1], color='gray', linewidth=1, alpha=0.4)
    ax.scatter(x, y, c=colors, s=50)
    ax.set_title(r'$\beta = 3, H=1, N_{relax}= 10^9 $')
    ax.set_aspect('equal')
    plt.show()
#plot_ising_configuration(preslikana)


def load_data():
    data = []
    ind = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2])
    for i in range(0,10):
        filename = f"/home/ziga/Desktop/FMF/magisterij/modelska_1/Zakljucna_potts_na_heksagonalni_mrezi/potts32_q3_h_{ind[i]}0.txt"
        file_data = np.loadtxt(filename, skiprows=1)
        data.append(file_data)

    return np.array(data)

data = load_data()

def plot_contour(matrix, beta, H):

    # Create a meshgrid from matrix dimensions
    matrix = matrix.T
    print(matrix)
    X, Y = np.meshgrid(range(matrix.shape[1]), range(matrix.shape[0]))

    # Flatten the matrix and corresponding coordinates
    matrix_flattened = matrix.flatten()
    X_flattened = X.flatten()
    Y_flattened = Y.flatten()

    # Plot the scatter plot
    plt.figure()
    plt.scatter(X_flattened, Y_flattened, c=matrix_flattened, cmap='viridis',s=100)

    plt.xticks(range(beta.size),  [str(tick) if i % 3 == 0 else '' for i, tick in enumerate(beta)])
    plt.yticks(range(matrix.shape[0]),  H)
    plt.colorbar(label=r'$\chi$')#label=r'$\langle M \rangle / N^2$')
    plt.ylabel('H')
    plt.xlabel(r'$\beta$')
    plt.title(r'$J=1, 32x32, N_{relax} = 50^{9}, q=3$')
    plt.tight_layout()
    plt.show()

H = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.2])
beta = data[0][:,0]

mat = np.zeros((data[0][:,0].size, 10))

for i in range(mat[0,:].size):
    mat[:,i] = ((1/np.max((beta)*(data[i][:,4]-(data[i][:,2]**2))))*(beta)*(data[i][:,4]-(data[i][:,2]**2)))
    #mat[:, i] = data[i][:,2]



#plot_contour(mat, beta, H)
