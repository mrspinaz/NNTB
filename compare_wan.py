import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D
from itertools import product





def Create_Kpath(a,b):
    kx_size = 32
    ky_size = 32

    kx_new = np.linspace(0,np.pi/a,kx_size)
    ky_new = np.linspace(0,np.pi/b,ky_size)

    kx_2D, ky_2D = np.meshgrid(kx_new,ky_new) 

    kx_GX = kx_2D[0,:]
    ky_GX = ky_2D[0,:]

    kx_XS = kx_2D[:,-1]
    ky_XS = ky_2D[:,-1]

    kx_SY = np.flip(kx_2D[-1,:])
    ky_SY = ky_2D[-1,:]

    kx_YG = kx_2D[:,0]
    ky_YG = np.flip(ky_2D[:,0])

    kx = np.concatenate((kx_GX,kx_XS,kx_SY,kx_YG),axis=None)
    ky = np.concatenate((ky_GX,ky_XS,ky_SY,ky_YG),axis=None)

    return kx, ky

def Load_H_From_Save(H_filename):
    ham = np.loadtxt(H_filename, dtype=complex)
    dims = np.shape(ham)
    sqdim = int(np.sqrt(dims[0]))
    if(dims[1] == 5):
        alpha = ham[:,0].reshape(sqdim,sqdim)
        beta = ham[:,1].reshape(sqdim,sqdim)
        gamma = ham[:,2].reshape(sqdim,sqdim)
        delta11 = ham[:,3].reshape(sqdim,sqdim)
        delta1_min1 = ham[:,4].reshape(sqdim,sqdim)

        beta_dagger = np.transpose(beta)
        gamma_dagger = np.transpose(gamma)
        delta11_dagger = np.transpose(delta11)
        delta1_min1_dagger = np.transpose(delta1_min1)

        Hamiltonian = [alpha, beta, gamma, delta11, delta1_min1, beta_dagger, gamma_dagger, delta11_dagger, delta1_min1_dagger]
    elif(dims[1] == 8):
        alpha = ham[:,0].reshape(sqdim,sqdim)
        beta = ham[:,1].reshape(sqdim,sqdim)
        gamma = ham[:,2].reshape(sqdim,sqdim)
        gamma2 = ham[:,3].reshape(sqdim,sqdim)
        delta11 = ham[:,4].reshape(sqdim,sqdim)
        delta12 = ham[:,5].reshape(sqdim,sqdim)
        delta1_min1 = ham[:,6].reshape(sqdim,sqdim)
        delta1_min2 = ham[:,7].reshape(sqdim,sqdim)

        beta_dagger = np.transpose(beta)
        gamma_dagger = np.transpose(gamma)
        gamma2_dagger = np.transpose(gamma2)
        delta11_dagger = np.transpose(delta11)
        delta12_dagger = np.transpose(delta12)
        delta1_min1_dagger = np.transpose(delta1_min1)
        delta1_min2_dagger = np.transpose(delta1_min2)

        Hamiltonian = [alpha, beta, gamma, delta11, delta1_min1, beta_dagger, gamma_dagger, delta11_dagger, delta1_min1_dagger,
                    gamma2, delta12, delta1_min2, gamma2_dagger, delta12_dagger, delta1_min2_dagger]

    return Hamiltonian

def Calculate_Eigvals(a,b,kx,ky,hamiltonian):
    ham_dims = np.shape(hamiltonian)
    if(ham_dims[0] == 9):

        alpha = hamiltonian[0]
        beta = hamiltonian[1]
        gamma = hamiltonian[2]
        delta11 = hamiltonian[3]
        delta1_min1 = hamiltonian[4]

        beta_dagger = hamiltonian[5]
        gamma_dagger = hamiltonian[6]
        delta11_dagger = hamiltonian[7]
        delta1_min1_dagger = hamiltonian[8]

        E = np.zeros((ham_dims[1],len(kx)))
        for ii in range(len(kx)):
            H = alpha + beta*np.exp(1j*kx[ii]*a) + gamma*np.exp(1j*ky[ii]*b) + \
                delta11*np.exp(1j*kx[ii]*a + 1j*ky[ii]*b) + delta1_min1*np.exp(1j*kx[ii]*a - 1j*ky[ii]*b) + \
                beta_dagger*np.exp(-1j*kx[ii]*a) + gamma_dagger*np.exp(-1j*ky[ii]*b) + \
                delta11_dagger*np.exp(-1j*kx[ii]*a - 1j*ky[ii]*b) + delta1_min1_dagger*np.exp(-1j*kx[ii]*a + 1j*ky[ii]*b)
            eigvals, eigvecs = np.linalg.eig(H)

            E[:,ii] = np.real(np.sort((eigvals.T)))

    elif(ham_dims[0]== 15):
        alpha = hamiltonian[0]
        beta = hamiltonian[1]
        gamma = hamiltonian[2]
        delta11 = hamiltonian[3]
        delta1_min1 = hamiltonian[4]

        beta_dagger = hamiltonian[5]
        gamma_dagger = hamiltonian[6]
        delta11_dagger = hamiltonian[7]
        delta1_min1_dagger = hamiltonian[8]

        gamma2 = hamiltonian[9]
        delta12 = hamiltonian[10]
        delta1_min2 = hamiltonian[11]
        gamma2_dagger = hamiltonian[12]
        delta12_dagger = hamiltonian[13]
        delta1_min2_dagger = hamiltonian[14]


        E = np.zeros((ham_dims[1],len(kx)))
        for ii in range(len(kx)):
            H = alpha + beta*np.exp(1j*kx[ii]*a) + gamma*np.exp(1j*ky[ii]*b) + gamma2*np.exp(2j*ky[ii]*b) + \
                delta11*np.exp(1j*kx[ii]*a + 1j*ky[ii]*b) + delta12*np.exp(1j*kx[ii]*a + 2j*ky[ii]*b)  + \
                delta1_min1*np.exp(1j*kx[ii]*a - 1j*ky[ii]*b) + delta1_min2*np.exp(1j*kx[ii]*a - 2j*ky[ii]*b) + \
                beta_dagger*np.exp(-1j*kx[ii]*a) + gamma_dagger*np.exp(-1j*ky[ii]*b) + gamma2_dagger*np.exp(-2j*ky[ii]*b) + \
                delta11_dagger*np.exp(-1j*kx[ii]*a - 1j*ky[ii]*b) + delta12_dagger*np.exp(-1j*kx[ii]*a - 2j*ky[ii]*b) + \
                delta1_min1_dagger*np.exp(-1j*kx[ii]*a + 1j*ky[ii]*b) + delta1_min2_dagger*np.exp(-1j*kx[ii]*a + 2j*ky[ii]*b)
            
            eigvals, eigvecs = np.linalg.eig(H)

            E[:,ii] = np.real(np.sort((eigvals.T)))

    return E



def Plot_Bands(wan_filename,ml_filename, a, b):

    kx,ky = Create_Kpath(a,b)

    wan_ham = Load_H_From_Save(wan_filename)
    ml_ham = Load_H_From_Save(ml_filename)

    wan_E = Calculate_Eigvals(a,b,kx,ky,wan_ham)
    ml_E = Calculate_Eigvals(a,b,kx,ky,ml_ham)

    DeltaEc = min(ml_E[12,:])
    DeltaEv = -1*max(ml_E[11,:])
    print("Ec - Ef = ", DeltaEc, " eV")
    print("Ef - Ev = ", DeltaEv, " eV")
    print("Eg = ", DeltaEc + DeltaEv, " eV")

    plt.figure()
    x_vals = np.linspace(0,1,len(kx))
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['$\Gamma$', 'X', 'S', 'Y', '$\Gamma$'])
    plt.ylabel("E [eV]")
    legend_elements = [Line2D([0], [0], color='r', lw=3, label='Wannier Bands'),
                   Line2D([0], [0],color='b',lw=3,linestyle='dashed', label='ML Bands')]

    plt.legend(handles=legend_elements)

    plt.plot(x_vals, wan_E.T,'r')
    plt.plot(x_vals,ml_E.T,'b--')
    plt.show()

def Plot_Hamiltonian(H_filename):
    hamiltonian = Load_H_From_Save(H_filename)
    ham_dims = np.shape(hamiltonian)
    for i in range(ham_dims[0]):
        hamiltonian[i] = np.real(hamiltonian[i])

    ham_dims = np.shape(hamiltonian)
    if(ham_dims[0] == 9):
        num_bands = hamiltonian[0].shape[1]

        #Plotting matrix elements 
        H_map = np.zeros((num_bands*3,num_bands*3))
        #Delta1_min1_dagger
        H_map[:num_bands, :num_bands] = hamiltonian[8]

        #Beta_dagger
        H_map[num_bands:num_bands*2, :num_bands] = hamiltonian[5]

        #Delta11_dagger
        H_map[num_bands*2:num_bands*3, :num_bands] = hamiltonian[7]

        #Gamma
        H_map[:num_bands, num_bands:num_bands*2] = hamiltonian[2]

        #Alpha
        H_map[num_bands:num_bands*2, num_bands:num_bands*2] = hamiltonian[0]

        #Gamma_dagger
        H_map[num_bands*2:num_bands*3, num_bands:num_bands*2] = hamiltonian[6]

        #Delta11
        H_map[:num_bands, num_bands*2:num_bands*3] = hamiltonian[3]

        #Beta
        H_map[num_bands:num_bands*2, num_bands*2:num_bands*3] = hamiltonian[1]

        #Delta1_min1
        H_map[num_bands*2:num_bands*3, num_bands*2:num_bands*3] = hamiltonian[4]


        plt.figure()
        im = plt.imshow(np.real(H_map), cmap="seismic",vmin=-3, vmax=3)
        plt.colorbar(im)
        plt.axhline(y=num_bands-0.5,color='k')
        plt.axhline(y=num_bands*2-0.5,color='k')
        plt.axvline(x=num_bands-0.5,color='k')
        plt.axvline(x=num_bands*2-0.5,color='k')
        plt.show()

    elif(ham_dims[0] == 15):
        num_bands = hamiltonian[0].shape[1]

        #Plotting matrix elements 
        H_map = np.zeros((num_bands*5,num_bands*3))

        #Delta1_min2_dagger
        H_map[:num_bands, :num_bands] = hamiltonian[14]

        #Delta1_min1_dagger
        H_map[num_bands:num_bands*2, :num_bands] = hamiltonian[8]

        #Beta_dagger
        H_map[num_bands*2:num_bands*3, :num_bands] = hamiltonian[5]

        #Delta11_dagger
        H_map[num_bands*3:num_bands*4, :num_bands] = hamiltonian[7]

        #Delta12_dagger
        H_map[num_bands*4:num_bands*5, :num_bands] = hamiltonian[13]

        #Gamma2
        H_map[:num_bands, num_bands:num_bands*2] = hamiltonian[9]

        #Gamma
        H_map[num_bands:num_bands*2, num_bands:num_bands*2] = hamiltonian[2]

        #Alpha
        H_map[num_bands*2:num_bands*3, num_bands:num_bands*2] = hamiltonian[0]

        #Gamma_dagger
        H_map[num_bands*3:num_bands*4, num_bands:num_bands*2] = hamiltonian[6]

        #Gamma2_dagger
        H_map[num_bands*4:num_bands*5, num_bands:num_bands*2] = hamiltonian[12]

        #Delta12
        H_map[:num_bands, num_bands*2:num_bands*3] = hamiltonian[10]

        #Delta11
        H_map[num_bands:num_bands*2, num_bands*2:num_bands*3] = hamiltonian[3]

        #Beta
        H_map[num_bands*2:num_bands*3, num_bands*2:num_bands*3] = hamiltonian[1]

        #Delta1_min1
        H_map[num_bands*3:num_bands*4, num_bands*2:num_bands*3] = hamiltonian[4]

        #Delta1_min2
        H_map[num_bands*4:num_bands*5, num_bands*2:num_bands*3] = hamiltonian[11]



        plt.figure()
        im = plt.imshow(np.real(H_map), cmap="seismic",vmin=-3, vmax=3)
        plt.colorbar(im)
        plt.axhline(y=num_bands-0.5,color='k')
        plt.axhline(y=num_bands*2-0.5,color='k')
        plt.axhline(y=num_bands*3-0.5,color='k')
        plt.axhline(y=num_bands*4-0.5,color='k')
        plt.axvline(x=num_bands-0.5,color='k')
        plt.axvline(x=num_bands*2-0.5,color='k')
        plt.show()


a = 6.290339029863558E-10
b = 3.631729244941925E-10

Plot_Bands('inputs/HfS2_1L_BigGamma.dat','H_output/HfS2_fitwan_6.dat',a,b)
Plot_Hamiltonian('H_output/HfS2_fitwan_6.dat')
