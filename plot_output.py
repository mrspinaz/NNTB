import numpy as np
import re
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b):

        file_dir = 'inputs/' + bands_filename

        with open(file_dir) as file:
            first_line = file.readline().strip()
           
            temp = re.findall('\d+',first_line)
            nband = int(temp[0])
            nks = int(temp[1])
            kpoints = np.zeros((2,nks))
            abinit_bands = np.zeros(nband*nks)

            k_count = 0
            start = 0
            for line in file:
                line_split = line.split()
                line_split = [float(x) for x in line_split]
                if( len(line_split) == 3 and line_split[2] == 0.0 ):
                    kpoints[:,k_count] = line_split[0], line_split[1]
                    k_count += 1
                    
                else:
                    #reading energy eigenvals
                    abinit_bands[start:start + len(line_split)] = line_split
                    start += len(line_split)

            abinit_bands = abinit_bands - Ef
            abinit_bands = abinit_bands.reshape(nks,nband)
            truncated_bands = abinit_bands[:, skip_bands:(skip_bands + target_bands)]        
            

        file.close()

        #Convert k_points to their actual values
        ky_factor = a/b
        kpoints[0,:] =  (kpoints[0,:])*(2.0*np.pi/a)
        kpoints[1,:] =  (kpoints[1,:]/ky_factor)*(2.0*np.pi/b)

        return nband, nks, truncated_bands, kpoints


def Plot_CB_Surf(bands_filename, skip_bands, target_bands, Ef, a, b):
    nband, nks, truncated_bands, kpoints = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)
    kx = kpoints[0,:].reshape(31,31)
    ky = kpoints[1,:].reshape(31,31)

  
    Z = truncated_bands[:,1].reshape(len(kx),len(ky))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kx,ky,Z)
    plt.show()


def Plot_Bands(bands_filename, skip_bands, target_bands, Ef, a, b):

    alpha = np.real(np.loadtxt('H_output/alpha.txt', dtype=complex))
    beta = np.loadtxt('H_output/beta.txt', dtype=complex)
    gamma = np.loadtxt('H_output/gamma.txt', dtype=complex)
    delta11 = np.loadtxt('H_output/delta11.txt', dtype=complex)
    delta1_min1 = np.loadtxt('H_output/delta1_min1.txt', dtype=complex)

    beta_dagger = np.loadtxt('H_output/beta_dagger.txt', dtype=complex)
    gamma_dagger = np.loadtxt('H_output/gamma_dagger.txt', dtype=complex)
    delta11_dagger = np.loadtxt('H_output/delta11_dagger.txt', dtype=complex)
    delta1_min1_dagger = np.loadtxt('H_output/delta1_min1_dagger.txt', dtype=complex)

    a = 6.290339483e-10
    b = 3.631729507E-10


    kx_GX = np.linspace(0,np.pi/a, 32)
    ky_GX = np.linspace(0,0, 32)

    kx_XS = np.linspace(np.pi/a,np.pi/a, 32)
    ky_XS = np.linspace(0,np.pi/(b), 32)

    kx_SY = np.linspace(np.pi/a,0, 32)
    ky_SY = np.linspace(np.pi/b,np.pi/b, 32)

    kx_YG = np.linspace(0,0, 33)
    ky_YG = np.linspace(np.pi/b,0, 33)


    kx = np.concatenate((kx_GX,kx_XS,kx_SY,kx_YG), axis=None)
    ky = np.concatenate((ky_GX,ky_XS,ky_SY,ky_YG), axis=None)

    E = np.zeros((alpha.shape[1],len(kx)))


    for ii in range(len(kx)):
        H = alpha + beta*np.exp(1j*kx[ii]*a) + gamma*np.exp(1j*ky[ii]*b) + \
            delta11*np.exp(1j*kx[ii]*a + 1j*ky[ii]*b) + delta1_min1*np.exp(1j*kx[ii]*a - 1j*ky[ii]*b) + \
            beta_dagger*np.exp(-1j*kx[ii]*a) + gamma_dagger*np.exp(-1j*ky[ii]*b) + \
            delta11_dagger*np.exp(-1j*kx[ii]*a - 1j*ky[ii]*b) + delta1_min1_dagger*np.exp(-1j*kx[ii]*a + 1j*ky[ii]*b)
        eigvals, eigvecs = np.linalg.eig(H)

        E[:,ii] = np.sort((eigvals.T))

    plt.figure()
    plt.plot(E.T,'b--')
    plt.show()


Plot_CB_Surf('HfS2_IBZ_bands.dat',12,18,-2.5834, 6.290339483e-10, 3.631729507E-10)