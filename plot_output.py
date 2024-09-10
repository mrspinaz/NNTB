import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product

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

        #shift band energies to set fermi level at midgap
        first_eigval_set = truncated_bands[1,:]
        pos_eigvals = [a for a in first_eigval_set if a> 0]
        neg_eigvals = [a for a in first_eigval_set if a < 0]

        pos_smallest = min(pos_eigvals, key=abs)
        neg_smallest = min(neg_eigvals, key=abs)
        c = int(np.where(first_eigval_set == pos_smallest)[0])
        v = int(np.where(first_eigval_set == neg_smallest)[0])

        conduction_band = truncated_bands[:,c]
        valence_band = truncated_bands[:,v]
        print(c,v)
        bandgap = np.min(conduction_band) - np.max(valence_band)
        

        #Additional energy shift applied to position Ef at midgap
        cb_shift = abs(abs(np.min(conduction_band)) - bandgap/2.0)
        vb_shift = abs(abs(np.max(valence_band)) - bandgap/2.0)
        print(cb_shift, " ", vb_shift)
        if( abs(np.min(conduction_band)) < abs(np.max(valence_band))):
            truncated_bands[:,c:] += cb_shift
            truncated_bands[:,0:v+1] += vb_shift
        else:
            truncated_bands[:,c:] -= cb_shift
            truncated_bands[:,0:v+1] -= vb_shift   
            

        file.close()

        #Convert k_points to their actual values
        ky_factor = a/b
        kpoints[0,:] =  abs((kpoints[0,:])*(2.0*np.pi/a))
        kpoints[1,:] =  abs((kpoints[1,:]/ky_factor)*(2.0*np.pi/b) ) 

        return nband, nks, truncated_bands, kpoints,c,v

def Load_Hamiltonian():
    alpha = np.real(np.loadtxt('H_output/alpha.txt', dtype=complex))
    beta = np.loadtxt('H_output/beta.txt', dtype=complex)
    gamma = np.loadtxt('H_output/gamma.txt', dtype=complex)
    delta11 = np.loadtxt('H_output/delta11.txt', dtype=complex)
    delta1_min1 = np.loadtxt('H_output/delta1_min1.txt', dtype=complex)

    beta_dagger = np.loadtxt('H_output/beta_dagger.txt', dtype=complex)
    gamma_dagger = np.loadtxt('H_output/gamma_dagger.txt', dtype=complex)
    delta11_dagger = np.loadtxt('H_output/delta11_dagger.txt', dtype=complex)
    delta1_min1_dagger = np.loadtxt('H_output/delta1_min1_dagger.txt', dtype=complex)

    Hamiltonian = [alpha, beta, gamma, delta11, delta1_min1, beta_dagger, gamma_dagger, delta11_dagger, delta1_min1_dagger]

    return Hamiltonian

def Load_H_From_Save(H_filename):
    ham = np.loadtxt('H_output/' + H_filename, dtype=complex)
    dims = np.shape(ham)
    sqdim = int(np.sqrt(dims[0]))
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

    return Hamiltonian

def Plot_CB_Surf(bands_filename,H_filename, skip_bands, target_bands, Ef, a, b):
    nband, nks, truncated_bands, kpoints,c, v = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)
    
    adjust_bandgap = True
    experimental_bandgap = 2.07 #0.74 for 3L_Te
    if(adjust_bandgap):
        first_eigval_set = truncated_bands[1,:]
        pos_eigvals = [a for a in first_eigval_set if a> 0]
        neg_eigvals = [a for a in first_eigval_set if a < 0]

        pos_smallest = min(pos_eigvals, key=abs)
        neg_smallest = min(neg_eigvals, key=abs)
        c = int(np.where(first_eigval_set == pos_smallest)[0])
        v = int(np.where(first_eigval_set == neg_smallest)[0])

        conduction_band = truncated_bands[:,c]
        valence_band = truncated_bands[:,v]

        bandgap = np.min(conduction_band) - np.max(valence_band)
        print("Ec = " , np.min(conduction_band) , "Ev = " , np.max(valence_band))
        bandgap_shift = experimental_bandgap - bandgap
        
        truncated_bands[:,c:] += bandgap_shift/2 
        truncated_bands[:,0:v+1] -= bandgap_shift/2 
    
    
    kx = kpoints[0,:]
    ky = kpoints[1,:]
    
    nks = len(kpoints[0,:])

    kx_plot = kx.reshape(int(np.sqrt(nks)),int(np.sqrt(nks)))*(a/(2*np.pi))
    ky_plot = ky.reshape(int(np.sqrt(nks)),int(np.sqrt(nks)))*(b/(2*np.pi))
    #Z2 = truncated_bands[:,12].reshape(len(kx),len(ky))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Valence: ", truncated_bands[(np.abs(truncated_bands[:,13] - Ef)).argmin(),13])
    print("Conduction: ", truncated_bands[(np.abs(truncated_bands[:,14] - Ef)).argmin(),14])

    #ax.plot_surface(kx_plot,ky_plot,truncated_bands[:,c].reshape(len(kx_plot),len(ky_plot)), cmap=cm.viridis)
    #ax.plot_surface(kx_plot,ky_plot,truncated_bands[:,v].reshape(len(kx_plot),len(ky_plot)), cmap=cm.viridis)
    #ax.plot_surface(kx_plot,ky_plot,truncated_bands[:,14].reshape(len(kx_plot),len(ky_plot)), color='blue')
    


    hamiltonian = Load_H_From_Save(H_filename)
    alpha = hamiltonian[0]
    beta = hamiltonian[1]
    gamma = hamiltonian[2]
    delta11 = hamiltonian[3]
    delta1_min1 = hamiltonian[4]

    beta_dagger = hamiltonian[5]
    gamma_dagger = hamiltonian[6]
    delta11_dagger = hamiltonian[7]
    delta1_min1_dagger = hamiltonian[8]

    kx_new = np.linspace(-1*np.pi/a,1*np.pi/a,31)
    ky_new = np.linspace(-1*np.pi/b,1*np.pi/b,31)

    kx_2D, ky_2D = np.meshgrid(kx_new,ky_new)
    kx = np.reshape(kx_2D, np.size(kx_2D),order='C')
    ky = np.reshape(ky_2D, np.size(ky_2D),order='C')


    E = np.zeros((target_bands,len(kx)))
    for ii in range(len(kx)):
        H = alpha + beta*np.exp(1j*kx[ii]*a) + gamma*np.exp(1j*ky[ii]*b) + \
            delta11*np.exp(1j*kx[ii]*a + 1j*ky[ii]*b) + delta1_min1*np.exp(1j*kx[ii]*a - 1j*ky[ii]*b) + \
            beta_dagger*np.exp(-1j*kx[ii]*a) + gamma_dagger*np.exp(-1j*ky[ii]*b) + \
            delta11_dagger*np.exp(-1j*kx[ii]*a - 1j*ky[ii]*b) + delta1_min1_dagger*np.exp(-1j*kx[ii]*a + 1j*ky[ii]*b)
        eigvals, eigvecs = np.linalg.eig(H)

        E[:,ii] = np.sort((eigvals.T))
    
    E = E.T


    kx_2D = kx_2D*(a/(2*np.pi))
    ky_2D = ky_2D*(b/(2*np.pi))

    #ax.scatter(kx_2D,ky_2D,E[:,c].reshape(len(kx_plot),len(ky_plot)), s=5, color='red')
    #ax.scatter(kx_2D,ky_2D,E[:,v].reshape(len(kx_plot),len(ky_plot)), s=5, color='red')
    #ax.scatter(kx_2D,ky_2D,E[:,14].reshape(len(kx_plot),len(ky_plot)), s=3, color='red')
    

    #ax.plot_surface(kx_2D,ky_2D,E[:,c].reshape(len(kx_plot),len(ky_plot)), cmap=cm.viridis)
    ax.plot_surface(kx_2D,ky_2D,E[:,v].reshape(31,31), cmap=cm.viridis)
    #ax.plot_surface(kx_plot,ky_plot,E[:,14].reshape(len(kx_plot),len(ky_plot)), color='red')



    #ax.plot_surface(kx,ky,Z2,cmap=cm.coolwarm)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E [eV]')
    plt.show()


def Plot_Bands(bands_filename, H_filename, skip_bands, target_bands, Ef, a, b):

    nband, nks, truncated_bands, kpoints,c,v = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)

    adjust_bandgap = True
    experimental_bandgap = 1.89638 #0.74 for 3L_Te
    if(adjust_bandgap):
        first_eigval_set = truncated_bands[1,:]
        pos_eigvals = [a for a in first_eigval_set if a> 0]
        neg_eigvals = [a for a in first_eigval_set if a < 0]

        pos_smallest = min(pos_eigvals, key=abs)
        neg_smallest = min(neg_eigvals, key=abs)
        c = int(np.where(first_eigval_set == pos_smallest)[0])
        v = int(np.where(first_eigval_set == neg_smallest)[0])

        conduction_band = truncated_bands[:,c]
        valence_band = truncated_bands[:,v]

        bandgap = np.min(conduction_band) - np.max(valence_band)
        print("Ec = " , np.min(conduction_band) , "Ev = " , np.max(valence_band))
        bandgap_shift = experimental_bandgap - bandgap
        
        truncated_bands[:,c:] += bandgap_shift/2 
        truncated_bands[:,0:v+1] -= bandgap_shift/2 

        #For testing
        new_conduction_band = truncated_bands[:,c]
        new_valence_band = truncated_bands[:,v]
        print("Ec = " , np.min(new_conduction_band) , "Ev = " , np.max(new_valence_band))
    
    kx = kpoints[0,:]
    ky = kpoints[1,:]
    print(len(ky))

    hamiltonian = Load_H_From_Save(H_filename)
    alpha = hamiltonian[0]
    beta = hamiltonian[1]
    gamma = hamiltonian[2]
    delta11 = hamiltonian[3]
    delta1_min1 = hamiltonian[4]

    beta_dagger = hamiltonian[5]
    gamma_dagger = hamiltonian[6]
    delta11_dagger = hamiltonian[7]
    delta1_min1_dagger = hamiltonian[8]

    E = np.zeros((target_bands,len(kx)))
    for ii in range(len(kx)):
        H = alpha + beta*np.exp(1j*kx[ii]*a) + gamma*np.exp(1j*ky[ii]*b) + \
            delta11*np.exp(1j*kx[ii]*a + 1j*ky[ii]*b) + delta1_min1*np.exp(1j*kx[ii]*a - 1j*ky[ii]*b) + \
            beta_dagger*np.exp(-1j*kx[ii]*a) + gamma_dagger*np.exp(-1j*ky[ii]*b) + \
            delta11_dagger*np.exp(-1j*kx[ii]*a - 1j*ky[ii]*b) + delta1_min1_dagger*np.exp(-1j*kx[ii]*a + 1j*ky[ii]*b)
        eigvals, eigvecs = np.linalg.eig(H)

        E[:,ii] = np.real(np.sort((eigvals.T)))
    
    DeltaEc = min(E[c,:])
    DeltaEv = -1*max(E[v,:])
    print("Ec - Ef = ", DeltaEc, " eV")
    print("Ef - Ev = ", DeltaEv, " eV")
    print("Eg = ", DeltaEc + DeltaEv, " eV")

    plt.figure()
    x_vals = np.linspace(0,1,len(kx))
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['$\Gamma$', 'X', 'S', 'Y', '$\Gamma$'])
    plt.ylabel("E [eV]")
    plt.plot(x_vals, truncated_bands,'r')
    plt.plot(x_vals,E.T,'b--')
    plt.show()

def Plot_Hamiltonian(H_filename):
    hamiltonian = Load_H_From_Save(H_filename)
    dims = np.shape(hamiltonian)
    for i in range(dims[0]):
        hamiltonian[i] = np.real(hamiltonian[i])

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


#old fermi was -2.5834
plt.close("all")


a = 6.290339029863558E-10
b = 3.631729244941925E-10
Ef = -3.3312
target_bands = 22
skip_bands = 12
H_filename = 'HfS2_DFTfit_noreg.dat'
DFT_filename = 'HfS2_bands.dat'

#Plot_CB_Surf('HfS2_31x31_bands.dat',12,18, -2.5834 , 6.290339483e-10, 3.631729507E-10)
#Plot_Bands('HfS2_GXSYG_bands.dat',12,18, -2.5834, 6.290339483e-10, 3.631729507E-10)
Plot_Bands(DFT_filename,H_filename,skip_bands,target_bands,Ef, a, b)
#Plot_CB_Surf(DFT_filename,H_filename,30,24, 1.3267, a, b)
Plot_Hamiltonian(H_filename)


#3L vals
a = 5.84536306E-10
b = 4.36037009E-10
Ef = -0.5292