import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
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

        return nband, nks, truncated_bands, kpoints

def Load_Hamiltonian():
    alpha = np.real(np.loadtxt('H_output/alpha.txt', dtype=complex))
    beta = np.loadtxt('H_output/beta.txt', dtype=complex)
    gamma = np.loadtxt('H_output/gamma.txt', dtype=complex)
    gamma2 = np.loadtxt('H_output/gamma2.txt', dtype=complex)
    delta11 = np.loadtxt('H_output/delta11.txt', dtype=complex)
    delta1_min1 = np.loadtxt('H_output/delta1_min1.txt', dtype=complex)
    delta12 = np.loadtxt('H_output/delta12.txt', dtype=complex)
    delta1_min2 = np.loadtxt('H_output/delta1_min2.txt', dtype=complex)

    beta_dagger = np.loadtxt('H_output/beta_dagger.txt', dtype=complex)
    gamma_dagger = np.loadtxt('H_output/gamma_dagger.txt', dtype=complex)
    gamma2_dagger = np.loadtxt('H_output/gamma2_dagger.txt', dtype=complex)
    delta11_dagger = np.loadtxt('H_output/delta11_dagger.txt', dtype=complex)
    delta1_min1_dagger = np.loadtxt('H_output/delta1_min1_dagger.txt', dtype=complex)
    delta12_dagger = np.loadtxt('H_output/delta12_dagger.txt', dtype=complex)
    delta1_min2_dagger = np.loadtxt('H_output/delta1_min2_dagger.txt', dtype=complex)

    Hamiltonian = [alpha, beta, gamma, delta11, delta1_min1, beta_dagger, gamma_dagger, delta11_dagger, delta1_min1_dagger,
                   gamma2, delta12, delta1_min2, gamma2_dagger, delta12_dagger, delta1_min2_dagger]

    return Hamiltonian

def Load_H_From_Save(H_filename):
    ham = np.loadtxt('H_output/' + H_filename, dtype=complex)
    dims = np.shape(ham)
    sqdim = int(np.sqrt(dims[0]))
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
    

def Plot_CB_Surf(bands_filename, skip_bands, target_bands, Ef, a, b):
    nband, nks, truncated_bands, kpoints = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)
    kx = kpoints[0,:]
    ky = kpoints[1,:]
    print(ky/(np.pi/b))
    nks = len(kpoints[0,:])

    kx_plot = kx.reshape(int(np.sqrt(nks)),int(np.sqrt(nks)))*(a/(2*np.pi))
    ky_plot = ky.reshape(25,25)*(b/(2*np.pi))
    #Z2 = truncated_bands[:,12].reshape(len(kx),len(ky))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Valence: ", truncated_bands[(np.abs(truncated_bands[:,5] - Ef)).argmin(),5])
    print("Conduction: ", truncated_bands[(np.abs(truncated_bands[:,6] - Ef)).argmin(),6])


    ax.plot_surface(kx_plot,ky_plot,truncated_bands[:,5].reshape(len(kx_plot),len(ky_plot)), cmap=cm.viridis)
    #ax.plot_surface(kx_plot,ky_plot,truncated_bands[:,6].reshape(len(kx_plot),len(ky_plot)), color='blue')
    plt.show()

    '''
    hamiltonian = Load_Hamiltonian()
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

        E[:,ii] = np.sort((eigvals.T))
    
    E = E.T
    ax.scatter(kx_plot,ky_plot,E[:,11].reshape(len(kx_plot),len(ky_plot)), s=2, color='red')
    ax.scatter(kx_plot,ky_plot,E[:,12].reshape(len(kx_plot),len(ky_plot)), s=2, color='red')



    #ax.plot_surface(kx,ky,Z2,cmap=cm.coolwarm)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E [eV]')
    plt.show()
'''

def Plot_Bands(bands_filename, H_filename, skip_bands, target_bands, Ef, a, b):

    nband, nks, truncated_bands, kpoints = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)

    adjust_bandgap = True
    experimental_bandgap = 1.9
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

    gamma2 = hamiltonian[9]
    delta12 = hamiltonian[10]
    delta1_min2 = hamiltonian[11]
    gamma2_dagger = hamiltonian[12]
    delta12_dagger = hamiltonian[13]
    delta1_min2_dagger = hamiltonian[14]


    E = np.zeros((target_bands,len(kx)))
    for ii in range(len(kx)):
        H = alpha + beta*np.exp(1j*kx[ii]*a) + gamma*np.exp(1j*ky[ii]*b) + gamma2*np.exp(2j*ky[ii]*b) + \
            delta11*np.exp(1j*kx[ii]*a + 1j*ky[ii]*b) + delta12*np.exp(1j*kx[ii]*a + 2j*ky[ii]*b)  + \
            delta1_min1*np.exp(1j*kx[ii]*a - 1j*ky[ii]*b) + delta1_min2*np.exp(1j*kx[ii]*a - 2j*ky[ii]*b) + \
            beta_dagger*np.exp(-1j*kx[ii]*a) + gamma_dagger*np.exp(-1j*ky[ii]*b) + gamma2_dagger*np.exp(-2j*ky[ii]*b) + \
            delta11_dagger*np.exp(-1j*kx[ii]*a - 1j*ky[ii]*b) + delta12_dagger*np.exp(-1j*kx[ii]*a - 2j*ky[ii]*b) + \
            delta1_min1_dagger*np.exp(-1j*kx[ii]*a + 1j*ky[ii]*b) + delta1_min2_dagger*np.exp(-1j*kx[ii]*a + 2j*ky[ii]*b)
        
        eigvals, eigvecs = np.linalg.eig(H)

        E[:,ii] = np.sort((eigvals.T))

    E[c:,:] =  E[c:,:] + 0.6
    E[0:v+1,:] =  E[0:v+1,:] +0.05

    plt.figure(figsize=(8, 8),dpi=100)
    plt.rcParams.update({'font.size': 22})
    x_vals = np.linspace(0,1,len(kx))
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['$\Gamma$', 'X', 'S', 'Y', '$\Gamma$'])
    plt.ylabel("E - E$_F$ [eV]")
    plt.plot(x_vals, truncated_bands,'r')
    plt.plot(x_vals,E.T,'go',markevery=3,markersize=6,alpha=0.8)
    plt.ylim([-3,3])
    plt.margins(x=0)

    line = Line2D([0], [0], label='DFT', color='r')
    dot = Line2D([0], [0], label='MLWF', color='g',marker='o',markersize=5,linestyle='')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [line, dot]
    plt.legend(handles=handles)
    plt.show()

def Plot_Hamiltonian(H_filename):
    hamiltonian = Load_H_From_Save(H_filename)


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




a = 6.29033902986356e-10
b = 3.631729244941925e-10
#old fermi was -2.5834
plt.close("all")
#Plot_CB_Surf('HfS2_31x31_bands.dat',12,18, -2.5834 , 6.290339483e-10, 3.631729507E-10)
#Plot_Bands('HfS2_GXSYG_bands.dat',12,18, -2.5834, 6.290339483e-10, 3.631729507E-10)
Plot_Bands('HfS2_bands.dat','HfS2_1L_BigGammaWan_unadjusted_2.dat',12,22,-3.3312, a, b)
#Plot_CB_Surf('2L_Te_31x31_bands.dat',12,12, -1.9463, 5.7885636E-10, 4.3185190E-10)
#Plot_Hamiltonian('3L_Te_DoubleGamma_FullIBZFit_L1e-5_weighted_21x21_27bands_14.dat')
