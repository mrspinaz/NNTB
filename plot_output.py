import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import cm
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
        kpoints[0,:] =  (kpoints[0,:])*(2.0*np.pi/a)
        kpoints[1,:] =  (kpoints[1,:]/ky_factor)*(2.0*np.pi/b)  

        return nband, nks, truncated_bands, kpoints

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



def Plot_CB_Surf(bands_filename, skip_bands, target_bands, Ef, a, b):
    nband, nks, truncated_bands, kpoints = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)
    kx = kpoints[0,:]
    ky = kpoints[1,:]
    nks = len(kpoints[0,:])

    kx_plot = kx.reshape(int(np.sqrt(nks)),int(np.sqrt(nks)))*(a/(2*np.pi))
    ky_plot = ky.reshape(31,31)*(b/(2*np.pi))
    #Z2 = truncated_bands[:,12].reshape(len(kx),len(ky))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Valence: ", truncated_bands[(np.abs(truncated_bands[:,11] - Ef)).argmin(),11])
    print("Conduction: ", truncated_bands[(np.abs(truncated_bands[:,12] - Ef)).argmin(),12])


    ax.plot_surface(kx_plot,ky_plot,truncated_bands[:,11].reshape(len(kx_plot),len(ky_plot)), color='blue')
    ax.plot_surface(kx_plot,ky_plot,truncated_bands[:,12].reshape(len(kx_plot),len(ky_plot)), color='blue')


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

def Compare_TB_Bands(hamiltonian_filename,a,b):


    file_name = hamiltonian_filename
    data = np.loadtxt(file_name)
    num_neighbours = int(len(data[0,:]))
    dim = int(np.sqrt(len(data[:,0])))
    if(num_neighbours == 5):
        #nearest neighbour
        alpha = np.reshape(data[:,0],(dim,dim))
        beta = np.reshape(data[:,1],(dim,dim))
        gamma = np.reshape(data[:,2],(dim,dim))
        delta11 = np.reshape(data[:,3],(dim,dim))
        delta1_min_1 = np.reshape(data[:,4],(dim,dim))

    
    ml_ham = Load_Hamiltonian()
    ml_alpha = ml_ham[0]
    ml_beta = ml_ham[1]
    ml_gamma = ml_ham[2]
    ml_delta11 = ml_ham[3]
    ml_delta1_min1 = ml_ham[4]

    ml_beta_dagger = ml_ham[5]
    ml_gamma_dagger = ml_ham[6]
    ml_delta11_dagger = ml_ham[7]
    ml_delta1_min1_dagger = ml_ham[8]

    surface_plot = False
    if(surface_plot == True):


        grid_density = 30
        kxx = np.linspace(0,np.pi/a,grid_density)
        kyy = np.linspace(0,np.pi/b,grid_density)
        xx,yy = np.meshgrid(kxx,kyy)

        nks = grid_density**2
        kpoints = np.zeros((2,nks))
        count = 0
        for row in range(grid_density):
            for col in range(grid_density):
                kpoints[0,count] =  xx[row,col]
                kpoints[1,count] =  yy[row,col]
                count += 1

        mat_dims = np.shape(alpha)
        E = np.zeros((len(kpoints[0]),mat_dims[0]))
        ml_E = np.zeros((len(kpoints[0]),ml_alpha.shape[0]))
        for ii in range(len(kpoints[0])):
            H = alpha + beta*np.exp(1j*kpoints[0][ii]*a) + gamma*np.exp(1j*kpoints[1][ii]*b) + \
                delta11*np.exp(1j*kpoints[0][ii]*a + 1j*kpoints[1][ii]*b) + delta1_min_1*np.exp(1j*kpoints[0][ii]*a - 1j*kpoints[1][ii]*b) + \
                (beta*np.exp(1j*kpoints[0][ii]*a) + gamma*np.exp(1j*kpoints[1][ii]*b) + \
                delta11*np.exp(1j*kpoints[0][ii]*a + 1j*kpoints[1][ii]*b) + delta1_min_1*np.exp(1j*kpoints[0][ii]*a - 1j*kpoints[1][ii]*b)).conj().T
            eigvals, eigvecs = np.linalg.eig(H)
            E[ii,:] = np.sort((eigvals))

            H2 = ml_alpha + ml_beta*np.exp(1j*kpoints[0][ii]*a) + ml_gamma*np.exp(1j*kpoints[1][ii]*b) + \
                ml_delta11*np.exp(1j*kpoints[0][ii]*a + 1j*kpoints[1][ii]*b) + ml_delta1_min1*np.exp(1j*kpoints[0][ii]*a - 1j*kpoints[1][ii]*b) + \
                ml_beta_dagger*np.exp(-1j*kpoints[0][ii]*a) + ml_gamma_dagger*np.exp(-1j*kpoints[1][ii]*b) + \
                ml_delta11_dagger*np.exp(-1j*kpoints[0][ii]*a - 1j*kpoints[1][ii]*b) + ml_delta1_min1_dagger*np.exp(-1j*kpoints[0][ii]*a + 1j*kpoints[1][ii]*b)
            
            ml_eigvals, eigvecs = np.linalg.eig(H2)

            ml_E[ii,:] = np.sort((ml_eigvals))


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        print(yy.shape)
        ax.plot_surface(xx,yy,E[:,0].reshape(grid_density,grid_density),color='blue')
        ax.plot_surface(xx,yy,E[:,1].reshape(grid_density,grid_density),color='blue')
        ax.plot_surface(xx,yy,E[:,2].reshape(grid_density,grid_density),color='blue')
        ax.plot_surface(xx,yy,E[:,3].reshape(grid_density,grid_density),color='blue')

        ax.scatter(xx,yy,ml_E[:,0].reshape(grid_density,grid_density),color='red')
        ax.scatter(xx,yy,ml_E[:,1].reshape(grid_density,grid_density),color='red')
        ax.scatter(xx,yy,ml_E[:,2].reshape(grid_density,grid_density),color='red')
        ax.scatter(xx,yy,ml_E[:,3].reshape(grid_density,grid_density),color='red')
        plt.show()


    else:
        nks = 200
        nk_segement = int(((nks-1))/4)

        kx_GX = np.linspace(0,np.pi/a, nk_segement)
        ky_GX = np.linspace(0,0, nk_segement)

        kx_XS = np.linspace(np.pi/a,np.pi/a, nk_segement)
        ky_XS = np.linspace(0,np.pi/(b), nk_segement)

        kx_SY = np.linspace(np.pi/a,0, nk_segement)
        ky_SY = np.linspace(np.pi/b,np.pi/b, nk_segement)

        kx_YG = np.linspace(0,0, nk_segement)
        ky_YG = np.linspace(np.pi/b,0, nk_segement)

        kx_IBZ = np.concatenate((kx_GX,kx_XS,kx_SY,kx_YG), axis=None)
        ky_IBZ = np.concatenate((ky_GX,ky_XS,ky_SY,ky_YG), axis=None)

        kpoints = np.zeros((2,len(kx_IBZ)))
        kpoints[0,:] = kx_IBZ
        kpoints[1,:] = ky_IBZ

        mat_dims = np.shape(alpha)
        E = np.zeros((len(kpoints[0]),mat_dims[0]))
        ml_E = np.zeros((len(kpoints[0]),ml_alpha.shape[0]))
        for ii in range(len(kpoints[0])):
            H = alpha + beta*np.exp(1j*kpoints[0][ii]*a) + gamma*np.exp(1j*kpoints[1][ii]*b) + \
                delta11*np.exp(1j*kpoints[0][ii]*a + 1j*kpoints[1][ii]*b) + delta1_min_1*np.exp(1j*kpoints[0][ii]*a - 1j*kpoints[1][ii]*b) + \
                (beta*np.exp(1j*kpoints[0][ii]*a) + gamma*np.exp(1j*kpoints[1][ii]*b) + \
                delta11*np.exp(1j*kpoints[0][ii]*a + 1j*kpoints[1][ii]*b) + delta1_min_1*np.exp(1j*kpoints[0][ii]*a - 1j*kpoints[1][ii]*b)).conj().T
            eigvals, eigvecs = np.linalg.eig(H)
            E[ii,:] = np.sort((eigvals))

            H2 = ml_alpha + ml_beta*np.exp(1j*kpoints[0][ii]*a) + ml_gamma*np.exp(1j*kpoints[1][ii]*b) + \
                ml_delta11*np.exp(1j*kpoints[0][ii]*a + 1j*kpoints[1][ii]*b) + ml_delta1_min1*np.exp(1j*kpoints[0][ii]*a - 1j*kpoints[1][ii]*b) + \
                ml_beta_dagger*np.exp(-1j*kpoints[0][ii]*a) + ml_gamma_dagger*np.exp(-1j*kpoints[1][ii]*b) + \
                ml_delta11_dagger*np.exp(-1j*kpoints[0][ii]*a - 1j*kpoints[1][ii]*b) + ml_delta1_min1_dagger*np.exp(-1j*kpoints[0][ii]*a + 1j*kpoints[1][ii]*b)
            
            ml_eigvals, eigvecs = np.linalg.eig(H2)

            ml_E[ii,:] = np.sort((ml_eigvals))
        
        xvals = np.linspace(0,1,len(kx_IBZ))
        plt.plot(xvals, E,'b')
        plt.plot(xvals, ml_E, 'r--')
        plt.show()
    


def Plot_Bands(bands_filename, skip_bands, target_bands, Ef, a, b):

    nband, nks, truncated_bands, kpoints = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)

    adjust_bandgap = True
    experimental_bandgap = 1.5419
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

    plt.figure()
    x_vals = np.linspace(0,1,len(kx))
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['$\Gamma$', 'X', 'S', 'Y', '$\Gamma$'])
    plt.ylabel("E [eV]")
    plt.plot(x_vals, truncated_bands,'r')
    plt.plot(x_vals,E.T,'b--')
    plt.show()

def Plot_Hamiltonian():
    hamiltonian = Load_Hamiltonian()


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
#Plot_CB_Surf('HfS2_31x31_bands.dat',12,18, -2.5834 , 6.290339483e-10, 3.631729507E-10)
#Plot_Bands('HfS2_GXSYG_bands.dat',12,18, -2.5834, 6.290339483e-10, 3.631729507E-10)
#Plot_Hamiltonian()
Compare_TB_Bands('inputs/BP/BP.dat',4.43e-10,3.27e-10)
Plot_Hamiltonian()
