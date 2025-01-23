import numpy as np
import re
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from itertools import product
import scipy.optimize as opt
import seaborn as sns

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
        
        pos_eigvals = [a for a in first_eigval_set if a > 0]
        neg_eigvals = [a for a in first_eigval_set if a < 0]
        
        pos_smallest = min(pos_eigvals, key=abs)
        neg_smallest = min(neg_eigvals, key=abs)

       
      
        c = int(np.where(first_eigval_set == pos_smallest)[0][0])
        v = int(np.where(first_eigval_set == neg_smallest)[0][0])
        print('valence index ', v)
        conduction_band = truncated_bands[:,c]
        valence_band = truncated_bands[:,v]
        #print(c,v)
        bandgap = np.min(conduction_band) - np.max(valence_band)
        

        #Additional energy shift applied to position Ef at midgap
        #cb_shift = abs(abs(np.min(conduction_band)) - bandgap/2.0)
        #vb_shift = abs(abs(np.max(valence_band)) - bandgap/2.0)
        #print(cb_shift, " ", vb_shift)
        center_val = (np.min(conduction_band) + np.max(valence_band))/2
        shift = -1*center_val
        truncated_bands[:,c:] += shift
        truncated_bands[:,0:v+1] += shift 

        file.close()

        #Convert k_points to their actual values
        ky_factor = a/b
        kpoints[0,:] =  (kpoints[0,:])*(2.0*np.pi/a)
        kpoints[1,:] =  (kpoints[1,:]/ky_factor)*(2.0*np.pi/b)  

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
    nband, nks, truncated_bands, kpoints, c, v = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)
    
    adjust_bandgap = False
    experimental_bandgap = 2.07 #0.74 for 3L_Te
    if(adjust_bandgap):
        first_eigval_set = truncated_bands[1,:]
        pos_eigvals = [a for a in first_eigval_set if a> 0]
        neg_eigvals = [a for a in first_eigval_set if a < 0]

        pos_smallest = min(pos_eigvals, key=abs)
        neg_smallest = min(neg_eigvals, key=abs)
        c = int(np.where(first_eigval_set == pos_smallest)[0][0])
        v = int(np.where(first_eigval_set == neg_smallest)[0][0])

        conduction_band = truncated_bands[:,c]
        valence_band = truncated_bands[:,v]

        bandgap = np.min(conduction_band) - np.max(valence_band)
        print("Ec = " , np.min(conduction_band) , "Ev = " , np.max(valence_band))
        bandgap_shift = experimental_bandgap - bandgap
        
        truncated_bands[:,c:] += bandgap_shift/2 
        truncated_bands[:,0:v+1] -= bandgap_shift/2 

        print("new bandgap = ", np.min(conduction_band) - np.max(valence_band))
    
    kx = kpoints[0,:]
    ky = kpoints[1,:]
    
    #nks = len(kpoints[0,:])

    kx_plot = kx.reshape(int(np.sqrt(nks)),int(np.sqrt(nks)))
    ky_plot = ky.reshape(int(np.sqrt(nks)),int(np.sqrt(nks)))
    #Z2 = truncated_bands[:,12].reshape(len(kx),len(ky))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print("Valence: ", truncated_bands[(np.abs(truncated_bands[:,c] - Ef)).argmin(),c])
    print("Conduction: ", truncated_bands[(np.abs(truncated_bands[:,v] - Ef)).argmin(),v])
    #ax.contourf(kx_plot,ky_plot,truncated_bands[:,c].reshape(len(kx_plot),len(ky_plot)), cmap=cm.hsv)
    #ax.plot_surface(kx_plot,ky_plot,truncated_bands[:,v].reshape(len(kx_plot),len(ky_plot)), cmap=cm.hsv)
    #ax.set_zlim(-1.2,-1.0)
    #ax.scatter(kx_plot,ky_plot,truncated_bands[:,c].reshape(len(kx_plot),len(ky_plot)), cmap=cm.viridis)
    #ax.scatter(kx_plot,ky_plot,truncated_bands[:,v].reshape(len(kx_plot),len(ky_plot)),color='red')
    #ax.scatter(kx_plot,ky_plot,truncated_bands[:,v].reshape(len(kx_plot),len(ky_plot)), color='red')
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

    kx_new = np.linspace(-np.pi/a,np.pi/a,100)
    ky_new = np.linspace(-np.pi/b,np.pi/b,100)

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

    DeltaEc = min(E[:,c])
    DeltaEv = max(E[:,v])
    print("Ec - Ef = ", DeltaEc, " eV")
    print("Ef - Ev = ", DeltaEv, " eV")

    #kx_2D = kx_2D*(a/(np.pi))
    #ky_2D = ky_2D*(b/(np.pi))

    #ax.scatter(kx_2D,ky_2D,E[:,c].reshape(len(kx_plot),len(ky_plot)), s=5, color='red')
    #ax.scatter(kx_2D,ky_2D,E[:,v].reshape(len(kx_plot),len(ky_plot)), s=5, color='red')
    #ax.scatter(kx_2D,ky_2D,E[:,14].reshape(len(kx_plot),len(ky_plot)), s=3, color='red')
    

    ax.plot_surface(kx_2D,ky_2D,E[:,v].reshape(100,100), cmap=cm.hsv)
    #ax.plot_surface(kx_2D,ky_2D,E[:,c+1].reshape(100,100), cmap=cm.viridis)
    #ax.plot_surface(kx_2D,ky_2D,E[:,c+2].reshape(100,100), cmap=cm.viridis)
    #ax.plot_surface(kx_2D,ky_2D,E[:,v].reshape(100,100), cmap=cm.viridis)
    #ax.plot_surface(kx_2D,ky_2D,E[:,v-1].reshape(50,50), cmap=cm.viridis)
    #ax.scatter(2*np.pi/(3*a),0, 4, color='red', s=50)
    ax.set_xlim([-np.pi/a,np.pi/a])
    ax.set_ylim([-np.pi/a,np.pi/a])
    #ax.plot_surface(kx_plot,ky_plot,E[:,14].reshape(len(kx_plot),len(ky_plot)), color='red')



    #ax.plot_surface(kx,ky,Z2,cmap=cm.coolwarm)
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    ax.set_zlabel('E [eV]')
    plt.show()


def Plot_Bands(bands_filename, H_filename, skip_bands, target_bands, Ef, Eg, a, b):

    nband, nks, truncated_bands, kpoints,c,v = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)
    adjust_bandgap = True
    experimental_bandgap = Eg #0.74 for 3L_Te
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
        #print("Ec = " , np.min(conduction_band) , "Ev = " , np.max(valence_band))
        bandgap_shift = experimental_bandgap - bandgap
        
        truncated_bands[:,c:] += bandgap_shift/2 
        truncated_bands[:,0:v+1] -= bandgap_shift/2 

        #For testing
        new_conduction_band = truncated_bands[:,c]
        new_valence_band = truncated_bands[:,v]
        #print("Ec = " , np.min(new_conduction_band) , "Ev = " , np.max(new_valence_band))
    
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

    plt.figure(figsize=(8, 8),dpi=100)
    plt.rcParams.update({'font.size': 22})
    x_vals = np.linspace(0,1,len(kx))
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ['$\Gamma$', 'X', 'S', 'Y', '$\Gamma$'],fontsize=22)
    #plt.xticks([0, 0.333333, 0.666666, 1.0], ['K', '$\Gamma$', 'M', 'K'],fontsize=22)
    plt.ylabel("E - E$_F$ (eV)", fontsize=22)
    plt1 = plt.plot(x_vals, truncated_bands,'r',linewidth=2)
    plt2 = plt.plot(x_vals,E[:].T,'bo',markevery=3,markersize=6,alpha=0.8)
    #plt2 = plt.plot(x_vals,E[6:,:].T,'bo',markevery=3,markersize=6,alpha=0.8)
    plt.margins(x=0)
    plt.ylim([-2,2])
    plt.xlim([0,1])

    line = Line2D([0], [0], label='DFT', color='r')
    dot = Line2D([0], [0], label='MLTB', color='b',marker='o',markersize=5,linestyle='')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    handles = [line, dot]
    #plt.legend(handles=handles)


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
    plt.xticks([])
    plt.yticks([])
    plt.rcParams.update({'font.size': 15})
    im = plt.imshow(np.real(H_map), cmap="seismic",vmin=-3, vmax=3)

    plt.colorbar(im,aspect=10)
    plt.axhline(y=num_bands-0.5,color='k')
    plt.axhline(y=num_bands*2-0.5,color='k')
    plt.axvline(x=num_bands-0.5,color='k')
    plt.axvline(x=num_bands*2-0.5,color='k')
    plt.show()

def Extract_Effective_Mass_From_DFT(bands_filename, H_filename, skip_bands, target_bands, Ef, a, b, central_k, fitting_area):
    nband, nks, truncated_bands, kpoints,c,v = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)

    #Extract Effective mass from DFT Bands:
    kx = kpoints[0,:]
    ky = kpoints[1,:]

    kx_2D = kpoints[0,:].reshape(int(np.sqrt(nks)),int(np.sqrt(nks)))
    ky_2D = kpoints[1,:].reshape(int(np.sqrt(nks)),int(np.sqrt(nks)))  

    CB = truncated_bands[:,c].T
    VB = truncated_bands[:,v].T

    central_kx, central_ky = central_k
    kx_2D = (kx_2D - central_kx)*a/np.pi #For plotting
    ky_2D =(ky_2D - central_ky)*b/np.pi #For plotting
    kx_centered = kx - central_kx
    ky_centered = ky - central_ky

    k_r = np.sqrt(kx_centered**2 + ky_centered**2)
    mask = k_r < 0.13*np.pi/a #Change radius where fitting is performed
    k_r_filt = k_r[mask]
    k_theta = np.arctan2(ky_centered,kx_centered)  
    k_theta_filt = k_theta[mask]

    E_filt = truncated_bands[mask,c]
    
    theta_range = np.arange(0,195,15)

    tol = 15
    eff_m_radial = np.zeros(len(theta_range))
    q = 1.602e-19
    hbar = (6.626e-34/(2*np.pi))
    me = 9.109e-31
    for ind, theta in enumerate(theta_range):
       
        extracted_indices_1 = np.where(np.logical_and(k_theta_filt > np.deg2rad(theta - tol), k_theta_filt < np.deg2rad(theta + tol)))
        extracted_indices_2 = np.where(np.logical_and(k_theta_filt > np.deg2rad(-(180-theta) - tol), k_theta_filt < np.deg2rad(-(180-theta) + tol)))
        extracted_indices = np.concatenate((extracted_indices_1, extracted_indices_2),axis=1)
        k_r_slice = np.take(k_r_filt,extracted_indices)
        k_theta_slice = np.take(k_theta_filt,extracted_indices)
        E_slice = (np.take(E_filt,extracted_indices))*q
        kx_ptoc = k_r_slice*np.cos(k_theta_slice)
        ky_ptoc = k_r_slice*np.sin(k_theta_slice)

        make_negative_indices = np.where(k_theta_slice < 0)
    
        for i in make_negative_indices[1]:
            k_r_slice[0,i] *= -1
        
        coeffs = np.polyfit(k_r_slice[0],E_slice[0],2)
        eff_m_radial[ind] = ((hbar**2)/(2*coeffs[0]))/me
        x_vals = np.linspace(min(kx_ptoc),max(ky_ptoc),40)
        y_vals = coeffs[0]*x_vals**2 + coeffs[1]*x_vals + coeffs[2]

    fig3 = plt.figure(figsize=(8, 8),dpi=100)
    plt.rcParams.update({'font.size': 28})
    ax = fig3.add_subplot(111, projection='polar')
    
    eff_m_rev = np.flip(eff_m_radial)
    print(theta_range)
    eff_m_radial_360 = np.abs(np.hstack((eff_m_radial, eff_m_rev)).ravel())
    print(eff_m_rev)
    
    theta_plot = np.linspace(0,360,len(eff_m_radial_360))
    #for i in range(len(eff_m_radial_360)-1):
    #    eff_m_radial_360[i] = (eff_m_radial_360[i-1] + eff_m_radial_360[i] + eff_m_radial_360[i+1])/3.0
    
    ax.plot(np.deg2rad(theta_plot),eff_m_radial_360,linewidth=5,linestyle="-",color='r')
    radial_ticks = [0.4, 0.8, 1.2, 1.6]
    ax.set_yticks(radial_ticks)
    ax.set_xticklabels(['0°','', '90°', '','180°','','270°'])
    G_to_K = 1/((1/3)*(1/eff_m_rev[2] + 1/eff_m_rev[6] + 1/eff_m_rev[10]))
    G_to_M = 1/((1/3)*(1/eff_m_rev[0] + 1/eff_m_rev[4] + 1/eff_m_rev[4]))
    print("Gamma to K:", G_to_K)
    print("Gamma to M:", G_to_M)
    
    plt.show()

def Extract_Effective_Mass(bands_filename, H_filename, skip_bands, target_bands, Ef, a, b, central_k, fitting_area):

    nband, nks, truncated_bands, kpoints,c,v = Extract_Abinit_Eigvals(bands_filename, skip_bands, target_bands, Ef, a, b)
    
    #Extract effective mass from  MLTB Hamiltonian
    kgrid_size = 50
    central_kx, central_ky = central_k

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

    kx_range = np.linspace(central_kx - fitting_area*np.pi/a, central_kx + fitting_area*np.pi/a,kgrid_size)
    ky_range = np.linspace(central_ky - (0.1 + fitting_area)*np.pi/b, central_ky + (fitting_area + 0.1)*np.pi/b,kgrid_size)

    kx_2D, ky_2D = np.meshgrid(kx_range,ky_range)
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

    #Adjust coordinate system such that central k-point is the new origin.
    kx_2D = (kx_2D - central_kx)*a/np.pi #For plotting
    ky_2D =(ky_2D - central_ky)*b/np.pi #For plotting
    kx_centered = kx - central_kx
    ky_centered = ky - central_ky

    #Convert all points to their polar coordinate form.
    k_r = np.sqrt(kx_centered**2 + ky_centered**2)
    mask = k_r < 0.2*np.pi/a #Change radius where fitting is performed
    k_r_filt = k_r[mask]
    k_theta = np.arctan2(ky_centered,kx_centered)  
    k_theta_filt = k_theta[mask]

    E_filt = E[mask,v]
    
    #Define small range for each theta slice, get k-points within that range.
    theta_range = np.arange(0,180,50)
    tol = 10
    eff_m_radial = np.zeros(len(theta_range))
    q = 1.602e-19
    hbar = (6.626e-34/(2*np.pi))
    me = 9.109e-31
    for ind, theta in enumerate(theta_range):
       
        extracted_indices_1 = np.where(np.logical_and(k_theta_filt > np.deg2rad(theta - tol), k_theta_filt < np.deg2rad(theta + tol)))
        extracted_indices_2 = np.where(np.logical_and(k_theta_filt > np.deg2rad(-(180-theta) - tol), k_theta_filt < np.deg2rad(-(180-theta) + tol)))
        
        extracted_indices = np.concatenate((extracted_indices_1, extracted_indices_2),axis=1)
        k_r_slice = np.take(k_r_filt,extracted_indices)
        k_theta_slice = np.take(k_theta_filt,extracted_indices)
        E_slice = (np.take(E_filt,extracted_indices))*q
        kx_ptoc = k_r_slice*np.cos(k_theta_slice)
        ky_ptoc = k_r_slice*np.sin(k_theta_slice)

        make_negative_indices = np.where(k_theta_slice < 0)
    
        for i in make_negative_indices[1]:
            k_r_slice[0,i] *= -1
        
        coeffs = np.polyfit(k_r_slice[0],E_slice[0],2)
        eff_m_radial[ind] = ((hbar**2)/(2*coeffs[0]))/me
        x_vals = np.linspace(min(kx_ptoc),max(ky_ptoc),40)
        y_vals = coeffs[0]*x_vals**2 + coeffs[1]*x_vals + coeffs[2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(kx_2D*np.pi/a,ky_2D*np.pi/b,E[:,v].reshape(kgrid_size,kgrid_size), cmap=cm.rainbow)
    ax.set_xlim([-0.2*np.pi/a,0.2*np.pi/a])
    ax.set_ylim([-0.2*np.pi/a,0.2*np.pi/a])
    ax.set_xlabel('kx')
    ax.set_ylabel('ky')
    #ax.scatter(kx_ptoc,ky_ptoc,E_slice,color='red')


    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.scatter(k_r_slice,E_slice,color='red')
    ax.scatter(x_vals,y_vals)
    plt.show()

    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection='polar')
    
    eff_m_rev = np.flip(eff_m_radial)
    eff_m_radial_360 = np.abs(np.hstack((eff_m_radial, eff_m_rev)).ravel())
    
    theta_plot = np.linspace(0,360,len(eff_m_radial_360))
    print(len(eff_m_radial_360))
    ax.plot(np.deg2rad(theta_plot),eff_m_radial_360,linewidth=2)
    ax.set_ylim([0,1.75])

    #ax.set_xlim([central_kx - 0.2*np.pi/a, central_kx + 0.2*np.pi/a])
    #ax.set_ylim([central_ky - 0.2*np.pi/a, central_ky + 0.2*np.pi/a])
    plt.show()

def Shift_Ef(H_filename,output_hamiltonian_name,shift=0):
    hamiltonian = Load_H_From_Save(H_filename)
    basis_size = hamiltonian[0].shape[1]
    hamiltonian[0] = hamiltonian[0] + np.identity(basis_size)*shift
    directory = 'H_output'

    H_save = np.zeros((hamiltonian[0].shape[1]**2, 9))
    for i,mat in enumerate(hamiltonian):
        flat_mat = np.real(mat.flatten('F')) #Flatten in column-major order.
        H_save[:,i] = flat_mat
    np.savetxt(os.path.join(directory, output_hamiltonian_name), H_save, delimiter='\t')

def DOS(H_filename):
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

    kx_new = np.linspace(-3*np.pi/a,3*np.pi/a,150)
    ky_new = np.linspace(-3*np.pi/b,3*np.pi/b,150)

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

    dE = 7e-3
    E_bins = np.arange(-2,2,dE)

    sns.kdeplot(np.array(E.flatten()),vertical=True, bw=0.005)
    sns.kdeplot(np.array(-1*E.flatten()),vertical=True, bw=0.005)
    plt.ylim(0,2)
    #plt.xlim(0, 0.1)

    #plt.hist(E.flatten(),bins=E_bins,histtype='step',density=True)
    #plt.hist(-1*E.flatten(),bins=E_bins,histtype='step',density=True)
    plt.show()


#old fermi was -2.5834
plt.close("all")


a = 2*6.149189998598033e-10
b = 2*6.149189998598033e-10
Ef = -2.4557 + 0.1
Eg = 1.2
target_bands = 24
skip_bands = 80
H_filename = 'PdTe2_2x2_24B_H1.dat'
DFT_filename = 'PdTe2_2x2_GXSYG_bands_PAW.dat'

#Additional settings for effective mass extraction
band_extremum_kpoint = (2*np.pi/(3*a),0)
fitting_area = 0.3

#Plot_CB_Surf('HfS2_31x31_bands.dat',12,18, -2.5834 , 6.290339483e-10, 3.631729507E-10)
Plot_Bands(DFT_filename,H_filename,skip_bands,target_bands,Ef, Eg, a, b)
#Plot_CB_Surf(DFT_filename,H_filename,skip_bands,target_bands, Ef, a, b)
#Plot_Hamiltonian(H_filename)
#Extract_Effective_Mass(DFT_filename, H_filename, skip_bands, target_bands, Ef, a, b, band_extremum_kpoint, fitting_area)
#Extract_Effective_Mass_From_DFT(DFT_filename, H_filename, skip_bands, target_bands, Ef, a, b, band_extremum_kpoint, fitting_area)
#Shift_Ef(H_filename,output_hamiltonian_name='PdTe2_2x1_18B_H2_shifted.dat',shift=(0.6138456122132121-0.6005620505655475)/2.0)
#DOS(H_filename)

'''
Extra Code:

1L WSi2N4:
a = 2.912061868832271e-10
b = 5.04382216226827e-10
Ef = 1.3267 + 0.2
Eg = 2.07
target_bands = 24
skip_bands = 30
H_filename = 'WSi2N4_H_24B_18.dat'
DFT_filename = 'WSi2N4_Supercell_60x60_halfBZ_bands.dat'

#Additional settings for effective mass extraction
band_extremum_kpoint = (2*np.pi/(3*a),0)
fitting_area = 0.3

1L HfS2:
a = 6.29033902986356e-10
b = 3.63172924494192e-10
Ef = -3.2370 
Eg = 1.90
target_bands = 22
skip_bands = 12
H_filename = 'HfS2_DFTfit_1.dat'
DFT_filename = 'HfS2_GXSYG_bands.dat'

Pent PdTe2:
a = 6.149189998598033e-10
b = 6.149189998598033e-10
Ef = -1.7314 + 0.1
Eg = 1.2
target_bands = 18
skip_bands = 36
H_filename = 'PdTe2_18B_H1.dat'
DFT_filename = 'PdTe2_GXSYG_bands.dat'

'''