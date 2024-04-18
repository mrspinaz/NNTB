import numpy as np
import math as m
import scipy.linalg as lin
import matplotlib.pyplot as plt

#directional cosines
def l_dirc(vec):
    return vec[0]/np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def m_dirc(vec):
    return vec[1]/np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def n_dirc(vec):
    return vec[2]/np.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)

def overlap_elements(vec):
    #parameters taken from https://doi.org/10.1103/PhysRevB.82.245412
    Ssss = 0.102
    Ssps = -0.171
    Spps = -0.377
    Sppp = 0.07

    l = l_dirc(vec)
    m = m_dirc(vec)
    n = n_dirc(vec)

    S_As_Bs = Ssss
    S_As_Bpx = l*Ssps
    S_As_Bpy = m*Ssps
    S_As_Bpz = n*Ssps

    S_Apx_Bpx = (l**2)*Spps + (1 - l**2)*Sppp
    S_Apx_Bpy = l*m*(Spps - Sppp)
    S_Apx_Bpz = l*n*(Spps - Sppp)

    S_Apy_Bpx = l*m*(Spps - Sppp)
    S_Apy_Bpy = (m**2)*Spps + (1 - m**2)*Sppp
    S_Apy_Bpz = m*n*(Spps - Sppp)

    S_Apz_Bpx = l*n*(Spps - Sppp)
    S_Apz_Bpy = m*n*(Spps - Sppp)
    S_Apz_Bpz = (n**2)*Spps + (1 - n**2)*Sppp

    #element list is ordered in row major order for the matrix.
    elements_list = [S_As_Bs, S_As_Bpx, S_As_Bpy, S_As_Bpz, S_Apx_Bpx, S_Apx_Bpy, S_Apx_Bpz, S_Apy_Bpy, S_Apy_Bpz, S_Apz_Bpz ]
    return elements_list

#calculating matrix elements 
def off_diag_elements(vec):
    
    #parameters taken from https://doi.org/10.1103/PhysRevB.82.245412

    
    Vsss = -5.729
    Vsps = 5.618
    Vpps = 6.05
    Vppp = -3.07

    l = l_dirc(vec)
    m = m_dirc(vec)
    n = n_dirc(vec)

    E_As_Bs = Vsss
    E_As_Bpx = l*Vsps
    E_As_Bpy = m*Vsps
    E_As_Bpz = n*Vsps

    E_Apx_Bpx = (l**2)*Vpps + (1 - l**2)*Vppp
    E_Apx_Bpy = l*m*(Vpps - Vppp)
    E_Apx_Bpz = l*n*(Vpps - Vppp)

    E_Apy_Bpx = l*m*(Vpps - Vppp)
    E_Apy_Bpy = (m**2)*Vpps + (1 - m**2)*Vppp
    E_Apy_Bpz = m*n*(Vpps - Vppp)

    E_Apz_Bpx = l*n*(Vpps - Vppp)
    E_Apz_Bpy = m*n*(Vpps - Vppp)
    E_Apz_Bpz = (n**2)*Vpps + (1 - n**2)*Vppp

    #element list is ordered in row major order for the matrix.
    elements_list = [E_As_Bs, E_As_Bpx, E_As_Bpy, E_As_Bpz, E_Apx_Bpx, E_Apx_Bpy, E_Apx_Bpz, E_Apy_Bpy, E_Apy_Bpz, E_Apz_Bpz ]
    
    return elements_list

def k_path(v1,v2):

    a = abs(v2[0])
    b = abs(v2[1])

    len_GM = np.pi/b
    len_MK = np.tan(np.deg2rad(30))*len_GM
    len_KG = len_MK/np.sin(np.deg2rad(30))
    total_path_len = len_GM + len_MK + len_KG

    numk = 100
    
    kx_GM = np.linspace(0,0, int(len_GM*numk/total_path_len))
    ky_GM = np.linspace(0,np.pi/b, int(len_GM*numk/total_path_len))

    kx_MK = np.linspace(0,np.pi/(3.0*a), int(len_MK*numk/total_path_len))
    ky_MK = np.linspace(np.pi/b,np.pi/b, int(len_MK*numk/total_path_len))

    kx_KG = np.linspace(np.pi/(3.0*a),0, int(len_KG*numk/total_path_len))
    ky_KG = np.linspace(np.pi/b,0, int(len_KG*numk/total_path_len))

    kx = np.concatenate((kx_KG,kx_GM,kx_MK), axis=None)
    ky = np.concatenate((ky_KG,ky_GM,ky_MK), axis=None)

    kz = np.zeros(len(kx))

    k_path_coords = np.column_stack((kx, ky, kz))

    return k_path_coords
    

#Cell vectors
v1 = 1e-10*np.array([1.2306, -2.13146, 0])
v2 = 1e-10*np.array([1.2306, 2.13146, 0])
v3 = 1e-10*np.array([0,0,6.07])

#atomic coordinates
c1 = 1e-10*np.array([1.2306,-0.710487,3.3545])
c2 = 1e-10*np.array([1.2306,0.710487,3.3545])


#alpha diagonal
basis_size = len(["s", "px", "py", "pz"])

Es = -8.37
Ep = 0.0

orbital_energies = np.array([Es, Ep, Ep, Ep])
diag = np.diag(orbital_energies)
alpha_diag = lin.block_diag(diag, diag)
S_alpha_diag = np.eye(len(orbital_energies)*2, dtype=float)

#alpha off diagonal
AB_dist = c2 - c1
print("AB_dist: ", AB_dist)
alpha_od_AB = np.zeros((basis_size,basis_size))
S_alpha_od_AB = np.zeros((basis_size,basis_size))
AB_elements = off_diag_elements(AB_dist)
AB_overlap = overlap_elements(AB_dist)

list_ind = 0
for i in range(alpha_od_AB.shape[0]):
    for j in range(i,alpha_od_AB.shape[0]):
        alpha_od_AB[i,j] = AB_elements[list_ind]
        alpha_od_AB[j,i] = AB_elements[list_ind]

        S_alpha_od_AB[i,j] = AB_overlap[list_ind]
        S_alpha_od_AB[j,i] = AB_overlap[list_ind]
        list_ind += 1

#changing sign of s/p terms 
alpha_od_AB[1:-1,0] *= -1
S_alpha_od_AB[1:-1,0] *= -1

alpha_od_BA = np.transpose(alpha_od_AB)
S_alpha_od_BA = np.transpose(S_alpha_od_AB)


alpha_od = np.block([[np.zeros((basis_size,basis_size)), alpha_od_AB],
                     [alpha_od_BA, np.zeros((basis_size,basis_size))]])

S_alpha_od = np.block([[np.zeros((basis_size,basis_size)), S_alpha_od_AB],
                     [S_alpha_od_BA, np.zeros((basis_size,basis_size))]])


alpha = alpha_diag + alpha_od
S_alpha = S_alpha_diag + S_alpha_od

#beta matrix
c3_A = c1 + v2

BA_beta = c3_A - c2
print("BA_beta: ",BA_beta)

BA_beta_elements = off_diag_elements(BA_beta)
BA_overlap = overlap_elements(BA_beta)

beta_od_BA = np.zeros((basis_size,basis_size))
S_beta_od_BA = np.zeros((basis_size,basis_size))
list_ind = 0
for i in range(beta_od_BA.shape[0]):
    for j in range(i,beta_od_BA.shape[0]):
        beta_od_BA[i,j] = BA_beta_elements[list_ind]
        beta_od_BA[j,i] = BA_beta_elements[list_ind]

        S_beta_od_BA[i,j] = BA_overlap[list_ind]
        S_beta_od_BA[j,i] = BA_overlap[list_ind]
        list_ind += 1

#changing sign of p/s terms 
beta_od_BA[1:-1,0] *= -1
S_beta_od_BA[1:-1,0] *= -1


beta = np.block([[np.zeros((basis_size,basis_size)), np.zeros((basis_size,basis_size))],
                 [beta_od_BA ,np.zeros((basis_size,basis_size))]])

S_beta = np.block([[np.zeros((basis_size,basis_size)), np.zeros((basis_size,basis_size))],
                 [S_beta_od_BA ,np.zeros((basis_size,basis_size))]])



#gamma
c4_B = c2 + v1
AB_gamma = c4_B - c1
AB_gamma_elements = off_diag_elements(AB_gamma)
S_AB_gamma = overlap_elements(AB_gamma)
gamma_od_AB = np.zeros((basis_size,basis_size))
S_gamma_od_AB = np.zeros((basis_size,basis_size))
list_ind = 0
for i in range(gamma_od_AB.shape[0]):
    for j in range(i,gamma_od_AB.shape[0]):
        gamma_od_AB[i,j] = AB_gamma_elements[list_ind]
        gamma_od_AB[j,i] = AB_gamma_elements[list_ind]

        S_gamma_od_AB[i,j] = S_AB_gamma[list_ind]
        S_gamma_od_AB[j,i] = S_AB_gamma[list_ind]
        list_ind += 1

#changing sign of s/p terms 
gamma_od_AB[1:-1,0] *= -1
S_gamma_od_AB[1:-1,0] *= -1


gamma = np.block([[np.zeros((basis_size,basis_size)), gamma_od_AB ],
                 [np.zeros((basis_size,basis_size)) ,np.zeros((basis_size,basis_size))]])

S_gamma = np.block([[np.zeros((basis_size,basis_size)), S_gamma_od_AB ],
                 [np.zeros((basis_size,basis_size)) ,np.zeros((basis_size,basis_size))]])

k_path_coords = k_path(v1,v2)

print("Det beta: ", np.linalg.det(S_beta))


bands = np.zeros((alpha.shape[0], k_path_coords.shape[0]))
for i,k in enumerate(k_path_coords):

    H = alpha + beta*np.exp(1j*np.dot(v2,k)) + gamma*np.exp(1j*np.dot(v1,k)) + beta.T*np.exp(-1j*np.dot(v2,k)) + gamma.T*np.exp(-1j*np.dot(v1,k))
    
    S = S_alpha + S_beta*np.exp(1j*np.dot(v2,k)) + S_gamma*np.exp(1j*np.dot(v1,k)) + S_beta.T*np.exp(-1j*np.dot(v2,k)) + S_gamma.T*np.exp(-1j*np.dot(v1,k))
   
    eig_vals, eig_vecs = lin.eigh(H,S)
    bands[:,i] = eig_vals

    #checking matrices at gamma point
    if i == 0:
        print(alpha[4:8,0:4])
        print(beta[4:8,0:4])
        print(gamma.T[4:8,0:4])
        print(np.round(np.real(H[4:8,0:4]),4))
        print("Hamiltonian at Gamma: \n", np.round(np.real(H),3))

#Output Bands File
np.savetxt(('SK_input/graphene_sp3_bands.txt'), bands.T)

#Convert from eV to Ha
bands[:,:] *= 0.0367492929        
                
#Plotting
b = v2[1]
len_GM = np.pi/b
len_MK = np.tan(np.deg2rad(30))*len_GM
len_KG = len_MK/np.sin(np.deg2rad(30))
total_path_len = len_GM + len_MK + len_KG

x_vals = np.linspace(0,1,k_path_coords.shape[0])
print("Eigenvalues at Gamma: \n", bands[:,0])
plt.xticks([0, (len_KG/total_path_len), (len_KG+ len_GM)/total_path_len, 1.0], ['K', '$\Gamma$', 'M' ,'K'])
plt.plot(x_vals,bands[:,:].T,'b')
plt.ylabel("E [Ha]")
plt.ylim([-0.8,0.6])
plt.show()


#old code 

# alpha_test = np.array([[0,-3.07],
#                        [-3.07,0]])
# beta_test = np.array([[0, 0 ],
#                       [-3.07,0]])
# gamma_test = np.array([[0, -3.07],
#                        [0,0]])


# bands_pz = np.zeros((alpha_test.shape[0], k_path_coords.shape[0]))
# for i,k in enumerate(k_path_coords):
#     H = alpha_test + beta_test*np.exp(1j*np.dot(v2,k)) + gamma_test*np.exp(1j*np.dot(v1,k)) + beta_test.T*np.exp(-1j*np.dot(v2,k)) + gamma_test.T*np.exp(-1j*np.dot(v1,k))
#     eig_vals, eig_vecs = lin.eig(H)
#     bands_pz[:,i] = eig_vals
