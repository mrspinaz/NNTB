import numpy as np
import os
### Inputs ###
hamiltonian_name = "WSi2N4_H_24B_15.dat"


### Code ###

H = np.loadtxt("H_output/" + hamiltonian_name)
sq_size = int(np.sqrt(len(H[:,0])))
alpha = H[:,0].reshape((sq_size,sq_size))
beta =  H[:,1].reshape((sq_size,sq_size))
gamma = H[:,2].reshape((sq_size,sq_size))
delta11 = H[:,3].reshape((sq_size,sq_size))
delta1_min1 = H[:,4].reshape((sq_size,sq_size))

beta_dagger = np.transpose(beta)
delta1_min1_dagger = np.transpose(delta1_min1)

#New Hamiltonian
new_name = hamiltonian_name.replace(".dat", "_trans.dat")
new_beta = gamma
new_delta1_min1 = delta11
new_gamma = beta_dagger
new_delta11 = delta1_min1_dagger
H_transverse = [alpha, new_beta, new_gamma, new_delta11, new_delta1_min1]

H_save = np.zeros((H_transverse[0].shape[1]**2, 5))
for i,mat in enumerate(H_transverse):
    flat_mat = np.real(mat.flatten('F')) #Flatten in column-major order.
    H_save[:,i] = flat_mat
np.savetxt(os.path.join("H_output/", new_name), H_save, delimiter='\t')