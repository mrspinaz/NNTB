import re
import os
import numpy as np
import tensorflow as tf

class TBNN_V2_fitwan:
    #Maybe change code at some point to extract these from scf, bands, etc output files, other than boolean commands.
    def __init__(self, a, b, restart, target_bands, converge_target, max_iter, learn_rate, regularization_factor ,bands_filename, output_hamiltonian_name):
        self.a = a
        self.b = b
        self.a_tens = tf.convert_to_tensor(a,dtype=tf.complex64)
        self.b_tens = tf.convert_to_tensor(b,dtype=tf.complex64)


        self.restart = restart
        self.target_bands = target_bands
        self.converge_target = converge_target
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.regularization_factor = regularization_factor
        self.bands_filename = bands_filename
        self.output_hamiltonian_name = output_hamiltonian_name

    def _Extract_Wannier_Eigvals(self,a,b, bands_filename):

        
        ham = np.loadtxt('inputs/' + bands_filename, dtype=float)
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

        kgrid_size = 21
        kx_new = np.linspace(0,np.pi/self.a,kgrid_size)
        ky_new = np.linspace(0,np.pi/self.b,kgrid_size)

        kx_2D, ky_2D = np.meshgrid(kx_new,ky_new)
        nks = np.size(kx_2D)
        kx = np.reshape(kx_2D, (1,np.size(kx_2D)),order='F').flatten()
        ky = np.reshape(ky_2D, (1,np.size(ky_2D)),order='F').flatten()
        kpoints = np.vstack((kx,ky))

        bands = np.zeros((nks,sqdim))
        for ii in range(len(kx)):
            H = alpha + beta*np.exp(1j*kx[ii]*a) + gamma*np.exp(1j*ky[ii]*b) + gamma2*np.exp(2j*ky[ii]*b) + \
            delta11*np.exp(1j*kx[ii]*a + 1j*ky[ii]*b) + delta12*np.exp(1j*kx[ii]*a + 2j*ky[ii]*b)  + \
            delta1_min1*np.exp(1j*kx[ii]*a - 1j*ky[ii]*b) + delta1_min2*np.exp(1j*kx[ii]*a - 2j*ky[ii]*b) + \
            beta_dagger*np.exp(-1j*kx[ii]*a) + gamma_dagger*np.exp(-1j*ky[ii]*b) + gamma2_dagger*np.exp(-2j*ky[ii]*b) + \
            delta11_dagger*np.exp(-1j*kx[ii]*a - 1j*ky[ii]*b) + delta12_dagger*np.exp(-1j*kx[ii]*a - 2j*ky[ii]*b) + \
            delta1_min1_dagger*np.exp(-1j*kx[ii]*a + 1j*ky[ii]*b) + delta1_min2_dagger*np.exp(-1j*kx[ii]*a + 2j*ky[ii]*b)
            eigvals, eigvecs = np.linalg.eig(H)
            bands[ii,:] = np.sort((eigvals))

        first_eigval_set = bands[1,:]
        pos_eigvals = [a for a in first_eigval_set if a> 0]
        neg_eigvals = [a for a in first_eigval_set if a < 0]

        pos_smallest = min(pos_eigvals, key=abs)
        neg_smallest = min(neg_eigvals, key=abs)
        c = int(np.where(first_eigval_set == pos_smallest)[0])
        v = int(np.where(first_eigval_set == neg_smallest)[0])

        conduction_band = bands[:,c]
        valence_band = bands[:,v]

        bandgap = np.min(conduction_band) - np.max(valence_band)

        #add dummy bands as needed, default to 10 eV as dummy value.
        extra_bands = self.target_bands - len(first_eigval_set)
        dummy_bands = 10*np.ones((len(conduction_band),extra_bands))
        bands = np.concatenate((bands,dummy_bands),axis=1)
        #Ec and Ev should have the same magnitude
        print("Ec = " , np.min(bands[:,c])  , "Ev = " , np.max(bands[:,v]))
        
        
        kpoints = tf.convert_to_tensor(kpoints, dtype =tf.complex64)
        bands = tf.convert_to_tensor(bands, dtype =tf.float32)

        #To expand the fitting basis, add dummy eigvals to bands here, then you can just expand mat dims in Initialize_Hamiltonian.

        return nks, bands, kpoints , c, v
    

    def _Initialize_Hamiltonian(self):


        #Generated random TB matrix:
        alpha_rand = tf.cast(tf.random.normal([self.target_bands,self.target_bands]),  dtype=tf.complex64)
        beta_rand = tf.cast(tf.random.normal([self.target_bands,self.target_bands]),  dtype=tf.complex64)
        gamma_rand = tf.cast(tf.random.normal([self.target_bands,self.target_bands]),  dtype=tf.complex64)
        delta11_rand = tf.cast(tf.random.normal([self.target_bands,self.target_bands]),  dtype=tf.complex64)
        delta1_min1_rand = tf.cast(tf.random.normal([self.target_bands,self.target_bands]),  dtype=tf.complex64)

        #Ensure the matrix is hermatian
        #A matrix plus it's conjugate transpose is hermatian.
        alpha_rand = alpha_rand + tf.transpose(alpha_rand)
        beta_rand_dagger = tf.transpose(beta_rand)
        gamma_rand_dagger = tf.transpose(gamma_rand)
        delta11_rand_dagger = tf.transpose(delta11_rand)
        delta1_min1_rand_dagger = tf.transpose(delta1_min1_rand)

        #Combine variables into one list.
        alpha_tensor = tf.Variable(alpha_rand, dtype=tf.complex64)
        beta_tensor = tf.Variable(beta_rand, dtype=tf.complex64)
        gamma_tensor = tf.Variable(gamma_rand, dtype=tf.complex64)
        delta11_tensor = tf.Variable(delta11_rand, dtype=tf.complex64)
        delta1_min1_tensor = tf.Variable(delta1_min1_rand, dtype=tf.complex64)

        beta_tensor_dagger = tf.Variable(beta_rand_dagger, dtype=tf.complex64)
        gamma_tensor_dagger = tf.Variable(gamma_rand_dagger, dtype=tf.complex64)
        delta11_tensor_dagger = tf.Variable(delta11_rand_dagger, dtype=tf.complex64)
        delta1_min1_tensor_dagger = tf.Variable(delta1_min1_rand_dagger, dtype=tf.complex64)
        
        H_trainable = [alpha_tensor, beta_tensor, gamma_tensor, delta11_tensor, delta1_min1_tensor, beta_tensor_dagger, gamma_tensor_dagger, delta11_tensor_dagger, delta1_min1_tensor_dagger]

        return H_trainable 
    
    def _Initialize_MLWF(self):
        """
        Reads initial values from the .dat file output by extractTB2st_biggamma.m script.
        """
        #Generated random TB matrix:
        MLWF_file = np.loadtxt('inputs/3L_Te_Wannier_H.dat')
        alpha = tf.convert_to_tensor(np.reshape(MLWF_file[:,0], (self.target_bands,self.target_bands)), dtype=tf.complex64 )
        beta = tf.convert_to_tensor(np.reshape(MLWF_file[:,1], (self.target_bands,self.target_bands)), dtype=tf.complex64 )
        gamma = tf.convert_to_tensor(np.reshape(MLWF_file[:,2], (self.target_bands,self.target_bands)), dtype=tf.complex64 )
        delta11 = tf.convert_to_tensor(np.reshape(MLWF_file[:,3], (self.target_bands,self.target_bands)), dtype=tf.complex64 )
        delta1_min1 = tf.convert_to_tensor(np.reshape(MLWF_file[:,4], (self.target_bands,self.target_bands)), dtype=tf.complex64 )

        beta_dagger = tf.transpose(beta)
        gamma_dagger = tf.transpose(gamma)
        delta11_dagger = tf.transpose(delta11)
        delta1_min1_dagger = tf.transpose(delta1_min1)

        #Combine variables into one list.
        alpha_tensor = tf.Variable(alpha, dtype=tf.complex64)
        beta_tensor = tf.Variable(beta, dtype=tf.complex64)
        gamma_tensor = tf.Variable(gamma, dtype=tf.complex64)
        delta11_tensor = tf.Variable(delta11, dtype=tf.complex64)
        delta1_min1_tensor = tf.Variable(delta1_min1, dtype=tf.complex64)

        beta_tensor_dagger = tf.Variable(beta_dagger, dtype=tf.complex64)
        gamma_tensor_dagger = tf.Variable(gamma_dagger, dtype=tf.complex64)
        delta11_tensor_dagger = tf.Variable(delta11_dagger, dtype=tf.complex64)
        delta1_min1_tensor_dagger = tf.Variable(delta1_min1_dagger, dtype=tf.complex64)

        H_trainable = [alpha_tensor, beta_tensor, gamma_tensor, delta11_tensor, delta1_min1_tensor, beta_tensor_dagger, gamma_tensor_dagger, delta11_tensor_dagger, delta1_min1_tensor_dagger]
        
        return H_trainable
        

    def _Reinitialize(self):
        """
        Re-initialize the hamiltonian elements. Called if Restart == True.
        """
        alpha = np.real(np.loadtxt('H_output/alpha.txt', dtype=complex))
        beta = np.loadtxt('H_output/beta.txt', dtype=complex)
        gamma = np.loadtxt('H_output/gamma.txt', dtype=complex)
        delta11 = np.loadtxt('H_output/delta11.txt', dtype=complex)
        delta1_min1 = np.loadtxt('H_output/delta1_min1.txt', dtype=complex)
        
        alpha = np.diag(np.diag(alpha)) + np.tril(alpha,-1) + np.transpose(np.tril(alpha,-1))

        alpha_tensor = tf.convert_to_tensor(alpha, dtype=tf.complex64)
        beta_tensor = tf.convert_to_tensor(beta, dtype=tf.complex64)
        gamma_tensor = tf.convert_to_tensor(gamma, dtype=tf.complex64)
        delta11_tensor = tf.convert_to_tensor(delta11, dtype=tf.complex64)
        delta1_min1_tensor = tf.convert_to_tensor(delta1_min1, dtype=tf.complex64)

        alpha_tensor = tf.Variable(alpha_tensor, dtype=tf.complex64)
        beta_tensor = tf.Variable(beta_tensor, dtype=tf.complex64)
        gamma_tensor = tf.Variable(gamma_tensor, dtype=tf.complex64)
        delta11_tensor = tf.Variable(delta11_tensor, dtype=tf.complex64)
        delta1_min1_tensor = tf.Variable(delta1_min1_tensor, dtype=tf.complex64)
        
        gamma_tensor = tf.Variable(gamma_tensor, dtype=tf.complex64)
        beta_tensor_dagger = tf.Variable(tf.transpose(beta_tensor), dtype=tf.complex64)
        gamma_tensor_dagger = tf.Variable(tf.transpose(gamma_tensor), dtype=tf.complex64)
        delta11_tensor_dagger = tf.Variable(tf.transpose(delta11_tensor), dtype=tf.complex64)
        delta1_min1_tensor_dagger = tf.Variable(tf.transpose(delta1_min1_tensor), dtype=tf.complex64)

        H_trainable = [alpha_tensor, beta_tensor, gamma_tensor, delta11_tensor, delta1_min1_tensor, beta_tensor_dagger, gamma_tensor_dagger, delta11_tensor_dagger, delta1_min1_tensor_dagger]
        
        return H_trainable
    

    def Calculate_Energy_Eigenvals(self, H_train, kpoints, nks):

            alpha = H_train[0]
            beta = H_train[1]
            gamma = H_train[2]
            delta11 = H_train[3]
            delta1_min1 = H_train[4]
            beta_dagger = H_train[5]
            gamma_dagger = H_train[6]
            delta11_dagger = H_train[7]
            delta1_min1_dagger = H_train[8]

           
            E = tf.zeros([nks, self.target_bands], dtype=tf.complex64)

            for ii in range(nks):
                H = alpha + \
                    beta*tf.exp(1j*kpoints[0][ii]*self.a_tens) + \
                    gamma*tf.exp(1j*kpoints[1][ii]*self.b_tens) + \
                    delta11*tf.exp(1j*kpoints[0][ii]*self.a_tens + 1j*kpoints[1][ii]*self.b_tens) + \
                    delta1_min1*tf.exp(1j*kpoints[0][ii]*self.a_tens - 1j*kpoints[1][ii]*self.b_tens) + \
                    beta_dagger*tf.exp(-1j*kpoints[0][ii]*self.a_tens) + \
                    gamma_dagger*tf.exp(-1j*kpoints[1][ii]*self.b_tens) + \
                    delta11_dagger*tf.exp(-1j*kpoints[0][ii]*self.a_tens - 1j*kpoints[1][ii]*self.b_tens) + \
                    delta1_min1_dagger*tf.exp(-1j*kpoints[0][ii]*self.a_tens + 1j*kpoints[1][ii]*self.b_tens)
                
                eigvals, eigvecs = tf.linalg.eig(H) #tf.linalg.eigh(H)
                eigvals = tf.reshape(eigvals, shape=[1,self.target_bands])
                
                E = E + tf.scatter_nd([[ii]],eigvals,[nks,self.target_bands])

            #Flatten numpy array and convert back to tensor. The math.real() operation converts datatype to float32.
            E = tf.math.real(E)
            E = tf.sort(E, axis=1)
            #E = E[:, self.added_bands//2 : self.added_bands//2 + self.original_num_TBbands]
            #E_frac = E.numpy()
            #E_frac = E_frac[:,self.added_bands//2 : self.added_bands//2 + self.original_num_TBbands]
            #E_frac = tf.convert_to_tensor(E_frac, dtype=tf.float32)
            return E

    def _Save_Output(self, H_trainable):  
            
        directory = 'H_output'
        if not os.path.exists(directory):
            os.mkdir(directory)

        np.savetxt(os.path.join(directory,'alpha.txt'), H_trainable[0].numpy())
        np.savetxt(os.path.join(directory,'beta.txt'), H_trainable[1].numpy())
        np.savetxt(os.path.join(directory,'gamma.txt'), H_trainable[2].numpy())
        np.savetxt(os.path.join(directory,'delta11.txt'), H_trainable[3].numpy())
        np.savetxt(os.path.join(directory,'delta1_min1.txt'), H_trainable[4].numpy())
        np.savetxt(os.path.join(directory,'beta_dagger.txt'), H_trainable[5].numpy())
        np.savetxt(os.path.join(directory,'gamma_dagger.txt'), H_trainable[6].numpy())
        np.savetxt(os.path.join(directory,'delta11_dagger.txt'), H_trainable[7].numpy())
        np.savetxt(os.path.join(directory,'delta1_min1_dagger.txt'), H_trainable[8].numpy())


        #For plotting
        self.H_final = [H_trainable[0].numpy(), H_trainable[1].numpy(), H_trainable[2].numpy(), H_trainable[3].numpy(), H_trainable[4].numpy(), H_trainable[5].numpy(), H_trainable[6].numpy(), H_trainable[7].numpy(), H_trainable[8].numpy()]


        #For export to NEGF
        H_oneside = self.H_final[0:5] #5 is hard-coded for nearest neighbour.
        
        H_save = np.zeros((H_trainable[0].shape[1]**2, 5))
        for i,mat in enumerate(H_oneside):
            flat_mat = np.real(mat.flatten('F')) #Flatten in column-major order.
            H_save[:,i] = flat_mat
        np.savetxt(os.path.join(directory, self.output_hamiltonian_name), H_save, delimiter='\t')
    
    def create_weight_mat(self,c,v):
        num_cb = self.target_bands - c
        num_vb = self.target_bands - num_cb
        cb_weights = np.array([2,2,2,1,1,1,0.0005,0.0000,0.0000,0.000,0,0,0,0])
        vb_weights = np.array([0.0000,0.0000,0.0000,0.0000,0,0,1,1,1,2,2,2])
        band_weights = np.concatenate((vb_weights, cb_weights))
        print(band_weights)
        return tf.convert_to_tensor(band_weights, dtype=tf.float32)
       

    def fit_bands(self):
        """ 
        This fuction is for training the bands from randomly generated Hamiltonian.
        Each loop the loss function is calculated and used to extract the gradients.
        Then the gradients are symmetrized and applied to the Hamiltonian elements.
        
        """
        
            

        #abinit_bands and kpoints are already tensor objects
        nks, wan_bands, kpoints, c, v = self._Extract_Wannier_Eigvals(self.a, self.b, self.bands_filename)
        
        #initialize Hamiltonian
        if(self.restart):
            H_trainable = self._Reinitialize()
        else:
            H_trainable = self._Initialize_Hamiltonian()
        
        #Truncate bands
        

        opt = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)  
        loss = 10000
        loss_list = []
        count = 0

        band_weights = self.create_weight_mat(c,v)
        L1_mats = [0,1,2,3,4,5,6,7,8]
        while(loss > self.converge_target and count < self.max_iter):
            
            with tf.GradientTape() as tape:
                
                E_tb_pred = self.Calculate_Energy_Eigenvals(H_trainable, kpoints, nks)
                
                diff_tens = tf.square(E_tb_pred - wan_bands)
                diff_tens = diff_tens * band_weights

                loss1 = tf.reduce_mean(diff_tens)
                
                loss_reg = 0
                #loss_L2 = 0

                for i in L1_mats:
                    loss_reg += tf.cast(tf.reduce_sum(tf.abs((tf.math.real(H_trainable[i])))), dtype=tf.float32) 
                #alpha_loss_reg = tf.cast(tf.reduce_sum(tf.abs((tf.math.real(H_trainable[0])))), dtype=tf.float32)
                #beta_loss_reg = tf.cast(tf.reduce_sum(tf.abs((tf.math.real(H_trainable[1])))), dtype=tf.float32) 
                #beta_dag_loss_reg = tf.cast(tf.reduce_sum(tf.abs((tf.math.real(H_trainable[5])))), dtype=tf.float32) 
                #gamma_loss_reg = tf.cast(tf.reduce_sum(tf.abs((tf.math.real(H_trainable[2])))), dtype=tf.float32) 
                #gamma_dag_loss_reg = tf.cast(tf.reduce_sum(tf.abs((tf.math.real(H_trainable[6])))), dtype=tf.float32) 
                #for i in range(len(H_trainable)):
                #    loss_L2 += tf.cast(tf.reduce_sum(tf.square((tf.math.real(H_trainable[i])))), dtype=tf.float32) 

                loss = loss1 + self.regularization_factor*loss_reg  #1.7e-5*alpha_loss_reg #+ self.L2_factor*loss_L2
                print('Iteration: ', count,' Loss: ' , (loss).numpy())

            grad = tape.gradient(loss, H_trainable)
            
            grad_real = tf.cast(tf.math.real(grad), dtype=tf.complex64)
            

            sym_grad = []
            
            #Alpha
            diagonal = tf.linalg.tensor_diag_part(grad_real[0])
            diag_tensor = tf.linalg.diag(diagonal)

            upper_trig = tf.linalg.band_part(grad_real[0],0,-1)
            upper_trig_diag = tf.linalg.diag(tf.linalg.tensor_diag_part(upper_trig))
            upper_trig_nodiag = upper_trig - upper_trig_diag
    
            lower_trig_nodiag = tf.transpose(upper_trig_nodiag)
            
            sym_grad.append(diag_tensor + upper_trig_nodiag + lower_trig_nodiag)
        
            #Beta
            sym_grad.append(grad_real[1])
    
            #Gamma
            sym_grad.append(grad_real[2])
    
            #Delta11
            sym_grad.append(grad_real[3])
    
            #Delta1_min1
            sym_grad.append(grad_real[4])
            
            #Transposed matrix blocks
            sym_grad.append(tf.transpose(grad_real[1]))
            sym_grad.append(tf.transpose(grad_real[2]))
            sym_grad.append(tf.transpose(grad_real[3]))
            sym_grad.append(tf.transpose(grad_real[4]))
            sym_grad_tens = tf.stack(sym_grad)
            
            opt.apply_gradients(zip(sym_grad_tens, H_trainable))
            loss_list.append(loss)
            count+=1

        if count == self.max_iter:
            print('Max iterations reached.')
        elif loss <= self.converge_target:
            print('Convergence target reached.')
        else:
            print('Unknown exit condition.')
        
        self._Save_Output(H_trainable)

        
    




        



                
