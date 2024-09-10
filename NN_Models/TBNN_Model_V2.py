import re
import os
import numpy as np
import tensorflow as tf

class TBNN_V2:
    #Maybe change code at some point to extract these from scf, bands, etc output files, other than boolean commands.
    def __init__(self, a, b, Ef, restart, skip_bands, target_bands, converge_target, max_iter, learn_rate, regularization_factor ,bands_filename, output_hamiltonian_name, adjust_bandgap, experimental_bandgap):
        self.a = a
        self.b = b
        self.a_tens = tf.convert_to_tensor(a,dtype=tf.complex64)
        self.b_tens = tf.convert_to_tensor(b,dtype=tf.complex64)


        self.Ef = Ef
        self.restart = restart
        self.skip_bands = skip_bands
        self.target_bands = target_bands
        self.converge_target = converge_target
        self.max_iter = max_iter
        self.learn_rate = learn_rate
        self.regularization_factor = regularization_factor
        self.bands_filename = bands_filename
        self.output_hamiltonian_name = output_hamiltonian_name

        self.adjust_bandgap = adjust_bandgap
        self.experimental_bandgap = experimental_bandgap

    def _Extract_Abinit_Eigvals(self, bands_filename):

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

            abinit_bands = abinit_bands - self.Ef
            abinit_bands = abinit_bands.reshape(nks,nband)
            truncated_bands = abinit_bands[:, self.skip_bands:(self.skip_bands + self.target_bands)]        
            

        file.close()

        #Convert k_points to their actual values
        ky_factor = self.a/self.b
        kpoints[0,:] =  np.abs((kpoints[0,:])*(2.0*np.pi/self.a))
        kpoints[1,:] =  np.abs((kpoints[1,:]/ky_factor)*(2.0*np.pi/self.b))

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


        #Ec and Ev should have the same magnitude
        print("Ec = " , np.min(truncated_bands[:,c])  , "Ev = " , np.max(truncated_bands[:,v]))
        
        if(self.adjust_bandgap):

            bandgap_shift = self.experimental_bandgap - bandgap
            
            truncated_bands[:,c:] += bandgap_shift/2
            truncated_bands[:,0:v+1] -= bandgap_shift/2

            
            #For testing
            #new_conduction_band = truncated_bands[:,c]
            #new_valence_band = truncated_bands[:,v]
            #print("Ec = " , np.min(new_conduction_band) , "Ev = " , np.max(new_valence_band))

        
        kpoints = tf.convert_to_tensor(kpoints, dtype =tf.complex64)
        truncated_bands = tf.convert_to_tensor(truncated_bands, dtype =tf.float32)
        print("Ec = " , np.min(truncated_bands[:,c])  , "Ev = " , np.max(truncated_bands[:,v]))
        
        return nband, nks, truncated_bands, kpoints
    
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
    
    def adjust_bandgap(self, extracted_bands):

        first_eigval_set = extracted_bands[1,:]
        pos_eigvals = [a for a in first_eigval_set if a> 0]
        neg_eigvals = [a for a in first_eigval_set if a < 0]

        pos_smallest = min(pos_eigvals, key=abs)
        neg_smallest = min(neg_eigvals, key=abs)
        c = int(np.where(first_eigval_set == pos_smallest)[0])
        v = int(np.where(first_eigval_set == neg_smallest)[0])

        conduction_band = extracted_bands[:,c]
        valence_band = extracted_bands[:,v]

        bandgap = np.min(conduction_band) - np.max(valence_band)
        print("Ec = " , np.min(conduction_band) , "Ev = " , np.max(valence_band))
        bandgap_shift = self.experimental_bandgap - bandgap
        
        extracted_bands[:,c:] += bandgap_shift/2
        extracted_bands[:,0:v+1] -= bandgap_shift/2

        #For testing
        new_conduction_band = extracted_bands[:,c]
        new_valence_band = extracted_bands[:,v]
        print("Ec = " , np.min(new_conduction_band) , "Ev = " , np.max(new_valence_band))


    def fit_bands(self):
        """ 
        This fuction is for training the bands from randomly generated Hamiltonian.
        Each loop the loss function is calculated and used to extract the gradients.
        Then the gradients are symmetrized and applied to the Hamiltonian elements.
        
        """
        
            

        #abinit_bands and kpoints are already tensor objects
        nband, nks, truncated_bands, kpoints = self._Extract_Abinit_Eigvals(self.bands_filename)
        
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

        while(loss > self.converge_target and count < self.max_iter):
            
            with tf.GradientTape() as tape:
                
                E_tb_pred = self.Calculate_Energy_Eigenvals(H_trainable, kpoints, nks)

                loss1 = tf.reduce_mean(tf.square(E_tb_pred - truncated_bands))
                
                loss_reg = 0
                for i in range(len(H_trainable)):
                    loss_reg += tf.cast(tf.reduce_sum(tf.abs((tf.math.real(H_trainable[i])))), dtype=tf.float32)
                loss = loss1 + self.regularization_factor*loss_reg
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

        
    




        



                
