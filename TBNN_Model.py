import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re 
import os  
import warnings

warnings.simplefilter("ignore", np.ComplexWarning)

class TBNN:

    def __init__ (self, a, b, bands_filename, Ef, experiemental_bandgap, num_TBbands, skip_bands, do_train, do_restart, do_shift, learn_rate, converge_target, max_iter):
        
        self.a = a
        self.b = b
        #Convert to complex datatype so values can be applied used in complex-valued tensor operations, ie calculating energy eigenvalues.
        self.a_tens = tf.convert_to_tensor(a, dtype=tf.complex64)
        self.b_tens = tf.convert_to_tensor(b, dtype=tf.complex64)

        self.bands_filename = bands_filename
        self.Ef = Ef
        self.experiemental_bandgap = experiemental_bandgap
        self.num_TBbands = num_TBbands
        self.skip_bands = skip_bands

        self.do_train = do_train
        self.do_restart = do_restart
        self.do_shift = do_shift

        self.learn_rate = learn_rate
        self.converge_target = converge_target
        self.max_iter = max_iter


    def Calculate_Energy_Eigenvals(self, H_train):

        alpha = H_train[0]
        beta = H_train[1]
        gamma = H_train[2]
        delta11 = H_train[3]
        delta1_min1 = H_train[4]
        beta_dagger = H_train[5]
        gamma_dagger = H_train[6]
        delta11_dagger = H_train[7]
        delta1_min1_dagger = H_train[8]

        E = tf.zeros([len(self.kx), self.num_TBbands], dtype=tf.complex64)

        #alpha = alpha + tf.transpose(alpha)
        #beta_dagger = tf.transpose(beta)
        #gamma_dagger = tf.transpose(gamma)
        #delta11_dagger = tf.transpose(delta11)
        #delta1_min1_dagger = tf.transpose(delta1_min1)


        for ii in range(len(self.kx)):
            H = alpha + beta*tf.exp(1j*self.kx[ii]*self.a_tens) + gamma*tf.exp(1j*self.ky[ii]*self.b_tens) + \
                delta11*tf.exp(1j*self.kx[ii]*self.a_tens + 1j*self.ky[ii]*self.b_tens) + delta1_min1*tf.exp(1j*self.kx[ii]*self.a_tens - 1j*self.ky[ii]*self.b_tens) + \
                beta_dagger*tf.exp(-1j*self.kx[ii]*self.a_tens) + gamma_dagger*tf.exp(-1j*self.ky[ii]*self.b_tens) + \
                delta11_dagger*tf.exp(-1j*self.kx[ii]*self.a_tens - 1j*self.ky[ii]*self.b_tens) + delta1_min1_dagger*tf.exp(-1j*self.kx[ii]*self.a_tens + 1j*self.ky[ii]*self.b_tens)
            eigvals, eigvecs = tf.linalg.eig(H) #tf.linalg.eigh(H)
            eigvals = tf.reshape(eigvals, shape=[1,self.num_TBbands])
            
            E = E + tf.scatter_nd([[ii]],eigvals,[len(self.kx),self.num_TBbands])

        #Flatten numpy array and convert back to tensor. The math.real() operation converts datatype to float32.
        E = tf.math.real(E)
        E = tf.sort(E, axis=1)
        E = E[:, self.added_bands//2 : self.added_bands//2 + self.original_num_TBbands]
        #E_frac = E.numpy()
        #E_frac = E_frac[:,self.added_bands//2 : self.added_bands//2 + self.original_num_TBbands]
        #E_frac = tf.convert_to_tensor(E_frac, dtype=tf.float32)
        return E



    def Extract_Abinit_Bands(self,plot_bands=False):
        """
        Extracts ab-initio bands from the .bands Quantum Espresso file.
        """

        with open(self.bands_filename) as data:
            lines = data.readlines()
            data.close()

        #Extract number of bands and kpoints.
        temp = re.findall('\d+',lines[0])
        self.nband = int(temp[0])
        self.nks = int(temp[1])
        print(self.nks)

        #Extract Kpoint list
        kpoints_arr = np.zeros((self.nks,3))
        all_kpoints_text = [line for line in lines if line.startswith("            ")]

        for text_line in range(self.nks):
            kpoints_oneline = re.findall('[-]?\d+.\d+', all_kpoints_text[text_line])
            kpoints_oneline_arr = np.array(kpoints_oneline, dtype=np.float32)
            kpoints_arr[text_line,:] = kpoints_oneline_arr


        #Extract bandstructure data from Quantum Espresso bands.dat output file.
        self.extracted_bands = np.zeros((self.nband,self.nks))
        all_bands_text = [line for line in lines if not (line.startswith("            ") or line.startswith(' &plot'))] 
        text_line = 0

        for k in range(self.nks):
            listed_energies = []
            listed_energies_length = len(listed_energies)

            #Append energy values to list until all energies are appended, then assign values to extracted_bands. This repeats for every kpoint.
            while listed_energies_length < self.nband:
                energy_vals_oneline = re.findall('[-]?\d+.\d+', all_bands_text[text_line])
                listed_energies = listed_energies + energy_vals_oneline
                listed_energies_length = len(listed_energies)
                text_line += 1

            #Convert strings to floats and list to numpy array
            listed_energies = np.array(listed_energies, dtype=np.float32)
            self.extracted_bands[:,k] = listed_energies
            
        #Ef = -3.2834
        self.extracted_bands = self.extracted_bands - self.Ef
        self.extracted_bands = self.extracted_bands.T

        #Bandgap Shift. Shifting conduction and valence bands by equal magnitude in opposite direction to obtain experiental bandgap.
        do_shift = False
        if(do_shift):
            first_eigval_set = self.extracted_bands[1,:]
            pos_eigvals = [a for a in first_eigval_set if a> 0]
            neg_eigvals = [a for a in first_eigval_set if a < 0]

            pos_smallest = min(pos_eigvals, key=abs)
            neg_smallest = min(neg_eigvals, key=abs)
            c = int(np.where(first_eigval_set == pos_smallest)[0])
            v = int(np.where(first_eigval_set == neg_smallest)[0])

            conduction_band = self.extracted_bands[:,c]
            valence_band = self.extracted_bands[:,v]

            bandgap = np.min(conduction_band) - np.max(valence_band)
            bandgap_shift = self.experiemental_bandgap - bandgap
            
            self.extracted_bands[:,c:-1] += bandgap_shift/2
            self.extracted_bands[:,0:v+1] -= bandgap_shift/2
            



        #Matrix of only the bands we want to consider in our tight binding model. Generally this is the same as our input in main(), but the basis can be increased if needed.
        self.original_num_TBbands = self.num_TBbands
        self.original_skip_bands = self.skip_bands
        self.added_bands = self.num_TBbands - self.original_num_TBbands

        self.truncated_abinit_bands = self.extracted_bands[:, self.original_skip_bands:(self.original_skip_bands + self.original_num_TBbands) ]
        self.truncated_abinit_bands_tens = tf.convert_to_tensor(self.truncated_abinit_bands, dtype=tf.float32)
        

        if(plot_bands == True):
            plt.figure(0)
            red_patch = mpatches.Patch(color='red', label='target bands')
            blue_patch = mpatches.Patch(color='blue', label='abinit bands')
            plt.title('Fitting Bands')
            plt.plot(self.truncated_abinit_bands, 'r+')
            plt.plot(self.extracted_bands,'b')
            plt.legend(loc='upper right', handles=[red_patch,blue_patch])
            plt.ylim(-6,6)
            plt.show()

            #plt.figure(1)
            #plt.title('All TB Bands')
            #plt.plot(self.all_TB_bands, 'r+')
            #plt.plot(self.extracted_bands, 'b')
            #plt.ylim(-6,6)
            #plt.show()


    def Generate_K_Points(self):
        """
        Generate tensor of k-points for calculating energy eigenvalues.
        Assumes K-path is G -> X -> S -> Y -> G
        Must be a tensor with complex datatype.
        """

        a = self.a_tens
        b = self.b_tens

        a = np.real(a.numpy())
        b = np.real(b.numpy())

        nk_segement = int(((self.nks-1))/4)

        kx_GX = np.linspace(0,np.pi/a, nk_segement)
        ky_GX = np.linspace(0,0, nk_segement)

        kx_XS = np.linspace(np.pi/a,np.pi/a, nk_segement)
        ky_XS = np.linspace(0,np.pi/(b), nk_segement)

        kx_SY = np.linspace(np.pi/a,0, nk_segement)
        ky_SY = np.linspace(np.pi/b,np.pi/b, nk_segement)

        kx_YG = np.linspace(0,0, nk_segement+1)
        ky_YG = np.linspace(np.pi/b,0, nk_segement+1)


        self.kx = tf.convert_to_tensor(np.concatenate((kx_GX,kx_XS,kx_SY,kx_YG), axis=None), dtype=tf.complex64)
        self.ky = tf.convert_to_tensor(np.concatenate((ky_GX,ky_XS,ky_SY,ky_YG), axis=None), dtype=tf.complex64)
        



    def Initialize_MLWF(self):
        """
        Reads initial values from the .dat file output by extractTB2st_biggamma.m script.
        """
        if(self.do_restart):
            self.Reinitialize()
        else:
            #Generated random TB matrix:
            MLWF_file = np.loadtxt('HfS2_1L_SmallGamma.dat')
            alpha = tf.convert_to_tensor(np.reshape(MLWF_file[:,0], (self.num_TBbands,self.num_TBbands)), dtype=tf.complex64 )
            beta = tf.convert_to_tensor(np.reshape(MLWF_file[:,1], (self.num_TBbands,self.num_TBbands)), dtype=tf.complex64 )
            gamma = tf.convert_to_tensor(np.reshape(MLWF_file[:,2], (self.num_TBbands,self.num_TBbands)), dtype=tf.complex64 )
            delta11 = tf.convert_to_tensor(np.reshape(MLWF_file[:,3], (self.num_TBbands,self.num_TBbands)), dtype=tf.complex64 )
            delta1_min1 = tf.convert_to_tensor(np.reshape(MLWF_file[:,4], (self.num_TBbands,self.num_TBbands)), dtype=tf.complex64 )

            #alpha_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)
            #beta_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)
            #gamma_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)
            #delta11_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)
            #delta1_min1_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)

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

            self.H_trainable = [alpha_tensor, beta_tensor, gamma_tensor, delta11_tensor, delta1_min1_tensor, beta_tensor_dagger, gamma_tensor_dagger, delta11_tensor_dagger, delta1_min1_tensor_dagger]
            #H_init is never modified. We use it for our loss function modifier.
            self.H_init = [alpha_tensor, beta_tensor, gamma_tensor, delta11_tensor, delta1_min1_tensor, beta_tensor_dagger, gamma_tensor_dagger, delta11_tensor_dagger, delta1_min1_tensor_dagger]

    def Initialize_Random(self):
        """
        Initialize random Hamiltonian. This is currently hard-coded for the small gamma model. Could generalize code in the future.
        """
        if(self.do_restart):
            self.Reinitialize()
            
        else:
            #Generated random TB matrix:
            alpha_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)
            beta_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)
            gamma_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)
            delta11_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)
            delta1_min1_rand = tf.cast(tf.random.normal([self.num_TBbands,self.num_TBbands]),  dtype=tf.complex64)

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
            self.H_trainable = [alpha_tensor, beta_tensor, gamma_tensor, delta11_tensor, delta1_min1_tensor, beta_tensor_dagger, gamma_tensor_dagger, delta11_tensor_dagger, delta1_min1_tensor_dagger]

    def Reinitialize(self):
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

        self.H_trainable = [alpha_tensor, beta_tensor, gamma_tensor, delta11_tensor, delta1_min1_tensor, beta_tensor_dagger, gamma_tensor_dagger, delta11_tensor_dagger, delta1_min1_tensor_dagger]

    def Train_TB(self):
        """ 
        This fuction is for training the bands from randomly generated Hamiltonian.
        Each loop the loss function is calculated and used to extract the gradients.
        Then the gradients are properly symmetrized and applied to the Hamiltonian elements.
        
        """
        opt = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)  
            
        self.loss = 100
        self.count = 0
        self.loss_list = []
        loss_factor = 1/1000

        if(self.do_train):
            while(self.loss > self.converge_target and self.count < self.max_iter):
                
                with tf.GradientTape() as tape:
                    
                    self.E_tb_pred = self.Calculate_Energy_Eigenvals(self.H_trainable)

                    self.loss = tf.reduce_mean(tf.square(self.E_tb_pred - self.truncated_abinit_bands_tens))
                    print(self.count, (self.loss).numpy())
                
                grad = tape.gradient(self.loss, self.H_trainable)
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
           

                opt.apply_gradients(zip(sym_grad_tens, self.H_trainable))
                self.loss_list.append(self.loss)
                self.count+=1

    def Train_MLWF(self):

        """ 
        This fuction is for training the bands starting from a MLWF Hamiltonian.
        Each loop the loss function is calculated and used to extract the gradients.
        An extra term is added to the loss function which penalizes large deviations from the original Hamiltonian elements.
        This extra term compares the current iteration of the hamiltonian elements to their initial values.
        Then the gradients are properly symmetrized and applied to the Hamiltonian elements.
        
        """
        
        opt = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)   
        self.loss = 100
        self.count = 0
        self.loss_list = []
        loss_factor = 1/1000

        if(self.do_train):
            while(self.loss > self.converge_target and self.count < self.max_iter):
                
                with tf.GradientTape() as tape:
                    
                    self.E_tb_pred = self.Calculate_Energy_Eigenvals(self.H_trainable)

                    self.loss1 = tf.reduce_mean(tf.square(self.E_tb_pred - self.truncated_abinit_bands_tens))
                    self.loss2 = 0
                    for i in range(len(self.H_init)):
                        self.loss2 += tf.cast(tf.reduce_mean(tf.square(tf.math.real(self.H_init[i] - self.H_trainable[i]))), dtype=tf.float32)
                    self.loss = self.loss1 + loss_factor*self.loss2

                    print(self.count, (self.loss).numpy())
                
                grad = tape.gradient(self.loss, self.H_trainable)
                grad_real = tf.cast(tf.math.real(grad), dtype=tf.complex64)

                sym_grad = []
                       
                diagonal = tf.linalg.tensor_diag_part(grad_real[0])
                diag_tensor = tf.linalg.diag(diagonal)

                upper_trig = tf.linalg.band_part(grad_real[0],0,-1)
                upper_trig_diag = tf.linalg.diag(tf.linalg.tensor_diag_part(upper_trig))
                upper_trig_nodiag = upper_trig - upper_trig_diag
        
                lower_trig_nodiag = tf.transpose(upper_trig_nodiag)
                
                sym_grad.append(diag_tensor + upper_trig_nodiag + lower_trig_nodiag)
            
                #beta_grad = grad_real[i]
                sym_grad.append(grad_real[1])
        
                #gamma_grad = grad_real[i]
                sym_grad.append(grad_real[2])
        
                #delta11_grad = grad_real[i]
                sym_grad.append(grad_real[3])
        
                #delta1_min1_grad = grad_real[i]
                sym_grad.append(grad_real[4])
                
                sym_grad.append(tf.transpose(grad_real[1]))
                sym_grad.append(tf.transpose(grad_real[2]))
                sym_grad.append(tf.transpose(grad_real[3]))
                sym_grad.append(tf.transpose(grad_real[4]))
                sym_grad_tens = tf.stack(sym_grad)

                opt.apply_gradients(zip(sym_grad_tens, self.H_trainable))
                self.loss_list.append(self.loss)
                self.count+=1

    def Save_Output(self):  
            
        directory = 'H_output'
        if not os.path.exists(directory):
            os.mkdir(directory)

        np.savetxt(os.path.join(directory,'alpha.txt'), self.H_trainable[0].numpy())
        np.savetxt(os.path.join(directory,'beta.txt'), self.H_trainable[1].numpy())
        np.savetxt(os.path.join(directory,'gamma.txt'), self.H_trainable[2].numpy())
        np.savetxt(os.path.join(directory,'delta11.txt'), self.H_trainable[3].numpy())
        np.savetxt(os.path.join(directory,'delta1_min1.txt'), self.H_trainable[4].numpy())
        np.savetxt(os.path.join(directory,'beta_dagger.txt'), self.H_trainable[5].numpy())
        np.savetxt(os.path.join(directory,'gamma_dagger.txt'), self.H_trainable[6].numpy())
        np.savetxt(os.path.join(directory,'delta11_dagger.txt'), self.H_trainable[7].numpy())
        np.savetxt(os.path.join(directory,'delta1_min1_dagger.txt'), self.H_trainable[8].numpy())

        #For plotting
        self.H_final = [self.H_trainable[0].numpy(), self.H_trainable[1].numpy(), self.H_trainable[2].numpy(), self.H_trainable[3].numpy(), self.H_trainable[4].numpy(), self.H_trainable[5].numpy(),self.H_trainable[6].numpy(), self.H_trainable[7].numpy(), self.H_trainable[8].numpy()]
        
        #For export to NEGF
        H_oneside = self.H_final[0:5]
        print()
        H_save = np.zeros((self.num_TBbands**2, 5))
        for i,mat in enumerate(H_oneside):
            flat_mat = np.real(mat.flatten('F')) #Flatten in column-major order.
            H_save[:,i] = flat_mat
        np.savetxt(os.path.join(directory,'HfS2_Small_Gamma_MLTB.dat'), H_save, delimiter='\t')  

    
    def Plot_Bands(self, plot_loss=False):

        self.E_tb_pred_np = (self.E_tb_pred).numpy()

        plt.figure(0)
        plt.plot(self.truncated_abinit_bands,'b')
        
        plt.plot(self.E_tb_pred_np, 'r--')
        plt.ylim(-6,6)
        plt.ylabel('E (eV)')
        plt.title('HfS2 TB Model')
        plt.show()

        #Plotting matrix elements 
        H_map = np.zeros((self.num_TBbands*3,self.num_TBbands*3))
        #Delta1_min1_dagger
        H_map[:self.num_TBbands, :self.num_TBbands] = self.H_final[8]

        #Beta_dagger
        H_map[self.num_TBbands:self.num_TBbands*2, :self.num_TBbands] = self.H_final[5]

        #Delta11_dagger
        H_map[self.num_TBbands*2:self.num_TBbands*3, :self.num_TBbands] = self.H_final[7]

        #Gamma
        H_map[:self.num_TBbands, self.num_TBbands:self.num_TBbands*2] = self.H_final[2]

        #Alpha
        H_map[self.num_TBbands:self.num_TBbands*2, self.num_TBbands:self.num_TBbands*2] = self.H_final[0]

        #Gamma_dagger
        H_map[self.num_TBbands*2:self.num_TBbands*3, self.num_TBbands:self.num_TBbands*2] = self.H_final[6]

        #Delta11
        H_map[:self.num_TBbands, self.num_TBbands*2:self.num_TBbands*3] = self.H_final[3]

        #Beta
        H_map[self.num_TBbands:self.num_TBbands*2, self.num_TBbands*2:self.num_TBbands*3] = self.H_final[1]

        #Delta1_min1
        H_map[self.num_TBbands*2:self.num_TBbands*3, self.num_TBbands*2:self.num_TBbands*3] = self.H_final[4]


        plt.figure()
        im = plt.imshow(np.real(H_map), cmap="OrRd")
        plt.colorbar(im)
        plt.axhline(y=self.num_TBbands-0.5,color='k')
        plt.axhline(y=self.num_TBbands*2-0.5,color='k')
        plt.axvline(x=self.num_TBbands-0.5,color='k')
        plt.axvline(x=self.num_TBbands*2-0.5,color='k')
        plt.show()

        if(plot_loss):
            plt.figure(1)
            plt.plot(self.loss_list)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()









    


