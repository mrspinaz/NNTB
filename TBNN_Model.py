import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re   

class TBNN:

    def __init__ (self, a, b, bands_filename, Ef, num_TBbands, skip_bands, do_train, do_restart, learn_rate, converge_target, max_iter):
        
        self.a = a
        self.b = b
        self.a_tens = tf.convert_to_tensor(a, dtype=tf.complex64)
        self.b_tens = tf.convert_to_tensor(b, dtype=tf.complex64)

        self.bands_filename = bands_filename
        self.Ef = Ef
        self.num_TBbands = num_TBbands
        self.skip_bands = skip_bands

        self.do_train = do_train
        self.do_restart = do_restart

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

        with open(self.bands_filename) as data:
            lines = data.readlines()
            data.close()

        #Extract number of bands and kpoints.
        temp = re.findall('\d+',lines[0])
        self.nband = int(temp[0])
        self.nks = int(temp[1])

        #Extract Kpoint list
        kpoints_arr = np.zeros((self.nks,3))
        all_kpoints_text = [line for line in lines if line.startswith("            ")]

        for text_line in range(self.nks):
            kpoints_oneline = re.findall('[-]?\d+.\d+', all_kpoints_text[text_line])
            kpoints_oneline_arr = np.array(kpoints_oneline, dtype=np.float32)
            kpoints_arr[text_line,:] = kpoints_oneline_arr

        #self.ky = kpoints_arr[:,1]*2*np.pi/self.a
        #self.ky = tf.convert_to_tensor(self.ky, dtype=tf.complex64)
        
        #self.kx = kpoints_arr[:,0]*2*np.pi/self.b
        #self.kx = tf.convert_to_tensor(self.kx, dtype=tf.complex64)
        

        

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

        #Matrix of only the bands we want to consider in our tight binding model.
        self.original_num_TBbands = 14
        self.original_skip_bands = 16
        self.added_bands = self.num_TBbands - self.original_num_TBbands

        #self.truncated_abinit_bands = self.extracted_bands[:, self.skip_bands:(self.skip_bands + self.num_TBbands) ]
        #self.truncated_abinit_bands_tens = tf.convert_to_tensor(self.truncated_abinit_bands, dtype=tf.float32)

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

        a = self.a_tens
        b = self.b_tens

        a = np.real(a.numpy())
        b = np.real(b.numpy())


        kx_GX = np.linspace(0,np.pi/a, 32)
        ky_GX = np.linspace(0,0, 32)

        kx_XS = np.linspace(np.pi/a,np.pi/a, 32)
        ky_XS = np.linspace(0,np.pi/(b), 32)

        kx_SY = np.linspace(np.pi/a,0, 32)
        ky_SY = np.linspace(np.pi/b,np.pi/b, 32)

        kx_YG = np.linspace(0,0, 33)
        ky_YG = np.linspace(np.pi/b,0, 33)


        self.kx = tf.convert_to_tensor(np.concatenate((kx_GX,kx_XS,kx_SY,kx_YG), axis=None), dtype=tf.complex64)
        self.ky = tf.convert_to_tensor(np.concatenate((ky_GX,ky_XS,ky_SY,ky_YG), axis=None), dtype=tf.complex64)
        


    def Train_TB(self):

        opt = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)

        if(self.do_restart):
            #Read values from alpha.txt and beta.txt
            alpha = np.real(np.loadtxt('alpha.txt', dtype=complex))
            beta = np.loadtxt('beta.txt', dtype=complex)
            gamma = np.loadtxt('gamma.txt', dtype=complex)
            delta11 = np.loadtxt('delta11.txt', dtype=complex)
            delta1_min1 = np.loadtxt('delta1_min1.txt', dtype=complex)
            
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
            print(alpha)
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
            
        self.loss = 100
        self.count = 0
        self.loss_list = []

        if(self.do_train):
            while(self.loss > self.converge_target and self.count < self.max_iter):
                
                with tf.GradientTape() as tape:
                    #Shouldn't acutally need the tape.watch() calls
                    tape.watch(alpha_tensor)
                    tape.watch(beta_tensor)
                    tape.watch(gamma_tensor)
                    tape.watch(delta11_tensor)
                    tape.watch(delta1_min1_tensor)
                    tape.watch(beta_tensor_dagger)
                    tape.watch(gamma_tensor_dagger)
                    tape.watch(delta11_tensor_dagger)
                    tape.watch(delta1_min1_tensor_dagger)
                    self.E_tb_pred = self.Calculate_Energy_Eigenvals(self.H_trainable)

                    self.loss = tf.reduce_mean(tf.square(self.E_tb_pred - self.truncated_abinit_bands_tens))
                    print(self.count, (self.loss).numpy())
                
                grad = tape.gradient(self.loss, self.H_trainable)
                grad_real = tf.cast(tf.math.real(grad), dtype=tf.complex64)
                opt.apply_gradients(zip(grad_real, self.H_trainable))
                self.loss_list.append(self.loss)
                self.count+=1
                
            self.E_tb_pred_np = (self.E_tb_pred).numpy()
            
            np.savetxt('alpha.txt', alpha_tensor.numpy())
            np.savetxt('beta.txt', beta_tensor.numpy())
            np.savetxt('gamma.txt', gamma_tensor.numpy())
            np.savetxt('delta11.txt', delta11_tensor.numpy())
            np.savetxt('delta1_min1.txt', delta1_min1_tensor.numpy())
            np.savetxt('beta_dagger.txt', beta_tensor_dagger.numpy())
            np.savetxt('gamma_dagger.txt', gamma_tensor_dagger.numpy())
            np.savetxt('delta11_dagger.txt', delta11_tensor_dagger.numpy())
            np.savetxt('delta1_min1_dagger.txt', delta1_min1_tensor_dagger.numpy())

            
        
    def Plot_Bands(self, plot_loss=False):

        plt.figure(0)
        plt.plot(self.truncated_abinit_bands,'b')
        
        plt.plot(self.E_tb_pred_np, 'r--')
        plt.ylim(-6,6)
        plt.ylabel('E (eV)')
        plt.title('Graphene 2pz TB Model')
        plt.show()

        if(plot_loss):
            plt.figure(1)
            plt.plot(self.loss_list)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()









    


