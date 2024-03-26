import re
import numpy as np
import tensorflow as tf

class TBNN_V2:
    #Maybe change code at some point to extract these from scf, bands, etc output files, other than boolean commands.
    def __init__(self, a, b, Ef, restart, skip_bands, target_bands, converge_target, max_iter, learn_rate):
        self.a = a
        self.b = b
        self.Ef = Ef
        self.restart = restart
        self.skip_bands = skip_bands
        self.target_bands = target_bands
        self.converge_target = converge_target
        self.max_iter = max_iter
        self.learn_rate = learn_rate

    def Extract_Abinit_Eigvals(self, bands_filename):

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

            abinit_bands = abinit_bands.reshape(nks,nband)        
            

        file.close()

        #Convert k_points to their actual values
        ky_factor = self.a/self.b
        kpoints[0,:] =  (kpoints[0,:])*(2.0*np.pi/self.a)
        kpoints[1,:] =  (kpoints[1,:]/ky_factor)*(2.0*np.pi/self.b)
        
        return abinit_bands, kpoints
    
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

        H_trainable = [alpha_tensor, beta_tensor, gamma_tensor, delta11_tensor, delta1_min1_tensor, beta_tensor_dagger, gamma_tensor_dagger, delta11_tensor_dagger, delta1_min1_tensor_dagger]
        
        return H_trainable
    
    def fit_bands(self):
        """ 
        This fuction is for training the bands from randomly generated Hamiltonian.
        Each loop the loss function is calculated and used to extract the gradients.
        Then the gradients are properly symmetrized and applied to the Hamiltonian elements.
        
        """
        opt = tf.keras.optimizers.Adam(learning_rate=self.learn_rate)  
            
        loss = 100
        count = 0
        loss_list = []

        H_trainable = self._Initialize_Hamiltonian()

        
        
        while(self.loss > self.converge_target and count < self.max_iter):
            
            with tf.GradientTape() as tape:
                
                self.E_tb_pred = self.Calculate_Energy_Eigenvals(H_trainable)

                self.loss = tf.reduce_mean(tf.square(self.E_tb_pred - self.truncated_abinit_bands_tens))
                print('Iteration: ', count,' Loss: ' , (self.loss).numpy())
            
            grad = tape.gradient(self.loss, H_trainable)
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

        if self.count == self.max_iter:
            print('Max iterations reached.')
        elif self.loss <= self.converge_target:
            print('Convergence target reached.')
        else:
            print('Unknown exit condition.')
    




        



                
