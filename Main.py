import numpy as np
import tensorflow as tf
from TBNN_Model import TBNN

'''
The following code generates tight-binding parameters for 2D materials using a machine learning approach. 
The user must provide the ab-initio band structure and unit cell dimensions.

========INPUTS========
a , b : X and Y dimensions of the unit cell, respectively.

H_abinit : The ab-initio Hamiltonian.

fit_bands : Set to TRUE to run tight-binding fitting. Can be disabled if you just want to plot the ab-initio bands.

restart : Set to TRUE if you want to run the band fitting procedure starting from the most recently generated tight-binding parameters
          that the program ended at.

learn_rate : Learning rate argument for the Adam Optimizer. In the range of 0.1-0.0001 is ideal. Larger learning rate works better for initial fitting.
             and smaller learning rate better for fine tuning when loss is already small.

converge_target : Convergence target for the band fitting. The fitting procedure will end when the loss reaches this value.

max_iter : Maximum number of iterations for the band fitting procedure.

=======OUTPUTS=======
Graphs : Graph of the ab-initio and TB bands overlayed on each other.

alpha.txt, beta.txt, beta_dagger.txt : TB parameters for the real-space Hamiltonian. beta_dagger.txt is just output to verify that
                                        the generated Hamiltonian is hermatian.

'''

#Cell Dimensions
a = 6.290339483e-10
b = 3.631729507E-10

#Band Structure
bands_filename = 'HfS2_bands.dat'
Ef = -2.5834
num_TBbands = 24
skip_bands = 13

#Routines to perform
#plot_tb = True #Keep False if Running on cluster.
fit_bands = True
restart = False

#Learning Parameters
learn_rate = 0.01
converge_target = 1e-6
max_iter = 1000


def main():
    tbnn = TBNN(a, b, bands_filename, Ef, num_TBbands, skip_bands, fit_bands, restart, learn_rate, converge_target, max_iter)
    tbnn.Extract_Abinit_Bands(plot_bands=False)
    tbnn.Generate_K_Points()
    tbnn.Train_TB()
    tbnn.Plot_Bands()

if __name__ == '__main__':
    main()

