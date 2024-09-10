import numpy as np
import tensorflow as tf
from NN_Models.TBNN_Model_V2 import TBNN_V2
from NN_Models.TBNN_Model_V2_extended import TBNN_V2_DoubleGamma
from NN_Models.TBNN_Model_V2_weighted import TBNN_V2_weighted
from NN_Models.TBNN_Model_V2_fitwan import TBNN_V2_fitwan
from NN_Models.TBNN_Model_V2_fitwan_extended import TBNN_V2_fitwan_extended
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

'''
The following code generates tight-binding parameters for 2D materials using a machine learning approach. 
The user must provide the ab-initio band structure and unit cell dimensions.
If Maximally Localized Wannier Functions (MLWFs) are used as the initial guess, the MLWF hamiltonian must be provided. This can be generated using the extract_gamma Matlab script.

========INPUTS========
a , b : X and Y dimensions of the unit cell, respectively.

bands_filename : The name of the ab-initio .bands file output from Quantum Espresso.

output_hamiltonian : Name of the output file to be exported for NEGF calculations.

Ef : The fermi energy.

experimental_bandgap : The experimentaly determined bandgap of the system.

target_bands : The number of ab-initio bands to consider for the fitting. The included bands start from the first band with energy higher than the skipped bands.
              The dimensions of each Hamiltonian block alpha, beta, etc is (target_bands, target_bands).

skip_bands : The number of ab-intio bands to skip, starting from the lowest energy band. These bands will not be used for the fitting.


plot_abinit_bands : Set to TRUE to plot the ab-initio bands. The bands being considered in the fitting are marked by the red dashed lines.

fit_MLWF : Set to TRUE if you want to use a wannier model as the initial guess. In this case set numTBbands and skip_bands to the same values you set in the .win file.
           Set to FALSE is you want to generate a tight binding model starting from a randomly initialized Hamiltonian.

restart : Set to TRUE if you want to run the band fitting procedure starting from the most recently generated tight-binding parameters
          that the program ended at.

bandgap_correction : Set to TRUE if you want to use the experimental bandgap value, rather than the DFT-calculated bandgap.



learn_rate : Learning rate argument for the Adam Optimizer. In the range of 0.01-0.0001 is ideal. Larger learning rate works better for initial fitting.
             and smaller learning rate better for fine tuning when loss is already small.

converge_target : Convergence target for the band fitting. The fitting procedure will end when the loss reaches this value. 
                  Generally smaller than 5e-5 produces a good fit.

max_iter : Maximum number of iterations for the band fitting procedure.



fit_bands : Set to TRUE to run tight-binding fitting. Can be disabled if you just want to plot the ab-initio bands.

=======OUTPUTS=======
Graphs : Graph of the ab-initio and TB bands overlayed on each other.
         Visual representation of Hamiltonian element values.

Files: Your_Hamiltonian.dat <-- use this for NEGF code. Found in the H_output folder.

'''

#Cell Dimensions
a = 6.290339029863558E-10
b = 3.631729244941925E-10

#Band Structure
bands_filename = 'HfS2_23x23_bands.dat'
output_hamiltonian_name = 'HfS2_DFTfit_noreg.dat'
Ef = -3.3312 #-2.5 - 0.126
experimental_bandgap = 1.89638 #[eV]s
target_bands = 22
skip_bands = 12

#Routines to perform
fit_MLWF = False
restart = True
bandgap_correction = True

#set max iter to 0 if using threshold
do_threshold = False
threshold_val = 0.02


#Learning Parameters
learn_rate = 0.006
regularization_factor = 0e-5 #Controls Hamiltonian sparsity. Adjust as needed.
regularization_factor_ext = 0e-5 #For double gamma, should be larger than regularization_factor
L2_factor = 0e-5
converge_target = 1e-8
max_iter = 600

def main():
    #tbnn2 = TBNN_V2(a, b, Ef, restart, skip_bands, target_bands, converge_target, max_iter, learn_rate, regularization_factor , bands_filename, output_hamiltonian_name, bandgap_correction, experimental_bandgap)
    #tbnn2.fit_bands()

    #tbnn2 = TBNN_V2_DoubleGamma(a, b, Ef, restart, skip_bands, target_bands, converge_target, max_iter, learn_rate, 
    #                            regularization_factor, regularization_factor_ext, bands_filename, output_hamiltonian_name, bandgap_correction, experimental_bandgap, do_threshold, threshold_val)
    #tbnn2.fit_bands()

    tbnn2 = TBNN_V2_weighted(a, b, Ef, restart, skip_bands, target_bands, converge_target, max_iter, learn_rate, 
                             regularization_factor , bands_filename, output_hamiltonian_name, bandgap_correction, experimental_bandgap, 
                             L2_factor, do_threshold, threshold_val, fit_MLWF)
    tbnn2.fit_bands()

    #tbnn2 = TBNN_V2_fitwan(a, b, restart, target_bands, converge_target, max_iter, learn_rate, 
    #                         regularization_factor , bands_filename, output_hamiltonian_name)
    #tbnn2.fit_bands()

    #tbnn2 = TBNN_V2_fitwan_extended(a, b, restart, target_bands, converge_target, max_iter, learn_rate, 
    #                         regularization_factor , bands_filename, output_hamiltonian_name)
    #tbnn2.fit_bands()


if __name__ == '__main__':
    main()

