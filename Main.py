import numpy as np
import tensorflow as tf
from TBNN_Model import TBNN
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

num_TBbands : The number of ab-initio bands to consider for the fitting. The included bands start from the first band with energy higher than the skipped bands.
              The size of each Hamiltonian block (alpha, beta, etc) is num_TBbands*numTBbands.

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
a = 6.290339483e-10
b = 3.631729507E-10

#Band Structure
bands_filename = 'HfS2_bands.dat'
output_hamiltonian = 'HfS2_Small_Gamma_MLTB_EgAdjusted.dat'
Ef = -2.5834
experimental_bandgap = 1.8858 #[eV]
num_TBbands = 18 
skip_bands = 12

#Routines to perform
plot_abinit_bands = True
fit_MLWF = True
restart = True
bandgap_correction = True

#Learning Parameters
learn_rate = 0.005
converge_target = 1e-4
max_iter = 200

#Parameters for Testing
fit_bands = True 

def main():
    tbnn = TBNN(a, b, bands_filename, output_hamiltonian, Ef, experimental_bandgap, num_TBbands, skip_bands, fit_bands, restart, bandgap_correction,  learn_rate, converge_target, max_iter)
    tbnn.Extract_Abinit_Bands(plot_abinit_bands)
    tbnn.Generate_K_Points()
    if(fit_MLWF):
        tbnn.Initialize_MLWF()
        tbnn.Train_MLWF()
    else:
        tbnn.Initialize_Random()
        tbnn.Train_TB()
    tbnn.Save_Output()
    tbnn.Plot_Bands()

if __name__ == '__main__':
    main()

