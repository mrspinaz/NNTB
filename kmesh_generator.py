import numpy as np
import os

'''
Creates a grid of k-points in the irriducible Brillouin zone for a 2D simple orthorhombic cell.

The generated grid of k-points should be copy/pasted to the end of the '.bands' file for Quantum Espresso 
to obtain the input data for training.


'''

filename = 'kmesh.txt'
kx_size = 40
ky_size = 40

kx_new = np.linspace(0,0.5,kx_size)
ky_new = np.linspace(-0.5,0.5,ky_size)

kx_2D, ky_2D = np.meshgrid(kx_new,ky_new)
nks = np.size(kx_2D)
weight = 1.0/nks

kx = np.reshape(kx_2D, (np.size(kx_2D),1),order='F')
ky = np.reshape(ky_2D, (np.size(ky_2D),1),order='F')
kz = np.zeros((nks,1))
w = np.ones((nks,1))*weight


all_kw = np.concatenate([kx, ky, kz, w], axis=1)
cwd = os.getcwd()
cwd = cwd + '/kmeshes/'
filepath = os.path.join(cwd,filename)
if not os.path.exists(cwd):
    os.makedirs(cwd)
f = open(filepath,'w')
lines = ["K_POINTS crystal \n", str(nks) + '\n'  ]
f.writelines(lines)


for i in range(nks):
    line = all_kw[i,:]
    string_line = ["%.8f" % number + '  ' for number in line]
    #print(string_line)
    string_line.append('\n')
    f.writelines(string_line)
    #np.savetxt(filepath,all_kw,delimiter='\t',fmt='%f')
f.close()



