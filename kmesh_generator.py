import numpy as np
import os

filename = 'kmesh.txt'

kx_new = np.linspace(0,0.5,21)
ky_new = np.linspace(0,0.5,21)

kx_2D, ky_2D = np.meshgrid(kx_new,ky_new)
kx = np.reshape(kx_2D, np.size(kx_2D),order='C')
ky = np.reshape(ky_2D, np.size(ky_2D),order='C')

nks = np.size(kx)

cwd = os.getcwd()
cwd = cwd + '/kmeshes/'
print(cwd)

filepath = os.path.join(cwd,filename)
f = open(filepath,'w')
lines = ["K_POINTS crystal \n", str(nks) ]
f.writelines(lines)
f.close()



