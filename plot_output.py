import numpy as np
import matplotlib.pyplot as plt

alpha = np.real(np.loadtxt('alpha.txt', dtype=complex))
beta = np.loadtxt('beta.txt', dtype=complex)
gamma = np.loadtxt('gamma.txt', dtype=complex)
delta11 = np.loadtxt('delta11.txt', dtype=complex)
delta1_min1 = np.loadtxt('delta1_min1.txt', dtype=complex)

beta_dagger = np.loadtxt('beta_dagger.txt', dtype=complex)
gamma_dagger = np.loadtxt('gamma_dagger.txt', dtype=complex)
delta11_dagger = np.loadtxt('delta11_dagger.txt', dtype=complex)
delta1_min1_dagger = np.loadtxt('delta1_min1_dagger.txt', dtype=complex)

a = 6.290339483e-10
b = 3.631729507E-10


kx_GX = np.linspace(0,np.pi/a, 32)
ky_GX = np.linspace(0,0, 32)

kx_XS = np.linspace(np.pi/a,np.pi/a, 32)
ky_XS = np.linspace(0,np.pi/(b), 32)

kx_SY = np.linspace(np.pi/a,0, 32)
ky_SY = np.linspace(np.pi/b,np.pi/b, 32)

kx_YG = np.linspace(0,0, 33)
ky_YG = np.linspace(np.pi/b,0, 33)


kx = np.concatenate((kx_GX,kx_XS,kx_SY,kx_YG), axis=None)
ky = np.concatenate((ky_GX,ky_XS,ky_SY,ky_YG), axis=None)

E = np.zeros((alpha.shape[1],len(kx)))


for ii in range(len(kx)):
    H = alpha + beta*np.exp(1j*kx[ii]*a) + gamma*np.exp(1j*ky[ii]*b) + \
        delta11*np.exp(1j*kx[ii]*a + 1j*ky[ii]*b) + delta1_min1*np.exp(1j*kx[ii]*a - 1j*ky[ii]*b) + \
        beta_dagger*np.exp(-1j*kx[ii]*a) + gamma_dagger*np.exp(-1j*ky[ii]*b) + \
        delta11_dagger*np.exp(-1j*kx[ii]*a - 1j*ky[ii]*b) + delta1_min1_dagger*np.exp(-1j*kx[ii]*a + 1j*ky[ii]*b)
    eigvals, eigvecs = np.linalg.eig(H)

    E[:,ii] = np.sort((eigvals.T))

plt.figure()
plt.plot(E.T,'b--')
plt.show()


