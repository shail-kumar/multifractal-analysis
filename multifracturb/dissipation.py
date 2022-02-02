#!/usr/bin/env python
# coding: utf-8

# ### Computation of the derivative of velocity field using FFT

# In[ ]:


"""
Created on 2021-06-20 00:21:11

@author: Shailendra K Rathor

"""


# In[2]:


import numpy as np
import h5py
import os


# ### Fourier Transform 

# In[3]:


def FT(A):
    N = A.shape[0]
    A = A / N**3		# Normalization
    return np.fft.rfftn(A)

# Inverse Fourier Transform
def IFT(A):
    N = A.shape[0]
    A = A * N**3		# Normalization
    return np.fft.irfftn(A)


# ### Loading of velocity data

# In[4]:


def expand_thirdComponent(Ux, Uy, Uz):
    """
    Uz is only the velocity field of the plane kz = 0.
    This function expands the third component of velocity to match the array size of the first two components.
    This is achieved from the incompressibility condition: U3 = (-kxUx -kyUy)/kz; kz!= 0.
    
    Parameters
    ----------
    Ux, Uy, Uz :  numpy array
            Input array, taken to be complex. The components of velocity field.
            
    Returns
    -------
    out : numpy array
        Third component of size N * N * (N/2 + 1).

    """
    kxUx = Ux.copy()
    N = Ux.shape[0]
    kx_list = np.fft.fftfreq(N, 1./N)
    for i, kx in enumerate(kx_list):
        kxUx[i,:,:] = kx * Ux[i,:,:]
        
    kyUy = Uy.copy()
    N = Uy.shape[0]
    ky_list = np.fft.fftfreq(N, 1./N)
    for j, ky in enumerate(ky_list):
        kyUy[:,j,:] = ky * Uy[:,j,:]
        
    U3 = -(kxUx + kyUy)
    U3[:,:,0] = Uz[:,:,0]

    kz_list = np.fft.rfftfreq(N, 1./N)
    for k, kz in enumerate(kz_list):
        if(kz != 0):
            U3[:,:,k] = U3[:,:,k]/kz
    
    return U3


# In[5]:


def read_complexVelocityField(input_dir = "."):
    """
    Reads the hdf5 files and returns the components of complex velocity field.
    Parameter
    ---------
    input_dir : string
        The directory containing the .h5 files of data.
        
    Returns
    -------
    out :  numpy array
        All three components of velocity, so 3 arrays.
        
    """
    f = h5py.File(input_dir + '/U.V1.h5', 'r')
    V1 = f['U.V1']['real'] + 1j * f['U.V1']['imag']
    f.close()
    
    f = h5py.File(input_dir + '/U.V2.h5', 'r')
    V2 = f['U.V2']['real'] + 1j * f['U.V2']['imag']
    f.close()
    
    f = h5py.File(input_dir + '/U.V3kz0.h5', 'r')
    V3_kz0 = f['U.V3kz0']['real'] + 1j * f['U.V3kz0']['imag']
    f.close()
    
    V3 = expand_thirdComponent(V1, V2, V3_kz0)
    
    return V1, V2, V3


# ### Derivatives:
# $\partial_i u_j = IFT(i k_i \hat{u_j})$, where $IFT$ is inverse Fourier transform.

# In[6]:


def derivatives(Uj):
    """
    Derivative components dU/dx, dU/dy, and dU/dz of a given velocity component U.
    Input: Complex velocity field component U (a numpy array) of size N*N*(N/2+1)
    Returns: 3 numpy arrays of size N*N*N
    
    Parameters
    ----------
    Uj : numpy array
        Input array, taken to be complex. The component of velocity field of which 
        the derivatives along three components are to be computed.
        
    Returns:
    out : numpy array
        Three arrays, real.

    """
    
    ## Derivative of Uj wrt x
    kxUj = Uj.copy()
    N = Uj.shape[0]
    kx_list = np.fft.fftfreq(N, 1./N)
    for i, kx in enumerate(kx_list):
        kxUj[i,:,:] = kx * Uj[i,:,:]
    kxUj = 1j * kxUj
    dxuj = IFT(kxUj)

    ## Derivative of Uj wrt y
    kyUj = Uj.copy()
    N = Uj.shape[1]
    ky_list = np.fft.fftfreq(N, 1./N)
    for j, ky in enumerate(ky_list):
        kyUj[:,j,:] = ky * Uj[:,j,:]
    kyUj = 1j * kyUj
    dyuj = IFT(kyUj)

    ## Derivative of Uj wrt z
    kzUj = Uj.copy()
    N = 2*(Uj.shape[2]-1)
    kz_list = np.fft.rfftfreq(N, 1./N)
    for k, kz in enumerate(kz_list):
        kzUj[:,:,k] = kz * Uj[:,:,k]
    kzUj = 1j * kzUj
    dzuj = IFT(kzUj)

    return dxuj, dyuj, dzuj


# ### Calculation of local dissipation rate:
# $\varepsilon(x) = (\nu/2) * \sum_{i,j} (\partial_i u_j + \partial_j u_i)^2 $

# In[7]:


def dissipationRate(U1, U2, U3, nu, save = True, save_in_file = 'dissipation_rate.h5'):
    """
    Computes the local energy dissipation rate eps(x).
    $\epsilon(x) = (\nu/2) * \sum_{i,j} (\partial_i u_j + \partial_j u_i)^2 $
    
    Parameters
    ----------
    U1, U2, U3 : numpy array
        Three components of velocity field.
    nu : number
        viscosity.
    save : bool, optional
        Saves the dissipation rate in file kwarg save_in_file, if set to True.
    save_in_file = string, optional
        Saves the dissipation rate in this file. Default is 'dissipation_rate.h5' and dataset name is 'eps'.
    
    Returns
    -------
    out : numpy array
        One array, real. 

    """
    # 	U1, U2, U3 = read_complexVelocityField()

    d1u1, d2u1, d3u1 = derivatives(U1)
    d1u2, d2u2, d3u2 = derivatives(U2)
    d1u3, d2u3, d3u3 = derivatives(U3)
    
    e11 = 2*d1u1
    e22 = 2*d2u2
    e33 = 2*d3u3
    
    e12 = d1u2 + d2u1
    e13 = d1u3 + d3u1
    e23 = d2u3 + d3u2
    
    eps = (e11**2 + e22**2 + e33**2) + 2 * (e12**2 + e13**2 + e23**2)
    eps = 0.5 * nu * eps
    
    # save dissipation rate
    if save:
        f = h5py.File(save_in_file, 'w')
        f.create_dataset('eps', data = eps, compression = 'gzip')
        print('Dissipation rate saved in ', os.path.join(os.getcwd(), save_in_file))
    return eps


# In[ ]:


if __name__=='__main__':
    print("This is a module!")

