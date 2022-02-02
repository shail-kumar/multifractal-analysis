#!/usr/bin/env python
# coding: utf-8

# ### Multifractal analysis of a dataset in 3-dimensions.

# In[1]:


"""
Created on 2021-06-20 03:20:36 

@author: Shailendra K Rathor

This script is to compute Multifractal spectrum of a 3-D data set.
INPUT: A 3-D dataset.
OUTPUT: A 2-D array of order of moment and generalized dimension.
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import h5py


# In[2]:


def load_synthetic_data(run, gridSize = 32):
    """ Load data obtained from shell model data using Jensen scheme."""
    grid_size = gridSize
    A = np.zeros((grid_size,grid_size,grid_size))
    for x_plane in range(grid_size):
        #~ print(x_plane)
        filename = '../output/nu-3/dissRate-'+str(run)+'/x-'+str(x_plane)+'.h5'
        f = h5py.File(filename, 'r')
        A[x_plane,:,:] = f['eps'][:grid_size, :grid_size]
        f.close()

    return A


# ### Working with measure

# In[3]:


def measure(data, boxSize):
    """ 
    Compute the measure in ith box of size r.
    
    Parameters
    ----------
    data : numpy array
        Input array, taken to be real. Values of observables in 3D.
    boxSize : number
        Size of box as power of 2, i.e. 1, 2, 4, 8,..., size of the data array.
    
    Returns
    -------
    out : numpy array
        Array of values of measure mu of a box of given box size. 
    """
    S = data
    r = boxSize
    Norm = np.sum(S)
    if(boxSize == 1):
        return S/Norm
    else:
        L = S.shape[0]
        linear_boxes = int(L/r)       # Number of intervals in a dimension
        mu = np.zeros((linear_boxes))
        for i in range(linear_boxes):
            mu[i] = np.sum(S[i*r:(i+1)*r])
        return mu/Norm


# In[4]:


def normalized_measure(Pl, q):
    """
    Normalizes the measure.
    
    Parameters
    ----------
    Pl : numpy array
        Input array, taken to be real. Array of integrated Pprobability of box-size l. 
    q : number
        Order of moment.
        
    Returns
    ------
    measure_ql : numpy array
        array of real elements.
    norm : number
        normalization constant
        
    """
    #~ Pl = Pl.astype(np.float128) 

    norm = np.sum(Pl**q)
    numerator = Pl**q

    measure_ql = numerator/norm
    return measure_ql, norm


# In[5]:


def compute_measures(data, save=False, save_in_file="mu.h5"):
    """
    Computes measures for all scales for the given data.
    
    Parameters
    ----------
    data : numpy array
        Input 3D array of real elements.
        
    save : bool, optional
        Saves the measure in a file if True, Default is False.
        
    save_in_file : string, optional
        The file for saving the output. Default is 'mu.h5'
        
    Returns
    -------
    mu_list : list, if save = True
        List of numpy arrays of differnt sizes determined by the box size.
    1 : integer, if the input array is not cubical i.e all three axes are not equal.
    
    """
    Nx = data.shape[-1]
    scales = int(np.log(Nx) / np.log(2))
    print("Generating measures... Wait...")
    mu_list = []
    if save:
        f = h5py.File(save_in_file, 'w')
    for i in range(scales + 1):
        print("scale = ",i)
        box_size = 2**i
        Er = measure(data, box_size)
        mu_list.append(Er)
        if save:
            f.create_dataset('mu_'+str(box_size), data = Er)
    if save:
        f.close()
    
    return mu_list


# In[6]:


def load_measures(measure_file):
    """
    Reads the measure_file and returns a list of arrays of measure for at different scales.
    
    Parameters
    ----------
    measure_file : string
        Name of the file (with path) in HDF5 format.
    
    Returns
    -------
    mu_list : list of the arrays of measures at all scales.
    """
    f = h5py.File(measure_file,'r')
    dset_list = list(f.keys())
    scales = len(dset_list)
    mu_list = []
    for i in range(scales):
        box_size = 2**i
#         print(box_size)
        mu_list.append(f['mu_'+str(box_size)][:])
    f.close()
    return mu_list


# In[7]:


def q_list(q, dq):
    """ 
    Returns the set of moments.
    
    Parameters
    ---------
    q : number
        float, in general. The maximum value of the order of moments.
    dq : number
        float, in general. The separation between two consecutive moments.
        
    Returns
    -------
    out : numpy array
        array of reals. The list of moments.
        
    """
    dq = float(dq)
    moments = np.arange(-q,q+dq,dq)  # q+dq is taken to include the endpoint +q.
    for i in range(len(moments)):
        if(moments[i] == 1):
            moments[i] = 0.999
    return moments


# ### Formulas for multifractal analysis:
# Generalised dimension:
# $D_q = \frac{1}{q-1} \lim_N \frac{log}{log(N)}$
# 
# ### Computation of multifractal spectrum

# In[8]:


def mf_data(data, q_max, dq = 1., save=True, save_in_dir='.'):
    """
    Computes the multifractal spectrum of the 3D data.
    
    Parameters
    ----------
    data : ndarray, list
        ndarray for 3D data to compute measures and list for already computed measures.
    q_max : number
        Maximum order of moment. Accordingly, minimum of order of moment is taken -q.
    dq : float number, optional, default is 1.0
        Seperation between two consecutive orders of moment.
    save : bool, optional
        saves the output in files if set to True.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, infer the data
    type from the other input arguments.

    Returns
    -------
    mf_spectrum : ndarray
        Three arrays in order of: Dq_numerator, alpha_numerator, f_numerator
    
    NOTE
    ----
    In the output array, the first row are denominators, the first column are moments 
    and the rest are the numerators.
    
    """
    ### Data
    if(type(data)==np.ndarray):
        mu_list = compute_measures(data)
        if(mu_list == 1):
            print("Program terminating...")
            sys.exit(0)
    elif(type(data) == list):
        mu_list = data
    else:
        print("Data should be either list or numpy.ndarray. Terminating program...")
        sys.exit(0)
    
    ### Moments
    moments = q_list(q_max, dq)
    number_of_moments = len(moments)

    ### scales
    number_of_scales = len(mu_list)
    scales = range(number_of_scales)
    
    denominator = np.zeros(number_of_scales)
    for i in scales:  # Loop over Scales
        Er = mu_list[i]
        N = Er.shape[0]
        denominator[i] = (-1)*np.log(N)
        
    Dq_numerator = np.zeros((number_of_moments+1, number_of_scales+1))
    Dq_numerator[0,1:] = denominator
    Dq_numerator[1:,0] = moments
    
    alpha_numerator = np.zeros((number_of_moments+1, number_of_scales+1))
    alpha_numerator[0,1:] = denominator
    alpha_numerator[1:,0] = moments
    
    f_numerator = np.zeros((number_of_moments+1, number_of_scales+1))
    f_numerator[0,1:] = denominator
    f_numerator[1:,0] = moments
    
    for iq, q in enumerate(moments):  # Loop over moments
        print("q \t scale")
        for i in scales:  # Loop over Scales
            print(q,"\t", i)
            Er = mu_list[i]
            N = Er.shape[0]
            mu, norm = normalized_measure(Er,q)
            
            Dq_numerator[iq+1,i+1] = np.log(norm)/(q-1)
            alpha_numerator[iq+1,i+1] = np.sum(mu * np.log(Er))
            f_numerator[iq+1,i+1] = np.sum(mu * np.log(mu))
        print("-------------")
        
    if save:
        dir_nod = os.path.exists(save_in_dir)
        if(dir_nod == False):
            os.makedirs(save_in_dir)
        np.savetxt(save_in_dir + "/Dq_data.d",Dq_numerator)
        np.savetxt(save_in_dir + "/alpha_data.d",alpha_numerator)
        np.savetxt(save_in_dir + "/f_data.d",f_numerator)
        print("Data saved in dir ", os.path.join(os.getcwd(), save_in_dir))
    return Dq_numerator, alpha_numerator, f_numerator


# In[9]:


def plots_for_scaling_range(mf_data, data_of, plots_dir="."):
    """
    Plots the multifractal data with log(l) to determine the linear scaling range, manually.
    The individual plots are generated for all moments and saved in the plots_dir.
    
    Parameters
    ----------
    mf_data : numpy array
        Input array, taken to be real. The first row contains log(l), the first 
        column contains momemts (q), and the rest of the rows are the mf quantity 
        corressponding to q of that row.
    data_of : string
        This specifies the multifractal quantity of interset, namely Dq, \alpah or f(\alpha).
        This name is only used to name the file for saving the plot.
    plots_dir : string
        The name of the directory for saving the plots. If directory does not exist, it is created.
        Default is current directry ('.').
    
    Return
    ------
    out : It only saves '.png' files.
    
    """
    dir_nod = os.path.exists(plots_dir)
    if(dir_nod == False):
        os.makedirs(plots_dir)
        
    moments = mf_data[1:,0]
    denominator = mf_data[0,1:]
    
    for iq, q in enumerate(moments):
        numerator = mf_data[iq+1, 1:]
        z,cov = np.polyfit(denominator, numerator, 1, cov='unscaled')
        
        plt.plot(denominator, numerator,'o', label='q = '+str(q))
        
        y = z[0] * denominator + z[1]
        plt.plot(denominator, y)
        
        plt.legend()
        plt.title(data_of)
        plt.xlabel(r'$\log(l)$')
        plt.ylabel(data_of)
        
        plt.savefig(plots_dir+'/'+data_of+'_'+str(q)+'.png')
        plt.close()


# In[10]:


def plotsGrid_for_scaling_range(mf_data, data_of, nrows, ncols, save=True, plots_dir="."):
    """
    Plots the multifractal data with log(l) to determine the linear scaling range, manually.
    The plot is generated on a grid for all moments and saved in the plots_dir.
    
    Parameters
    ----------
    mf_data : numpy array
        Input array, taken to be real. The first row contains log(l), the first 
        column contains momemts (q), and the rest of the rows are the mf quantity 
        corressponding to q of that row.
    data_of : string
        This specifies the multifractal quantity of interset, namely Dq, \alpah or f(\alpha).
        This name is only used to name the file for saving the plot.
    plots_dir : string
        The name of the directory for saving the plots. If directory does not exist, it is created.
        Default is current directry ('.').
    
    Return
    ------
    out : It only saves a '.png' file.
    
    """
    moments = mf_data[1:,0]
    denominator = mf_data[0,1:]
    
    width = 7 * nrows
    height = (width/1.4) * nrows / ncols
    fig, ax = plt.subplots(nrows, ncols, True, True, figsize=(width, height))
    i = -1 
    for iq, q in enumerate(moments):
        numerator = mf_data[iq+1, 1:]
        z,cov = np.polyfit(denominator, numerator, 1, cov='unscaled')

        # indices (i,j) of the subplot of the grid
        # ----------
        j = iq%(nrows)
        if(j==0):
            i=i+1
        # ----------
#         print(i,j)
        ax[i,j].plot(denominator, numerator,'o', label=q)

        y = z[0] * denominator + z[1]
        ax[i,j].plot(denominator, y)

        ax[i,j].legend()
        ax[i,j].set_xlabel(r'$\log(l)$')
        ax[i,j].set_ylabel(data_of)
    if save:
        dir_nod = os.path.exists(plots_dir)
        if(dir_nod == False):
            os.makedirs(plots_dir)

        plt.savefig(plots_dir+'/'+data_of+'.png')


# In[11]:


def slopes(data, range_min, range_max, save = True, data_of = "test"):
    """
    """
    moments = data[1:,0]
    denominator = data[0,1:]
    m = np.zeros((len(moments), 2))
    m[:,0]=moments
    for iq, q in enumerate(moments):
        numerator = data[iq+1, 1:]
        z,cov = np.polyfit(denominator[range_min:range_max], numerator[range_min:range_max], 1, cov='unscaled')
        m[iq, 1] = z[0]
    if save:
        np.savetxt(data_of+'.d', m)
    return m


# In[12]:


def intermittency_exponent(tauq, q):
	i = 0		# index for q = 0, as determined in next block of code.
	for j, q_value in enumerate(q):
		if q_value == 0:
			i = j
	d2tauq = tauq[i+1] - 2*tauq[i] + tauq[i-1]
	dq2 = (q[i] - q[i-1])**2
	mu = (-1)*d2tauq / dq2
	return mu
	
# In[13]:


if __name__=='__main__':
    print ("This is a module.")

