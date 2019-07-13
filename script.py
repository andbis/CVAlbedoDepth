import ps_utils as ps
from library import *

import numpy as np
from numpy.linalg import pinv
from numpy.linalg import inv

from skimage import img_as_float
from skimage.io import imshow, imsave
from skimage.filters import gaussian


### loading the data ###

I, m, S = ps.read_data_file('Beethoven.mat')
I2, m2, S2 = ps.read_data_file('Buddha.mat')


### Beethoven dataset ###

S_inv = inv(S)

J = np.array([get_j(I[:,:,k], m) for k in range(3)])

M = np.dot(S_inv, J)


n_0 = np.divide(M[0], np.linalg.norm(M[0]))
n_1 = np.divide(M[1], np.linalg.norm(M[1]))
n_2 = np.divide(M[2], np.linalg.norm(M[2]))
          
n_0_matrix = back_to_the_matrix(n_0, m)
n_1_matrix = back_to_the_matrix(n_1, m)
n_2_matrix = back_to_the_matrix(n_2, m)

N = ps.unbiased_integrate(n_0_matrix, n_1_matrix, n_2_matrix, m)

A = get_albedo(M, m, I)
imsave('beethoven_albedo.png', A)


"""
As the didn't manage to install mayavi package and matplotlib 3d plotting is very slow we used other way of 3d visualization.
Works only in jupyter notebook, from where we got the images for the report.

3d_display(N)
"""



#### Buddha dataset with Moore-Penrose pseudo-inversion ###


S2_pinv = pinv(S2)


I2s = [I2[:,:,k] for k in range(10)] 

J2 = np.array([get_j(I2s[k], m2) for k in range(10)])

M2 = np.dot(S2_pinv, J2)


n2_0 = np.divide(M2[0], np.linalg.norm(M2[0]))
n2_1 = np.divide(M2[1], np.linalg.norm(M2[1]))
n2_2 = np.divide(M2[2], np.linalg.norm(M2[2]))
    
n2_0_matrix = back_to_the_matrix(M2[0], m2)
n2_1_matrix = back_to_the_matrix(M2[1], m2)
n2_2_matrix = back_to_the_matrix(M2[2], m2)


N2 = ps.unbiased_integrate(n2_0_matrix, n2_1_matrix, n2_2_matrix, m2)

A2 = get_albedo(M2, m2, I2[:,:,:3])
imsave('buddha_albedo.png', A2)

    
"""
As the didn't manage to install mayavi package and matplotlib 3d plotting is very slow we used other way of 3d visualization.
Works only in jupyter notebook, from where we got the images for the report.

3d_display(N2)
"""



### This part doesn't work ###
# Buddha dataset with normal inversion for best 3 candidates

"""

#looking for 3 best vectors
S2_square = get_independent(S2)
S2_inv = inv(S2_square)

#getting the corresponding images
I2_0 = I2[:,:,np.where(S2==S2_square[0])[0][0]]
I2_1 = I2[:,:,np.where(S2==S2_square[1])[0][0]]
I2_2 = I2[:,:,np.where(S2==S2_square[2])[0][0]]


S2_inv = inv(S2[:3])

I2_0 = I2[:,:,0]
I2_1 = I2[:,:,1]
I2_2 = I2[:,:,2]


j2_0 = get_j(I2_0, m2)
j2_1 = get_j(I2_1, m2)
j2_2 = get_j(I2_2, m2)

J2 = np.array([j2_0, j2_1, j2_2])


M2 = np.dot(S2_inv, J2)


n2_0 = np.divide(M2[0], np.linalg.norm(M2[0]))
n2_1 = np.divide(M2[1], np.linalg.norm(M2[1]))
n2_2 = np.divide(M2[2], np.linalg.norm(M2[2]))

N2_inv = unbiased_integrate(n2_0, n2_1, n2_2, m2)
"""
