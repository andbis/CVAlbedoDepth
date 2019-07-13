import numpy as np
import ps_utils as ps

from skimage.io import imshow
from skimage import img_as_float

from itertools import combinations
from numpy.linalg import det

def back_to_the_matrix(n, mask):
    n_list=list(n)
    matrix = np.zeros(mask.shape)
    for (i,j), val in np.ndenumerate(mask):
        if val:
            matrix[i,j] = n_list.pop(0)
    return matrix


def get_j(img, mask):
    mask_list = ps.tolist(mask)
    img_list = ps.tolist(img)
    j = []
    for m_val, i_val in zip(mask_list, img_list):
        if m_val:
            j.append(i_val)
    return j


def get_independent(vectors):
    #probably should be normalized before but they're not
    combs = combinations(vectors, 3)
    #0 89 
    best_vecs = []
    best_det = 0
    for vecs3 in combs:
        print(det(vecs3))
        if det(vecs3) > best_det:
            best_vecs = vecs3
            best_det = det(vecs3)
    print('best det', best_det)
    return best_vecs


def show(img):
    imshow(img, cmap='gray')
    return


def get_albedo(M, m, I):
    #Creating list of M arrays
    hodl = [M[0], M[1], M[2]]

    #Creating temporary matrix to hold x,y,z values
    MM = np.zeros(I.shape)
    indel = 0

    #Iterating thorugh temp matrix and adding values from hodl list
    for (i, j, c), _ in np.ndenumerate(MM):
        if m[i, j]:
            hodl[c][indel]
            MM[i, j, c] = hodl[c][indel]
            if c == 2:
                indel += 1

    #Creating Albedo matrix to length of vector values
    Albedo = np.zeros(m.shape)

    #Iterating through Albedo matrix to calculate and add length of vector
    for (i, j), _ in np.ndenumerate(Albedo):
        if m[i, j]:
            x = MM[i, j, 0]
            y = MM[i, j, 1]
            z = MM[i, j, 2]
            Albedo[i, j] = np.sqrt(x**2 + y**2 + z**2)

    
    
    # normalization
    
    #Extracting and normalizing the values in Albedo matrix
    Albedo_nlist = normalize(ps.tolist(Albedo), (0,1))

    #Creating matrix to hold normalized values
    Albedo_normalized = np.zeros(Albedo.shape)
    indel = 0

    #Iterating through normalized albedo matrix to add normalized values 
    for (i, j), _ in np.ndenumerate(Albedo_normalized):
        if m[i, j] and Albedo_nlist[indel] != 0:
            Albedo_normalized[i, j] = Albedo_nlist[indel]
        indel += 1
        
    return Albedo_normalized


def normalize(differences, range=(0,1.0)):
    #linear rescaling
    max_val = max(differences)
    min_val = min(differences)

    return np.multiply(np.subtract(range[1], range[0]), np.divide( np.subtract(differences, min_val), np.subtract(max_val, min_val)))


def display_3d(N):
    # the code commented below is taken from here: https://plot.ly/python/3d-surface-plots/
    # and it works only in jupyter notebook

    import pandas as pd
    import plotly.plotly as py
    import plotly.graph_objs as go
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    init_notebook_mode(connected=True)


    data = [go.Surface(z=N)]
    layout = go.Layout(title='3D plot', autosize=True)
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    