"""
File handling helper functions
"""
import numpy as np
from scipy.io import loadmat

def read_2d_matrix_from_csv(filename, dtype='double'):
    """
    Read 2d matrix from csv file
    """
    matrix = np.genfromtxt(open(filename, "rb"), delimiter=",", dtype=dtype)
    return matrix

def read_matrix_from_mat(filename):
    """
    Read matrix from .mat file as numpy ndarray
    """
    matrix = loadmat(filename)['matrix']
    return matrix
