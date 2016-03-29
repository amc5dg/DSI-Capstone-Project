import numpy as np
import scipy.stats as scs


def rotate_90(ary):
    '''
    input: np array
    rotates array by 90 degrees
    output: np array
    '''
    return ary.swapaxes(0,1)[:,::-1]


def rotate_180(ary):
    '''
    input: np array
    rotates array by 180 degrees
    output: np array
    '''
    return ary[::-1,::-1]


def rotate_270(ary):
    '''
    input: np array
    rotates array by 270 degrees
    output: np array
    '''
    return ary.swapaxes(0,1)[::-1,:]


def transpose(ary):
    '''
    input: np array
    flips array over diagonal
    output: np array
    '''
    return ary.T


def reflect_diagonal(ary):
    '''
    input: np array
    flips array over top-right/bottom-left diagonal
    output: np array
    '''
    x.T[::-1, ::-1]


def horiz_mirror(ary):
    '''
    input: np array
    flips array horizontally
    output: np array
    '''
    return ary[:,::-1]


def vert_mirror(ary):
    '''
    input: np array
    flips array vertically
    output: np array
    '''
    return ary[::-1,:]


def translate_and_crop(ary, f = 2):
    '''
    input: np array, f (fraction of cropping desired)
    flips array vertically
    output: np array
    '''
    # center coordinates (and half-widths) of array
    xc = ary.shape[0]/2
    yc = ary.shape[1]/2
    # randomly choose x and y center of cropped image from a normal distribution around the actual image center
    x_center = int(round(scs.norm(xc, np.sqrt(xc)).rvs(), 0))
    y_center = int(round(scs.norm(yc, np.sqrt(yc)).rvs(), 0))
    # returns array with new center and cropped by factor f
    return ary[x_center - xc/f:x_center + xc/f, y_center - yc/f:y_center + yc/f]
