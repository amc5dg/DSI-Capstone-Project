'''
sorts through spirals dataframe and plots images of spirals with different
confidence levels so I know where to make the cut for neural net
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

path_to_project_data = '/Users/tarynheilman/science/DSI/DSI-Capstone-Project/data/'
spiral_types = ['face_on_spiral', 'edge_on_disk']

def load_dfs():
    '''
    input: None
    loads csvs as DataFrames and sorts by confidence level of class
    output: face_on_spiral (pd DataFrame), edge_on_disk (pd DataFrame)
    '''
    # reads in
    face_on = pd.read_csv(path_to_project_data+'face_on_spiral.csv')
    edge_on = pd.read_csv(path_to_project_data+'edge_on_disk.csv')
    # sorts probability values
    face_on.sort_values('face_on_spiral', inplace=True)
    edge_on.sort_values('edge_on_disk', inplace=True)
    # resets index
    face_on.reset_index(drop=True, inplace=True)
    edge_on.reset_index(drop=True, inplace=True)
    return face_on, edge_on


def load_image(ra, dec, folder='spiral'):
    '''
    input: ra(float), dec (float), folder (str)
    output: image (np array)
    '''
    filename = path_to_project_data+'{}_images/{}_{}.jpg'.format(folder, ra, dec)
    return io.imread(filename)


def pick_images(df, typ):
    '''
    input: df (pd DataFrame), typ (str - 'face_on_spiral' or 'edge_on_disk')
    output: df (pd DataFrame)
    '''
    # creates list of 20 evenly spaced indicies in DataFrame
    inds = ((len(df) - 1) * np.arange(.05,1.01,.05)).astype(int)
    # gets RA, DEC, and probability from these
    data = df.ix[inds][['RA', 'DEC', typ]]
    # rounds RA and DEC to match image files
    data.RA = data.RA.apply(lambda x: round(x, 3))
    data.DEC = data.DEC.apply(lambda x: round(x, 3))
    # resets index
    data.reset_index(drop=True, inplace=True)
    return data

def plot_images(df, typ, plotname=None):
    '''
    input: df (pd DataFrame), typ (str - 'face_on_spiral' or 'edge_on_disk')
    output: None (shows matplotlib plot or writes to disk)
    '''
    images_df = pick_images(df, typ)
    for i, ra, dec, prob in images_df.itertuples():
        image = load_image(ra, dec)
        plt.subplot(4, 5, i+1)
        plt.imshow(image)
        plt.title(round(prob, 2))
        plt.axis('off')

    if plotname:
        plt.savefig(plotname)
    else:
        plt.show()



if __name__ == '__main__':
    face_on, edge_on = load_dfs()
    plot_images(face_on, 'face_on_spiral', plotname=path_to_project_data+'face_on_probs.png')
    plot_images(edge_on, 'edge_on_disk', plotname=path_to_project_data+'edge_on_probs.png')
