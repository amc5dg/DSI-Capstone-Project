'''
sorts through spirals dataframe and plots images of spirals with different
confidence levels so I know where to make the cut for neural net
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

path_to_project_data = '/Users/tarynheilman/science/DSI/DSI-Capstone-Project/data/'



if __name__ == '__main__':
    spirals_data = pd.read_csv(path_to_project_data+'spiral_galaxies.csv')
