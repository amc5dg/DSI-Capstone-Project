import numpy as np
import pandas as pd

path_to_project_data = '~/science/DSI/DSI-Capstone-Project/data/'

def clean_galaxies():
    '''
    reads in csv file of metadata and returns cleaned csv file
    output:
    '''
    # reads in csv file as pandas dataframe
    galaxies = pd.read_csv(path_to_project_data+'galaxy_zoo_no_spectra.csv')
    # change column names for readability
    galaxies.columns = ['RA', 'DEC', 'nvotes', 'elliptical', 'clockwise', \
        'anticlockwise', 'edge_on_disk', 'dont_know', 'merger', 'combined_spiral']
    # creates spiral column with all spiral classes
    galaxies['spiral'] = galaxies.clockwise + galaxies.anticlockwise + \
        galaxies.edge_on_disk + galaxies.combined_spiral
    # dropping other spiral columns
    galaxies.drop(['clockwise', 'anticlockwise', 'edge_on_disk', \
        'combined_spiral'], axis=1, inplace=True)
    # adds column listing the max class
    max_class = max_col(galaxies, ['spiral', 'elliptical', 'merger', 'dont_know'])
    galaxies = pd.concat([galaxies, max_class], axis=1)
    return galaxies


def max_col(df, col_names):
    '''
    input: df (pd DataFrame), col_names (list)
    output: pd.Series
    determines the max class from the given columns
    '''
    # subsets DataFrame to only the columns we care about
    types = df[col_names]
    # creates an array of the maximum class
    max_class = np.array(col_names)[types.values.argmax(axis=1)]

    return pd.Series(max_class, name='class')


def get_coords_file(galaxies, outfile='final_galaxy_coords.csv'):
    '''
    input: pd DataFrame
    output: None
    gets csv file with ra, dec coords to use for SDSS bulk search
    '''
    coords = galaxies[['RA', 'DEC']]
    coords.to_csv(path_to_project_data+outfile, header=False)


if __name__ == '__main__':
    galaxies = clean_galaxies()
    #get_coords_file(galaxies)
