import numpy as np
import pandas as pd

path_to_project_data = '/Users/tarynheilman/science/DSI/DSI-Capstone-Project/data/'
galaxy_types = ['face_on_spiral', 'edge_on_disk', 'elliptical', 'merger', 'dont_know']
#galaxy_types = ['face_on_spiral', 'edge_on_disk', 'elliptical', 'merger']

def clean_galaxies():
    '''
    reads in csv file of metadata and returns cleaned csv file
    output:
    '''
    # reads in csv files as pandas dataframes
    galaxies = pd.read_csv(path_to_project_data+'galaxy_zoo_no_spectra.csv')
    galaxies2 = pd.read_csv(path_to_project_data+'galaxy_zoo_with_spectra.csv')
    galaxies = pd.concat([galaxies, galaxies2], axis=0, ignore_index=True)
    # change column names for readability
    galaxies.columns = ['RA', 'DEC', 'nvotes', 'elliptical', 'clockwise', \
        'anticlockwise', 'edge_on_disk', 'dont_know', 'merger', 'combined_spiral']
    # creates face on spiral column with clockwise spiral classes
    galaxies['face_on_spiral'] = galaxies[['clockwise', 'anticlockwise']].max(axis=1)
    # dropping directional spiral columns and combined for now
    galaxies.drop(['clockwise', 'anticlockwise', 'combined_spiral'], axis=1, inplace=True)
    # gets rid of random nan row with nans
    galaxies.dropna(axis=0, inplace=True)
    # adds column listing the max class
    max_class = max_col(galaxies, galaxy_types)
    galaxies = pd.concat([galaxies, max_class], axis=1)
    # normalizes probabilities of remaining columns
    galaxies = renorm_probs(galaxies)
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

    return pd.Series(max_class, name='type')


def renorm_probs(df):
    '''
    input: df (pd DataFrame)
    output: df (pd DataFrame)
    '''
    df.norm = df[galaxy_types].sum(axis=1)
    for typ in galaxy_types:
        df[typ] = df[typ] / df.norm
    return df


def separate_galaxies(galaxies, clss, thresh=0.6):
    '''
    input: galaxies (pd DataFrame), type (str)
    output: subsetted DataFrame
    subsets galaxies by class and takes only those with greater than 50% confidence
    '''
    df = galaxies[(galaxies.type == clss) & (galaxies[clss] >= thresh)]
    df.to_csv(path_to_project_data+'{}.csv'.format(clss), index=False)
    return df


def get_coords_file(galaxies, outfile='final_galaxy_coords.csv'):
    '''
    input: pd DataFrame
    output: None
    gets csv file with ra, dec coords to use for SDSS bulk search
    '''
    coords = galaxies[['RA', 'DEC']]
    coords.to_csv(path_to_project_data+outfile, index=False)


if __name__ == '__main__':
    galaxies = clean_galaxies()

    for typ in galaxy_types:
        df = separate_galaxies(galaxies, typ, thresh=0.666)
        get_coords_file(df, outfile='{}_coords.csv'.format(typ))
