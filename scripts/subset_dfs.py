'''
subsets dataframes into smaller sets for testing net
'''

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

path_to_project_data = '/home/ubuntu/DSI-Capstone-Project/data/'
galaxy_types = ['edge_on_disk', 'elliptical', 'face_on_spiral', 'merger']

def subset_df(infile, outfile, max_rows = 6250):
    '''
    input: infile (str), outfile(str), optional: max_rows (int)
    output: None (writes new file to outfile)
    '''
    df = pd.read_csv(infile)
    df.RA = df.RA.apply(lambda x: round(x, 3))
    df.DEC = df.DEC.apply(lambda x: round(x, 3))
    split_frac = float(max_rows) / len(df)
    if split_frac <= 1:
        split_0, split_1 = train_test_split(df, test_size=split_frac)
        split_1.to_csv(outfile)
    else:
        df.to_csv(outfile)


if __name__ == "__main__":
    for typ in galaxy_types:
        subset_df(path_to_project_data+'{}.csv'.format(typ), path_to_project_data+'{}_test.csv'.format(typ))
