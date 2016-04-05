import pandas as pd
import urllib
import os
import boto
import time

# loading AWS keys
access_key = os.environ['AWS_ACCESS_KEY_ID']
secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
path_to_project_data = '/Users/tarynheilman/science/DSI/DSI-Capstone-Project/data/'
galaxy_types = ['spiral', 'elliptical', 'merger']

def img_download(ra, dec):
    '''
    downloading galaxy images and storing them locally, for now
    '''
    url = 'http://skyservice.pha.jhu.edu/DR8/ImgCutout/getjpeg.aspx?ra={}&dec={}&scale=0.40&width=120&height=120&opt='.format(ra, dec)
    raf = round(ra, 3)
    decf = round(dec, 3)
    #outfile = path_to_project_data+'{}_images/{}_{}.jpg'.format(typ, raf, decf)
    outfile = '{}_{}.jpg'.format(raf, decf)
    image = urllib.URLopener()
    image.retrieve(url, outfile)


def write_to_aws(new_filename, local_file, bucket_name='rynsbucket'):
    conn = boto.connect_s3(access_key, secret_access_key)

    # Check if bucket exist. If exist get bucket, else create one
    if conn.lookup(bucket_name) is None:
        b = conn.create_bucket(bucket_name, policy='public-read')
    else:
        b = conn.get_bucket(bucket_name)

    # creates a key for the file in AWS
    a = b.new_key(new_filename)

    # puts the contents of your local file into the AWS key
    a.set_contents_from_filename(local_file)


def download_images():
    for typ in galaxy_types:
        os.chdir(path_to_project_data+'{}_images/'.format(typ))
        coords = pd.read_csv(path_to_project_data+'{}_coords.csv'.format(typ))
        for ra, dec in coords.itertuples(index=False):
            img_download(ra, dec)
            time.sleep(.01)
    #    write_to_aws('SDSS_images/{}_{}.jpg'.format(ra, dec), path_to_project_data+'test_images/{}_{}.jpg'.format(ra, dec))


if __name__ == '__main__':
    download_images()
