import numpy as np
import scipy.stats as scs
import pandas as pd
from image_transformations import *
from skimage import io, transform, color
from sklearn.cross_validation import train_test_split

'''
Separates data into training and test sets. Loads each subsetted metadata dataframe
and splits individually, compiles images from coordinates, augments training set
to combat class imbalance, resizes and crops images, randomly recentering training set.
'''

path_to_project_data = '/home/ubuntu/DSI-Capstone-Project/data/'
galaxy_types = ['edge_on_disk', 'elliptical', 'face_on_spiral', 'merger']

def load_images(df, typ):
    '''
    input: df (pd DataFrame), typ(str) (galaxy type)
    output: list of n_images, each with dimensions (n_rows, n_cols, n_channels), filelist
    '''
    if typ == 'edge_on_disk' or typ == 'face_on_spiral':
	filelist = [path_to_project_data+'spiral_images/{}_{}.jpg'.format(ra, dec) for ra, dec in df[['RA', 'DEC']].itertuples(index=False)]
    else:
	filelist = [path_to_project_data+'{}_images/{}_{}.jpg'.format(typ, ra, dec) for ra, dec in df[['RA', 'DEC']].itertuples(index=False)]
    return io.imread_collection(filelist), filelist


def resize_image(image, output_shape):
    '''
    input: image (np array), output_shape (tuple)
    output: np array with shape = output_shape
    '''
    return transform.resize(image, output_shape)


def hsv_image(image):
    '''
    input: image (np array)
    output: np array
    '''
    return color.rgb2hsv(image)


def get_train_test_splits(df):
    '''
    input: df (pd DataFrame)
    output: X_train (list of image arrays), X_test (list of image arrays),
    y_train (1D np array of targets), y_test (1D np array of targets)
    (FROM ONE SUBSETTED DATA FRAME)
    '''
    # create target column
    y = df.pop('type')
    # gets typ variable
    typ = y.unique()[0]
    # gets train, test indices for subsetted df
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2)
    # converts gets images for RA and DEC
    train_ims, train_files = load_images(X_train, typ) 
    test_ims, test_files = load_images(X_test, typ)
    return train_ims, test_ims, y_train.values, y_test.values, train_files, test_files


def modify_images(image, n):
    '''
    input: image (np array), n (int)
    output: list of modified images with length n
    '''
    # list of 8 function to transform images
    transforms = np.array([rotate_0, rotate_90, rotate_180, rotate_270, \
                transpose, reflect_diagonal, horiz_mirror, vert_mirror])
    # list of n transformations on images
    mod_images = [transforms[np.random.randint(0,8)](image) for i in xrange(n)]
    # crops images and translates randomly
    return [translate_and_crop(im, xform=True) for im in mod_images]


def augment_data(im_list, filelist, targets, test=False, hsv=False):
    '''
    input: im_list (list of np arrays of images), filelist (list of images in order of imlist,
    targets (1D np array), test (Bool, optional, default = False), 
    hsv (Bool, optional, default = False)
    rotate, translate, skew, crop images to put into neural net
    output: cropped (modified image list), files (list of files), targets
    '''
    # resize image to (90, 90)
    # resized = [resize_image(im, (90, 90)) for im in im_list]
    if test:
        # crop image in center to (45, 45)
        # cropped = [translate_and_crop(im) for im in resized]
        cropped = [translate_and_crop(im) for im in im_list]
        files = filelist
    else:
        # number of copies of images to make to get balanced classes in training set
        # n_copies = int(round(333000./len(im_list), 0))
        n_copies = min(25, int(round(10000./len(im_list), 0)))
        # extends list of targets
        targets = np.array(targets.flatten().tolist()*n_copies)
        # randomly transforms and crops each image, making n_copies altered images, 
        # and keeps track of image files
        cropped, files = [], []
        for i, im in enumerate(im_list):
            cropped.extend(modify_images(im, n_copies))
            files.extend([filelist[i]]*n_copies)
    if hsv:
         # tranforms images to hsv space and stacks onto image channels if selected	   
         cropped = [np.concatenate([im, hsv_image(im)], axis=2) for im in cropped]
    
    return cropped, files, targets


def get_data(hsv=False):
    '''
    input: hsv (Bool, optional, default=False)
    reads in dataframes, splits into train and test sets, processes images, and
    combines train and test sets.
    output: X_train (list of image arrays), X_test (list of image arrays),
    y_train (1D np array of targets), y_test (1D np array of targets)
    (COMBINED), train_files (list of files of training images), test_files (list of files for testing images)
    '''
    train_images, test_images, train_targets, test_targets, train_files, test_files = [], [], [], [], [], []
    for typ in galaxy_types:
        # read in dataframe
        # df = pd.read_csv(path_to_project_data+'{}_galaxies.csv'.format(typ))
        df = pd.read_csv(path_to_project_data+'{}_test.csv'.format(typ))
        # get un-augmented image lists and target arrays
        X_train, X_test, y_train, y_test, train_list, test_list = get_train_test_splits(df)
        # get augmented training and test images and targets
        train_im, train_file, train_targ = augment_data(X_train, train_list, y_train, test=False, hsv=hsv)
        test_im, test_file, test_targ = augment_data(X_test, test_list, y_test, test=True, hsv=hsv)
        # extend all lists
        train_images.extend(train_im)
        test_images.extend(test_im)
        train_targets.extend(train_targ)
        test_targets.extend(test_targ)
	train_files.extend(train_file)
	test_files.extend(test_file)
    # array of indices to shuffle training data
    inds = np.random.choice(np.arange(len(train_images)), size=len(train_images), replace=False)
    return np.array(train_images)[inds], np.array(test_images), np.array(train_targets)[inds], np.array(test_targets), np.array(train_files)[inds], np.array(test_files)


def save_outputs(X_train, X_test, y_train, y_test, train_files, test_files, extension):
    '''
    input: X_train, X_test, y_train, y_test, train_files, test_files (np arrays), extension (str)
    output: None (saves arrays to file)
    '''
    names = ['X_train', 'X_test', 'y_train', 'y_test']
    for i, ary in enumerate([X_train, X_test, y_train, y_test]):
        np.save('{}{}_{}.npy'.format(path_to_project_data, names[i], extension), ary)
    np.save(path_to_project_data+'training_image_files_{}.npy'.format(extension), train_files)
    np.save(path_to_project_data+'test_image_files_{}.npy'.format(extension), test_files)


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, train_files, test_files = get_data()
    # saving files so that they don't need to be re-written each time
    save_outputs(X_train, X_test, y_train, y_test, train_files, test_files, 'small')
