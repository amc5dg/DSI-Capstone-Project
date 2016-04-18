'''
script to plot figures used in presentation
'''

import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import pandas as pd
from image_transformations import *
# import seaborn

path_to_project_data = '/Users/tarynheilman/science/DSI/DSI-Capstone-Project/data/'
path_to_project_presentation = '/Users/tarynheilman/science/DSI/DSI-Capstone-Project/presentation/'
names = ['Edge On Disk', 'Elliptical', 'Face On Spiral', 'Merger']


def load_images():
    '''
    input: None
    output: all_classes (image collection), mergers (image collection)
    '''
    # gets list of files for one example image for each class, and more merger examples
    new_names = [name.lower().replace(' ', '_') for name in names]
    all_class_files = [path_to_project_presentation+'{}0.jpg'.format(name) for name in new_names]
    merger_files = [path_to_project_presentation+'merger{}.jpg'.format(i+1) for i in xrange(6)]
    # loads image collections
    all_classes = io.imread_collection(all_class_files)
    mergers = io.imread_collection(merger_files)
    return all_classes, mergers


def make_square(image):
    '''
    input: image (np array)
    output: image (np array), resized
    '''
    x, y, z = image.shape
    side = min(x, y)
    return transform.resize(image, (side, side, z))


def galaxy_examples(class_images, plotname):
    '''
    input: class_images (image collection), plotname (str)
    output: None (saves plot to disk)
    '''
    plt.clf()
    fig = plt.figure(figsize=(7, 8))
    # width and spacing between axes
    dx, dy, sx, sy = 0.485, 0.45, 0.01, 0.05
    for i, image in enumerate(class_images):
        # plt.subplot(2, 2, i+1, axisbg='k')
        fig.add_axes([.01 + (i%2)*(dx + sx), .5 - (i/2)*(dy + sy), dx, dy])
        plt.imshow(make_square(image))
        plt.title(names[i], fontsize='x-large')
        plt.axis('off')
    plt.savefig(plotname)


def plot_mergers(merge_ims, plotname):
    '''
    input: merge_ims (list of 6 np array images), plotname (str)
    output: None (saves plot to disk)
    '''
    plt.clf()
    fig = plt.figure()
    # width and spacing between axes
    dx, dy, sx, sy = 0.32, 0.47, 0.01, 0.02
    for i, image in enumerate(merge_ims):
        fig.add_axes([.01 + (i%3)*(dx + sx), .02 + (i%2)*(dy + sy), dx, dy])
        # plt.subplot(2, 3, i+1)
        plt.imshow(make_square(image))
        plt.axis('off')
    #plt.tight_layout(pad = 0.05, h_pad = 0.05, w_pad=0.05)
    plt.savefig(plotname)


def autolabel(ax, rects):
    '''
    input: ax (matplotlib subplots object), rects (matplotlib bar plots)
    attaches text labels
    '''
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., .95*height, '%.2f'%(height),
                ha='center', va='bottom')


def plot_model_results(recalls, precisions, plotname):
    '''
    input: recalls (list), precisions (list)
    output: None (saves plot to disk)
    '''
    x = np.arange(4)
    labels = ['Edge on Disk\n test size: 306', 'Elliptical\n test size: 1450', 'Face on Spiral\n test size: 1092', 'Merger\n test size: 9']
    fig, ax = plt.subplots()
    rects1 = ax.bar(x, recalls, 0.25, color='#6495ED', alpha=0.5)
    rects2 = ax.bar(x+0.25, precisions, 0.25, color='#DB7093', alpha=0.5)
    ax.set_ylabel('Scores')
    #ax.set_title('Model Validation Scores')
    ax.set_xticks(x+0.25)
    ax.set_xticklabels(labels)
    ax.legend( (rects1[0], rects2[0]), ('Recall', 'Precision'), bbox_to_anchor=(0.95, 1.15))
    autolabel(ax, rects1)
    autolabel(ax, rects2)
    plt.savefig(plotname)


def preprocess(image):
    '''
    input: image (np array)
    downsamples image to 120 by 120, randomly rotates or mirrors, then re-centers and crops to 60x60
    output: image (np array)
    '''
    resize_im = transform.resize(image, (120, 120))
    # list of possible image transformations
    transforms = np.array([rotate_0, rotate_90, rotate_180, rotate_270, \
                transpose, reflect_diagonal, horiz_mirror, vert_mirror])
    # randomly modifies image
    mod_image = transforms[np.random.randint(0,8)](resize_im)
    # randomly recenters and crops
    return translate_and_crop(mod_image)


def visualize_convolution(image, model):
    '''
    input: image (np array), model (keras model),
    output: image (np array)
    '''
    pass


def plt_all_image(name):
    '''
    input: name(str)
    output: image (np array)
    '''
    name = name.lower().replace(' ', '_')
    # grabs dataframe belonging to name class
    df = pd.read_csv(path_to_project_data+'{}.csv'.format(name))
    # gets top entry with 100% probability of class membership
    data = df.query('{} == 1.0'.format(name))
    data.RA = data.RA.apply(lambda x: round(x, 3))
    data.DEC = data.DEC.apply(lambda x: round(x, 3))
    if name == 'edge_on_disk' or name == 'face_on_spiral':
        filelist = [path_to_project_data+'spiral_images/{}_{}.jpg'.format(ra, dec) for ra, dec in data[['RA', 'DEC']].itertuples(index=False)]
    else:
        filelist = [path_to_project_data+'{}_images/{}_{}.jpg'.format(name, ra, dec) for ra, dec in data[['RA', 'DEC']].itertuples(index=False)]
    images = io.imread_collection(filelist)
    rows = int(np.sqrt(len(images))) + 1
    plt.clf()
    for i, image in enumerate(images):
        plt.subplot(rows, rows, i+1, axisbg='k')
        plt.imshow(image)
        plt.axis('off')
    plt.savefig('perfect_{}_galaxies.jpg'.format(name))


def plot_pipeline(image, plotname):
    '''
    input: image (np array)
    plots image at all stages of processing
    output: None (saves plot to disk)
    '''
    # instantiates figure
    fig = plt.figure()
    # shows original image
    fig.add_axes([0.1, 0.55, 0.35, 0.35])
    plt.imshow(image)
    plt.axis('off')
    plt.title('Original Image')
    # shows downsampled, translated and cropped image
    fig.add_axes([0.55, 0.55, 0.35, 0.35])
    mod_image = preprocess(image)
    plt.imshow(mod_image)
    plt.axis('off')
    plt.title('Image after preprocessing')
    # shows convolved image
    fig.add_axes([0.55, 0.1, 0.35, 0.35])
    conv_image = mod_image
    plt.imshow(conv_image)
    plt.axis('off')
    plt.title('Image after convolutional layers')

    # maybe show predictions, maybe features?
    plt.savefig(plotname)

if __name__ == '__main__':
    class_images, mergers = load_images()
    # galaxy_examples(class_images, path_to_project_presentation+'galaxy_examples.png')
    # plot_mergers(mergers, path_to_project_presentation+'merger_examples.png')
    # plot_model_results([0.96, 0.96, 0.84, 0.89], [0.95, 1.00, 0.99, 0.03], path_to_project_presentation+'model_results.png')
    andromeda = io.imread(path_to_project_presentation+'Andromeda.jpg')
    plot_pipeline(andromeda, path_to_project_presentation+'testing_andromeda.png')
