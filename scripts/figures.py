'''
script to plot figures used in presentation
'''

import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import pandas as pd
from image_transformations import *
# import seaborn
from scipy.ndimage import filters

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


def vis_preprocess(image, outname1, outname2):
    '''
    input: image (np array)
    downsamples image to 120 by 120, randomly rotates or mirrors, then
    re-centers and crops to 60x60
    output: None (saves images to disk)
    '''
    # downsamples image
    resize_im = transform.resize(image, (120, 120))
    # saves resized to disk
    io.imsave(outname1, resize_im)
    # list of possible image transformations
    transforms = np.array([rotate_0, rotate_90, rotate_180, rotate_270, \
                transpose, reflect_diagonal, horiz_mirror, vert_mirror])
    # randomly modifies image
    mod_image = transforms[np.random.randint(0,8)](resize_im)
    # randomly recenters and crops
    crop_im = translate_and_crop(mod_image)
    # saves cropped image
    io.imsave(outname2, crop_im)


if __name__ == '__main__':
    # class_images, mergers = load_images()
    # galaxy_examples(class_images, path_to_project_presentation+'galaxy_examples.png')
    # plot_mergers(mergers, path_to_project_presentation+'merger_examples.png')
    # plot_model_results([0.96, 0.96, 0.84, 0.89], [0.95, 1.00, 0.99, 0.03], path_to_project_presentation+'model_results.png')
    andromeda = io.imread(path_to_project_presentation+'Andromeda.jpg')
    vis_preprocess(andromeda, path_to_project_presentation+'downsampled_andromeda.jpg', path_to_project_presentation+'cropped_andromeda.jpg')
