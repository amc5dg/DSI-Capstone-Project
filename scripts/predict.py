import numpy as np
from keras.models import Sequential, model_from_json
from keras import backend as K
from skimage import io, transform
from image_transformations import translate_and_crop

path_to_project_data = '/home/ubuntu/DSI-Capstone-Project/data/'

classes = ['Edge-on Disk', 'Elliptical', 'Face-on Spiral', 'Merger']

def pre_process(image):
    '''
    input: image (np array)
    resizes and crops image to proper size for net prediction
    output: np array
    '''
    # rescaling input to (120, 120)
    res_im = transform.resize(image, (120, 120)     
    # translating and cropping to (60, 60)
    return translate_and_crop(res_im)


def load_image(files, collection=False):
    '''
    input: files (str), collection (Bool, optional, default=False)
    if collection = True, files is a list of filenames to be read in as a collection)
    output: image (np array with dimensions (60,60,3) if collection=False, or dimensions (n_files, 60, 60, 3)
    '''
    if collection:
	images = io.imread_collection(files)
	return np.array([pre_process(image) for image in images])
    else:
	image = io.imread(files)
	return pre_process(image)


def load_model(model_file, weights_file):
    '''
    input: model_file (str), weights_file (str) {paths to model and weight files}
    output: model (keras model)
    '''
    # loads model
    with open(path_to_project_data + model_file, 'rb') as f:
        json_string = f.read()
    model = model_from_json(json_string)
    # loads weights
    model.load_weights(path_to_project_data + weights_file)
    return model


def predict_one(model, image, predict_probs=True):
    '''
    input: model (keras sequential), image (np array), predict_probs (Bool, default=True)
    output: predictions (np array) if predict_probs=True; predicted class (str) if predict_probs = False
    '''
    # reshapes image adding first dimension to satisfy model
    ary = image.reshape(1, 60, 60, 3)
    if predict_probs:
        return model.predict_proba(image)
    else:
	return classes[model.predict_classes(image)]


def predict_many(model, images, predict_probs=True):
    '''
    input: model (keras sequential), images (np array with dimensions (n_images, 60, 60, 3)), predict_probs (Bool, default=True)
    output: predictions (np array with dimensons (n_images, 4) if predict_probs=True; predicted class (np array of str) if predict_probs = False
    '''
    if predict_probs:
	return model.predict_proba(images)
    else:
	return np.array(classes)[model.predict_classes(images)]


if __name__ == "__main__":
    model = load_model('CNN_model_architecture.json', 'model_weights.h5')

    # to predict on one image
    # image = load_image(your_filename_here)
    # predict_one(model, image, predict_probs=False) # to predict class name
    # predict_one(model, image) # to predict probability of membership in each class 

    # to predict on many images
    # images = load_image(your_list_of_files_here, collection=True)
    # predict_many(model, images)
