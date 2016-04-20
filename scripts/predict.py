import numpy as np
from keras.models import Sequential, model_from_json

path_to_project_data = ''









if __name__ == "__main__":
    # loads model
    with open(path_to_project_data+'CNN_model_architecture.json', 'rb') as f:
	json_string = f.read()
    model = model_from_json(json_string)
    # loads weights
    model.load_weights(path_to_project_data+'model_weights.h5')
