import numpy as np
import pandas as pd
import keras.backend as K
from loadmat_stackoverflow import loadmat
from keras_ssd512 import ssd_512

def expand_to_4d(array):
    return np.reshape(array, [1, 1, array.shape[0], array.shape[1]])

img_height = 512
img_width = 512

# Set input model name (without .mat) here.
model_name="head-detector"

model = ssd_512(image_size=(img_height, img_width, 3), n_classes=1, min_scale=0.07, max_scale=1, mode='inference')

model_ori = loadmat(model_name + ".mat")

layers = pd.read_csv('layers.csv', sep=';')

for index, layer in layers.iterrows():

    print("Assigning layer " + layer['name'] + "...")

    if layer['name'] in ["fc7", "conv6_1", "conv7_1", "conv8_1", "conv9_1", "conv10_1"]:
        K.set_value(model.get_layer(layer['name']).weights[0], expand_to_4d(model_ori['vars'][layer['weights'] - 1]))
    else:
        K.set_value(model.get_layer(layer['name']).weights[0], model_ori['vars'][layer['weights'] - 1])

    print("Weights...")

    if layer['biases'] != -1:
        print("Biases...")
        K.set_value(model.get_layer(layer['name']).weights[1], model_ori['vars'][layer['biases'] - 1])

    print("Done.")

model.save_weights(model_name + ".h5")