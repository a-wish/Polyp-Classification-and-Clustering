from __future__ import division
import os
import sys
from neupy import algorithms, environment
from imutils import paths
from utils import iter_neighbours
import cv2
from keras.preprocessing.image import img_to_array

CURRENT_DIR = os.path.abspath(os.path.dirname(__name__))
CNN_EXAMPLE_FILES = os.path.join(CURRENT_DIR,'pick_result')
WEIGHTS_FILE = os.path.join(CNN_EXAMPLE_FILES,'network.pickle')
IMAGE_DIR = os.path.join(CURRENT_DIR, '2class')  #change to 2class for 2 clusters, 4class for 4


sys.path.append(CNN_EXAMPLE_FILES)


os.listdir(IMAGE_DIR)


Benign_images = os.listdir(os.path.join(IMAGE_DIR, "Benign"))
Benign_images[:10]


import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


random.seed(0)
images = []
index = 1


fig = plt.figure(figsize=(12, 9))

class_parameters = [
    dict(
        marker='o',
        markeredgecolor='#E24A33',
        markersize=11,
        markeredgewidth=2,
        markerfacecolor='None',
    ),
    dict(
        marker='x',
        markeredgecolor='#348ABD',
        markersize=14,
        markeredgewidth=2,
        markerfacecolor='None',
    ),
    dict(
        marker='p',
        markeredgecolor='#33e256',
        markersize=14,
        markeredgewidth=2,
        markerfacecolor='None',
    ),
        dict(
        marker='s',
        markeredgecolor='#dfe233',
        markersize=14,
        markeredgewidth=2,
        markerfacecolor='None',
    ),

]

def compute_heatmap(weight):
    heatmap = np.zeros((GRID_HEIGHT, GRID_WIDTH))
    for (neuron_x, neuron_y), neighbours in iter_neighbours(weight):
        total_distance = 0

        for (neigbour_x, neigbour_y) in neighbours:
            neuron_vec = weight[:, neuron_x, neuron_y]
            neigbour_vec = weight[:, neigbour_x, neigbour_y]

            distance = np.linalg.norm(neuron_vec - neigbour_vec)
            total_distance += distance

        avg_distance = total_distance / len(neighbours)
        heatmap[neuron_x, neuron_y] = avg_distance

    return heatmap

def compute_heatmap_expanded(weight):
    heatmap = np.zeros((2 * GRID_HEIGHT - 1, 2 * GRID_WIDTH - 1))
    for (neuron_x, neuron_y), neighbours in iter_neighbours(weight):
        for (neigbour_x, neigbour_y) in neighbours:
            neuron_vec = weight[:, neuron_x, neuron_y]
            neigbour_vec = weight[:, neigbour_x, neigbour_y]

            distance = np.linalg.norm(neuron_vec - neigbour_vec)

            if neuron_x == neigbour_x and (neigbour_y - neuron_y) == 1:
                heatmap[2 * neuron_x, 2 * neuron_y + 1] = distance

            elif (neigbour_x - neuron_x) == 1 and neigbour_y == neuron_y:
                heatmap[2 * neuron_x + 1, 2 * neuron_y] = distance

    return heatmap


for name in os.listdir(IMAGE_DIR):
    path = os.path.join(IMAGE_DIR, name)
    
    if os.path.isdir(path):
        image_name = random.choice(os.listdir(path))
        image_path = os.path.join(path, image_name)
        
        image = mpimg.imread(image_path)
        
        plt.subplot(3, 3, index)
        plt.title(name.capitalize().replace('_', ' '))
        #plt.imshow(image)
        plt.axis('off')
        
        index += 1
        
fig.tight_layout()




from tools import download_file, load_image, deprocess

import theano
theano.config.floatX = 'float32'


from network import network
net =network()


import os
from neupy import storage
storage.load(net, WEIGHTS_FILE)


import numpy as np
import matplotlib.pyplot as plt


images = []
image_paths = []
target=[]

for path, directories, image_names in os.walk(IMAGE_DIR):
    for image_name in image_names:
        image_path = os.path.join(path, image_name)
        image = load_image(
            image_path,
            image_size=(224, 224),
            crop_size=(224, 224))
        
        images.append(image)
        image_paths.append(image_path)

        label = image_path.split(os.path.sep)[-2]
        if label == "Benign":
            label=3;
        if label == "malignant":
            label=2;
        if label == "benignwhite":
            label=1;
        if label == "malignantwhite":
            label=0;
        
	    
	    	 
        target.append(label)
target=np.array(target)
#print(target)        
images = np.concatenate(images, axis=0)
image_paths = np.array(image_paths)
images.shape


# Note: It's important to use dense layer, because SOFM expect to see vectors
dense_2 = net.end('dense_2')

# Compile Theano function that we can use to
# propagate image through the network

dense_2_propagete = dense_2.compile()
#dense_2_propagete=net.compile()
probabilities=dense_2_propagete(images)
probabilities=np.array(probabilities)
dense_2_output = dense_2_propagete(images)

dense_2_output.shape


from neupy import algorithms, environment

environment.reproducible()
# print(probabilities)

data = dense_2_output
sofm = algorithms.SOFM(
    n_inputs=data.shape[1],
    
    # Feature map grid is 2 dimensions and has
    # 400 output clusters (20 * 20).
    features_grid=(20, 20),
    
    # Closest neuron (winning neuron) measures
    # using cosine similarity
    distance='cos',
    
    # Sample weights from the data.
    # Every weight vector will be just a sample
    # from the input data. In this way we can
    # ensure that initialized map will cover data
    # at the very beggining.
    weight='sample_from_data',

    # Defines radius within we consider near by
    # neurons as neighbours relatively to the
    # winning neuron
    learning_radius=6,
    # Large radius is efficient only for the first
    # iterations, that's why we reduce it by 1
    # every 5 epochs.
    reduce_radius_after=5,

    # The further the neighbour neuron from the winning
    # neuron the smaller learning rate for it. How much
    # smaller the learning rate controls by the `std`
    # parameter. The smaller `std` the smaller learning
    # rate for neighboring neurons.
    std=1,
    # Neighbours within 
    reduce_std_after=5,
    
    # Learning rate
    step=0.1,
    # Learning rate is going to be reduced every 5 epochs
    reduce_step_after=5,

     
    
)
sofm.train(data, epochs=32)

clusters = sofm.predict(data).argmax(axis=1)

plt.figure(figsize=(13, 13))
plt.title("NB=3s NM=2p WB=1x WM=0o")

GRID_HEIGHT = 20
GRID_WIDTH = 20

for actual_class, cluster_index in zip(target, clusters):
    cluster_x, cluster_y = divmod(cluster_index, GRID_HEIGHT)
    parameters = class_parameters[actual_class]

    
    plt.plot(2 * cluster_x, 2 * cluster_y, **parameters)
    
    plt.plot(cluster_x, cluster_y, **parameters)
weight = sofm.weight.reshape((sofm.n_inputs, GRID_HEIGHT, GRID_WIDTH))


heatmap1 = compute_heatmap_expanded(weight)

heatmap2 = compute_heatmap(weight)

plt.imshow(heatmap1, cmap='Greys_r', interpolation='nearest')
plt.imshow(heatmap2, cmap='Greys_r', interpolation='nearest')
plt.axis('off')
plt.colorbar()
plt.show()

from scipy.misc import imread
import matplotlib.gridspec as gridspec



def draw_grid(sofm, images, output_features):
    data = images
    clusters = sofm.predict(output_features).argmax(axis=1)
    grid_height, grid_weight = sofm.features_grid
    plt.figure(figsize=(16, 16))
    grid = gridspec.GridSpec(grid_height, grid_weight)
    grid.update(wspace=0, hspace=0)
    for row_id in range(grid_height):
        print("Progress: {:.2%}".format(row_id / grid_weight))
        for col_id in range(grid_weight):
            index = row_id * grid_height + col_id
            clustered_samples = data[clusters == index]
            if len(clustered_samples) > 0:
                # We take the first sample, but it can be any
                # sample from this cluster (random or the one
                # that closer to the center)
                sample = -deprocess(clustered_samples[0])
            else:
                # If we don't have samples in cluster then
                # it means that there is a gap in space
                sample = np.zeros((224, 224, 3))
            plt.subplot(grid[index])
            plt.imshow(sample)
            plt.axis('off')
    print("Progress: 100%")
    return sample

sample = draw_grid(sofm, images, dense_2_output)

plt.show()
