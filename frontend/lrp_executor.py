import numpy as np
import matplotlib.pyplot as plt

'''
import sys
sys.path.append('../lrp_toolbox')
sys.path.append('../lrp_toolbox/python')
'''

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../lrp_toolbox"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../lrp_toolbox/python"))

import model_io
import data_io
import render

print(sys.path)



def applyLRP(data, labels, number):
    nn = model_io.read('C:/Martin/ML/MLProjects/LRP/lrp_toolbox/models/MNIST/long-rect.nn')
    X = data
    Y = labels
    Y = Y[:, np.newaxis]

    #X = data_io.read('C:/Martin/ML/MLProjects/LRP/lrp_toolbox/data/MNIST/test_images.npy')
    #Y = data_io.read('C:/Martin/ML/MLProjects/LRP/lrp_toolbox/data/MNIST/test_labels.npy')

    # transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
    X = X / 0.5 - 1

    # transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
    I = Y[:, 0].astype(int)
    Y = np.zeros([X.shape[0], np.amax(I) + 1])
    Y[np.arange(Y.shape[0]), I] = 1

    # permute data order for demonstration. or not. your choice.
    I = np.arange(X.shape[0])
    # I = np.random.permutation(I)

    heatmaps = []
    for i in range(0, number):
        x = X[np.newaxis, i, :]

        # forward pass and prediction
        ypred = nn.forward(x)
        #print('True Class:     ', np.argmax(Y[i]))
        #print('Predicted Class:', np.argmax(ypred), '\n')

        # compute first layer relevance according to prediction
        # R = nn.lrp(ypred)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
        R = nn.lrp(ypred, 'epsilon', 1.)  # as Eq(58) from DOI: 10.1371/journal.pone.0130140
        # R = nn.lrp(ypred,'alphabeta',2)    #as Eq(60) from DOI: 10.1371/journal.pone.0130140


        # R = nn.lrp(Y[na,i]) #compute first layer relevance according to the true class label


        '''
        yselect = 3
        yselect = (np.arange(Y.shape[1])[na,:] == yselect)*1.
        R = nn.lrp(yselect) #compute first layer relvance for an arbitrarily selected class
        '''

        # undo input normalization for digit drawing. get it back to range [0,1] per pixel
        x = (x + 1.) / 2.

        # render input and heatmap as rgb images
        hm = render.hm_to_rgb(R, X=x, scaling=3, sigma=2)

        heatmaps.append(R[0])
        R = R.reshape(28,28)
        # display the image as written to file
        #plt.imshow(hm, interpolation='none')
       # plt.axis('off')
        #plt.show()
    return np.array(heatmaps)

