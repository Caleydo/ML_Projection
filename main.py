from frontend.data_loader import loadDataSet
from frontend.plot import View
from frontend.lrp_executor import applyLRP
from frontend.serialize import serialize, deserialize
from sklearn.utils import shuffle
import os

import argparse
import numpy as np

ProjectionMethods = ['pca', 't-sne']

def firstFunctionEver():
    loadData()

def loadData(path):
    return loadDataSet('MNIST', path);

def parseArguments():
    parser = argparse.ArgumentParser(description='Creates projections for input images and LRP activations')
    parser.add_argument('integers', metavar='N', type=int, nargs='?',
                        help='number of images to process (12...10000)')
    parser.add_argument('-p', '--projection_method', help = 'Select used projection method', choices=['t-sne', 'pca'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    path = os.path.dirname(os.path.realpath(__file__))
    args = parseArguments()
    if args.integers:
        instanceCount = args.integers
        if instanceCount < 12 or instanceCount > 10000:
            raise ValueError('The instance count is out of range')
        data = loadData(path)
        X_train, y_train = data.data[:70000] / 255., data.target[:70000]
        X_train, y_train = shuffle(X_train, y_train)

        images = X_train[:instanceCount]
        labels = y_train[:instanceCount]

        heatmaps, predictedLabels = applyLRP(images, labels, instanceCount)
        serialize(images, heatmaps, labels, predictedLabels, path)
    else:
        x = deserialize(path)
        instanceCount = len(x)
        print('Number of instances: ', instanceCount)
        images = np.array(list((map(lambda y: y['input_image'] , x))))
        heatmaps = np.array(list(map(lambda y: y['heatmap'] , x)))
        labels = np.array(list(map(lambda y: y['label'] , x)))
        predictedLabels = np.array(list(map(lambda y: y['predicted_label'] , x)))

        projectionMethod = 'pca'
        if args.projection_method:
            projectionMethod = args.projection_method
        view = View()
        view.plotInput(images, instanceCount, projectionMethod, labels, predictedLabels)
        view.plotActivations(heatmaps, instanceCount, projectionMethod, labels, predictedLabels)
        view.show()

