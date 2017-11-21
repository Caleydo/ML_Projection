from frontend.data_loader import loadDataSet
from frontend.plot import View
from frontend.lrp_executor import applyLRP
from frontend.serialize import serialize, deserialize
from sklearn.utils import shuffle

import argparse
import numpy as np
def firstFunctionEver():
    loadData()

def loadData():
    return loadDataSet('MNIST');

def parseArguments():
    parser = argparse.ArgumentParser(description='Creates projections for input images and LRP activations')
    parser.add_argument('integers', metavar='N', type=int, nargs='?',
                        help='number of images to process (12...10000)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()
    if args.integers:
        instanceCount = args.integers
        if instanceCount < 12 or instanceCount > 10000:
            raise ValueError('The instance count is out of range')
        data = loadData()
        X_train, y_train = data.data[:70000] / 255., data.target[:70000]
        X_train, y_train = shuffle(X_train, y_train)

        images = X_train[:instanceCount]
        labels = y_train[:instanceCount]

        heatmaps = applyLRP(images, labels, instanceCount)
        serialize(images, heatmaps, labels)
    else:
        x = deserialize()
        instanceCount = len(x)
        images = np.array(list((map(lambda y: y['input_image'] , x))))
        heatmaps = np.array(list(map(lambda y: y['heatmap'] , x)))
        labels = np.array(list(map(lambda y: y['label'] , x)))

        view = View()
        view.plotInput(images, labels, instanceCount)
        view.plotActivations(heatmaps, labels, instanceCount)
        view.show()

