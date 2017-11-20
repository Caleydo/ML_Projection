from frontend.data_loader import loadDataSet
from frontend.plot import View
from frontend.lrp_executor import applyLRP

from sklearn.utils import shuffle

def firstFunctionEver():
    loadData()

def loadData():
    return loadDataSet('MNIST');

if __name__ == '__main__':
    instanceCount = 100
    data = loadData()
    X_train, y_train = data.data[:70000] / 255., data.target[:70000]
    X_train, y_train = shuffle(X_train, y_train)

    images = X_train[:instanceCount]
    labels = y_train[:instanceCount]

    view = View()

    view.plotInput(images, labels, instanceCount)

    heatmaps = applyLRP(images, labels, instanceCount)
    view.plotActivations(heatmaps, labels, instanceCount)

    view.show()
