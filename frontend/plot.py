import matplotlib.pyplot as plt
from .pca import project

class View:

    def __init__(self):
        self.plotCounter = 0

    def gallery(self, array, ncols=3):
        array = array[:12]
        nindex, height, width = array.shape

        nrows = nindex//ncols
        assert nindex == nrows*ncols
        # want result.shape = (height*nrows, width*ncols, intensity)
        result = (array.reshape(nrows, ncols, height, width)
                  .swapaxes(1,2)
                  .reshape(height*nrows, width*ncols))
        return result

    def plotInput(self, images, labels, number):
        self.plotCounter += 1
        plt.subplot(2, 2, self.plotCounter)
        images_reshaped = images.reshape(number, 28, 28)
        images_reshaped = self.gallery(images_reshaped)


        plt.imshow(images_reshaped, cmap='gray')

        self.plotCounter += 1
        X_transformed = project(images)
        plt.subplot(2, 2, self.plotCounter)
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels)
        plt.prism()

    def show(self):
        plt.show()