import matplotlib.pyplot as plt
from frontend.pca import project
import numpy as np
class View:

    def __init__(self):
        self.plotCounter = 0
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 10))
        '''y = np.sin(x ** 2)
        self.ax1.plot(x, y)
        self.ax2.scatter(x, y)
        self.ax3.scatter(x, 2 * y ** 2 - 1, color='r')
        self.ax4.plot(x, 2 * y ** 2 - 1, color='r')
        plt.show()'''

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
        images_reshaped = images.reshape(number, 28, 28)
        images_reshaped = self.gallery(images_reshaped)


        self.ax1.imshow(images_reshaped, cmap='gray')

        X_transformed = project(images)
        plt.prism()
        self.ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels)

    def plotActivations(self, images, labels, number):
        images_reshaped = images.reshape(number, 28, 28)
        images_reshaped = self.gallery(images_reshaped)


        self.ax3.imshow(images_reshaped, cmap='gray')

        X_transformed = project(images)
        plt.prism()
        self.ax4.scatter(X_transformed[:, 0], X_transformed[:, 1], c=labels)

    def show(self):
        plt.show()