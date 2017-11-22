import matplotlib.pyplot as plt
from frontend.pca import project
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
0
class View:

    def __init__(self):
        self.plotCounter = 0
        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 10))

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

    def plotInput(self, images, labels, instanceCount, projectionMethod):
        self.plotRow(images, instanceCount, projectionMethod, self.ax1, self.ax2)

    def plotActivations(self, images, labels, instanceCount, projectionMethod):
        self.plotRow(images, instanceCount, projectionMethod, self.ax3, self.ax4)

    def plotRow(self, images, instanceCount, projectionMethod, axes1, axes2):
        X_transformed = project(images, projectionMethod)
        images_reshaped = images.reshape(instanceCount, 28, 28)
        self.plotImages(axes1, images_reshaped)
        self.plotProjection(axes2, images_reshaped, X_transformed)

    def plotImages(self, axes, images):
        images_rendering = self.gallery(images)
        axes.imshow(images_rendering)

    def plotProjection(self, axes, images, xTransformed):
        for index, (x0, y0) in enumerate(zip(xTransformed[:, 0], xTransformed[:, 1])):
            im = OffsetImage(images[index], zoom=0.5)
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            axes.add_artist(ab)

        axes.update_datalim(np.column_stack([xTransformed[:, 0], xTransformed[:, 1]]))
        axes.autoscale()

    def show(self):
        plt.show()