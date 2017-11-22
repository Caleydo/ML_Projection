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

    def plotInput(self, images, instanceCount, projectionMethod, labels, predictedLabels):
        self.plotRow(images, instanceCount, projectionMethod, labels, predictedLabels, self.ax1, self.ax2)

    def plotActivations(self, images, instanceCount, projectionMethod, labels, predictedLabels):
        self.plotRow(images, instanceCount, projectionMethod, labels, predictedLabels, self.ax3, self.ax4)

    def plotRow(self, images, instanceCount, projectionMethod, labels, predictedLabels, axes1, axes2):
        X_transformed = project(images, projectionMethod)
        images_reshaped = images.reshape(instanceCount, 28, 28)
        self.plotImages(axes1, images_reshaped)
        self.plotProjection(axes2, images_reshaped, X_transformed, labels, predictedLabels)

    def plotImages(self, axes, images):
        images_rendering = self.gallery(images)
        axes.imshow(images_rendering)

    def plotProjection(self, axes, images, xTransformed, labels, predictedLabels):
        for index, (x0, y0) in enumerate(zip(xTransformed[:, 0], xTransformed[:, 1])):
            im = self.modifyMisclassified(images[index], labels[index], predictedLabels[index])
            im = OffsetImage(im, zoom=0.5)
            ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
            axes.add_artist(ab)

        axes.update_datalim(np.column_stack([xTransformed[:, 0], xTransformed[:, 1]]))
        axes.autoscale()

    def modifyMisclassified(self, image, label, predictedLabel):
        if label != predictedLabel:
            image[:] = 0.000
        return image


    def show(self):
        plt.show()