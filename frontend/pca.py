from sklearn import decomposition
from sklearn.manifold import TSNE as tsne

def project(images, projectionMethod):
    if projectionMethod == 'pca':
        return applyPCA(images)
    else:
        return applyTsne(images)

def applyPCA(images):
    pca = decomposition.PCA(n_components=2)
    pca.fit(images)
    return pca.transform(images)

def applyTsne(images):
    return tsne(n_components=2).fit_transform(images)