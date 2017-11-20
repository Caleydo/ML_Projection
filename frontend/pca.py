from sklearn import decomposition


def project(images):
    pca = decomposition.PCA(n_components=2)
    pca.fit(images)
    return pca.transform(images)