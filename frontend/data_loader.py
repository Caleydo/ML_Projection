from sklearn.datasets import fetch_mldata
import os
def loadDataSet(name, path):
    dataset = None
    if(name == 'MNIST'):
        dataset = fetch_mldata('MNIST original', data_home=os.path.join(path, 'minst_data'))
    return dataset


