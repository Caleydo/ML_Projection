from sklearn.datasets import fetch_mldata

def loadDataSet(name):
    dataset = None
    if(name == 'MNIST'):
        dataset = fetch_mldata('MNIST original', data_home='minst_data')
    return dataset


