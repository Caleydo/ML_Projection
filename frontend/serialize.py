import csv
import os
import json

foldername = 'data'
filename = 'foo.json'

def createJSON(images, heatmaps, labels):
    instances = []
    for i in range(0, len(images)):
        instances.append({
            "input_image": images[i].tolist(),
            "heatmap": heatmaps[i].tolist(),
            "label": labels[i]
        })
    return instances

def createDataFolder(foldername):
    if not os.path.exists(foldername):
        os.makedirs(foldername)

def serialize(images, heatmaps, labels):
    if len(images) != len(heatmaps) and len(heatmaps) != len(labels):
        raise ValueError('The dimensions of the input/activities/labels are not compatible')
    createDataFolder(foldername)


    with open(foldername + '/' + filename, 'w') as f:
       json.dump(createJSON(images, heatmaps, labels), f)

def deserialize():
    with open(foldername + '/' + filename, 'r') as f:
        return json.load(f)

