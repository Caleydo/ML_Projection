import csv
import os
import json

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

def serialize(a, b, c):
    if len(a) != len(b) and len(b) != len(c):
        raise ValueError('The dimensions of the input/activities/labels are not compatible')
    foldername = 'data'
    filename = 'foo.json'
    createDataFolder(foldername)


    with open(foldername + '/' + filename, 'w') as f:
       json.dump(createJSON(a, b, c), f)
