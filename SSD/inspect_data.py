from SSD import *
import json
import os
from tqdm import tqdm
dataset = CustomDataset("/dataset", "ssd_data.pt")

label_map = json.load(open("labels.json", "r"))

decode = {}
for k, v in label_map.items():
    decode[v] = k
img_counts = {label:0 for label in label_map}
counts = {label:0 for label in label_map}
labels = dataset.labels
images = dataset.images
for i in tqdm(range(len(labels))):
    temp_label = labels[i]['labels'][0]
    img_counts[decode[temp_label]] += 1
    #img_path = images[i]
   
    for label in labels[i]['labels']:
        counts[decode[label]] += 1

print("bbox count", counts)
print("img counts", img_counts)
