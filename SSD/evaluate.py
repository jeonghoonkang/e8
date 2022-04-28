import argparse
parser = argparse.ArgumentParser(description='plot predictions on imgs of all classes')
parser.add_argument('--conf', default=0.5, type=float,help='confidence threshold')
parser.add_argument('--num', default=1, type=int, help='total number of plots for each label')
parser.add_argument('--data', default="ssd_data.pt", type=str, help="dataset.pt filename")
parser.add_argument('--model', default="ssd_model_130.pt", type=str, help="ssd_model.pt filename")
args = parser.parse_args()
confidence = args.conf
num = args.num
assert 0<confidence<1

import json
dic = json.load(open("dic.json","r"))
label_map = json.load(open("labels.json", "r"))

decode = {}
for k, v in label_map.items():
    decode[v] = k

    
colors = {
    'struct_crack_best':'#94CC5A',
    'struct_crack_normal':'#73936F',
    'struct_crack_faulty':'#71020B',
    'struct_peel_best':'#017256',
    'struct_peel_normal':'#42138B',
    'struct_peel_faulty':'#CE80EC',
    'struct_rebar_best':'#E99E1C',
    'struct_rebar_normal':'#030063',
    'struct_rebar_faulty':'#BEFD57',
    'ground_best':'#B17088',
    'ground_normal':'#07A12B',
    'ground_faulty':'#8AA0DB',
    'finish_best':'#1199BC',
    'finish_normal':'#BE21E9',
    'finish_faulty':'#33BA1B',
    'window_best':'#02CB30',
    'window_normal':'#59D248',
    'window_faulty':'#8B570F',
    'living_best':'#7FA0BE',
    'living_normal':'#0494C7',
    'living_faulty':'#AE5EEB'
} 


import torch
from SSD import *
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import draw_bounding_boxes
import random

#saved_datasets = [file for file in os.listdir() if file.endswith(".pt") and "dataset" in file]
#saved_datasets = sorted(saved_datasets, key=lambda filename:int(filename.strip("dataset_").strip(".pt")))
#most_recent_dataset = saved_datasets[-1]

dataset = CustomDataset("/dataset", args.data)
N = len(dataset.images) # size of dataset

#saved_models = [file for file in os.listdir() if file.endswith(".pt") and "ssd" in file]
#saved_models = sorted(saved_models, key=lambda filename:int(filename.strip("ssd_model_").strip(".pt")))
#most_recent_model = "ssd_model_125.pt"#saved_models[-1]
print(f"evaluate {args.model} with {args.data}")

model = Model(num_classes=dataset.num_classes, device = "cpu", parallel = False, model_name=args.model,  batch_size=1)

for it in range(num):
    
    
    temp_labels = {}
    temp_imgs = {}

    keys = list(dataset.labels.keys())
    random.shuffle(keys)
    for img_id in keys:
        img, target = dataset[img_id]
        temp_label = target['labels'][0] #take any label as one image contains only one label 
        if temp_label not in temp_labels:
            temp_labels[temp_label] = target
            temp_imgs[temp_label] = img
            if len(temp_labels) == len(label_map):
                break

    model.model.eval()

    for label, target in tqdm(temp_labels.items(), desc="reading data"):
        image = temp_imgs[label]
        preds = model.model([image])
        image_to_plot = image * 255.0
        image_to_plot = image_to_plot.type(torch.uint8)

        gt_boxes = target['boxes']
        gt_labels = target['labels']

        preds_temp = preds[0]
        boxes = preds_temp['boxes']
        labels = preds_temp['labels']
        scores = preds_temp['scores']
        pred_result = {}
        for i, temp_label in enumerate(labels):
            temp_label = temp_label.item()
            if temp_label in pred_result:
                pred_result[temp_label]['scores'].append(scores[i])
                pred_result[temp_label]['boxes'].append(boxes[i])

            else:
                pred_result[temp_label]={'scores':[scores[i]], 'boxes':[boxes[i]]}

        
        gt_boxes_tensor = torch.zeros((len(gt_boxes), 4))
        gt_labels = [decode[label.item()] for _ in range(len(gt_boxes))]  # only one class exists in one photo (gt)
        gt_box_colors = [colors[temp] for temp in gt_labels]
        gt_labels = [gt_label + "_TRUE" for gt_label in gt_labels]
        for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            gt_boxes_tensor[i, :] = torch.tensor(gt_box)
            
        image_to_plot = draw_bounding_boxes(image_to_plot, gt_boxes_tensor, colors=gt_box_colors, labels=gt_labels)        
            
        for temp_label, pred in pred_result.items():
            score_idx = torch.as_tensor(pred['scores']) > confidence
            if torch.count_nonzero(score_idx) == 0:
                print("nothing detected on", decode[temp_label])
                continue

            boxes = pred['boxes']
            n = len(boxes)

            boxes_tensor = torch.zeros((n, 4))

            
            for i, box in enumerate(boxes):
                boxes_tensor[i, :] = box


            boxes = boxes_tensor[score_idx]


            temp_labels = [decode[temp_label] for _ in range(len(boxes))] # all boxes in 

            
            box_colors = [colors[temp] for temp in temp_labels]
            mask_colors = [colors[temp] for temp in temp_labels]
            image_to_plot = draw_bounding_boxes(image_to_plot, boxes, colors=box_colors, labels=temp_labels)


 
        plt.figure(1, figsize=(12, 12))
        image_to_plot = image_to_plot.permute(1, 2, 0).numpy()
        plt.imshow(image_to_plot)
        plt.savefig(f"predict_{decode[label.item()]}_{it}.png", dpi=600)
        plt.clf()
