import argparse
from re import S
parser = argparse.ArgumentParser(description='plot predictions on imgs of all classes')
parser.add_argument('--conf', default=0.5, type=float,help='confidence threshold')
parser.add_argument('--num', default=1, type=int, help='total number of plots for each label')
parser.add_argument('--data', default="mrcnn_data.pt", type=str, help="dataset.pt filename")
parser.add_argument('--model', default="mrcnn_model_75.pt", type=str, help="mrcnn_model.pt filename")
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
font_size = 8
alpha = 0.2
colors = {
    'crack_faulty': "#f700ad", 
    'crack_normal': "#24ad46",
    'crack_best': "#220aaa",
    'peel_faulty': "#a50616",
    'peel_normal': "#01873d",
    'peel_best': "#543daf",
    'rebar_faulty': "#382891",
    'rebar_normal': "#e5be12",
    'rebar_best': "#0d2808"
} 


import torch
from MRCNN import *
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

#saved_datasets = [file for file in os.listdir() if file.endswith(".pt") and "dataset" in file]
#saved_datasets = sorted(saved_datasets, key=lambda filename:int(filename.strip("dataset_").strip(".pt")))
#most_recent_dataset = saved_datasets[-1]

dataset = CustomDataset("/dataset", args.data)
N = len(dataset.images) # size of dataset

#saved_models = [file for file in os.listdir() if file.endswith(".pt") and "mrcnn" in file]
#saved_models = sorted(saved_models, key=lambda filename:int(filename.strip("mrcnn_model_").strip(".pt")))
#most_recent_model = saved_models[-1]
print(f"evaluate {args.model} with {args.data}")
model = Model(num_classes=dataset.num_classes, device = "cpu", parallel = False, model_name=args.model,  batch_size=8)

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
        gt_masks = target['masks']
        
        preds_temp = preds[0]
        boxes = preds_temp['boxes']
        labels = preds_temp['labels']
        scores = preds_temp['scores']
        masks = preds_temp['masks']
        
        # print(gt_boxes)
        # print(gt_labels)
        pred_result = {}
        for i, temp_label in enumerate(labels):
            temp_label = temp_label.item()
            if temp_label in pred_result:
                pred_result[temp_label]['scores'].append(scores[i])
                pred_result[temp_label]['boxes'].append(boxes[i])
                pred_result[temp_label]['masks'].append(masks[i])
            else:
                pred_result[temp_label]={'scores':[scores[i]], 'boxes':[boxes[i]], 'masks':[masks[i]]}
        #print(image_to_plot)

        
        gt_boxes_tensor = torch.zeros((len(gt_boxes), 4))
        gt_masks_tensor = torch.zeros((len(gt_masks), image.shape[-2], image.shape[-1]), dtype=torch.bool)
        gt_labels = [decode[label.item()] for _ in range(len(gt_boxes))]  # only one class exists in one photo (gt)
        gt_box_colors = [colors[temp] for temp in gt_labels]
        gt_mask_colors = [colors[temp] for temp in gt_labels]
        gt_labels = [gt_label + "_TRUE" for gt_label in gt_labels]
        for i, (gt_box, gt_mask, gt_label) in enumerate(zip(gt_boxes, gt_masks, gt_labels)):
            gt_boxes_tensor[i, :] = torch.tensor(gt_box)
            gt_masks_tensor[i, :, :] = torch.tensor(gt_mask).bool()
            
        image_to_plot = draw_bounding_boxes(image_to_plot, gt_boxes_tensor, colors=gt_box_colors, labels=gt_labels, font_size=font_size)
        image_to_plot = draw_segmentation_masks(image_to_plot, gt_masks_tensor, colors=gt_mask_colors, alpha = 0.4)
        
            
        for temp_label, pred in pred_result.items():
            score_idx = torch.as_tensor(pred['scores']) > confidence
            if torch.count_nonzero(score_idx) == 0:
                print("nothing detected on", decode[temp_label])
                continue

            boxes = pred['boxes']
            masks = pred['masks']
            n = len(boxes)

            boxes_tensor = torch.zeros((n, 4))
            masks_tensor = torch.zeros((n, image.shape[-2], image.shape[-1]), dtype=torch.bool)
            
            for i, box in enumerate(boxes):
                boxes_tensor[i, :] = box
            for i, mask in enumerate(masks):
                masks_tensor[i, :, :] = mask.bool()

            boxes = boxes_tensor[score_idx]
            masks = masks_tensor[score_idx]

            temp_labels = [decode[temp_label] for _ in range(len(boxes))] # all boxes in 

            
            box_colors = [colors[temp] for temp in temp_labels]
            mask_colors = [colors[temp] for temp in temp_labels]
            image_to_plot = draw_bounding_boxes(image_to_plot, boxes, colors=box_colors, labels=temp_labels, font_size=font_size)
            image_to_plot = draw_segmentation_masks(image_to_plot, masks, colors=mask_colors, alpha = alpha)

 
        plt.figure(1, figsize=(12, 12))
        image_to_plot = image_to_plot.permute(1, 2, 0).numpy()
        plt.imshow(image_to_plot)
        plt.savefig(f"predict_{decode[label.item()]}_{it}.png", dpi=600)
        plt.clf()
