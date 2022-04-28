#!/usr/bin/env python
# coding: utf-8
# %%
import os

from torch.serialization import save
#import shutil
from tqdm import tqdm
import json
import numpy as np
import torchvision

# %%
#categories = {'C':"균열", 'P':'박리,박락', 'X':'철근노출'}
#levels = {'1': '불량', '2': '보통', '3': '우수'} 
#level_en = {'보통': 'normal', '불량': 'faulty', '우수': 'best'}
#cat_en = {'철근노출': 'rebar',
# '박리,박락': 'peel',
# '균열': 'crack',
#}


# %%
#dic = {}
#for k1, v1 in categories.items():
#    for k2, v2 in cat_en.items():
#        if v1 == k2:
#            dic[k1] = v2

# %%
#for k1, v1 in levels.items():
#    for k2, v2 in level_en.items():
#        if v1 == k2:
#            dic[k1] = v2

# %%
#with open("dic.json", 'w') as f:
#    json.dump(dic, f)

# %%
dic = json.load(open("dic.json","r"))

# %%
#temp = dic
#temp_cat, temp_level = {}, {}
#for k, v in temp.items():
#    if k.isnumeric():
#        temp_level[k] = v
#    else:
#        temp_cat[k] = v

# %%
#labels = {}
#count = 0
#for k, v in temp_cat.items():
#    for k1, v1 in temp_level.items():
#        label=f"{v}_{v1}"
#        labels[label] = count
#        count += 1

# %%
#with open("labels.json", "w") as f:
#    json.dump(labels, f)

# %%
label_map = json.load(open("labels.json", "r"))

# %%


W, H = 1080, 1440
W_new, H_new = 360, 480

def process_dataset(data_dir):
    paths_filename = {}

    for root, dirs, files in tqdm(os.walk(data_dir), desc="Searching for files"):
        for file in files:
            paths_filename[file] = os.path.join(root, file)
    
    labels_names = [f for f in paths_filename.keys() if f.endswith(".json")]
    images_names = [f for f in paths_filename.keys() if f.endswith(".jpg")]

    labels_ids = [label.split("/")[-1].strip(".json") for label in labels_names]
    images_ids = [image.split("/")[-1].strip(".jpg") for image in images_names]
    labels_ids_idxs = {label_id: i for i, label_id in enumerate(labels_ids)}
    images_ids_idxs = {image_id: i for i, image_id in enumerate(images_ids)}

    labels_ids_set = set(labels_ids)
    images_ids_set = set(images_ids)
    import csv
    diff = list(labels_ids_set ^ images_ids_set)
    f = open('errors.csv', 'w', encoding='utf-8', newline='\n')
    wr = csv.writer(f)
    wr.writerows([c.strip() for c in r.strip(', ').split(',')]
                     for r in list(diff))
    f.close()
    print("Missing files written in errors.csv")
    del diff

    pairs = sorted(list(labels_ids_set & images_ids_set)) # matching pairs         
    f = open('pairs.csv', 'w', encoding='utf-8', newline='\n')
    wr = csv.writer(f)
    wr.writerows([c.strip() for c in r.strip(', ').split(',')]
                     for r in list(pairs))
    f.close()

    del labels_ids_set
    del images_ids_set
    labels_idxs = [labels_ids_idxs[pair] for pair in tqdm(pairs, desc="finding pairs for labels")]
    images_idxs = [images_ids_idxs[pair] for pair in tqdm(pairs, desc="finding pairs for images")]
    del labels_ids
    del images_ids

    labels_names = [labels_names[idx] for idx in labels_idxs]
    images_names = [images_names[idx] for idx in images_idxs]
    del labels_idxs
    del images_idxs

    for label_path, image_path in tqdm(zip(labels_names, images_names), desc="checking valid pairs"):
        assert (label_path.split("/")[-1].strip(".json") == image_path.split("/")[-1].strip(".jpg"))

    print("Done validating dataset")

    print("Collecting processed files...")
    labels_paths = [paths_filename[name] for name in tqdm(labels_names)]
    images_paths = [paths_filename[name] for name in tqdm(images_names)]
    print("Done Collection")

    images = []
    labels = []
    
    #classes = en_classes
    for img, label in tqdm(zip(images_paths, labels_paths), desc="Data processing"):
        info = {}
        info['boxes'], info['labels'], info['polys'] = [], [], []
        f = open(os.path.join(data_dir, label), "r")
        json_data = json.load(f)['Learning_Data_Info']
        f.close()
        annotations = json_data['Annotations']
    
        json_id = json_data['Json_Data_ID']
        _, _, cat, cls, img_type, _  = json_id.split("_")
        
        if img_type == "R":
            for ant in annotations:
                ant_type = ant['Type']
            
                # mask = np.zeros((H_new, W_new), dtype=np.uint8)
                if ant_type == "polygon":
                    temp_arr = np.array(ant[ant_type]).reshape(len(ant[ant_type])//2, 2)
                    xmin, ymin = np.min(temp_arr, axis=0)
                    xmax, ymax = np.max(temp_arr, axis=0)
                    
                
                    if xmin == xmax or ymin == ymax:
                        print(img, "bbox size error (too thin)")
                        continue
                    
                    for i in range(len(temp_arr)):
                        temp_arr[i][0] = int((temp_arr[i][0] / W) * W_new)
                        temp_arr[i][1] = int((temp_arr[i][1] / H) * H_new)
                    #cv2.fillPoly(img=mask, pts=[temp_arr], color=(1, 1, 1))

                if ant_type == "bbox":
                    #print("bbox pass")
                    continue

                xmin, xmax = xmin/W, xmax/W
                ymin, ymax = ymin/H, ymax/H

                xmin, xmax = int(W_new * xmin), int(W_new * xmax)
                ymin, ymax = int(H_new * ymin), int(H_new * ymax)
                
                if xmin == xmax or ymin == ymax:
                    #print("bbox error")
                    continue
                    
                info['polys'].append(temp_arr)
                info['boxes'].append([xmin, ymin, xmax, ymax])
                label = f"{dic[cat]}_{dic[cls]}"
                info['labels'].append(label_map[label])

            if info['boxes']:
                images.append(img.strip(data_dir))
                labels.append(info)
    
   
    temp_images = {i: img for i, img in enumerate(images)}
    temp_labels = {i: label for i, label in enumerate(labels)}
    #for i, img in enumerate(images):
    images = temp_images
    labels = temp_labels
    print("클래스, 인덱스 맵핑", label_map)
    num_classes = len(label_map)
    num_classes += 1 # background class
    return num_classes, images, labels


# %%


import cv2
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, data_name, transform=None):
        assert data_name.endswith(".pt"), "data_name must end with .pt"
        try:
            self.load(data_name)
            self.dir = data_dir
            print(f"Data loaded from {data_name} from directory {data_dir}")
            self.transform = transform # override transform to apply augmentation
        except:
            print(f"Error loading {data_name}, processing data from {data_dir}")
            self.dir = data_dir
            self.transform = transform
            self.num_classes, self.images, self.labels = process_dataset(data_dir)
            self.save(data_name)

        print("Dataset Initialized")
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.dir, self.images[index]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W_new, H_new), interpolation=cv2.INTER_AREA)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        img = img.transpose((2,0,1)).astype(np.float32)
        img = img / 255.0
        img = torch.as_tensor(img, dtype=torch.float32)
    
        boxes = torch.as_tensor(self.labels[index]['boxes'], dtype=torch.float32)
        labels = torch.as_tensor(self.labels[index]['labels'], dtype=torch.int64)

        polys = self.labels[index]['polys']

        masks = torch.zeros((len(polys),H_new, W_new), dtype=torch.uint8)
        for i, poly in enumerate(polys):
            mask = np.zeros((H_new, W_new), dtype=np.uint8)
            cv2.fillPoly(img=mask, pts=[poly], color=(1, 1, 1))
            masks[i, :, :] = torch.tensor(mask, dtype=torch.bool)
        
        image_id = torch.tensor([index])
        area = (boxes[:, 3]-boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        num_objs = len(boxes)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        assert len(boxes) == len(labels)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transform:
            img, target = self.transform(img, target)
        return img, target
    
    def save(self, data_name):
        temp = {
            "transform":self.transform,
            "images":self.images,
            "labels":self.labels,
            "num_classes":self.num_classes
        }
        
        torch.save(temp, data_name)
        print(f"dataset saved to {data_name}")

    def load(self, data_name):
        temp = torch.load(data_name)
        self.transform = temp["transform"]
        self.images = temp["images"]
        self.labels = temp["labels"]
        self.num_classes = temp["num_classes"]

# %%


import numpy as np
import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision.models.detection import MaskRCNN
from torch.utils.data import DataLoader
import torchvision.models.detection as detection
from torchvision.models.detection.anchor_utils import AnchorGenerator
#import torchvision.transforms as transforms
import os
#from sklearn.metrics import average_precision_score
from engine import train_one_epoch, evaluate
#from collections import Counter

def collate_fn(batch):
    imgs, targets = [], []
    for img, target in batch:
        imgs.append(img)
        targets.append(target)
        
    return imgs, targets
class Model(Module):
    def __init__(self, num_classes, device, parallel, model_name, batch_size=8):
        Module.__init__(self)
        self.batch_size=batch_size
        self.num_classes=num_classes
        self.model = self.build_model(num_classes, device, parallel)
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=0.005)
        
        self.device = device
        self.start_epoch = 0
        
        saved_models = [file for file in os.listdir() if file.endswith(".pt") and "mrcnn_model" in file]
        
        if model_name in os.listdir(): # if model name is found
            self.load(model_name)
            print(f"model loaded from {model_name}")
        
        else:
            saved_models = sorted(saved_models, 
                                key=lambda filename:int(filename.strip("mrcnn_model_").strip(".pt")))
            if saved_models:
                self.load(saved_models[-1])
                print(f"model loaded from {saved_models[-1]}")
                
    def build_model(self, num_classes, device, parallel):
        """
        mod:
            use backbone
        """
        backbone = torchvision.models.mobilenet_v2(pretrained=True).features
        backbone.out_channels = 1280
        anchor_generator = AnchorGenerator(sizes = ((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0), ))
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
        mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
        #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, mask_roi_pool=mask_roi_pooler)
        #model = detection.maskrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes, pretrained_backbone=True,trainable_backbone_layers=3)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        
        n_gpus = torch.cuda.device_count()
        if parallel:
            assert n_gpus >= 2
            model = torch.nn.DataParallel(model)
        model.to(device)
        return model

    def forward(self, images, targets):
        for target in targets:
            target['boxes'] = torch.unsqueeze(target['boxes'][0].to(self.device), 0)
            target['labels'] = torch.unsqueeze(target['labels'][0].to(self.device), 0)
            target['masks'] = torch.unsqueeze(target['masks'][0].to(self.device), 0)
            target['image_id'] = torch.unsqueeze(target['image_id'][0].to(self.device), 0)
            target['area'] = torch.unsqueeze(target['area'][0].to(self.device), 0)
            target['iscrowd'] = torch.unsqueeze(target['iscrowd'][0].to(self.device), 0)
            
        return self.model(images, targets)
    
    def fit(self, dataset, max_epochs):
        data_size = len(dataset)
        print("data_size", data_size)
        n_train = int(data_size * 0.8)
        n_valid = int(data_size * 0.9)
        split_idx = np.random.choice(data_size, data_size, replace=False)
        
        train_idx = split_idx[:n_train]
        val_idx = split_idx[n_train:n_valid]
        test_idx = split_idx[n_valid:]
        
        trainset = torch.utils.data.Subset(dataset, train_idx)
        valset = torch.utils.data.Subset(dataset, val_idx)
        testset = torch.utils.data.Subset(dataset, test_idx)

        trainloader = DataLoader(dataset = trainset, batch_size=self.batch_size, shuffle=True, num_workers=4,collate_fn=collate_fn)
        valloader = DataLoader(dataset = valset, batch_size=self.batch_size, shuffle=True, num_workers=4,collate_fn=collate_fn)

        for e in range(self.start_epoch, self.start_epoch + max_epochs):
            #for testing
            #evaluate(self.model, e, valloader, device=self.device)
            train_one_epoch(self.model, self.optimizer, trainloader, self.device, e, print_freq=100)
            
            if e % 5 == 0:
                evaluate(self.model, e, valloader, device=self.device)
                self.save(e)
        
        testloader = DataLoader(dataset = testset, batch_size=self.batch_size, shuffle=False, num_workers=4,collate_fn=collate_fn)
        evaluate(self.model, e+1, testloader, device=self.device)
     

    def save(self, epoch):
        path = f"mrcnn_model_{epoch}.pt"
        assert epoch != None
        if isinstance(self.model, torch.nn.DataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        torch.save({
            'epoch':epoch,
            'model_state_dict':  model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print("Model saved at", path)

    def load(self, path):
        print("Loading model", path)
        checkpoint = torch.load(path)
        self.start_epoch = checkpoint['epoch'] + 1
        print("Starting from epoch", checkpoint['epoch'] + 1)

        if isinstance(self.model, torch.nn.DataParallel):
            try:
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            except: 
                print("Failed to load from parallel ckpt")
                self.model = self.build_model(self.num_classes, self.device, parallel=False)
                self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=0.005) 
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])      
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.005 * (0.95 ** checkpoint['epoch'])

