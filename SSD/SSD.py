#!/usr/bin/env python
# coding: utf-8
# %%


import os
import shutil
from tqdm import tqdm
import json
import numpy as np


dic = json.load(open("dic.json","r"))
label_map = json.load(open("labels.json", "r"))

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
    
    del labels_ids_set
    del images_ids_set
    labels_idxs = [labels_ids_idxs[pair] for pair in tqdm(pairs, desc="finding pairs for labels")]
    images_idxs = [images_ids_idxs[pair] for pair in tqdm(pairs, desc="finding pairs for images")]
    del labels_ids
    del images_ids
    del labels_ids_idxs
    del images_ids_idxs
    
    labels_names = [labels_names[idx] for idx in labels_idxs]
    images_names = [images_names[idx] for idx in images_idxs]
    del labels_idxs
    del images_idxs
    
    # to check the pairs 
    for label_path, image_path in tqdm(zip(labels_names, images_names), desc="checking valid pairs"):
        assert (label_path.split("/")[-1].strip(".json") == image_path.split("/")[-1].strip(".jpg"))
    
    print("Done validating dataset")
    
    print("Collecting processed files...")
    labels_paths = [paths_filename[name] for name in tqdm(labels_names)]
    images_paths = [paths_filename[name] for name in tqdm(images_names)]
    print("Done Collection")
    
    images = []
    labels = []

    for img, label in tqdm(zip(images_paths, labels_paths), desc="Data processing"):
        info = {}
        info['boxes'], info['labels'] = [], []
        
        f = open(os.path.join(data_dir, label), "r")
        # json_data = json.load(f)['Learning data info']
        json_data = json.load(f)['Learning_Data_Info']
        annotations = json_data['Annotations']
    
        json_id = json_data['Json_Data_ID']
        _, _, cat, cls, img_type, _  = json_id.split("_")
        
        if img_type == "R":
            f.close()
            for ant in annotations:
                ant_type = ant['Type']
                if ant_type == "polygon":
                    temp_arr = np.array(ant[ant_type]).reshape(len(ant[ant_type])//2, 2)
                    xmin, ymin = np.min(temp_arr, axis=0)
                    xmax, ymax = np.max(temp_arr, axis=0)
                    
                    if xmin == xmax or ymin == ymax:
                        continue
                        
                    
                if ant_type == "bbox":
                    xmin, ymin, xmax, ymax = ant[ant_type]
                    
                    
                xmin, xmax = xmin/W, xmax/W
                ymin, ymax = ymin/H, ymax/H

                xmin, xmax = int(W_new * xmin), int(W_new * xmax)
                ymin, ymax = int(H_new * ymin), int(H_new * ymax)
                
                if xmin == xmax or ymin == ymax:
                    continue
                
                info['boxes'].append([xmin, ymin, xmax, ymax])
                label = f"{dic[cat]}_{dic[cls]}"
                info['labels'].append(label_map[label])

            if info['boxes']:
                images.append(img.strip(data_dir))
                labels.append(info)
    temp_images = {i: img for i, img in enumerate(images)}
    temp_labels = {i: label for i, label in enumerate(labels)}
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
        num_objs = len(boxes)
        labels = torch.as_tensor(self.labels[index]['labels'], dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3]-boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        assert len(boxes) == len(labels)
    
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

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

        
            


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
import torchvision.models.detection as detection
import os
from engine import train_one_epoch, evaluate


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
        self.optimizer = torch.optim.Adam(params, lr=0.0001)
        self.device = device
        self.start_epoch = 0
        
        saved_models = [file for file in os.listdir() if file.endswith(".pt") and "ssd_model" in file]
        
        if model_name in os.listdir(): # if model name is found
            self.load(model_name)
            print(f"model loaded from {model_name}")
        
        else:
            saved_models = sorted(saved_models, 
                                key=lambda filename:int(filename.strip("ssd_model_").strip(".pt")))
            if saved_models:
                self.load(saved_models[-1])
                print(f"model loaded from {saved_models[-1]}")
                
    def build_model(self, num_classes, device, parallel):
        model = detection.ssd300_vgg16(pretrained=False, num_classes=num_classes, pretrained_backbone=False)
        n_gpus = torch.cuda.device_count()
        if parallel:
            assert n_gpus >= 2
            # , f"Requires at least 2 GPUs to run, but got {n_gpus}"
            # print(f"Running DDP with model parallel example on rank {rank}.")
            #model = torch.nn.parallel.DistributedDataParallel(model)
            model = torch.nn.DataParallel(model)
            
        model.to(device)
        return model

    def forward(self, images, targets):
        for target in targets:
            target['boxes'] = torch.unsqueeze(target['boxes'][0].to(self.device), 0)
            target['labels'] = torch.unsqueeze(target['labels'][0].to(self.device), 0)
            target['image_id'] = torch.unsqueeze(target['image_id'][0].to(self.device), 0)
            target['area'] = torch.unsqueeze(target['area'][0].to(self.device), 0)
            target['iscrowd'] = torch.unsqueeze(target['iscrowd'][0].to(self.device), 0)
            
        return self.model(images, targets)
    
    def fit(self, dataset, max_epochs):
        data_size = len(dataset)
        n_train = int(data_size * 0.8)
        n_valid = int(data_size * 0.9)
        
        split_idx = np.random.choice(data_size, data_size, replace=False)
        
        train_idx = split_idx[:n_train]
        val_idx = split_idx[n_train:n_valid]
        test_idx = split_idx[n_valid:]
        
        trainset = torch.utils.data.Subset(dataset, train_idx)
        valset = torch.utils.data.Subset(dataset, val_idx)
        testset = torch.utils.data.Subset(dataset, test_idx) 
       
        
        trainloader = DataLoader(dataset = trainset, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
        valloader = DataLoader(dataset = valset, batch_size=self.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)
        


        for e in range(self.start_epoch, self.start_epoch + max_epochs):
            #evaluate(self.model, e, valloader, device=self.device)
            train_one_epoch(self.model, self.optimizer, trainloader, self.device, e, print_freq=100)
            
            if e % 5 == 0:
                evaluate(self.model, e, valloader, device=self.device)
                self.save(e)

        
        testloader = DataLoader(dataset = testset, batch_size=self.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn)
        evaluate(self.model, e+1, testloader, device=self.device)

    def save(self, epoch):
        path = f"ssd_model_{epoch}.pt"
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
        print("Loading model from", path)
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
        self.optimizer = torch.optim.Adam(params, lr=0.001 * (0.97 ** self.start_epoch))
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = 0.001 * (0.97 ** self.start_epoch)
