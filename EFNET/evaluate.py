import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn 
import argparse
def getTimestamp():
    import time, datetime
    timezone = 60*60*9 # seconds * minutes * utc + 9
    utc_timestamp = int(time.time() + timezone)
    date = datetime.datetime.fromtimestamp(utc_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    return utc_timestamp
parser = argparse.ArgumentParser(description="train efficientnet-b0")
parser.add_argument("--model", default="eff_net.pt", type=str, help="model name to load from")
args = parser.parse_args()

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)) # H, W
])

test_dir  = 'dataset/test'
testset = ImageFolder(root=test_dir, transform=transforms, target_transform=None)
testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)

PATH = args.model
dataiter = iter(testloader) 
images, labels = dataiter.next() # 실험용 데이터와 결과 출력 
def imsave(img):
    npimg = img.numpy()
    plt.figure(1, figsize=(12, 12))
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.savefig("evaluate.png", dpi=600)
    plt.clf()
imsave(torchvision.utils.make_grid(images)) 
print('GroundTruth: ', ' '.join('%5s' % testset.classes[label] for label in labels)) # 학습한 모델로 예측값 뽑아보기 

def build_net(num_classes):
    net = models.efficientnet_b0(pretrained=True)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    return net

net = build_net(len(testset.classes))
net.load_state_dict(torch.load(PATH)) 
#outputs = net(images)
#_, predicted = torch.max(outputs, 1) 
#print('Predicted: ', ' '.join('%5s' %  testset.classes[predict] for predict in predicted))

logs = {"start":getTimestamp()}
stats_by_class = {i:{"correct":0, "total":0} for i in range(11)} #11 classes
labels_by_class = {i:[] for i in range(11)}
preds_by_class = {i:[] for i in range(11)}

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
image_names = list(map(lambda img: img[0], testset.imgs))
with torch.no_grad(): 
    for img_idx, (image, label) in enumerate(testloader): 
        output = net(image)
        img_name = image_names[img_idx]
        _, predicted = torch.max(output.data, 1) 
        
        temp_label = label.item()
        temp_predict = predicted.item()
        is_correct = temp_label == temp_predict

        logs[img_name] = {"predict":temp_predict, "label": temp_label, "is_correct": is_correct, "class_stats":{}, "final_stats":{}}
        logs[img_name]["cumul_correct"] = stats_by_class[temp_label]["correct"]
        logs[img_name]["cumul_total"] = stats_by_class[temp_label]["total"]
        try:
            logs[img_name]["current_acc"] = stats_by_class[temp_label]["correct"] / stats_by_class[temp_label]["total"]
        except:
            logs[img_name]["current_acc"] = 0
        
        labels_by_class[temp_label].append(temp_label)
        preds_by_class[temp_label].append(temp_predict)

        temp_labels = labels_by_class[temp_label]
        temp_preds = preds_by_class[temp_label]
        #numpy_labels, numpy_preds = labels.numpy(), predicted.numpy()

        #batch_precision, batch_recall = precision_score(numpy_labels, numpy_preds), recall_score(numpy_labels, numpy_preds)
        cumul_precision, cumul_recall, cumul_f1 = precision_score(temp_labels, temp_preds), recall_score(temp_labels, temp_preds), f1_score(temp_labels, temp_preds)
        
        logs[img_name]["cumul_precision"] = cumul_precision
        logs[img_name]["cumul_recall"] = cumul_recall
        logs[img_name]["cumul_f1"] = cumul_f1
        
        total += 1
        correct += int(is_correct)
        stats_by_class[temp_label]["total"] += 1
        stats_by_class[temp_label]["correct"] += int(is_correct))
        
        if img_idx % 20 == 0:
            str_buffer = f"== Image Index {img_idx}==")
            for i in range(11):
                labels = labels_by_class[i]
                preds = labels_by_class[i]
                try:
                    acc = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds)
                    str_stats = f"Class {i}, Accuracy:{acc}, F1: {f1}"
                    str_buffer = f"{str_buffer}\n{str_stats}\n"
                except: 
                    str_stats = f"Class {i}, Accuracy:NaN, F1: NaN"
                    str_buffer = f"{str_buffer}\n{str_stats}\n"
            
            print(str_buffer)
    
    for i in range(11):
        labels = labels_by_class[i]
        preds = labels_by_class[i]
        logs["class_stats"][i] = {"acc": accuracy_score(labels, preds), "f1":f1_score(labels, preds)})
    
    mean_acc = np.mean(list(logs["class_stats"][i]["acc"] for i in range(11)))
    mean_f1 = np.mean(list(logs["class_stats"][i]["f1"] for i in range(11)))

    logs["final_stats"] = {"mean_acc":mean_acc, "mean_f1":mean_f1}

logs["end"] = getTimestamp()

json.dump(logs, open("detailed_metrics.json", "w"))


def write_to_excel(logs):
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["image_name", "predict", "label", "cumul_precision", "cumul_recall", "cumul_f1", "cumul_correct", "cumul_total"])
    for key, value in logs.items():
        if key not in ["final_stats", "end", "start", "class_stats"]:
            img_name = key
            predict, label, cumul_prec, cumul_recall, cumul_f1, cumul_correct, cumul_total = value["predict"]

            try:
                ws.append([img_name, str(value["predict"]), str(value["label"), str(value["cumul_precision"]), str(value["cumul_recall"), str(value["cumul_f1"]), str(value["cumul_correct"]), str(value["cumul_total"]))

            except:
                continue

    wb.save("test.xlsx")
    print("test.xlsx saved")

    print(f"Eval started : {logs['start']}, Eval ended : {logs['end']}")
    for i in range(11): # 데이터 전체가 아니라 list index 1-10 만 출력 -> 코드 결과 & 이유 확인 필요
        print(f"Class {i}, Accuracy: {logs['class_stats'][i]['acc']}, F1: {logs['class_stats'][i]['f1']}")

    print(f"Final mean Acc : {logs['final_stats']['mean_acc']}, mean F1 : {logs['final_stats']['mean_f1']}")

write_to_excel(logs)
