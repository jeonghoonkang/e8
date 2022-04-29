import numpy as np 
import json
from tqdm import tqdm
def compute_iou(cand_box, gt_box):
    # Calculate intersection areas
    x1 = np.maximum(cand_box[0], gt_box[0])
    y1 = np.maximum(cand_box[1], gt_box[1])
    x2 = np.minimum(cand_box[2], gt_box[2])
    y2 = np.minimum(cand_box[3], gt_box[3])
    
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    
    cand_box_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
    gt_box_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = cand_box_area + gt_box_area - intersection
    
    iou = intersection / union
    return iou

print("Reading detailed_metrics.json...")
f = open("detailed_metrics.json", "r")
metrics = json.load(f)
f.close()
print("Finished reading")

from sklearn.metrics import average_precision_score
def analysis(metrics):
    logs = {}
    print(f"Eval started : {metrics['start']}, Eval ended : {metrics['end']}")
    for image_name, result in tqdm(metrics.items(), desc="reading image result"):
        if isinstance(result, int):
            continue
        if image_name not in logs:
            logs[image_name] = {}
            
        for class_name, stats in result.items():
            if class_name not in logs[image_name]:
                logs[image_name][class_name] = {}
                logs[image_name][class_name]["gt_label"] = []
                logs[image_name][class_name]["label"] = []
                logs[image_name][class_name]["gt_bbox"] = []
                logs[image_name][class_name]["bbox"] = []
                logs[image_name][class_name]["conf"] = []
                logs[image_name][class_name]["iou"] = []
                logs[image_name][class_name]["correct"] = []
                
            gt_bbox = stats['gt_bbox']
            gt_label = stats['gt_label']
            pred_bbox = stats['bbox']
            pred_label = stats['label']
            conf = stats['conf']
            for i in range(len(gt_bbox)):
                for j in range(len(pred_bbox)):
                    iou = compute_iou(gt_bbox[i], pred_bbox[j])
                    if iou > 0.5:
                        logs[image_name][class_name]["gt_label"].append(gt_label[i])
                        logs[image_name][class_name]["gt_bbox"].append(gt_bbox[i])
                        logs[image_name][class_name]["label"].append(pred_label[j])
                        logs[image_name][class_name]['bbox'].append(pred_bbox[j])
                        logs[image_name][class_name]['conf'].append(conf[j])
                        logs[image_name][class_name]["iou"].append(iou)
                        logs[image_name][class_name]["correct"].append(pred_label[j] == gt_label[i])
                                                

    return logs

def write_to_excel(metrics):
    from openpyxl import Workbook
    is_correct_by_class = {}
    conf_by_class = {}
    iou_by_class = {}

    # 220429 변수 추가
    analysis_result = []

    # wb = Workbook()
    # ws = wb.active
    # ws.append(["image_name", "correct", "gt_label", "gt_bbox", "label","bbox", "conf", "iou", "cumul_average_iou", "cumul_average_ap", "", 'Class_name', 'Average IoU', 'Average Precision', "", 'mAP', 'mIoU'])
    for image_name, result in tqdm(analysis(metrics).items(), desc="image analysis"):
        if isinstance(result, int):
            continue

        for class_name, stats in result.items():
            if class_name not in is_correct_by_class:
                is_correct_by_class[class_name] = []
            
            if class_name not in conf_by_class:
                conf_by_class[class_name] = []
            
            if class_name not in iou_by_class:
                iou_by_class[class_name] = []

            for i in range(len(stats['label'])):
                try:
                    is_correct_by_class[class_name].append(stats['correct'][i])
                    conf_by_class[class_name].append(stats["conf"][i])
                    iou_by_class[class_name].append(stats["iou"][i])

                    cumul_average_iou = np.mean(iou_by_class[class_name])
                    cumul_average_ap = average_precision_score(is_correct_by_class[class_name], conf_by_class[class_name])
                    # ws.append([image_name, str(stats["correct"][i]), str(stats["gt_label"][i]), str(stats['gt_bbox'][i]), str(stats['label'][i]), str(stats['bbox'][i]), str(stats['conf'][i]), str(stats['iou'][i]), str(cumul_average_iou), str(cumul_average_ap)])
                    analysis_result.append([image_name, str(stats["correct"][i]), str(stats["gt_label"][i]), str(stats['gt_bbox'][i]), str(stats['label'][i]), str(stats['bbox'][i]), str(stats['conf'][i]), str(stats['iou'][i]), str(cumul_average_iou), str(cumul_average_ap)])

                except:
                    continue
    
    
    mAP = 0
    mIoU = 0
    count = 0
    cnt = 0
    for class_name in is_correct_by_class:
        is_cor = is_correct_by_class[class_name]
        conf = conf_by_class[class_name]
        ious = iou_by_class[class_name]

        class_average_iou = np.mean(ious)
        class_average_ap = average_precision_score(is_cor, conf)

        print(f"Class: {class_name}, Average IoU: {class_average_iou}, Average Precision: {class_average_ap}")
        analysis_result[i].extend(["", class_name, class_average_iou, class_average_ap])
        cnt+=1


        mIoU += class_average_iou
        mAP += class_average_ap
        count += 1
    print(f"Final mAP :{mAP/count}, Final mIoU : {mIoU/count}")
    analysis_result[0].extend(["", mAP/count, mIoU/count])

    wb = Workbook()
    ws = wb.active
    ws.append(["image_name", "correct", "gt_label", "gt_bbox", "label","bbox", "conf", "iou", "cumul_average_iou", "cumul_average_ap", "", 'Class_name', 'Average IoU', 'Average Precision', "", 'mAP', 'mIoU'])
    for list_elements in analysis_result:
        ws.append(list_elements)
    # wb.save("test.xlsx")
    wb.save("SSD_test_result.xlsx")


write_to_excel(metrics)
