import os
import json

labels = json.load(open("labels.json", 'r'))

metric_json = []
for filename in os.listdir():
    if "metrics" in filename and filename.endswith(".json"):
        metric_json.append(filename)

metric_json = sorted(metric_json, key=lambda x: int(x.strip("metrics_").strip(".json")))

import pandas as pd
data_miou = {label:[] for label in labels}
data_map =  {label:[] for label in labels}
data_macc =  {label:[] for label in labels}
data_mmiou =  {label:[] for label in labels}
for filename in metric_json:
    with open(filename, "r") as f:
        metric_log = json.load(f)
        epoch = filename.strip("metrics_").strip(".json")
        for metric_name, metrics in metric_log.items():
            #print(f"Metric {metric_name}")
            
            for label, value in metrics.items():
                if metric_name  == "mIOU":
                    data_miou[label].append(value)
                elif metric_name == "mAP":
                    data_map[label].append(value)
                elif metric_name =="meanAcc":
                    data_macc[label].append(value)
                

                #print(f"{label} : {value}")
                #data[label][metric_name].append(value)

            for diff in set(labels)-set(metrics.keys()):
                #print(f"{diff} :  missing")
                if metric_name  == "mIOU":
                    data_miou[label].append(0)
                elif metric_name == "mAP":
                    data_map[label].append(0)
                elif metric_name =="meanAcc":
                    data_macc[label].append(0)
        

#print(data_miou)
def save_to_json(json_data,metric_name):
    with open(metric_name+"_col.json", "w") as f:
        json.dump(json_data, f)
for data, metric_name in zip((data_miou, data_map, data_macc), ("mIOU", "mAP", "mAcc")):
    save_to_json(data, metric_name)

                 



