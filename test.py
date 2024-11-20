from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from itertools import product
import pandas as pd
import os
import matplotlib.pyplot as plt


columns = ["Precision", "Recall", "mAP@50", "mAP@50-95"]

results = pd.DataFrame(columns=columns)

# Initialize the YOLO model
model = YOLO('/home/abdullah/ml_work/0.0005_weight_decay__0.0005_lr/train/weights/best.pt')


# Test the model

val_results = model.val(data='/home/abdullah/ml_work/FoD_dataset/FoD_Dataset/data.yaml', split='test')
map50 = val_results.box.map50
map95 = val_results.box.map
precision = val_results.box.p.mean()
recall = val_results.box.r.mean()

data = pd.DataFrame([{
    "Precision" : precision,
    "Recall" : recall,
    "mAP@50" : map50,
    "mAP@50-95" : map95
    }])

results = pd.concat( [results, data], ignore_index=True)
print(results)
results.to_csv("test_results.csv", index=False)

