# Introduction

This repository contains my work on **object detection** using **YOLOv8** on the **FoD dataset**. This project focuses exclusively on experimenting with YOLOv8 for detection tasks, particularly aimed at identifying foriegn objects in images.

---

# Project Overview

### Detection

- **Base Model**: YOLOv8n.pt
- **Dataset**: FoD — A dataset consisting of images with guns as the primary object class.
- **Task**: Foriegn Object Detection

#### Dataset
- [Original Dataset](https://github.com/FOD-UNOmaha/FOD-data)
- [Modified Dataset for YOLOv8](https://universe.roboflow.com/yoloweapondetection/fod-0zljy/dataset/1)
#### Modifications
- Experimented with various hyperparameters including learning rates, weight decay, batch size.
- Performed grid search to select the best learning rate and weight decay

#### Performance:

##### On Validation set:
- **Precision**: 0.984
- **Recall**: 0.989
- **mAP@50**: 0.991
##### On Test set:
- **Precision**: 0.979
- **Recall**: 0.984
- **mAP@50**: 0.989
- **mAP@50-95**: 0.884
---

# Conclusion

This project demonstrates the versatility of YOLOv8 in detection task. Through grid search on hyperparameters, I aimed to optimize performance for detecting Foriegn objects.