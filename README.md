# Detection of Foreign Objects on FoD Dataset using YOLOv8

This repository contains my work on **object detection** using **YOLOv8** on the **FoD dataset**. The project focuses on detecting **foreign objects** in images, specifically guns, through parameter tuning and model optimization.

---

## Project Overview

### Detection

- **Base Model**: YOLOv8n.pt
- **Dataset**: FoD — A dataset of images with guns as the primary object class.
- **Task**: Foreign Object Detection

#### Dataset
- [Original Dataset](https://github.com/FOD-UNOmaha/FOD-data)
- [Modified Dataset for YOLOv8](https://universe.roboflow.com/yoloweapondetection/fod-0zljy/dataset/1)

#### Modifications
- Experimented with hyperparameters such as learning rates, weight decay, and batch size.
- Performed grid search to select the best learning rate and weight decay.

#### Performance:

##### On Validation Set:
- **Precision**: 0.984
- **Recall**: 0.989
- **mAP@50**: 0.991

##### On Test Set:
- **Precision**: 0.979
- **Recall**: 0.984
- **mAP@50**: 0.989
- **mAP@50-95**: 0.884

---

## Conclusion

This project demonstrates YOLOv8's effectiveness in **foreign object detection**. Through hyperparameter optimization using grid search, I achieved high performance in detecting foreign objects with the FoD dataset.

- [Best Model](https://github.com/abdullahejazjanjua/Foriegn_object_detection/blob/main/train/weights/best.pt)

---

## Test the Model

To test the trained model, clone the repository and run the following code:

```bash
git clone https://github.com/abdullahejazjanjua/Foriegn_object_detection.git
```

Then, use the following Python script to load and test the model:

```python
from ultralytics import YOLO

# Load the best model
model = YOLO("best.pt")

# Make predictions
model.predict("path_to_images_or_video")

# Evaluate the model on the validation set
model.val()
```