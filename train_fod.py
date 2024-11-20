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

# Initialize the YOLO model
model = YOLO('/home/abdullah/ml_work/yolov8n.pt')

# Hyperparameters for tuning
learning_rates = [0.001, 0.01, 0.0005]
weight_decays = [0.0005, 0.0001]

# Initialize variables for tracking the best model and performance
best_model = None
best_mAP = 0
best_model_state = None  # To store the state_dict of the best model
best_hyperparameters = None

# Create directory for saving models and graphs
os.makedirs('saved_models', exist_ok=True)
os.makedirs('graphs', exist_ok=True)

# Dataframe for logging results
columns = ['Learning Rate', 'Weight Decay', 'mAP', 'Precision', 'Recall']
results_df = pd.DataFrame(columns=columns)

# Loop through all combinations of hyperparameters
for lr, wd in product(learning_rates, weight_decays):
    print(f"Training with learning rate {lr} and weight decay {wd}")

    project_name = f"{wd}_weight_decay__{lr}_lr"

    # Start training the model
    try:
        results = model.train(
            data='/home/abdullah/ml_work/FoD_dataset/FoD_Dataset/data.yaml',
            epochs=50,
            imgsz=640,
            batch=32,
            device='0',
            project=project_name,
            lr0=lr,
            weight_decay=wd,
            patience=10
        )

        # Calculate average mAP, precision, and recall for 31 classes
        avg_mAP = sum(results.class_result(i)[2] for i in range(31)) / 31
        avg_precision = sum(results.class_result(i)[0] for i in range(31)) / 31
        avg_recall = sum(results.class_result(i)[1] for i in range(31)) / 31

        # Log the configuration and results to the dataframe
        cur_result = pd.DataFrame([{
            'Learning Rate': lr,
            'Weight Decay': wd,
            'mAP': avg_mAP,
            'Precision': avg_precision,
            'Recall': avg_recall
        }])

        results_df = pd.concat([results_df, cur_result], ignore_index=True)

    
        # Keep track of the best model based on mAP
        if avg_mAP > best_mAP:
            best_mAP = avg_mAP
            best_model_state = model.state_dict()  # Save the model's state_dict
            best_hyperparameters = {
                'Learning Rate': lr,
                'Weight Decay': wd
            }
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"OOM error with batch_size = {batch_size}, lr = {lr}, wd = {wd}, dropout = {drop}, skipping...")
            torch.cuda.empty_cache()
        else:
            raise

# Save the actual best model after all hyperparameter combinations are evaluated
if best_model_state is not None:
    best_model_path = 'saved_models/best_model.pt'
    torch.save(best_model_state, best_model_path)
    print(f"Actual best model saved at {best_model_path}")

    # Log the best hyperparameters
    print("Best Hyperparameters:")
    for key, value in best_hyperparameters.items():
        print(f"  {key}: {value}")

# Save the results dataframe to CSV
results_df.to_csv('training_results.csv', index=False)

# Plot metrics for each parameter
parameters = ['Learning Rate', 'Weight Decay']
metrics = ['mAP', 'Precision', 'Recall']

for param in parameters:
    plt.figure(figsize=(10, 6))
    for metric in metrics:
        plt.plot(
            results_df[param],
            results_df[metric],
            label=metric,
            marker='o'
        )
    plt.xlabel(param)
    plt.ylabel('Metric Value')
    plt.title(f'Performance Metrics vs {param}')
    plt.legend()
    plt.grid()
    plot_path = f'graphs/performance_metrics_vs_{param.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    print(f"Graph saved at {plot_path}")

# Display the best model's performance
print(f"Best model achieved an average mAP@50 of: {best_mAP:.4f}")

