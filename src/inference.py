import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import apply_and_plot_tsne, get_device, plot_cfm

MODEL_DIR = os.path.join(os.getcwd(), "models")

# Function for prediction of the trained model
def predict(test_dataset, model_filename, criteria="CrossEntropyLoss",
            batch_size=32, shuffle=True):
    # Initialize variables for tracking performance
    running_loss, running_correct, total, step = 0, 0, 0, 0
    predictions, targets = [], []
    precision, recall = [], []
    features = []

    # Get the computation device
    device = get_device()

    # DataLoader for the test dataset
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)

    # Load the trained model, or raise an exception if not found
    model_filepath = os.path.join(MODEL_DIR, f"{model_filename}.pth")
    if os.path.isfile(model_filepath):
        print("\nLoading Saved Model for Inferencing...")
        model = torch.load(model_filepath)
    else:
        raise Exception("Model not found!!")

    # Move model to the device
    model.to(device)

    # Set the loss criterion
    criterion = getattr(nn, criteria)()

    print("\nMaking Predictions on Test Dataset...")
    # Model set to evaluation mode
    model.eval()
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader, desc=f"Testing")):
            imgs, labels = imgs.to(device), labels.to(device)   # Move data to the appropriate device
            outputs = model(imgs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get the predicted labels
            loss = criterion(outputs, labels)  # Compute the loss
            features.append(outputs.cpu().numpy())#store the features

            # Update performance tracking variables
            running_loss += loss
            step += 1
            running_correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())


    # apply and plot T-SNE
    print("Apply t-SNE...")
    flattened_features = np.concatenate(features, axis=0)
    apply_and_plot_tsne(flattened_features, targets, model_filename, mode="test")

    print("\nComputing Model Performance on Test Dataset...")
    # Calculate precision and recall
    precision = precision_score(targets, predictions, average='macro')
    recall = recall_score(targets, predictions, average='macro')
    f1score = f1_score(targets, predictions, average='macro')
    cfm = confusion_matrix(targets, predictions)

    # Print the performance metrics
    print(
        "Loss : {:.4f} | Accuracy : {:.2f}% | Precision : {:.2f}% | Recall : {:.2f}% | F1 Score : {:.4f}".\
        format(
            running_loss/step,
            (running_correct/total)*100,
            (precision)*100,
            (recall)*100,
            f1score
        )
    )

    plot_cfm(cfm, model_filename)