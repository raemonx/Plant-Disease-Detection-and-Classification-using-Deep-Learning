import os
import tempfile
import time

import numpy as np
import ray
import ray.train
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageFile
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from models import _efficientnet_b0, _googlenet, _mobilenet_v2
from utils import apply_and_plot_tsne, get_device, save_model, save_train_data

ImageFile.LOAD_TRUNCATED_IMAGES = True


def train(config, params):
    # Initialize lists to store metrics for training and validation phases
    train_summary = {
        "train": {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        },
        "valid": {
            "loss": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        },
        "lr_update_per_epoch": []
    }
    # For storing features and labels for t-SNE
    tsne_features, tsne_labels = [], []

    # Get the device (CPU/GPU) for training
    device = get_device()
    print(f"Training model on : {device}")

    # Initialize data loaders for training and validation datasets
    train_loader = DataLoader(params["train_ds"],
                              batch_size=config["batch_size"],
                              shuffle=True)
    val_loader = DataLoader(params["valid_ds"],
                            batch_size=config["batch_size"],
                            shuffle=True)

    # Select model based on configuration
    if params["model"] == "googlenet":
        model = _googlenet(params["num_classes"],
                           params["pretrained"])

    if params["model"] == "efficientnet_b0":
        model = _efficientnet_b0(params["num_classes"],
                                 params["pretrained"])

    if params["model"] == "mobilenet_v2":
        model = _mobilenet_v2(params["num_classes"],
                              params["pretrained"])

    # Print model summary
    print("Model Summary:")
    summary(model, (3, 224, 224))

    model.to(device)

    # Initialize loss function, optimizer, and learning rate scheduler
    criterion = getattr(nn, params["criterion"])()
    optimizer = getattr(optim, params["optimizer"])(
        model.parameters(),
        lr=config["lr"],
        momentum=0.9
    )
    scheduler = getattr(optim.lr_scheduler, params["lr_scheduler"])(optimizer)

    # Load existing checkpoint through `get_checkpoint()` API.
    if ray.train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            state_dicts = torch.load(
                os.path.join(loaded_checkpoint_dir, f"{params['model_filename']}.pth")
            )


    print("\n\nStarting Model Training...\n")
    start_time = time.time()
    # Main Training loop
    for epoch in range(params["num_epochs"]):
        # Initialize variables for tracking progress
        t_running_loss, t_running_correct, t_total, t_steps = 0, 0, 0, 0
        v_running_loss, v_running_correct, v_total, v_steps = 0, 0, 0, 0
        t_predictions, t_labels = [], []
        v_predictions, v_labels = [], []

        # Model set to training mode
        model.train()
        for batch_idx, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']} - Training")):
            # Moving data to device
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero gradients
            outputs = model(imgs)  # Forward pass
            _, preds = torch.max(outputs, 1)  # Get predictions
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            # Update training metrics varuiables
            t_running_loss += loss.item()
            t_steps += 1
            t_running_correct += torch.sum(preds == labels.data).item()
            t_total += labels.size(0)
            t_predictions.extend(preds.cpu().numpy())
            t_labels.extend(labels.cpu().numpy())

            if epoch == params["num_epochs"] - 1:
                tsne_features.append(outputs.cpu().detach().numpy())  # Store the features
                tsne_labels = t_labels  # Assuming you want to capture labels from the loop

        # Compute precision and recall for training
        t_precision = precision_score(t_labels, t_predictions, average='macro')
        t_recall = recall_score(t_labels, t_predictions, average='macro')
        t_f1score = f1_score(t_labels, t_predictions, average='macro')

        # Update lists with epoch metrics for training
        train_summary['train']['loss'].append(t_running_loss / t_steps)
        train_summary['train']['accuracy'].append(t_running_correct / t_total)
        train_summary['train']['precision'].append(t_precision)
        train_summary['train']['recall'].append(t_recall)
        train_summary['train']['f1_score'].append(t_f1score)

        # Model set to evaluation mode
        model.eval()
        with torch.no_grad():  # Without gradient computation
            for batch_idx, (imgs, labels) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{params['num_epochs']} - Validation")):
                # Moving data to device
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)  # Forward pass
                _, preds = torch.max(outputs, 1)  # Get predictions
                loss = criterion(outputs, labels)  # Compute loss

                v_running_loss += loss.item()
                v_steps += 1
                v_running_correct += torch.sum(preds == labels.data).item()
                v_total += labels.size(0)

                v_predictions.extend(preds.cpu().numpy())
                v_labels.extend(labels.cpu().numpy())

        # Scheduler step based on validation loss
        scheduler.step(v_running_loss / v_steps)
        train_summary['lr_update_per_epoch'].append(scheduler.get_last_lr()[0])

        # Compute precision and recall for validation
        v_precision = precision_score(t_labels, t_predictions, average='macro')
        v_recall = recall_score(t_labels, t_predictions, average='macro')
        v_f1score = f1_score(v_labels, v_predictions, average='macro')

        # Update lists with epoch metrics for validation
        train_summary['valid']['loss'].append(v_running_loss / v_steps)
        train_summary['valid']['accuracy'].append(v_running_correct / v_total)
        train_summary['valid']['precision'].append(v_precision)
        train_summary['valid']['recall'].append(v_recall)
        train_summary['valid']['f1_score'].append(v_f1score)


        # Metrics for hyperparameter tuning with Ray
        if params["tune_hyperparams"]:
            # Here we save a checkpoint. It is automatically registered with
            # Ray Tune and will potentially be accessed through in ``get_checkpoint()``
            # in future iterations.
            # Note to save a file like checkpoint, you still need to put it under a directory
            # to construct a checkpoint.
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                path = os.path.join(checkpoint_dir, f"{params['model_filename']}.pth")
                torch.save(model, path)
                checkpoint = ray.train.Checkpoint.from_directory(checkpoint_dir)
                ray.train.report(
                    {"loss": (v_running_loss / v_steps), "accuracy": v_running_correct / v_total},
                    checkpoint=checkpoint,
                )

        # Print epoch summary
        print(
            """\nEpoch {}/{} -> Training -> Loss : {:.4f} | Accuracy : {:.4f} | Precision : {:.4f} | Recall : {:.4f} | F1 Score : {:.4f}
               Validation -> Loss : {:.4f} | Accuracy : {:.4f} | Precision : {:.4f} | Recall : {:.4f} | F1 Score: {:.4f}\n""".\
            format(
                epoch + 1,
                params["num_epochs"],
                t_running_loss/t_steps,
                t_running_correct/t_total,
                t_precision,
                t_recall,
                t_f1score,
                v_running_loss/v_steps,
                v_running_correct/v_total,
                v_precision,
                v_recall,
                v_f1score
            )
        )
    end_time = time.time()
    print("Completed Model Training...\n\n")

    print("="*20)
    print(f"Experiment Summary: {params['model_filename'].split('_')[0]}-{params['model_filename'].split('_')[1]}")
    print(f"Time Taken Per Epoch : {(end_time-start_time)/params['num_epochs']:.0f}s")
    print(f"Avg. Training Loss : {np.mean(train_summary['train']['loss']):.4f}")
    print(f"Avg. Training Accuracy : {np.mean(train_summary['train']['accuracy'])*100:.2f}%")
    print(f"Avg. Training Precision : {np.mean(train_summary['train']['precision'])*100:.2f}%")
    print(f"Avg. Training Recall : {np.mean(train_summary['train']['recall'])*100:.2f}%")
    print(f"Avg. Validation Loss : {np.mean(train_summary['valid']['loss']):.4f}")
    print(f"Avg. Validation Accuracy : {np.mean(train_summary['valid']['accuracy'])*100:.2f}%")
    print(f"Avg. Validation Precision : {np.mean(train_summary['valid']['precision'])*100:.2f}%")
    print(f"Avg. Validation Recall : {np.mean(train_summary['valid']['recall'])*100:.2f}%")
    print("="*20)

    # apply and plot T-SNE
    print("Applying t-SNE on features from last epoch...")
    flattened_features = np.concatenate(tsne_features, axis=0)
    apply_and_plot_tsne(flattened_features, tsne_labels, params['model_filename'], mode="train")

    print("\nSaving Training & Validation Metrics...")
    save_train_data(params["model_filename"], train_summary)

    print("\nSaving Model...\n")
    save_model(model, params['model_filename'])