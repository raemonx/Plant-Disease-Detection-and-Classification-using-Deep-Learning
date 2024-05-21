import json
import os
from tabnanny import verbose

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_roc
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve

MODEL_DIR = os.path.join(os.getcwd(), "models")
OUTPUT_DIR = os.path.join(os.getcwd(), "outputs")


# Check for the availability of a GPU device for training,
#   falling back on MPS (Apple Silicon GPU) if available,
#   or CPU otherwise. Returns the device object.
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def setup_output_dir():
    # Ensure the output directory exists
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print("Created Models Directory...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print("Created Output Directory...")


# Save the given model to disk at the specified filepath.
def save_model(model, model_filename):
    filepath = os.path.join(MODEL_DIR, f'{model_filename}.pth')
    torch.save(model.to("cpu"), filepath)


#  Save the training and validation accuracy and loss plots to disk.
def save_plots(train_acc, valid_acc, train_loss, valid_loss, filename):
    # Ensure the output directory exists
    if not os.path.exists(os.path.join(os.getcwd(), "outputs")):
        os.makedirs(os.path.join(os.getcwd(), "outputs"))

    # Plot and save the accuracy graph plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), f"outputs/accuracy_{filename}.png"))

    # Plot and save the loss graph
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(os.getcwd(), f"outputs/loss_{filename}.png"))


# Plot the Receiver Operating Characteristic (ROC) curve for multi-class classification and saves the plot to disk.
def plot_ROC(softmax_probs, num_classes, true_labels_one_hot, filename):
    # Ensure the output directory exists
    if not os.path.exists(os.path.join(os.getcwd(), "outputs")):
        os.makedirs(os.path.join(os.getcwd(), "outputs"))

    plt_roc.figure(figsize=(10, 7))
    for i in range(num_classes):
        fpr, tpr, thresholds = roc_curve(true_labels_one_hot[:, i], softmax_probs[:, i])
        auc = roc_auc_score(true_labels_one_hot[:, i], softmax_probs[:, i])
        plt_roc.plot(fpr, tpr, label=f'Class {i} (AUC = {auc:.2f})')

    plt_roc.title('Multiclass ROC Curve')
    plt_roc.xlabel('False Positive Rate')
    plt_roc.ylabel('True Positive Rate')
    plt_roc.legend(loc='best')
    plt_roc.plot([0, 1], [0, 1], 'k--')
    plt_roc.savefig(os.path.join(os.getcwd(), f"outputs/ROC_{filename}.png"))

def plot_cfm(cfm, filename):
    figsize = (5, 5) if len(cfm) == 5 else (10, 10)
    tick_range = range(0, 30, 1)
    if len(cfm) == 5:
        tick_range = range(0, 5, 1)
    if len(cfm) == 15:
        tick_range = range(0, 15, 1)

    fig, ax = plt.subplots(figsize=figsize)
    cfm_im = ax.imshow(cfm, cmap='viridis')
    ax.set_xticks(tick_range)
    ax.set_yticks(tick_range)
    for i in range(len(cfm)):
        for j in range(len(cfm)):
            text = ax.text(j, i, cfm[i, j],
                        ha="center", va="center", color="w")
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}_cfm.png"))


def save_train_data(model, train_summary):
    data = {}

    filepath = os.path.join(OUTPUT_DIR, "train_summary.json")

    if os.path.isfile(filepath):
        with open(filepath, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print("Error: Invalid JSON format in {}".\
                      format(filepath))

    data.update({
        f"{model}_train_loss": train_summary['train']['loss'],
        f"{model}_train_accuracy": train_summary['train']['accuracy'],
        f"{model}_valid_loss": train_summary['valid']['loss'],
        f"{model}_valid_accuracy": train_summary['valid']['accuracy'],
        f"{model}_lr_update_per_epoch": train_summary['lr_update_per_epoch']
    })

    # Open the file in write mode and dump the updated JSON data
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)  # Add indentation for readability (optional)


def apply_and_plot_tsne(features, labels, filename, mode):
    tsne = TSNE(n_components=2, verbose=0, perplexity=45, n_iter=1000)
    tsne_results = tsne.fit_transform(features)

    plt.figure(figsize=(10, 10))
    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=f'Class {label}', alpha=0.5)
    plt.legend()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}_tsne_{mode}.png")
