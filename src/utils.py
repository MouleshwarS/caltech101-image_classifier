import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import torch

from typing import Dict, List
from pathlib import Path
from collections import Counter

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def plot_loss_curves(results: Dict[str, List[float]], model_type):
    """Plots training and test loss/accuracy curves from a results dictionary."""
    
    train_loss = results['train_loss']
    test_loss = results['test_loss']
    train_acc = results['train_acc']
    test_acc = results['test_acc']
    
    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plotting the loss curves
    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
    axes[0].plot(epochs, test_loss, label='Test Loss', color='orange', linewidth=2)
    axes[0].set_title(f'Loss Curves ({model_type})', weight='bold', size=14)
    axes[0].set_xlabel('Epochs', size=12)
    axes[0].set_ylabel('Loss', size=12)
    axes[0].tick_params(axis='both', labelsize=12)
    axes[0].legend(fontsize=12)
    axes[0].grid(True)

    # Plot the accuracy curves
    axes[1].plot(epochs, train_acc, label='Train Accuracy', color='blue', linewidth=2)
    axes[1].plot(epochs, test_acc, label='Test Accuracy', color='orange', linewidth=2)
    axes[1].set_title(f'Accuracy Curves ({model_type})', weight='bold', size=14)
    axes[1].set_xlabel('Epochs', size=12)
    axes[1].set_ylabel('Accuracy', size=12)
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def predict_and_plot_random_image(model, dataset, class_names, device):
    """
    Picks a random image from the dataset, makes a prediction, 
    and plots the image with its predicted and actual labels.
    """
    # Pick a random index and extract the image and label
    random_idx = random.randint(0, len(dataset) - 1)
    img_tensor, actual_label_idx = dataset[random_idx]
    
    # Add a batch dimension and move to device
    img_batch = img_tensor.unsqueeze(0).to(device)
    
    # Make a prediction
    model.eval()
    with torch.inference_mode():
        # Handle AMP if the model was trained with float16
        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            logits = model(img_batch)
            probs = torch.softmax(logits, dim=1)
            
            # Get the highest probability and its corresponding index
            pred_prob, pred_label_idx = torch.max(probs, dim=1)
    
    # Convert from tensors to standard Python values
    pred_prob = pred_prob.item()
    pred_label_idx = pred_label_idx.item()
    
    pred_class = class_names[pred_label_idx]
    actual_class = class_names[actual_label_idx]
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    img_unnorm = img_tensor.cpu() * std + mean
    img_unnorm = torch.clamp(img_unnorm, 0, 1)
    img_plot = img_unnorm.permute(1, 2, 0).numpy()
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img_plot)
    plt.axis("off")
    
    title_color = "green" if pred_class == actual_class else "red"
    title_text = f"Pred: {pred_class} | Prob: {pred_prob:.2f} | Actual: {actual_class}"
    plt.title(title_text, color=title_color, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    plt.show()


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model's state_dict to a target directory."""
    # Creating target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Creating model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name
    
    # Saving the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    
def get_advanced_metrics(model, dataloader, device):
    """Runs inference on a dataloader and returns accuracy, recall, precision, and F1 scores."""
    model.eval()
    y_true, y_pred = [], []
    
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            if device_type == "cuda":
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits = model(X)
            else:
                logits = model(X)
                
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            
    # Computing metrics using weighted average for the imbalanced Caltech101 dataset
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    
    return acc, recall, precision, f1


def get_model_size_mb(model):
    """Calculates the size of a PyTorch model in Megabytes (MB)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)

    
def plot_speed_vs_performance(df_results, models_summary, test_dataloader):
    """Plots a bubble chart comparing inference speed, accuracy, and model size."""
    plot_data = []
    num_test_images = len(test_dataloader.dataset)

    # Extract dimensions from the DataFrame and model summary
    for index, row in df_results.iterrows():
        name = row["Model Name"]
        model, _, total_test_time = models_summary[name]
        
        size_mb = get_model_size_mb(model)
        time_per_image = total_test_time / num_test_images
        accuracy_percent = float(row["Test Accuracy"]) * 100
        
        plot_data.append({
            "Model": name,
            "Size (MB)": size_mb,
            "Time per image (s)": time_per_image,
            "Accuracy (%)": accuracy_percent
        })

    df_plot = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["blue", "orange", "green", "red", "purple"]

    scatter = ax.scatter(
        x=df_plot["Time per image (s)"], 
        y=df_plot["Accuracy (%)"], 
        s=df_plot["Size (MB)"] * 3, 
        c=colors[:len(df_plot)], 
        alpha=0.7
    )

    for i, row in df_plot.iterrows():
        ax.annotate(
            row["Model"], 
            (row["Time per image (s)"], row["Accuracy (%)"]), 
            xytext=(-25, 5), 
            textcoords='offset points', 
            fontsize=11,
            fontweight='bold'
        )

    ax.set_title("CalTech101 Classifier Inference Speed vs. Performance", fontsize=16, fontweight="bold")
    ax.set_xlabel("Prediction time per image (seconds)", fontsize=14)
    ax.set_ylabel("Test accuracy (%)", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5, num=4, func=lambda s: s / 3)
    
    ax.legend(
        handles, 
        labels, 
        title="Model size (MB)", 
        loc="lower center", 
        borderpad=1.5,       
        ncol=5,              
        columnspacing=1.5,   
        labelspacing=1,
        fontsize=12,         
        title_fontsize=14 
    )

    plt.tight_layout()
    plt.show()


def plot_top_errors(model, dataloader, class_names, device, bcol, top_k=10, model_name="Model"):
    """Plots a horizontal bar chart of the most frequently confused class pairs."""
    model.eval()
    errors = []
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    
    with torch.inference_mode():
        for X, y in dataloader:
            X = X.to(device, non_blocking=True)
            
            if device_type == "cuda":
                with torch.autocast(device_type=device_type, dtype=torch.float16):
                    logits = model(X)
            else:
                logits = model(X)
                
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            actuals = y.cpu().numpy()
            
            # Incorrect predictions
            for actual, pred in zip(actuals, preds):
                if actual != pred:
                    errors.append(f"Actual: {class_names[actual]} \nPred: {class_names[pred]}")
    
    # Count the frequencies of each specific mistake
    error_counts = Counter(errors).most_common(top_k)
    
    if not error_counts:
        print(f"No errors found for {model_name}!")
        return
        
    labels, counts = zip(*error_counts)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(labels)), counts, color=bcol, edgecolor='black', alpha=0.8)

    ax.tick_params(axis='x', labelsize=14)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()  # Highest frequency at the top
    
    ax.set_xlabel("Number of Misclassifications", fontsize=12)
    ax.set_title(f"Top {top_k} Most Confused Class Pairs ({model_name})", fontsize=16, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def plot_gradcam(model, target_layer, dataset, class_names, device):
    """Generates a Grad-CAM heatmap for a random image in the dataset."""
    import numpy as np
    model.eval()
    
    idx = random.randint(0, len(dataset)-1)
    img_tensor, label_idx = dataset[idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    # Targeting the actual ground-truth class
    targets = [ClassifierOutputTarget(label_idx)]
    
    # Generating the heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_unnorm = img_tensor.permute(1, 2, 0).numpy() * std + mean
    img_unnorm = np.clip(img_unnorm, 0, 1)
    
    # Overlaying the heatmap on the image
    visualization = show_cam_on_image(img_unnorm, grayscale_cam, use_rgb=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img_unnorm)
    axes[0].set_title(f"Original Image: {class_names[label_idx]}", fontweight="bold")
    axes[0].axis('off')
    
    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM Activation Heatmap", fontweight="bold")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()