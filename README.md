# CalTech-101 Image Classifier
An end-to-end image classification pipeline trained on the CalTech-101 dataset, featuring a comparative analysis of five deep learning architectures. The project culminates in a production-ready Gradio web application hosted on Hugging Face Spaces, powered by a fine-tuned ConvNeXt-Tiny model.

## 📖 Project Overview
This repository demonstrates the complete lifecycle of a computer vision project, from data preparation and model benchmarking to deployment. Five different architectures were trained and evaluated to identify the best-performing model for this specific classification task.

* **Hardware:** Intel i7 13700H / 32GB RAM / 8GB NVIDIA RTX 4060 GPU
* **Frameworks:** PyTorch, Torchvision, Gradio
* **Deployment:** Hugging Face Spaces (ZeroGPU)

## 📁 Dataset
The model is trained on the [CalTech-101](https://data.caltech.edu/records/mzrjq-6wc02) dataset, which contains images of objects belonging to 101 distinct categories, plus an additional background clutter class. 

* **Total Classes:** 102 (101 categories + 1 `BACKGROUND_Google` class)
* **Data Split:** 80% Training / 20% Testing (stratified with manual seeds for reproducibility)
* **Preprocessing:** Images are converted to RGB (to handle grayscale and RGBA inputs safely), resized to 224x224, converted to PyTorch tensors, and normalized using standard ImageNet statistics.

## 📊 Model Performance
All models were trained using PyTorch with `benchmark = True` to leverage cuDNN auto-tuning on the RTX 4060. The table below outlines the final evaluation metrics on the 20% test split:

| Model Name | Total Training Time | Total Testing Time | Test Accuracy | Test Recall | Test Precision | Test F1 Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Custom CNN | `7.68` min | `99.86` sec | `0.7563` | `0.7563` | `0.7616` | `0.7468` |
| VGG16 | `17.14` min | `199.02` sec | `0.9476` | `0.9476` | `0.9513` | `0.9452` |
| ResNet50 | `7.95` min | `112.34` sec | `0.9482` | `0.9482` | `0.9510` | `0.9479` |
| EfficientNet-V2 | `9.02` min | `121.45` sec | `0.9211` | `0.9211` | `0.9265` | `0.9200` |
| **ConvNeXt-Tiny** | **`7.60` min** | **`108.53` sec** | **`0.9712`** | **`0.9712`** | **`0.9743`** | **`0.9711`** |

**ConvNeXt-Tiny** was selected for production deployment as it achieved the highest accuracy (97.12%) and F1 Score (97.11%) with highly efficient training and inference times.

## 🚀 Deployment Pipeline
The final model is deployed as an interactive web application on Hugging Face Spaces: [Launch CalTech101-Classifier](https://huggingface.co/spaces/Phoenix3238/CalTech101-Classifier). 

### 💡 Key Implementation Details
* **Robust Image Handling:** A custom `ConvertToRGB` transformation step is included in the inference pipeline to prevent tensor dimension crashes when users upload grayscale or RGBA images.
* **Synchronized Transformation:** The Gradio app strictly mimics the evaluation-time data transformations (direct resize to 224x224 without center-cropping) to prevent domain shift and maintain the 97.12% benchmark accuracy in production.
* **Hardware Acceleration:** Inference is decorated with `@spaces.GPU` to utilize Hugging Face's ZeroGPU infrastructure, dynamically allocating CUDA resources to process incoming user requests instantly.