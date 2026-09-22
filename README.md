# CalTech-101 Image Classifier 🚀

A deep learning image classification pipeline utilizing a fine-tuned **ConvNeXt** architecture to classify images into 102 categories from the CalTech-101 dataset.

## 🌐 Live Demo
Experience the deployed model in action via an interactive Gradio web interface on Hugging Face Spaces: 
**[Launch CalTech-101 Classifier](https://huggingface.co/spaces/Phoenix3238/CalTech101-Classifier)**

## 📊 Model Selection & Architecture
The exploratory notebook in this repository contains a comprehensive comparative analysis of 5 different computer vision architectures:
* Custom CNN
* VGG-16
* ResNet-50
* EfficientNet
* ConvNeXt-Tiny

**ConvNeXt-Tiny** was ultimately selected for final deployment due to its superior performance metrics and efficient feature extraction on the CalTech-101 dataset. During fine-tuning, the base architecture's final classification layer was structurally modified with an `nn.Sequential` block (Dropout + Linear) to map to the 102 distinct output classes (including the background class).

## 📁 Project Structure
```text
CalTech101_Image_Classifier/
├── data/                  # Raw CalTech-101 dataset (Git-ignored)
├── notebooks/             # Comparative model analysis and training experiments
├── src/                   # Modular source code (data setup, model definitions, engine, utils)
├── models/                # Saved .pth model weights (Git-ignored)
├── huggingface_space/     # Extracted deployment assets (app.py, requirements.txt)
├── .gitignore             # Ignored directories and large binaries
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation