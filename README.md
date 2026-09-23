# CalTech-101 Image Classifier 🚀
A deep learning image classification pipeline utilizing a fine-tuned **ConvNeXt** architecture to classify images into 102 categories from the CalTech-101 dataset.

## Live Demo 🌐
Experience the deployed model in action via an interactive Gradio web interface on Hugging Face Spaces: 
**[Launch CalTech-101 Classifier](https://huggingface.co/spaces/Phoenix3238/CalTech101-Classifier)**

## Model Evaluation and Hardware 📊

### Hardware & Environment
* **Compute:** Intel i7 13700H machine equipped with 32 GB of RAM and an 8 GB NVIDIA RTX 4060 Laptop GPU
* **Framework:** PyTorch 2.13.0+cu132
* **Environment:** JupyterLab

### Comparative Analysis
The exploratory notebook in this repository contains a comprehensive comparative analysis of 5 different computer vision architectures.

| Model Name | Total Training Time | Total Testing Time | Test Accuracy | Test Recall | Test Precision | Test F1 Score |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Custom CNN | `6.04` min | `55.68` sec | `0.7494` | `0.7494` | `0.7582` | `0.7399` |
| VGG16 | `16.78` min | `198.14` sec | `0.9447` | `0.9447` | `0.9506` | `0.9429` |
| ResNet50 | `6.51` min | `99.18` sec | `0.9476` | `0.9476` | `0.9505` | `0.9473` |
| EfficientNet-V2 | `9.10` min | `134.51` sec | `0.9211` | `0.9211` | `0.9265` | `0.9200` |
| **ConvNeXt** | **`6.31` min** | **`107.51` sec** | **`0.9712`** | **`0.9712`** | **`0.9743`** | **`0.9711`** |

**ConvNeXt-Tiny** was selected for final deployment due to its superior performance metrics and efficient feature extraction on the CalTech-101 dataset. During fine-tuning, the base architecture's final classification layer was structurally modified with an `nn.Sequential` block (Dropout + Linear) to map to the 102 distinct output classes (including the background class).

## Project Structure 📁
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
```