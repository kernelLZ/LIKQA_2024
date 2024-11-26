
# LIKQA: Lightweight Image Data Quality Assessment via Iterative Optimization and KAN-Based Model

This repository contains the implementation of <<LIKQA: Lightweight Image Data Quality Assessment via Iterative Optimization and KAN-Based Models>>. 
The code is modularized for better clarity and usability, and it supports training, validation, and testing pipelines.

## File Structure

### Main Files and Scripts
- **config.py**: Contains the configuration settings for the training, validation, and testing pipelines, including model parameters and paths.
- **loss.py**: Defines custom loss functions used during model training.
- **mobile_mlp_net.py**: Implements a Mobile Multi-Layer Perceptron (MLP) neural network architecture.
- **net.py**: Contains the definition of additional neural network architectures or utilities.
- **train.py**: The main script for training the model, including data loading, model initialization, and logging.
- **train_dataloader.py**: Defines the data loading logic, including dataset preprocessing and augmentation.
- **__init__.py**: Marks the directory as a Python package and initializes necessary imports.

### Dataset and Labels
- **train_label_score.txt**: Contains labels and corresponding scores for the training data.
- **label.json**: JSON file mapping class indices to their corresponding labels.
- **val_label_score.txt**: Contains labels and corresponding scores for the validation data.
- **label.json**: JSON file mapping class indices to their corresponding labels.
- **test_label_score.txt**: Contains labels and corresponding scores for the test data.
- **label.json**: JSON file mapping class indices to their corresponding labels.
(The specific images can be downloaded from the public addresses of the four datasets mentioned in the paper.)

### Model File
- **LIKQA_241120.pt**: This is our pre-trained model file.

### Requirements
- **requirements.txt**.

### Directory Structure
- **img/**: Directory containing image data for training, validation, and testing. Ensure that the `train`, `val`, and `test` subdirectories are organized as follows:
  - `train/`: Contains training images.
  - `val/`: Contains validation images.
  - `test/`: Contains test images.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training
To start training the model, use the following command:
```bash
python train.py
```

### Configuration
Modify the `config.py` file to adjust model parameters, file paths, or hyperparameters.

### Dataset Preparation
1. Place the dataset files (`train_label_score.txt`, `label.json`, and image directories) in the correct structure as specified above.
2. Ensure the `config.py` file points to the correct paths.
