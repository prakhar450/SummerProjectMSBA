# Image Score Prediction with ResNet-50

Transfer learning project that predicts continuous quality scores from images using a fine-tuned ResNet-50 model.

## Tech Stack

- **Framework:** PyTorch
- **Model:** ResNet-50 (pretrained on ImageNet)
- **Training:** Transfer learning with L1 loss, Adam optimizer, StepLR scheduler
- **Data:** Custom dataset with CSV-based image-score pairs

## Approach

1. Load a pretrained ResNet-50 and replace the final fully connected layer with a single-output regression head
2. Apply data augmentation: RandomPerspective, RandomRotation, GaussianBlur, CenterCrop
3. Fine-tune with Adam (lr=0.01) and StepLR (decay 0.3 every 10 epochs)
4. Evaluate using Mean Absolute Error (L1 Loss)

## Getting Started

### Environment Setup

```bash
conda env create -f environment.yml
conda activate train_sl
```

### Training

```bash
python SL_Resnet.py
```

Or use the shell script with custom parameters:

```bash
bash run_supervised_learning.sh <model_name> <pretrained> <csv_name> <epochs> <batch_size>
```

### Classification Variant

```bash
python Classification_SL_Resnet.py
```

## Project Structure

```
SL_Resnet.py                  # Main supervised learning script
Classification_SL_Resnet.py   # Classification variant
Load_classification_model.py  # Model loading utility
train_resnet.py               # Training pipeline
full_train_resnet.py          # Full training script
GenerateImageScoreDF.ipynb    # Data preparation notebook
ImageScoresSL.ipynb            # Analysis notebook
environment.yml               # Conda environment
models_hub/                   # Modular implementation
  CustomDataset.py            # Dataset class
  CustomModel.py              # Model architecture
  TrainModel.py               # Training loop
  main.py                     # Entry point
```

## Key Details

- **Loss function:** L1Loss (MAE) for regression, CrossEntropyLoss for classification
- **Optimizer:** Adam with learning rate 0.01
- **Scheduler:** StepLR with gamma=0.3, step_size=10
- **Augmentation pipeline:** RandomPerspective, RandomRotation(30), GaussianBlur, CenterCrop(224)
