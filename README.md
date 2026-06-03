# Plant-Disease-Classification-MATLAB
Tomato plant disease classification using baseline CNN and ResNet-18 transfer learning in MATLAB.
# Plant Disease Classification Using Deep Learning and Transfer Learning

This repository contains the MATLAB implementation codes for tomato plant disease classification using deep learning and transfer learning techniques.

## Project Description
In this project, tomato plant disease classification was performed using the PlantVillage dataset. Two models were implemented and compared: a baseline CNN model and a ResNet-18 transfer learning model.

## Dataset
PlantVillage Dataset:  
https://github.com/spMohanty/PlantVillage-Dataset

Only tomato disease classes were used. To reduce computational cost, 300 images were randomly selected from each class.

## Implemented Models
- Baseline CNN
- ResNet-18 Transfer Learning

## Software
- MATLAB
- Deep Learning Toolbox

## Experimental Setup
- Image size: 224 × 224 × 3
- Train / validation / test split: 70% / 15% / 15%
- Epochs: 5
- Mini-batch size: 32
- Optimizer: Adam

## Results

| Model | Accuracy |
|---|---|
| Baseline CNN | 10.00% |
| ResNet-18 Transfer Learning | 91.78% |

## Files
- `final_project.m`: Baseline CNN implementation
- `resnet_transfer_learning.m`: ResNet-18 transfer learning implementation
- `CNN_Confusion_Matrix.png`: Confusion matrix of baseline CNN
- `ResNet18_ConfusionMatrix.png`: Confusion matrix of ResNet-18
- `Final_Project_Results.xlsx`: Baseline CNN result
- `ResNet18_Results.xlsx`: ResNet-18 result

## Author
Sevinç Taşer
