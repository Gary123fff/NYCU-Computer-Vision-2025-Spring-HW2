# NYCU-Computer-Vision-2025-Spring-HW2
Student ID: 313551139 

Name: 陳冠豪
## Introduction
# Image Classification Model using ResNeXt-101_64x4d

This repository contains an image classification model built using the pretrained `ResNeXt-101_64x4d` model. The goal is to achieve high performance on image classification tasks by applying advanced techniques such as data augmentation, class balancing, and smart training strategies.

 *Model Architecture*

- **Backbone**: `ResNeXt-101_64x4d` from `torchvision.models` is used as the backbone, providing strong feature extraction capabilities.
- **Global Average Pooling**: The model ends with Global Average Pooling (GlobalAvgPool) to handle variable input image sizes effectively.

 *Loss Function*

- **Focal Loss**: This loss function is used to address class imbalance by focusing more on hard-to-classify samples, improving model performance on imbalanced datasets.

 *Data Augmentation*

- **MixUp & CutMix**: These techniques are applied to blend images and labels, helping the model generalize better.
- **AutoAugment**: This method introduces more variations to the training images, further improving the model's robustness.

 *Data Loading*

- **Balanced Dataloader**: The `get_balanced_dataloader()` function uses `WeightedRandomSampler` to ensure class balance in each batch, improving performance on imbalanced datasets.

 *Training Process*

- **Optimizer**: AdamW optimizer is used, with different learning rates for different layers in ResNeXt-101 to optimize performance.
- **OneCycleLR**: This learning rate scheduler dynamically adjusts the learning rate during training to help the model avoid local minima.
- **Early Stopping**: The model monitors validation loss and stops training early if overfitting is detected, ensuring better generalization.

 *Goal*

The model is designed to improve generalization by utilizing smart data augmentation strategies, dynamic learning rate adjustment, and handling class imbalance through Focal Loss. These techniques combine to create a robust image classification model that performs well on challenging datasets.
## How to install
1. Clone the repository
```
git clone https://github.com/Gary123fff/NYCU-Computer-Vision-2025-Spring-HW1.git
cd NYCU-Computer-Vision-2025-Spring-HW1
```
2. Create a virtual environment
```
conda env create -f environment.yml
conda activate cv
```

3. Download the dataset 
- Download the dataset from the [LINK](https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view)
- Put it in the following structure
```
NYCU-Computer-Vision-2025-Spring-HW1
├── data
│   ├── test
│   ├── train
│   └── val
├── datas
│   ├── dataset.py
├── models
│   ├── resxnet_model.py
├── utils
│   ├── augmentations.py
│   ├── focal_loss.py
│   ├── metrics.py
│   .
│   .
│   .
├── inference.py
├── main.py
├── train.py
.
.
.
```
4. Train and test
if you want to choose train or test, you can change the parameter in main.py and call:
```
python main.py
```
## Performance snapshot
![Alt Text](best_model_v8.png)

### Performance
![Alt Text](per.png)
|                  | Accuracy(%)                                         |
|------------------|-----------------------------------------------------|
| Validation       | 92                                                  |
| Public Test      | 94                                                  |
