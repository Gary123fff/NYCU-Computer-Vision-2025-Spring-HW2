# NYCU-Computer-Vision-2025-Spring-HW2
Student ID: 313551139 

Name: 陳冠豪
## Introduction
This project aims to build a digit recognition system using **Faster R-CNN**, an advanced object detection model. The dataset consists of RGB images, each containing one or more digits. We address two main tasks:

- **Task 1**: Detect individual digits by identifying their class labels (`0–9`) and bounding boxes.
- **Task 2**: Recognize the full number by combining detected digits into a string, sorted from **left to right**.

Our approach leverages **Faster R-CNN's powerful detection capabilities**:
1. **Digit Detection** – We detect all digits in the image using the trained model.
2. **Number Reconstruction** – Detected digits are sorted by their bounding box x-coordinates to form the final number string.

This two-step design enables robust digit-level recognition and complete number interpretation.

##Method
### 1. Data Preprocessing

- **Format**: COCO-style JSON annotations

- **Paths**:
  - Training images: `./datasets/train`
  - Validation images: `./datasets/valid`
  - Annotation files: `train.json`, `valid.json`

- **Transforms**:
  - Convert images to tensors
  - Data augmentation: **Scaling** (applied only during training)

- **Custom Collate Function**:
  - Handles batches of variable-sized images required by Faster R-CNN.

- **Visualization Tool**:
  - `visualize_random_ground_truth()` randomly samples and displays images with their annotated bounding boxes for inspection.
 
### 2. Model Architecture

We build our detector using Faster R-CNN with ResNet-50 + FPN as the backbone.

- **Backbone**: ResNet-50 + FPN for multi-scale feature extraction

- **Region Proposal Network (RPN)**:
  - Custom anchor sizes: `[8, 16, 32, 32, 64]`
  - Aspect ratios: `[0.5, 1.0, 2.0]`

- **RoI Align**: MultiScaleRoIAlign with output size `7x7` and sampling ratio `2`

- **Detection Head**:
  - **Classification head**: Digit label (0–9 + background)
  - **Regression head**: Bounding box refinement

- **Input Size**:
  - Minimum: `512`
  - Maximum: `1024`

 ###  3. Training Settings

| Hyperparameter        | Value                     |
|-----------------------|---------------------------|
| Optimizer             | AdamW                     |
| Learning Rate         | 5e-5                      |
| Weight Decay          | 0.0005                    |
| Batch Size            | 2                         |
| Epochs                | 12                        |
| LR Scheduler (Epoch 1)| Warm-up (`LambdaLR`)      |
| LR Scheduler (2–12)   | `CosineAnnealingLR`       |
| Minimum LR            | 1e-6                      |
| Loss Function         | Cross-Entropy + Smooth L1 |

---
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
