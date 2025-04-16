"""
Module for training object detection models with COCO dataset.
Contains functions for model training, visualization, and learning rate scheduling.
"""
import os
import random
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from tqdm import tqdm

from dataset import COCODetection
from model import get_improved_faster_rcnn_model
from valid import validate_model


WARMUP_ITERS = 500


def collate_fn(batch):
    """
    Custom collate function for object detection data batches.
    
    Args:
        batch: A batch of data from the DataLoader
        
    Returns:
        Tuple of (images, targets) where each is a list
    """
    return tuple(zip(*batch))


def warmup_lambda(current_step):
    """
    Lambda function for warmup learning rate schedule.
    
    Args:
        current_step: Current training step
        
    Returns:
        Scaled learning rate factor
    """
    if current_step < WARMUP_ITERS:
        return float(current_step) / float(max(1, WARMUP_ITERS))
    return 1.0


def visualize_random_ground_truth(dataset, save_dir="./ground_truth", num_samples=5):
    """
    Visualize random ground truth bounding boxes from the training dataset.
    
    Args:
        dataset: Dataset containing images and bounding box information
        save_dir: Directory to save visualization results
        num_samples: Number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)


    dataset_size = len(dataset)
    random_indices = random.sample(
        range(dataset_size), min(num_samples, dataset_size))

    for i, idx in enumerate(random_indices):
      
        img, target = dataset[idx]

      
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        else:
            img_np = np.array(img)

   
        _, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img_np)

        for box, label in zip(target["boxes"], target["labels"]):
            if isinstance(box, torch.Tensor):
                x_min, y_min, x_max, y_max = box.numpy()
            else:
                x_min, y_min, x_max, y_max = box

            ax.add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                      fill=False, color='green', linewidth=2))

            
            label_text = f'{int(label)}' if isinstance(
                label, (int, float, torch.Tensor)) else str(label)
            ax.text(x_min, y_min, label_text, color='green', fontsize=12,
                   bbox={"facecolor": 'white', "alpha": 0.5})

       
        if "image_id" in target:
            image_id = int(target["image_id"].item()) if isinstance(
                target["image_id"], torch.Tensor) else target["image_id"]
            ax.set_title(f"Image ID: {image_id} (Dataset Index: {idx})")
        else:
            ax.set_title(f"Dataset Index: {idx}")

 
        save_path = os.path.join(save_dir, f"ground_truth_sample_{i+1}.jpg")
        plt.savefig(save_path)
        plt.close()

    print(f"Saved {num_samples} random ground truth visualization images to {save_dir}")


def train_model(model_path="fasterrcn_50_2.pth", get_train_transform=None, 
                get_val_transform=None):
    """
    Train an object detection model and evaluate on validation set.
    
    Args:
        model_path: Path to save the trained model
        get_train_transform: Function to get training data transforms
        get_val_transform: Function to get validation data transforms
        
    Returns:
        Trained model
    """
    print("Training model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = get_improved_faster_rcnn_model(num_classes=11).to(device)


    def prepare_data():
        """Helper function to prepare datasets and dataloaders to reduce local variable count"""
        train_dataset = COCODetection(
            './datasets/train', './datasets/train.json', transforms=get_train_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
        
      
        visualize_random_ground_truth(
            train_dataset, save_dir="./ground_truth", num_samples=5)
        
      
        val_dataset = COCODetection(
            './datasets/valid', './datasets/valid.json', transforms=get_val_transform)
        val_loader = DataLoader(val_dataset, batch_size=2,
                               shuffle=False, collate_fn=collate_fn)
        
        return train_loader, val_loader
    
    train_loader, val_loader = prepare_data()

   
    def setup_optimization():
        """Helper function to set up optimizer and schedulers to reduce local variable count"""
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=5e-5, weight_decay=0.0005)

        warmup_epochs = 1
        cosine_epochs = 7

       
        warmup_scheduler = LambdaLR(
            optimizer, lr_lambda=lambda epoch: float(epoch + 1) / warmup_epochs)

       
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=1e-6
        )

       
        lr_scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_epochs]
        )
        
        return optimizer, lr_scheduler
    
    optimizer, lr_scheduler = setup_optimization()

   
    train_losses = []
    val_losses = []
    val_maps = []
    best_val_map = 0.0

   
    for epoch in range(8):
        
        model.train()
        total_loss = 0
        print(f" Epoch {epoch+1}")
        
      
        for imgs, targets in tqdm(train_loader, desc=" Training", leave=False):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
           
            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
        
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            total_loss += losses.item()

       
        lr_scheduler.step()
        
      
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"  ➤ Train Loss: {avg_loss:.4f}")

      
        val_loss, val_metrics = validate_model(
            model, device, val_loader, num_classes=11)
        val_losses.append(val_loss)
        current_map = val_metrics["mAP"]
        val_maps.append(current_map)
        print("Validation Loss:", val_loss)
        print("mAP:", current_map)
        

        if current_map > best_val_map:
            best_val_map = current_map
            torch.save(model.state_dict(), f"{model_path}.best")
            print(f"  ➤ New best model saved (mAP: {best_val_map:.4f}).")

      
        torch.save(model.state_dict(), model_path)
        print("  ➤ Latest model saved.")

    
        plot_training_progress(train_losses, val_losses, val_maps)

 
    model.load_state_dict(torch.load(f"{model_path}.best"))
    return model


def plot_training_progress(train_losses, val_losses, val_maps):
    """
    Plot and save training and validation metrics.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        val_maps: List of validation mAP values per epoch
    """
    plt.figure(figsize=(12, 5))


    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1),
             train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1),
             val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(val_maps) + 1), val_maps,
             label='Validation mAP', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves._50_2.png", dpi=300)
    plt.close()