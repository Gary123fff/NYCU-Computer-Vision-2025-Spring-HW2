"""
Module to validate and evaluate object detection models using COCO metrics.
This includes functions for model validation, visualization and evaluation.
"""
import os
import random
import json
from typing import List, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def visualize_predictions(images, targets, outputs, save_dir, max_samples=5):
    """
    Visualize model predictions alongside ground truth.
    
    Args:
        images: List of input images tensors
        targets: List of target dictionaries with boxes and labels
        outputs: List of model prediction dictionaries
        save_dir: Directory to save visualization results
        max_samples: Maximum number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)
    for idx in range(min(len(images), max_samples)):
        image = images[idx].permute(1, 2, 0).cpu().numpy()
        image = np.clip(image * 255, 0, 255).astype(np.uint8)

        _, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

       
        for box, label in zip(targets[idx]["boxes"], targets[idx]["labels"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     edgecolor='green', linewidth=2, fill=False))
            ax.text(x1, y1 - 5, f"GT: {label.item()-1}", color='green', fontsize=10,
                    bbox={"facecolor": 'white', "alpha": 0.5})

      
        for box, label, score in zip(outputs[idx]["boxes"], outputs[idx]["labels"], outputs[idx]["scores"]):
            if score < 0.7:
                continue
            x1, y1, x2, y2 = box.cpu().detach().numpy()
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     edgecolor='red', linewidth=2, fill=False))
            ax.text(x1, y2 + 5, f"Pred: {label.item()-1} ({score:.2f})", color='red', 
                    fontsize=10, bbox={"facecolor": 'white', "alpha": 0.5})

        plt.title(f"Sample {idx + 1}")
        plt.savefig(os.path.join(save_dir, f"val_sample_{idx+1}.png"))
        plt.close()


def validate_model(model, device, dataloader, num_classes):
    """
    Validate a detection model and compute metrics.
    
    Args:
        model: The PyTorch detection model to evaluate
        device: The computation device (CPU/GPU)
        dataloader: DataLoader containing validation data
        num_classes: Number of object classes
        
    Returns:
        Tuple of (average loss, metrics dictionary)
    """
    val_loss = 0.0
    all_predictions = []
    all_targets = []
    
   
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Validating")):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
       
            model.eval()
            outputs = model(images)
            
           
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

      
            if batch_idx == 0:
                visualize_predictions(
                    images, targets, outputs, save_dir="val_vis", max_samples=5)

           
            for pred in outputs:
                all_predictions.append({
                    "boxes": pred["boxes"].detach().cpu(),
                    "scores": pred["scores"].detach().cpu(),
                    "labels": pred["labels"].detach().cpu()
                })
            for tgt in targets:
                all_targets.append({
                    "boxes": tgt["boxes"].detach().cpu(),
                    "labels": tgt["labels"].detach().cpu()
                })

    stats = calculate_coco_map(all_predictions, all_targets, num_classes=num_classes)
    avg_loss = val_loss / len(dataloader)
    return avg_loss, {"mAP": stats[0]}


def convert_to_coco_format(predictions, targets, num_classes):
    """
    Convert detection outputs to COCO format for evaluation.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        num_classes: Number of object classes
        
    Returns:
        Tuple of (ground truth COCO dict, detections COCO list)
    """
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, num_classes + 1)]
    }

    coco_dt = []
    annotation_lists = []  

    ann_id = 1
    for img_id, (pred, tgt) in enumerate(zip(predictions, targets)):
        coco_gt["images"].append({"id": img_id})

        for box, label in zip(tgt["boxes"], tgt["labels"]):
            x1, y1, x2, y2 = box.tolist()
            width, height = x2 - x1, y2 - y1
            coco_box = [x1, y1, width, height]
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(label),
                "bbox": coco_box,
                "area": width * height,
                "iscrowd": 0
            })
            ann_id += 1

        
        for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
            x1, y1, x2, y2 = box.tolist()
            width, height = x2 - x1, y2 - y1
            coco_box = [x1, y1, width, height]
            coco_dt.append({
                "image_id": img_id,
                "category_id": int(label),
                "bbox": coco_box,
                "score": float(score)
            })

    return coco_gt, coco_dt


def calculate_coco_map(predictions, targets, num_classes):
    """
    Calculate COCO mAP for detection predictions.
    
    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        num_classes: Number of object classes
        
    Returns:
        COCO evaluation statistics
    """
    coco_gt_dict, coco_dt_list = convert_to_coco_format(
        predictions, targets, num_classes)

  
    gt_path = "coco_gt.json"
    dt_path = "coco_dt.json"

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(coco_gt_dict, f)
    with open(dt_path, "w", encoding="utf-8") as f:
        json.dump(coco_dt_list, f)


    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(dt_path)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


    os.remove(gt_path)
    os.remove(dt_path)

    return coco_eval.stats