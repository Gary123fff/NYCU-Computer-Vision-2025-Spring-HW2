import os
import json
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.folder import default_loader
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import get_improved_faster_rcnn_model
from dataset import COCODetection


class TestImageFolder(torch.utils.data.Dataset):
    """Custom dataset for loading test images from a folder."""
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(root, fname)
            for fname in os.listdir(root)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ], key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = default_loader(path)
        if self.transform:
            image = self.transform(image)
        image_id = int(os.path.splitext(os.path.basename(path))[0])
        return image, {"image_id": torch.tensor(image_id)}

    def __len__(self):
        return len(self.image_paths)


def collate_fn(batch):
    """Collate function for data loader."""
    return tuple(zip(*batch))


def horizontal_flip(image):
    """Horizontal flip for test-time augmentation."""
    return torch.flip(image, [2])


def reverse_boxes(boxes, width):
    """Reverse the coordinates of boxes after horizontal flipping."""
    reversed_boxes = boxes.clone()
    reversed_boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    return reversed_boxes


def visualize_predictions(model, test_loader, device, save_dir="./predictions4", conf_threshold=0.7):
    """
    Visualize and save the model predictions on the test set.
    Green bounding boxes with predicted digits and confidence scores.
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc="Visualizing Predictions"):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for img, output, target in zip(imgs, outputs, targets):
                img_np = img.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * np.array([0.229, 0.224, 0.225]) +
                          np.array([0.485, 0.456, 0.406]))
                img_np = np.clip(img_np, 0, 1)

                scores = output["scores"].cpu()
                keep_idxs = scores > conf_threshold

                image_id = int(target["image_id"].item())

                if not keep_idxs.any():
                    plt.figure(figsize=(12, 9))
                    plt.imshow(img_np)
                    plt.title(f"Image ID: {image_id} - No detections")
                    plt.savefig(os.path.join(save_dir, f"image_{image_id}_no_detections.jpg"))
                    plt.close()
                    continue

                boxes = output["boxes"][keep_idxs].cpu()
                scores = scores[keep_idxs]
                labels = output["labels"][keep_idxs].cpu()

                sorted_indices = boxes[:, 0].argsort()
                boxes = boxes[sorted_indices]
                scores = scores[sorted_indices]
                labels = labels[sorted_indices]

                fig, ax = plt.subplots(1, figsize=(12, 9))
                ax.imshow(img_np)
                all_digits = []

                for box, score, label in zip(boxes, scores, labels):
                    x_min, y_min, x_max, y_max = box.tolist()
                    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         fill=False, color='green', linewidth=2)
                    ax.add_patch(rect)

                    digit = int(label.item())
                    all_digits.append(str(digit))
                    label_text = f'{digit} ({score:.2f})'
                    text_bg = {"boxstyle": 'round,pad=0.5', "facecolor": 'yellow', "alpha": 0.7}
                    ax.text(x_min, y_min - 5, label_text, color='black', fontsize=10, bbox=text_bg)

                if all_digits:
                    predicted_number = ''.join(all_digits)
                    ax.set_title(f"Image ID: {image_id} - Predicted: {predicted_number}")

                plt.savefig(os.path.join(save_dir, f"image_{image_id}.jpg"))
                plt.close()


def predict(model_path="fasterrcn_50.pth", visualize=True):
    """
    Run predictions with test-time augmentation.
    Task1: Generate COCO format predictions (pred.json)
    Task2: Generate digit recognition results (pred.csv)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_improved_faster_rcnn_model(num_classes=11).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestImageFolder("./datasets/test", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    if visualize:

        visualize_predictions(model, test_loader, device)

    results = []

    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc="Predicting with TTA"):
            img = imgs[0].to(device)
            image_id = int(targets[0]["image_id"].item())
            output = model([img])[0]

            for box, score, label in zip(output["boxes"], output["scores"], output["labels"]):
                if score < 0.7:
                    continue
                x_min, y_min, x_max, y_max = box.tolist()
                results.append({
                    "image_id": image_id,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "score": score.item(),
                    "category_id": int(label.item()+1)
                })

    with open("pred.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


    generate_pred_csv(results=results)
    return results


def generate_pred_csv(results=None, input_json="pred.json", output_csv="pred.csv", test_dir="./datasets/test"):
    """
    Task 2: Generate digit recognition CSV file.
    Uses image_ids from test folder image filenames.
    """

    if results is None:
        with open(input_json, encoding="utf-8") as f:
            results = json.load(f)

    all_image_ids = []
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image_id = int(os.path.splitext(filename)[0])
                all_image_ids.append(image_id)
            except ValueError:
                continue

    all_image_ids.sort()

    image_dict = defaultdict(list)
    for detection in results:
        image_id = detection["image_id"]
        x_min = detection["bbox"][0]
        category_id = detection["category_id"]
        image_dict[image_id].append((x_min, category_id))

    csv_results = []
    for image_id in all_image_ids:
        digits = sorted(image_dict.get(image_id, []), key=lambda x: x[0])
        pred_label = int("".join(str(d[1] - 1) for d in digits)) if digits else -1
        csv_results.append({"image_id": image_id, "pred_label": pred_label})

    pd.DataFrame(csv_results).to_csv(output_csv, index=False)

