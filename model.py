"""Custom Faster R-CNN model with modified anchor generator and ROI pooling."""
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


def get_improved_faster_rcnn_model(num_classes, freeze_backbone=True):
    """
    Returns a customized Faster R-CNN model.

    Args:
        num_classes (int): Number of object classes (including background).
        freeze_backbone (bool): Whether to freeze the backbone. (Currently unused.)

    Returns:
        torch.nn.Module: The customized Faster R-CNN model.
    """
 
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    base_model = fasterrcnn_resnet50_fpn_v2(weights=weights)
    backbone = base_model.backbone


    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (32,), (64,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )


    roi_pooler = MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=512,
        max_size=1024
    )


    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
