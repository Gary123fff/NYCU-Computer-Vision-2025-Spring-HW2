from train import train_model
from pred import predict

import torchvision.transforms.functional as F


class DetectionTransform:
    def __init__(self, size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], train=True):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.mean = mean
        self.std = std
        self.train = train  

    def __call__(self, image, target):
  
        original_width, original_height = image.size
        boxes = target["boxes"]

        if self.train:
            image = F.resize(image, self.size)

      
            scale_x = self.size[0] / original_width
            scale_y = self.size[1] / original_height

        
            if len(boxes) > 0:
                boxes = boxes.clone()
                boxes[:, 0] *= scale_x 
                boxes[:, 1] *= scale_y 
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y  
        else:
            image = F.resize(image, self.size)

        
            scale_x = self.size[0] / original_width
            scale_y = self.size[1] / original_height

            if len(boxes) > 0:
                boxes = boxes.clone()
                boxes[:, 0] *= scale_x
                boxes[:, 1] *= scale_y
                boxes[:, 2] *= scale_x
                boxes[:, 3] *= scale_y

  
        target["boxes"] = boxes

        if len(boxes) > 0:
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            target["boxes"] = boxes[keep]
            target["labels"] = target["labels"][keep]
            target["area"] = target["area"][keep]
            target["iscrowd"] = target["iscrowd"][keep]


        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.mean, std=self.std)

        return image, target


def get_train_transform(size=(256, 256)):
    return DetectionTransform(size, train=True)


def get_val_transform(size=(256, 256)):
    return DetectionTransform(size, train=False)


if __name__ == "__main__":

    train_model(get_train_transform=get_train_transform(),get_val_transform=get_val_transform())
  
    results = predict(visualize=False)
