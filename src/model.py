import torch
import torch.nn as nn
from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights

def get_model(num_classes=2, pretrained=True):
    weights = LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
    model = lraspp_mobilenet_v3_large(weights=weights)

    if isinstance(model.classifier.low_classifier, nn.Conv2d):
        in_channels = model.classifier.low_classifier.in_channels
        model.classifier.low_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    else:
        in_channels = model.classifier.low_classifier[0].in_channels
        model.classifier.low_classifier[0] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    if isinstance(model.classifier.high_classifier, nn.Conv2d):
        in_channels = model.classifier.high_classifier.in_channels
        model.classifier.high_classifier = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    else:
        in_channels = model.classifier.high_classifier[0].in_channels
        model.classifier.high_classifier[0] = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    return model
