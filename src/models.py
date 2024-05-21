import torch.nn as nn
import torchvision.models as models


def _googlenet(num_classes, pretrained, fine_tune=True):
    print("Model : GoogLeNet")
    if pretrained:
        print("Loading pre-trained weights...")
    else:
        print("Not loading pre-trained weights...")

    model = models.googlenet(pretrained=pretrained)
    model.aux_logits = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(num_features=num_ftrs),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=256),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    if fine_tune:
        print("Fine-tuning all the layers...")
        for params in model.parameters():
            params.requires_grad = True
    else:
        print("Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    return model

def _efficientnet_b0(num_classes, pretrained, fine_tune=True):
    print("Model : EfficientNet B0")
    if pretrained:
        print("Loading pre-trained weights...")
    else:
        print("Not loading pre-trained weights...")

    model = models.efficientnet_b0(pretrained=pretrained)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(num_features=num_ftrs),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=256),
        nn.Linear(256, num_classes)
    )

    if fine_tune:
        print("Fine-tuning all the layers...")
        for params in model.parameters():
            params.requires_grad = True
    else:
        print("Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    return model

def _mobilenet_v2(num_classes, pretrained, fine_tune=True):
    print("Model : MobileNetV2")
    if pretrained:
        print("Loading pre-trained weights...")
    else:
        print("Not loading pre-trained weights...")

    model = models.mobilenet_v2(pretrained=pretrained)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.BatchNorm1d(num_features=num_ftrs),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=256),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes)
    )

    if fine_tune:
        print("Fine-tuning all the layers...")
        for params in model.parameters():
            params.requires_grad = True
    else:
        print("Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    return model