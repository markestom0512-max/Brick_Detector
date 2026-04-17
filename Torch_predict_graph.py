import torchvision.models as models

# Charge ResNet18 pré-entraîné
resnet = models.resnet18(pretrained=True)

# Regarde l'architecture
print(resnet)
