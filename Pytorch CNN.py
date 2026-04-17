import torch 
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 canal d'entrée, 8 filtres, noyau 3x3
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 8 canaux d'entrée, 16 filtres
        self.pool = nn.MaxPool2d(2, 2)  # max pooling
        self.fc = nn.Linear(16*16*16, 1)
        
    def forward(self, x):
        
        print(f"Entrée : {x.shape}")
        
        x = self.pool(torch.relu(self.conv1(x)))  # conv1 + ReLU + pooling
        print(f"Après conv1 : {x.shape}")
        
        x = self.pool(torch.relu(self.conv2(x)))  # conv2 + ReLU + pooling
        print(f"Après conv2 : {x.shape}")
        
        x = torch.flatten(x, 1)  # aplatir pour la couche fully connected
        print(f"Après aplatissement : {x.shape}")
        
        x = torch.sigmoid(self.fc(x))  # couche fully connected + sigmoïde
        print(f"Sortie : {x.shape}")
        
        return x
    
    
image = torch.randn(1, 1, 64, 64)  # batch de 1 image, 1 canal, 64x64 pixels

cnn = CNN()
sortie = cnn(image)
print(f"Sortie finale : {sortie.item():.4f}")