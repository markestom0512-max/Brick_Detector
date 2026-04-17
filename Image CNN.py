import cv2 as cv
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader, random_split

class LegoDataset(Dataset):
    def __init__(self, dossier_lego, dossier_pas_lego):
        # liste tous les chemins d'images et leurs labels
        self.images = []
        self.labels = []
        
        # images lego → label 1
        for nom in os.listdir(dossier_lego):
            if nom.endswith(".jpg"):
                self.images.append(os.path.join(dossier_lego, nom))
                self.labels.append(1)
        
        # images pas_lego → label 0
        for nom in os.listdir(dossier_pas_lego):
            if nom.endswith(".jpg"):
                self.images.append(os.path.join(dossier_pas_lego, nom))
                self.labels.append(0)
    
    def __len__(self):
        return len(self.images)  # nombre total d'images
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)  # lire en niveaux de gris
        img = cv.resize(img, (64, 64))  # redimensionner à 64x64 pixels
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # normaliser et ajouter une dimension de canal
        return img_tensor, label
    
dataset = LegoDataset(
    dossier_lego=r"C:\Fichiers python\OpenCv\dataset\dataset_cnn\lego",
    dossier_pas_lego=r"C:\Fichiers python\OpenCv\dataset\dataset_cnn\pas_lego"
)

train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

# PARTIE 2 — DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=16,   # 16 images en même temps
    shuffle=True     # mélange les images à chaque epoch
)
# PARTIE 3 — Entraînement du CNN
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size = 3, padding=1)  # 1 canal d'entrée, 8 filtres, noyau 3x3
        self.conv2 = nn.Conv2d(8, 16, kernel_size = 3, padding=1) # 8 canaux d'entrée, 16 filtres
        self.pool = nn.MaxPool2d(2, 2)  # max pooling
        self.fc = nn.Linear(16*16*16, 1)  # couche fully connected
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # conv1 + ReLU + pooling
        x = self.pool(torch.relu(self.conv2(x)))  # conv2 + ReLU + pooling
        x = torch.flatten(x, 1)  # aplatir pour la couche fully connected
        x = torch.sigmoid(self.fc(x))  # couche fully connected + sigmoïde
        return x
    
reseau    = CNN()
optimizer = torch.optim.Adam(reseau.parameters(), lr=0.001)
critere   = nn.BCELoss()

for epoch in range(50):
    loss_totale = 0
    for imgs, labels in train_loader:  # ← itère sur les batches
        labels = labels.float().unsqueeze(1)  # forme (batch, 1)
        
        output = reseau(imgs)
        loss   = critere(output, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_totale += loss.item()
    loss_moyenne = loss_totale / len(dataloader)
    
    reseau.eval()
    with torch.no_grad():
        val_loss = 0
        for imgs, labels in val_loader:
            labels = labels.float().unsqueeze(1)
            output = reseau(imgs)
            val_loss += critere(output, labels).item()
        val_loss /= len(val_loader)

    reseau.train()  # remet en mode entraînement
        
    print(f"Epoch {epoch+1} | Train Loss: {loss_moyenne:.4f} | Val Loss: {val_loss:.4f}")