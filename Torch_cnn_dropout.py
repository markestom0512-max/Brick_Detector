import cv2 as cv
import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# Classe CNN — accessible à l'import sans déclencher l'entraînement
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool    = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(16*16*16, 1)
        
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x

# Tout le reste — exécuté UNIQUEMENT si on lance ce fichier directement
if __name__ == '__main__':

    torch.manual_seed(42)
    random.seed(42)

    dossier_lego     = r"C:\Fichiers python\OpenCv\dataset\dataset_cnn\lego"
    dossier_pas_lego = r"C:\Fichiers python\OpenCv\dataset\dataset_cnn\pas_lego"

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    tous_lego     = [os.path.join(dossier_lego, f)     for f in os.listdir(dossier_lego)     if f.endswith(".jpg")]
    tous_pas_lego = [os.path.join(dossier_pas_lego, f) for f in os.listdir(dossier_pas_lego) if f.endswith(".jpg")]

    random.shuffle(tous_lego)
    random.shuffle(tous_pas_lego)

    split_lego     = int(0.8 * len(tous_lego))
    split_pas_lego = int(0.8 * len(tous_pas_lego))

    train_chemins = tous_lego[:split_lego]     + tous_pas_lego[:split_pas_lego]
    train_labels  = [1] * split_lego           + [0] * split_pas_lego
    val_chemins   = tous_lego[split_lego:]     + tous_pas_lego[split_pas_lego:]
    val_labels    = [1] * (len(tous_lego) - split_lego) + [0] * (len(tous_pas_lego) - split_pas_lego)

    class LegoDataset(Dataset):
        def __init__(self, chemins, labels, transform=None):
            self.images    = chemins
            self.labels    = labels
            self.transform = transform

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = cv.imread(self.images[idx], cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (64, 64))
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

    dataset_train = LegoDataset(train_chemins, train_labels, transform=transform_train)
    dataset_val   = LegoDataset(val_chemins,   val_labels,   transform=transform_val)

    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_loader   = DataLoader(dataset_val,   batch_size=16, shuffle=False)

    print(f"Train : {len(dataset_train)} images")
    print(f"Val   : {len(dataset_val)} images")

    reseau    = CNN()
    optimizer = torch.optim.Adam(reseau.parameters(), lr=0.001)
    critere   = nn.BCELoss()
    meilleure_val_loss = float('inf')

    for epoch in range(200):
        loss_totale = 0
        reseau.train()
        for imgs, labels in train_loader:
            labels = labels.float().unsqueeze(1)
            output = reseau(imgs)
            loss   = critere(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_totale += loss.item()
        loss_moyenne = loss_totale / len(train_loader)

        reseau.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0 
            total = 0
            
            for imgs, labels in val_loader:
                labels = labels.float().unsqueeze(1)
                output = reseau(imgs)
                
                predictions = (output > 0.5).float()
                
                correct += (predictions == labels).sum().item()
                
                total += labels.size(0)
                
                val_loss += critere(output, labels).item()
            val_loss /= len(val_loader)
            accuracy = correct / total * 100
            print(f"Epoch {epoch+1} | Train Loss: {loss_moyenne:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {accuracy:.2f}%")

        if val_loss < meilleure_val_loss:
            meilleure_val_loss = val_loss
            torch.save(reseau.state_dict(), "meilleur_modele_dropout.pth")
            print(f"Nouvelle meilleure perte de validation : {meilleure_val_loss:.4f} à l'epoch {epoch+1}")

