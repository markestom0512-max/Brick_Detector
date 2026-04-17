import torch
import torch.nn as nn
import torchvision.models as models
import cv2 as cv
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class LegoDataset(Dataset):
    def __init__(self, chemins, labels, transform=None):
        self.images    = chemins
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = cv.imread(self.images[idx])
        img = cv.resize(img, (224, 224))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # ← long pour CrossEntropyLoss
        return img, label

if __name__ == '__main__':

    torch.manual_seed(42)
    random.seed(42)

    dossier_rouge = r"C:\Fichiers python\OpenCv\dataset\crops\Rouge"
    dossier_bleu  = r"C:\Fichiers python\OpenCv\dataset\crops\Bleu"
    dossier_jaune = r"C:\Fichiers python\OpenCv\dataset\crops\Jaune"

    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.ToTensor()
    ])

    transform_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    tous_rouge = [os.path.join(dossier_rouge, f) for f in os.listdir(dossier_rouge) if f.endswith(".jpg")]
    tous_bleu  = [os.path.join(dossier_bleu,  f) for f in os.listdir(dossier_bleu)  if f.endswith(".jpg")]
    tous_jaune = [os.path.join(dossier_jaune, f) for f in os.listdir(dossier_jaune) if f.endswith(".jpg")]

    random.shuffle(tous_rouge)
    random.shuffle(tous_bleu)
    random.shuffle(tous_jaune)

    split_rouge = int(0.8 * len(tous_rouge))
    split_bleu  = int(0.8 * len(tous_bleu))
    split_jaune = int(0.8 * len(tous_jaune))

    train_chemins = tous_rouge[:split_rouge] + tous_bleu[:split_bleu] + tous_jaune[:split_jaune]
    train_labels  = [0] * split_rouge        + [1] * split_bleu       + [2] * split_jaune

    val_chemins = tous_rouge[split_rouge:] + tous_bleu[split_bleu:] + tous_jaune[split_jaune:]
    val_labels  = [0] * (len(tous_rouge) - split_rouge) + \
                  [1] * (len(tous_bleu)  - split_bleu)  + \
                  [2] * (len(tous_jaune) - split_jaune)

    dataset_train = LegoDataset(train_chemins, train_labels, transform=transform_train)
    dataset_val   = LegoDataset(val_chemins,   val_labels,   transform=transform_val)

    train_loader = DataLoader(dataset_train, batch_size=16, shuffle=True)
    val_loader   = DataLoader(dataset_val,   batch_size=16, shuffle=False)

    print(f"Train : {len(dataset_train)} images")
    print(f"Val   : {len(dataset_val)} images")

    reseau = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in reseau.parameters():
        param.requires_grad = False
        
    for param in reseau.layer4.parameters():
        param.requires_grad = True
        
    for param in reseau.fc.parameters():
        param.requires_grad = True 
    reseau.fc = nn.Linear(512, 3)
    

    CLASSES = ["rouge", "bleu", "jaune"]  # rouge=0, bleu=1, jaune=2

    optimizer = torch.optim.Adam([
        {'params': reseau.layer4.parameters(), 'lr': 0.0001},  # lr petit
        {'params': reseau.fc.parameters(),     'lr': 0.001}    # lr normal
    ])  
    poids     = torch.tensor([55/35, 55/55, 55/35], dtype=torch.float32)  # rouge, bleu, jaune
    critere   = nn.CrossEntropyLoss(weight=poids)
    meilleure_val_loss = float('inf')

    for epoch in range(100):
        loss_totale = 0
        reseau.train()
        for imgs, labels in train_loader:
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
            correct  = 0
            total    = 0
            for imgs, labels in val_loader:
                output      = reseau(imgs)
                val_loss    += critere(output, labels).item()
                predictions = output.argmax(dim=1)
                correct     += (predictions == labels).sum().item()
                total       += labels.size(0)
            val_loss /= len(val_loader)
            accuracy  = correct / total * 100

        if val_loss < meilleure_val_loss:
            meilleure_val_loss = val_loss
            
            chemin_sauvegarde = os.path.abspath("meilleur_resnet_couleur.pth")
            torch.save(reseau.state_dict(), chemin_sauvegarde)
            print(f"  → Sauvegardé ici : {chemin_sauvegarde}")
            print(f"  → Sauvegardé ! Val Loss: {val_loss:.4f}")

        print(f"Epoch {epoch+1} | Train Loss: {loss_moyenne:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {accuracy:.1f}%")