import torch.nn as nn
import torch

X = torch.tensor([0.8, 0.3]) 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # 2 entrées, 3 neurones dans la couche cachée
        self.fc2 = nn.Linear(3, 1)  # 3 neurones → 1 sortie

    def forward(self, x):   
        x = torch.relu(self.fc1(x))  # activation ReLU pour la couche cachée
        x = torch.sigmoid(self.fc2(x))  # activation sigmoïde pour la couche de sortie
        return x

reseau = Net()
optimizer = torch.optim.Adam(reseau.parameters(), lr=0.1)  # optimiseur SGD
critere = nn.BCELoss()  # fonction de perte binaire
y = torch.tensor([1.0])  # étiquette de classe (1 pour positif)

for epoch in range(50):
    output = reseau(X)
    loss = critere(output, y)  # perte binaire

    optimizer.zero_grad()  # réinitialiser les gradients
    
    loss.backward()  # calculer les gradients
    
    optimizer.step()  # mettre à jour les poids
    
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | w1: {reseau.fc1.weight.data} | w2: {reseau.fc2.weight.data}")