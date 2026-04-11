import numpy as np
w = 0.5  # poids initial
import matplotlib.pyplot as plt

losses = []  # liste pour stocker la loss à chaque epoch
def sigmoide(z):
    return 1 / (1 + np.exp(-z))
for epoch in range(50):
    x = 1.0
    b = 0.0
    y = 1
    lr = 10

    # Forward pass
    Z = x * w + b
    A = sigmoide(Z)
    Loss = -(y * np.log(A) + (1-y) * np.log(1-A))

    # Backward pass
    dLoss_dA = -(y/A) + (1-y)/(1-A)
    dA_dZ    = A * (1 - A)
    dZ_dw    = x

    dLoss_dw = dLoss_dA * dA_dZ * dZ_dw
    losses.append(Loss) 

# Mise à jour
    w = w - lr * dLoss_dw

    
    print(f"Epoch {epoch+1} | Loss: {Loss:.4f} | w: {w:.4f}")
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Descente de gradient")
plt.show()