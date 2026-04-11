import numpy as np

def neurone(x, w, b):
    z = np.dot(w, x) + b  # calcul de la somme pondérée
    fz = sigmoide(z)  # fonction d'activation sigmoïde
    return fz

def sigmoide(z):
    return 1 / (1 + np.exp(-z))

def relu(z):
    return np.maximum(0, z)

def neurones_first_layer(X, W1, b1):
    Z1 = np.dot(X, W1) + b1  # calcule les 3 neurones EN MÊME TEMPS
    A = relu(Z1)
    return A

def neurone_output(A, W2, b2):
    Z = np.dot(A, W2) + b2  # calcul de la somme pondérée pour la couche de sortie
    FZ = sigmoide(Z)  # fonction d'activation sigmoïde pour la couche de sortie
    return FZ

def reseau(X, W1, b1, W2, b2):
    A = neurones_first_layer(X, W1, b1)
    output = neurone_output(A, W2, b2)
    return output

x = np.array([0.8, 0.3, 0.6])
w = np.array([0.5, 0.9, 0.2])
b = -5  # biais


X  = np.array([0.8, 0.3])       # 2 entrées
W1 = np.random.randn(2, 3)      # poids couche cachée (2 entrées → 3 neurones)
b1 = np.zeros(3)                # biais couche cachée
W2 = np.random.randn(3, 1)      # poids couche sortie (3 neurones → 1 sortie)
b2 = np.zeros(1)  


print("Sortie du réseau :", reseau(X, W1, b1, W2, b2))