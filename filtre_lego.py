import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Charge une de tes images de briques
img = cv.imread("C:\\Fichiers python\\OpenCv\\dataset\\images\\image_002.jpg")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # convertit en niveaux de gris

# Filtre détecteur de bords verticaux
filtre_vertical = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
], dtype=np.float32)

filtre_horizontal = np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1]
], dtype=np.float32)

# Applique le filtre
resultat_vertical = cv.filter2D(img_gray, -1, filtre_vertical)
resultat_horizontal = cv.filter2D(img_gray, -1, filtre_horizontal)

# Affiche les deux images côte à côte
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(resultat_vertical, cmap='gray')
axes[0].set_title("Bords verticaux")
axes[1].imshow(resultat_horizontal, cmap='gray')
axes[1].set_title("Bords horizontaux")
plt.show()