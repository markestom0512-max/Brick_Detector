import os
import shutil
import random

# Chemins source — où Label Studio a exporté
source_images = "C:\\Fichiers python\\OpenCv\\dataset\\Batch 2\\images"
source_labels = "C:\\Fichiers python\\OpenCv\\dataset\\Batch 2\\labels"

# Chemins destination — train et val dans le même dossier
train_images = "C:\\Fichiers python\\OpenCv\\dataset\\Batch 2\\train\\images"
train_labels = "C:\\Fichiers python\\OpenCv\\dataset\\Batch 2\\train\\labels"
val_images   = "C:\\Fichiers python\\OpenCv\\dataset\\Batch 2\\val\\images"
val_labels   = "C:\\Fichiers python\\OpenCv\\dataset\\Batch 2\\val\\labels"

# Crée les dossiers destination
os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images,   exist_ok=True)
os.makedirs(val_labels,   exist_ok=True)

# Liste toutes les images
images = [f for f in os.listdir(source_images) if f.endswith(".jpg")]

# Mélange aléatoire
random.shuffle(images)

# Séparation 80% train / 20% val
split = int(len(images) * 0.8)
train = images[:split]
val   = images[split:]

def copier_et_renommer(liste, dest_img, dest_lbl):
    for img in liste:
        nom_base = os.path.splitext(img)[0]  # "2e7317f1-image_025"
        
        # Supprime le préfixe si présent (ex: "2e7317f1-")
        if "-" in nom_base:
            nom_propre = nom_base.split("-", 1)[1]  # → "image_025"
        else:
            nom_propre = nom_base

        # Copie et renomme l'image
        shutil.copy(
            os.path.join(source_images, img),
            os.path.join(dest_img, nom_propre + ".jpg")
        )

        # Cherche le label avec ou sans préfixe
        label_original = nom_base + ".txt"
        label_propre   = nom_propre + ".txt"

        # Essaie les deux noms possibles
        for label_nom in [label_original, label_propre]:
            label_path = os.path.join(source_labels, label_nom)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(dest_lbl, label_propre))
                break

copier_et_renommer(train, train_images, train_labels)
copier_et_renommer(val,   val_images,   val_labels)

print(f"Train : {len(train)} images")
print(f"Val   : {len(val)} images")
print("Dataset prêt !")