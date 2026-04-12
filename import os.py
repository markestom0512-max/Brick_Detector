import os
import shutil
import random

source_images = "A:/Claude Code/OpenCv/dataset/Batch 1/images"
source_labels = "A:/Claude Code/OpenCv/dataset/Batch 1/labels"

train_images = "A:/Claude Code/OpenCv/dataset/Batch 1/train/images"
train_labels = "A:/Claude Code/OpenCv/dataset/Batch 1/train/labels"
val_images   = "A:/Claude Code/OpenCv/dataset/Batch 1/val/images"
val_labels   = "A:/Claude Code/OpenCv/dataset/Batch 1/val/labels"

os.makedirs(train_images, exist_ok=True)
os.makedirs(train_labels, exist_ok=True)
os.makedirs(val_images,   exist_ok=True)
os.makedirs(val_labels,   exist_ok=True)

images = [f for f in os.listdir(source_images) if f.endswith(".jpg")]
random.shuffle(images)
split = int(len(images) * 0.8)
train = images[:split]
val   = images[split:]

def copier_et_renommer(liste, dest_img, dest_lbl):
    for img in liste:
        nom_base = os.path.splitext(img)[0]
        if "-" in nom_base:
            nom_propre = nom_base.split("-", 1)[1]
        else:
            nom_propre = nom_base
        shutil.copy(
            os.path.join(source_images, img),
            os.path.join(dest_img, nom_propre + ".jpg")
        )
        for label_nom in [nom_base + ".txt", nom_propre + ".txt"]:
            label_path = os.path.join(source_labels, label_nom)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(dest_lbl, nom_propre + ".txt"))
                break

copier_et_renommer(train, train_images, train_labels)
copier_et_renommer(val,   val_images,   val_labels)

print(f"Train : {len(train)} images")
print(f"Val   : {len(val)} images")
print("Batch 1 prêt !")