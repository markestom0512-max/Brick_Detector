from ultralytics import YOLO
import cv2 as cv
import os

# Chemins
model = YOLO(r"C:\Fichiers python\runs\detect\lego_v12\weights\best.pt")
dossier_images = r"C:\Fichiers python\OpenCv\dataset\Batch 1"
dossier_crops  = r"C:\Fichiers python\OpenCv\dataset\crops"

os.makedirs(dossier_crops, exist_ok=True)

compteur = 0
for nom_image in os.listdir(dossier_images):
    if not nom_image.endswith(".jpg"):
        continue
    
    chemin = os.path.join(dossier_images, nom_image)
    img    = cv.imread(chemin)
    
    # Détection YOLO
    resultats = model(img, verbose=False)
    
    for r in resultats:
        for box in r.boxes:
            if box.conf.item() < 0.5:  # ignore les détections peu sûres
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Découpe la brique
            crop = img[y1:y2, x1:x2]
            
            # Sauvegarde
            chemin_crop = os.path.join(dossier_crops, f"crop_{compteur:04d}.jpg")
            cv.imwrite(chemin_crop, crop)
            compteur += 1

print(f"{compteur} crops sauvegardés dans {dossier_crops}")