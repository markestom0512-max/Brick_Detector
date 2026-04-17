from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2 as cv

CLASSES = ["rouge", "bleu", "jaune"]

# Couleurs BGR pour chaque classe
COULEURS = {
    "rouge": (0, 0, 255),
    "bleu":  (255, 0, 0),
    "jaune": (0, 255, 255)
}

# Transform pour le CNN — même que pendant l'entraînement
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Charge YOLO
model_yolo = YOLO(r"C:\Fichiers python\runs\detect\lego_v12\weights\best.pt")

# Charge ResNet couleur
reseau = models.resnet18()
reseau.fc = nn.Linear(512, 3)
reseau.load_state_dict(torch.load(r"C:\Users\marke\AppData\Local\Programs\Microsoft VS Code\meilleur_resnet_couleur.pth"))
reseau.eval()

capture = cv.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Étape 1 — YOLO détecte les briques
    resultats = model_yolo(frame, verbose=False)

    for r in resultats:
        for box in r.boxes:
            if box.conf.item() < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Étape 2 — Crop de la brique
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:  # ignore les crops vides
                continue

            # Étape 3 — ResNet classifie la couleur
            crop_rgb = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
            crop_tensor = transform(crop_rgb).unsqueeze(0)

            with torch.no_grad():
                output    = reseau(crop_tensor)
                classe_id = output.argmax(dim=1).item()
                couleur   = CLASSES[classe_id]
                confiance = output.softmax(dim=1)[0][classe_id].item()

            # Étape 4 — Affichage
            cv.rectangle(frame, (x1, y1), (x2, y2), COULEURS[couleur], 2)
            cv.putText(frame, f"{couleur} {confiance:.0%}", (x1, y1-10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, COULEURS[couleur], 2)

    cv.imshow("YOLO + ResNet", frame)
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()