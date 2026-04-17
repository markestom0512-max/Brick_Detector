import torch
import cv2 as cv
from Torch_cnn_dropout import CNN  # ← adapte avec le nom de ton fichier

# Charge le meilleur modèle
reseau = CNN()
reseau.load_state_dict(torch.load("meilleur_modele.pth"))
reseau.eval()

# Teste sur une image
img = cv.imread(r"C:\Fichiers python\OpenCv\dataset\dataset_cnn\pas_lego\image_003.jpg", cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (64, 64))
img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

with torch.no_grad():
    prediction = reseau(img_tensor)
    prob = prediction.item()
    print(f"Probabilité lego : {prob:.4f}")
    print("✅ Brique détectée !" if prob > 0.5 else "❌ Pas une brique !")