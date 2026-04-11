from ultralytics import YOLO

# Charge le modèle de base — on part de yolov8n pré-entraîné
model = YOLO("yolov8n.pt")

# Crée le data.yaml avec les bons chemins
yaml_content = """train: C:/Fichiers python/OpenCv/dataset/Label-studio/train/images
val: C:/Fichiers python/OpenCv/dataset/Label-studio/val/images
nc: 1
names:
  - lego
"""

with open("C:/Fichiers python/OpenCv/dataset/Label-studio/data.yaml", "w") as f:
    f.write(yaml_content)

print("data.yaml créé !")

# Lance l'entraînement
model.train(
    data="C:/Fichiers python/OpenCv/dataset/Label-studio/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    name="lego_v1"
)