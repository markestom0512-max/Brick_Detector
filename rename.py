import os

labels_dir = r"C:\Fichiers python\OpenCv\dataset\Nouveau dossier\labels"

for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        # Sépare le préfixe du vrai nom
        # "2e7317f1-image_025.txt" → ["2e7317f1", "image_025.txt"]
        parties = filename.split("-", 1)  # split sur le premier "-" uniquement
        
        if len(parties) == 2:
            nouveau_nom = parties[1]  # garde uniquement "image_025.txt"
            
            ancien_chemin = os.path.join(labels_dir, filename)
            nouveau_chemin = os.path.join(labels_dir, nouveau_nom)
            
            os.rename(ancien_chemin, nouveau_chemin)
            print(f"{filename} → {nouveau_nom}")

print("Renommage terminé !")