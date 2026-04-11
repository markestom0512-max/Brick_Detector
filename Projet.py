import cv2 as cv 
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

couleurs = {
    "rouge": {
        "bornes": [                                    # liste car deux plages pour le rouge
            (np.array([0, 150, 100]),   np.array([10, 255, 255])),
            (np.array([160, 150, 100]), np.array([180, 255, 255]))
        ],
        "affichage": (0, 0, 255)                       # couleur BGR pour le rectangle
    },
    "bleu": {
        "bornes": [                                    # une seule plage pour le bleu
            (np.array([100, 150, 100]), np.array([130, 255, 255]))
        ],
        "affichage": (255, 0, 0)
    },
    "jaune": {
        "bornes": [                                    # une seule plage pour le jaune  
            (np.array([25, 150, 100]), np.array([35, 255, 255]))
        ],
        "affichage": (0, 255, 255)
    },
}

kernel = np.ones((4, 4), np.uint8)  # grille 4x4 de 1

def mask_color(frame_HSV, couleur):
    masque = np.zeros(frame_HSV.shape[:2], dtype=np.uint8)  # masque vide
    for (borne_basse, borne_haute) in couleurs[couleur]["bornes"]:
        masque_partiel = cv.inRange(frame_HSV, borne_basse, borne_haute)  # masque pour la plage de couleur
        masque = cv.bitwise_or(masque, masque_partiel)  # combiner les masques partiels
    mask_cleaned = cv.erode(masque, kernel, iterations=1)
    mask_cleaned = cv.dilate(mask_cleaned, kernel, iterations=1)
    return mask_cleaned

def contour_valide(liste_contours, aire_min=500):
    liste_contours = list(liste_contours)  # s'assurer que contours est une liste
    valides = [c for c in liste_contours if cv.contourArea(c) > aire_min]  # filtrer les contours par aire
    if len(valides) > 0:
        return valides
    elif len(valides) == 0:
        return []

def filtrage(frame, resultats, confiance_min=0.5):
    for r in resultats:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # convertit en entiers
            label = r.names[int(box.cls)]   
            confiance = box.conf.item()
            if confiance >= confiance_min:
                frame = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                frame = cv.putText(frame, f"{label} {confiance:.2f}", (x1, y1-10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def centre_contour(max_contour):
    M = cv.moments(max_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)

def boxing(frame, centre, couleur, max_contour):
    texte = f"{couleur}"
    x, y, w, h = cv.boundingRect(max_contour)
    frame = cv.rectangle(frame, (x, y), (x+w, y+h), couleurs[couleur]["affichage"], 2)
    frame = cv.circle(frame, centre, 5, (255, 255, 255), -1)
    return frame, texte, x, y

capture = cv.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    
    frame_HSV = frame.copy() 
    frame_YOLO = frame.copy()
    frame_HSV_BGR = frame.copy()  
    
    resultats = model(frame_YOLO, verbose=False)
    frame_YOLO = filtrage(frame_YOLO, resultats)

    
    frame_HSV = cv.cvtColor(frame_HSV, cv.COLOR_BGR2HSV)
    
    for i, (nom, config) in enumerate(couleurs.items()):
        mask_cleaned = mask_color(frame_HSV, nom)
        contours, hierarchy = cv.findContours(mask_cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        valides = contour_valide(contours)
    
    # Compteur en haut — i donne déjà la position !
        y_pos = 30 + i * 30  # rouge=30, bleu=60, jaune=90
        frame_HSV_BGR = cv.putText(frame_HSV_BGR, f"{nom}: {len(valides)}", (10, y_pos),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, couleurs[nom]["affichage"], 2)
    
        if valides is not None:
            for contour in valides:
                centre = centre_contour(contour)
                frame_HSV_BGR, texte, x, y = boxing(frame_HSV_BGR, centre, nom, contour)
                frame_HSV_BGR = cv.putText(frame_HSV_BGR, texte, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
       
    cv.imshow('HSV', frame_HSV_BGR)
    cv.imshow("YOLO", frame_YOLO)
    cv.imshow("Original", frame_HSV)
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()