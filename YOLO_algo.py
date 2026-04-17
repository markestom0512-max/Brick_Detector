from ultralytics import YOLO
import cv2 as cv

model = YOLO(r"A:\Claude Code\OpenCv\runs\detect\lego_nano_v12\weights\best.pt") 

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

capture = cv.VideoCapture(0)
while True:
    ret, frame = capture.read()
    if not ret:
        break
    resultats = model(frame, verbose=False)
    frame = filtrage(frame, resultats)

    cv.imshow("YOLO", frame)
    if cv.waitKey(1) == ord('q'):
        break

capture.release()
cv.destroyAllWindows()