from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8m.pt")

    yaml_content = """train: A:/Claude Code/OpenCv/dataset/Batch 2/train/images
val: A:/Claude Code/OpenCv/dataset/Batch 2/val/images
nc: 1
names:
  - lego
"""

    with open("A:/Claude Code/OpenCv/dataset/Batch 2/data.yaml", "w") as f:
        f.write(yaml_content)

    print("data.yaml créé !")

    model.train(
        data="A:/Claude Code/OpenCv/dataset/Batch 2/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="lego_medium_v1"
    )