from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    yaml_content = """train: A:/Claude Code/OpenCv/dataset/Merged Batch/train/images
val: A:/Claude Code/OpenCv/dataset/Merged Batch/val/images
nc: 1
names:
  - lego
"""

    with open("A:/Claude Code/OpenCv/dataset/Merged Batch/data.yaml", "w") as f:
        f.write(yaml_content)

    print("data.yaml créé !")

    model.train(
        data="A:/Claude Code/OpenCv/dataset/Merged Batch/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="lego_nano_v1"
    )