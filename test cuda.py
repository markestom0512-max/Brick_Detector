import torch
import cv2
import numpy as np
import ultralytics


print("PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
print("GPU :", torch.cuda.get_device_name(0))
print("OpenCV :", cv2.__version__)
print("NumPy :", np.__version__)
print("Ultralytics :", ultralytics.__version__)