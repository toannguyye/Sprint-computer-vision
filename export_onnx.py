#!/usr/bin/env python3
"""Export yolo26n.pt to yolo26n.onnx for faster CPU inference."""
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx", imgsz=640, dynamic=True, simplify=True)
print("Done — yolo26n.onnx created. Use it automatically in the main app.")
