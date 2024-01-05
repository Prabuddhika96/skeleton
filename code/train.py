from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(data='data.yaml', batch=8, imgsz=640, epochs=10, workers=1)