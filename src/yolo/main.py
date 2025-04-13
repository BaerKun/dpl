from ultralytics import YOLO
from visualize import predict_show


cls_names = ['R', 'I', 'A']

model = YOLO("runs/pose/train8/weights/best.pt")

model.train(data='data.yaml', task='pose', epochs=5 , imgsz=416, batch=16, device=0)
predict_show(model, "datasets/images/test", cls_names)

# model.export(format='onnx', opset=13)
