from ultralytics import YOLO
from ultralytics.engine.results import Results
import os.path
import cv2
import utils

cls_names = ['R', 'I', 'A']


def show_result(_result: Results) -> int:
    img = _result.orig_img

    for box in _result.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(img, cls_names[int(box.cls)] + str(float(box.conf))[:4], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

    for kp in _result.keypoints:
        xy = kp.xy[0]
        for x, y in xy:
            cv2.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)

    cv2.imshow('result', img)
    return cv2.waitKey(0)


def predict_show(images_folder):
    import os

    img_list = os.listdir(images_folder)
    img_list.sort()
    for img_name in img_list:
        img_path = os.path.join(images_folder, img_name)
        result = model.predict(source=img_path, conf=0.5, iou=0.3, device='cpu')
        if show_result(result[0]) == 27:
            break

    cv2.destroyWindow('result')

model = YOLO(os.path.join(utils.project_root, "weights/yolo/yolo11n-pose.pt"))

predict_show("datasets/images/test")
model.train(data='data.yaml', task='pose', epochs=5 , imgsz=416, batch=16, device=0)
predict_show("datasets/images/test")

# model.export(format='onnx', opset=13)
