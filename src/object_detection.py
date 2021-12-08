import cv2
import numpy as np
import time

# Load Yolo
from src.lp_detector import LicensePlateDetector

net = cv2.dnn.readNet("../yolo/car_detection/yolov3.weights", "../yolo/car_detection/yolov3.cfg")
classes = []
with open("../yolo/car_detection/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Loading image
cap = cv2.VideoCapture('../data/video 15.mp4')
# cap = cv2.VideoCapture(0)
color = (0, 255, 0)

lpd = LicensePlateDetector(
    pth_weights='../yolo/lp_detection/model.weights',
    pth_cfg='../yolo/lp_detection/darknet-yolov3.cfg',
    pth_classes='../yolo/lp_detection/classes.names'
)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    roi = frame

    # Detecting objects
    blob = cv2.dnn.blobFromImage(roi, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if classes[class_id] not in ['car', 'motorbike', 'bus', 'truck', ]:
                continue
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # label = str(classes[class_ids[i]])
            x1, y1, w1, h1 = lpd.detect(roi[y:y+h, x:x+w])
            cv2.rectangle(roi, (x+x1, y+y1), (x+x1 + w1, y+y1 + h1), color, 2)

    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(roi, "FPS: " + str(round(fps, 2)), (10, 50), font, 4, (0, 255, 0), 3)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow("Image", roi)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
