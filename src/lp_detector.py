import cv2
import numpy as np
import matplotlib.pyplot as plt


class LicensePlateDetector:
    def __init__(self, pth_weights: str, pth_cfg: str, pth_classes: str):
        self.net = cv2.dnn.readNet(pth_weights, pth_cfg)
        self.classes = []
        with open(pth_classes, 'r') as f:
            self.classes = f.read().splitlines()
        self.font = cv2.FONT_HERSHEY_PLAIN
        self.color = (255, 0, 0)
        self.coordinates = None
        self.img = None
        self.fig_image = None
        self.roi_image = None

    def detect(self, img):
        orig = img
        self.img = orig
        img = orig.copy()
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
        self.net.setInput(blob)
        output_layer_names = self.net.getUnconnectedOutLayersNames()
        layer_outputs = self.net.forward(output_layer_names)
        boxes = []
        confidences = []
        class_ids = []
        x1, y1, w1, h1 = 0, 0, 0, 0
        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w1 = int(detection[2] * width)
                    h1 = int(detection[3] * height)
                    x1 = int(center_x - w1 / 2)
                    y1 = int(center_y - h1 / 2)

                    boxes.append([x1, y1, w1, h1])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

        if len(indexes) > 0:
            for i in indexes.flatten():
                x1, y1, w1, h1 = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = str(round(confidences[i], 2))
                cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), self.color, 15)
                cv2.putText(img, label + ' ' + confidence, (x1, y1 + 20), self.font, 3, (255, 255, 255), 3)
        self.fig_image = img
        # self.coordinates = (x1, y1, w1, h1)
        return x1, y1, w1, h1

    def crop_plate(self):
        x, y, w, h = self.coordinates
        roi = self.img[y:y + h, x:x + w]
        self.roi_image = roi
        return
