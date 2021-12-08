
import time
import cv2
import numpy as np
import sys

from src.lp_detector import LicensePlateDetector
from sklearn.cluster import DBSCAN

cap = cv2.VideoCapture("../data/video 3.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=100)

lpd = LicensePlateDetector(
    pth_weights='../yolo/lp_detection/model.weights',
    pth_cfg='../yolo/lp_detection/darknet-yolov3.cfg',
    pth_classes='../yolo/lp_detection/classes.names'
)

frame_id = 0


while True:
    starting_time = time.time()
    ret, frame = cap.read()
    frame_id += 1
    height, width, _ = frame.shape

    # Extract Region of interest
    # roi = frame[height // 2:, ::]
    roi = frame

    # 1. Object Detection
    mask = object_detector.apply(roi)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []

    # Bool array indicating which initial bounding rect has
    # already been used
    rectsUsed = []
    rect_center = []
    # Just initialize bounding rects and set all bools to false
    for cnt in contours:
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)

            rect_center.append((int(((x + w // 2) * 25) / width), int(((y + h // 2) * 25) / height)))
            rects.append([x, y, w, h])
    if not rect_center:
        continue
    clustering = DBSCAN(eps=3, min_samples=2).fit(np.array(rect_center))

    final_rect = []
    rect_grps = [[] for i in range(max(clustering.labels_) + 1)]
    for ndx, label in enumerate(clustering.labels_):
        if label == -1:
            final_rect.append(rects[ndx])
        else:
            rect_grps[label].append(rects[ndx])

    for grp in rect_grps:
        if not grp:
            break
        x_min, y_min, x_max, y_max = sys.maxsize, sys.maxsize, 0, 0
        for rec in grp:
            if rec[0] < x_min:
                x_min = rec[0]
            if rec[1] < y_min:
                y_min = rec[1]
            if rec[0] + rec[2] > x_max:
                x_max = rec[0] + rec[2]
            if rec[1] + rec[3] > y_max:
                y_max = rec[1] + rec[3]
        final_rect.append([x_min, y_min, x_max - x_min, y_max - y_min])

    acceptedRects = final_rect

    for cnt in acceptedRects:
        x, y, w, h = cnt
        # cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)
        print(x,y,w,h)
        x1, y1, w1, h1 = lpd.detect(roi[y:y + h, x:x + w])
        print(x1,y1,w1,h1)
        cv2.rectangle(roi, (x+x1, y+y1), (x+x1+w1, y+y1+h1), (0, 255, 0), 3)
        # detections.append([x, y, w, h])

    elapsed_time = time.time() - starting_time
    fps = 1 / elapsed_time
    cv2.putText(roi, "FPS: " + str(round(fps, 2)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 3)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    cv2.imshow("Frame", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
