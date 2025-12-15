import os
from pathlib import Path
import cv2
import numpy as np
import time

DATASET_DIR = "dataset"
MAX_IMAGES = 300
CONF_THRESHOLD = 0.6
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def load_face_detector(proto_path=None, model_path=None):
    net = None
    if proto_path and model_path and Path(proto_path).exists() and Path(model_path).exists():
        try:
            net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
            print("[INFO] DNN face detector cargado.")
        except Exception as e:
            print("[WARN] No se pudo cargar DNN:", e)
            net = None
    if net is None:
        cascade = cv2.CascadeClassifier(HAAR_PATH)
        if cascade.empty():
            raise RuntimeError("No se pudo cargar Haar Cascade.")
        print("[INFO] Usando Haar Cascade como detector.")
        return ("haar", cascade)
    return ("dnn", net)

def detect_faces(net_tuple, frame):
    mode, net_or_cascade = net_tuple
    if mode == "dnn":
        net = net_or_cascade
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []
        for i in range(0, detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf > CONF_THRESHOLD:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (sx, sy, ex, ey) = box.astype("int")
                sx = max(0, sx); sy = max(0, sy)
                ex = min(w - 1, ex); ey = min(h - 1, ey)
                rects.append((sx, sy, ex - sx, ey - sy, conf))
        return rects
    else:
        cascade = net_or_cascade
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
        return [(int(x), int(y), int(w), int(h), 1.0) for (x,y,w,h) in rects]

def make_user_folder_when_starting(next_uid):
    os.makedirs(DATASET_DIR, exist_ok=True)
    user_folder = os.path.join(DATASET_DIR, f"usuario{next_uid}")
    os.makedirs(user_folder, exist_ok=True)
    return user_folder

def next_available_uid():
    os.makedirs(DATASET_DIR, exist_ok=True)
    existing = [d for d in os.listdir(DATASET_DIR) if d.startswith("usuario")]
    nums = []
    for x in existing:
        try:
            nums.append(int(x.replace("usuario", "")))
        except:
            pass
    if nums:
        return max(nums) + 1
    return 1

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=80):
        self.nextObjectID = 1
        self.objects = dict()
        self.disappeared = dict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
        return self.nextObjectID - 1

    def deregister(self, objectID):
        if objectID in self.objects:
            del self.objects[objectID]
        if objectID in self.disappeared:
            del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return {oid: None for oid in self.objects.keys()}

        input_centroids = []
        for (x,y,w,h,conf) in rects:
            cx = int(x + w/2.0)
            cy = int(y + h/2.0)
            input_centroids.append((cx, cy))

        if len(self.objects) == 0:
            assigned = {}
            for i, cen in enumerate(input_centroids):
                oid = self.register(cen)
                assigned[oid] = rects[i]
            return assigned

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        D = np.zeros((len(objectCentroids), len(input_centroids)), dtype="float")
        for i in range(len(objectCentroids)):
            for j in range(len(input_centroids)):
                D[i, j] = np.linalg.norm(np.array(objectCentroids[i]) - np.array(input_centroids[j]))

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()
        assigned = {}

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue
            if D[row, col] > self.max_distance:
                continue
            oid = objectIDs[row]
            self.objects[oid] = input_centroids[col]
            self.disappeared[oid] = 0
            assigned[oid] = rects[col]
            usedRows.add(row)
            usedCols.add(col)

        for j in range(len(input_centroids)):
            if j not in usedCols:
                oid = self.register(input_centroids[j])
                assigned[oid] = rects[j]

        for i in range(len(objectCentroids)):
            if i not in usedRows:
                oid = objectIDs[i]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

        return assigned

def save_face_image(user_folder, count, face_img):
    if count >= MAX_IMAGES:
        return False
    filename = f"img_{count:04d}.jpg"
    path = os.path.join(user_folder, filename)
    cv2.imwrite(path, face_img)
    return True
