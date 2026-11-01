import cv2
import numpy as np
import os
import math
from .utils import get_project_root, download_file

MODEL_DIR = os.path.join(get_project_root(), "models", "detection")
CAFFE_MODEL = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
PROTOTXT = os.path.join(MODEL_DIR, "ssd_deploy.prototxt")
CAFFE_URL = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
PROTOTXT_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"

class FaceDetector:
    def __init__(self):
        if not os.path.exists(CAFFE_MODEL) or not os.path.exists(PROTOTXT):
            print("Downloading Face Detection models...")
            os.makedirs(MODEL_DIR, exist_ok=True)
            download_file(PROTOTXT_URL, PROTOTXT)
            download_file(CAFFE_URL, CAFFE_MODEL)
            
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFE_MODEL)
        self.confidence_threshold = 0.5
        
    def get_bboxes(self, image):
        height, width = image.shape[:2]
        # SSD uses 300x300 blob size typically
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        out = self.net.forward()
        
        bboxes = []
        # SSD output shape is [1, 1, N, 7]
        for i in range(out.shape[2]):
            conf = out[0, 0, i, 2]
            if conf > self.confidence_threshold:
                left = out[0, 0, i, 3] * width
                top = out[0, 0, i, 4] * height
                right = out[0, 0, i, 5] * width
                bottom = out[0, 0, i, 6] * height
                bboxes.append([int(left), int(top), int(right-left+1), int(bottom-top+1)])
        return bboxes

    def get_bbox(self, image):
        bboxes = self.get_bboxes(image)
        if not bboxes:
            return None
        return bboxes[0] # For backward compatibility

def get_face_detector():
    return FaceDetector()

def crop_face(image, bbox, margin=0.2):
    if bbox is None:
        return None
    x, y, w, h = bbox
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(image.shape[1], x + w + margin_x)
    y2 = min(image.shape[0], y + h + margin_y)
    
    return image[y1:y2, x1:x2]

def get_scaled_crop(image, bbox, scale=4.0):
    if bbox is None:
        return None
        
    x, y, w, h = bbox
    src_h, src_w = image.shape[:2]
    
    scale = min((src_h - 1) / h, min((src_w - 1) / w, scale))
    
    new_width = w * scale
    new_height = h * scale
    center_x, center_y = w / 2 + x, h / 2 + y
    
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
        
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
        
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
        
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
        
    left_top_x = int(left_top_x)
    left_top_y = int(left_top_y)
    right_bottom_x = int(right_bottom_x)
    right_bottom_y = int(right_bottom_y)
    
    return image[left_top_y:right_bottom_y+1, left_top_x:right_bottom_x+1]
