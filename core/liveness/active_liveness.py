import cv2
import math
import numpy as np
import os
import collections
from .utils import get_project_root, download_file
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None

TASK_MODEL_NAME = "face_landmarker.task"
TASK_MODEL_PATH = os.path.join(get_project_root(), "models", TASK_MODEL_NAME)
TASK_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

_shared_landmarker = None

def get_shared_landmarker():
    global _shared_landmarker
    if _shared_landmarker is None:
        if mp is None:
            raise RuntimeError("mediapipe not installed")
        if not os.path.exists(TASK_MODEL_PATH):
            os.makedirs(os.path.dirname(TASK_MODEL_PATH), exist_ok=True)
            download_file(TASK_MODEL_URL, TASK_MODEL_PATH)

        base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1)
        _shared_landmarker = vision.FaceLandmarker.create_from_options(options)
    return _shared_landmarker

class ActiveLivenessDetector:
    def __init__(self, detector=None):
        self.detector = detector if detector is not None else get_shared_landmarker()
        
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        self.TURN_THRESH_RIGHT = 0.43 
        self.TURN_THRESH_LEFT = 0.57
        self.REQUIRED_SUSTAINED_FRAMES = 3
        
        self.reset()

    def reset(self):
        self.blink_state = 0 
        self.head_turn_frames = 0
        self.head_turn_state = False
        self.ratio_history = collections.deque(maxlen=3)
        self.wrong_action_frames = 0
        self.violation_type = None

    def _calc_ear(self, landmarks, eye_indices):
        p = [landmarks[i] for i in eye_indices]
        v1 = math.hypot(p[1].x - p[5].x, p[1].y - p[5].y)
        v2 = math.hypot(p[2].x - p[4].x, p[2].y - p[4].y)
        h = math.hypot(p[0].x - p[3].x, p[0].y - p[3].y)
        if h == 0: return 0
        return (v1 + v2) / (2.0 * h)
        
    def _calc_head_turn(self, landmarks):
        nose = landmarks[1]
        edge_1 = landmarks[234]
        edge_2 = landmarks[454]
        
        # FIX: Ensure we are always measuring strictly left-to-right on screen
        min_x = min(edge_1.x, edge_2.x)
        max_x = max(edge_1.x, edge_2.x)
        
        total_width = max_x - min_x
        if total_width == 0: return 0.5
        
        raw_ratio = (nose.x - min_x) / total_width
        
        # Apply smoothing
        self.ratio_history.append(raw_ratio)
        smoothed_ratio = sum(self.ratio_history) / len(self.ratio_history)
        
        return smoothed_ratio

    def check_frontal_pose(self, image):
        """
        Quick check on a single frame to evaluate if the person is looking straight at camera with eyes open.
        Returns: (is_frontal, quality, info_dict)
        """
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = self.detector.detect(mp_image)
        if not detection_result.face_landmarks:
            return False, 0.0, {}
            
        landmarks = detection_result.face_landmarks[0]
        left_ear = self._calc_ear(landmarks, self.LEFT_EYE)
        right_ear = self._calc_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        turn_ratio = self._calc_head_turn(landmarks)
        
        center_dev = abs(turn_ratio - 0.50)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        # Criteria for clean frontal frame:
        # Centered pose (0.43 - 0.57), eyes open (EAR >= 0.22), low motion blur (sharpness >= 35.0)
        is_frontal = (center_dev <= 0.07) and (avg_ear >= 0.22) and (sharpness >= 35.0)
        quality = (1.0 - center_dev) * 100.0 + min(sharpness, 500.0)
        if avg_ear < 0.22:
            quality *= 0.1
            
        return is_frontal, quality, {
            "turn_ratio": round(turn_ratio, 3),
            "avg_ear": round(avg_ear, 3),
            "sharpness": round(sharpness, 2),
            "is_frontal": is_frontal
        }

    def process_frame(self, image, target_action):
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        detection_result = self.detector.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return False, None, image, {"details": "No face detected in camera"}
            
        landmarks = detection_result.face_landmarks[0]
        
        # Calculate shared metrics
        left_ear = self._calc_ear(landmarks, self.LEFT_EYE)
        right_ear = self._calc_ear(landmarks, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        turn_ratio = self._calc_head_turn(landmarks)
        
        center_dev = abs(turn_ratio - 0.50)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        
        is_frontal = (center_dev <= 0.07) and (avg_ear >= 0.22) and (sharpness >= 35.0)
        quality = (1.0 - center_dev) * 100.0 + min(sharpness, 500.0)
        if avg_ear < 0.22:
            quality *= 0.1
            
        frame_info = {
            "turn_ratio": round(turn_ratio, 3),
            "avg_ear": round(avg_ear, 3),
            "sharpness": round(sharpness, 2),
            "is_frontal": is_frontal,
            "quality": round(quality, 2),
            "warning": None
        }

        violation = None
        
        if target_action == 'blink':
            # Check if user turned head significantly instead of blinking
            if turn_ratio > self.TURN_THRESH_LEFT or turn_ratio < self.TURN_THRESH_RIGHT:
                self.wrong_action_frames += 1
                if self.wrong_action_frames >= 4:
                    violation = "HEAD_TURN_INSTEAD_OF_BLINK"
                    frame_info["warning"] = "Do not turn head! Please BLINK your eyes naturally."
            else:
                self.wrong_action_frames = max(0, self.wrong_action_frames - 1)

            if self.blink_state == 0 and avg_ear < 0.20:
                self.blink_state = 1
            elif self.blink_state == 1 and avg_ear > 0.25:
                self.blink_state = 2
                
            return (self.blink_state == 2), violation, image, frame_info
            
        elif target_action == 'turn_left':
            # Target is turn_ratio > 0.57 (User turning to their LEFT)
            if turn_ratio > self.TURN_THRESH_LEFT:
                self.head_turn_frames += 1
                self.wrong_action_frames = 0
                if self.head_turn_frames >= self.REQUIRED_SUSTAINED_FRAMES:
                    self.head_turn_state = True
            elif turn_ratio < self.TURN_THRESH_RIGHT:
                # User is turning RIGHT (Opposite action!)
                self.head_turn_frames = 0
                self.wrong_action_frames += 1
                frame_info["warning"] = "Wrong direction: You turned RIGHT, but instruction is LEFT!"
                if self.wrong_action_frames >= 4:
                    violation = "OPPOSITE_TURN_RIGHT_INSTEAD_OF_LEFT"
            else:
                # Face is center
                self.head_turn_frames = 0
                self.wrong_action_frames = max(0, self.wrong_action_frames - 1)

            return self.head_turn_state, violation, image, frame_info
            
        elif target_action == 'turn_right':
            # Target is turn_ratio < 0.43 (User turning to their RIGHT)
            if turn_ratio < self.TURN_THRESH_RIGHT:
                self.head_turn_frames += 1
                self.wrong_action_frames = 0
                if self.head_turn_frames >= self.REQUIRED_SUSTAINED_FRAMES:
                    self.head_turn_state = True
            elif turn_ratio > self.TURN_THRESH_LEFT:
                # User is turning LEFT (Opposite action!)
                self.head_turn_frames = 0
                self.wrong_action_frames += 1
                frame_info["warning"] = "Wrong direction: You turned LEFT, but instruction is RIGHT!"
                if self.wrong_action_frames >= 4:
                    violation = "OPPOSITE_TURN_LEFT_INSTEAD_OF_RIGHT"
            else:
                # Face is center
                self.head_turn_frames = 0
                self.wrong_action_frames = max(0, self.wrong_action_frames - 1)

            return self.head_turn_state, violation, image, frame_info
            
        return False, None, image, frame_info
