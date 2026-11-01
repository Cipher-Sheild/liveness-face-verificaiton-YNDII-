import os
import cv2
import numpy as np
import onnxruntime as ort
from .utils import get_project_root, download_file

MODELS = {
    '4.0': {
        'name': '4_0_0_80x80_MiniFASNetV1SE.onnx',
        'url': 'https://raw.githubusercontent.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/master/onnx/4_0_0_80x80_MiniFASNetV1SE.onnx'
    },
    '2.7': {
        'name': '2.7_80x80_MiniFASNetV2.onnx',
        'url': 'https://raw.githubusercontent.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/master/onnx/2.7_80x80_MiniFASNetV2.onnx'
    }
}

class AntiSpoofModel:
    def __init__(self):
        self.sessions = {}
        self.input_names = {}
        
        models_dir = os.path.join(get_project_root(), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for scale, info in MODELS.items():
            model_path = os.path.join(models_dir, info['name'])
            if not os.path.exists(model_path):
                print(f"Downloading Anti-Spoof ONNX model {info['name']}...")
                try:
                    download_file(info['url'], model_path)
                except Exception as e:
                    print(f"Error downloading ONNX model: {e}")
                    
            if os.path.exists(model_path):
                try:
                    session = ort.InferenceSession(model_path)
                    self.sessions[scale] = session
                    self.input_names[scale] = session.get_inputs()[0].name
                    print(f"Loaded Anti-Spoof ONNX model: {info['name']}")
                except Exception as e:
                    print(f"Error loading ONNX model: {e}")
            else:
                print(f"Warning: Model {info['name']} could not be loaded.")

    def preprocess(self, image):
        img = cv2.resize(image, (80, 80))
        img = img.astype(np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def predict_spoof(self, face_crops):
        scores = []
        for scale, session in self.sessions.items():
            if scale not in face_crops: continue
            crop = face_crops[scale]
            if crop is None or crop.size == 0: continue
            
            input_data = self.preprocess(crop)
            try:
                outputs = session.run(None, {self.input_names[scale]: input_data})
                logits = outputs[0][0]
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)
                live_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
                scores.append(live_score)
            except Exception as e:
                print(f'Error during ONNX inference on {scale}: {e}')
                
        if not scores:
            return -1.0
        return sum(scores) / len(scores)
