import cv2
import numpy as np

from .face_detector import get_face_detector, crop_face, get_scaled_crop
from .anti_spoof_model import AntiSpoofModel
from .heuristic_checks import blur_score, moire_score, reflection_score

WEIGHT_MODEL = 0.85
WEIGHT_HEURISTICS = 0.15
THRESHOLD_LIVE = 0.6
THRESHOLD_SPOOF = 0.4

_detector = None
_model = None

def init_pipeline():
    global _detector, _model
    if _detector is None:
        _detector = get_face_detector()
    if _model is None:
        _model = AntiSpoofModel()

def run_liveness_check(image):
    init_pipeline()
    
    result = {
        'result': 'uncertain',
        'confidence': 0.0,
        'details': {}
    }
    
    bboxes = _detector.get_bboxes(image)
    if not bboxes:
        result['details'] = 'No face detected'
        return result
        
    if len(bboxes) > 1:
        result['details'] = f'Multiple faces detected ({len(bboxes)}). Liveness check requires exactly one face.'
        result['result'] = 'spoofed' # Safe fail
        return result
        
    bbox = bboxes[0]
        
    tight_crop = crop_face(image, bbox, margin=0.2)
    crop_4 = get_scaled_crop(image, bbox, scale=4.0)
    crop_2_7 = get_scaled_crop(image, bbox, scale=2.7)
    crop_6 = get_scaled_crop(image, bbox, scale=6.0)
    
    if crop_6 is None: 
        crop_6 = image
        
    if tight_crop is None or tight_crop.size == 0:
        return result

    face_crops = {'4.0': crop_4, '2.7': crop_2_7}
    
    # Calculate fusion score directly
    model_score = _model.predict_spoof(face_crops)
    
    score_blur = blur_score(tight_crop)
    score_moire = moire_score(crop_6)
    score_reflection = reflection_score(crop_6)
    
    avg_heuristics = (score_blur + score_moire + score_reflection) / 3.0
    
    if model_score < 0:
        final_score = avg_heuristics
    else:
        final_score = (model_score * WEIGHT_MODEL) + (avg_heuristics * WEIGHT_HEURISTICS)
        
    confidence = abs(final_score - 0.5) * 2
    
    if final_score >= THRESHOLD_LIVE:
        verdict = 'live'
    elif final_score <= THRESHOLD_SPOOF:
        verdict = 'spoofed'
    else:
        verdict = 'uncertain'
        
    result['result'] = verdict
    result['confidence'] = round(float(confidence), 4)
    result['details'] = {
        'model_score': round(float(model_score), 4),
        'blur_score': round(float(score_blur), 4),
        'moire_score': round(float(score_moire), 4),
        'reflection_score': round(float(score_reflection), 4),
        'final_score': round(float(final_score), 4)
    }
    
    return result
