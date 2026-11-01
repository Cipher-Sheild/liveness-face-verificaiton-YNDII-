import cv2
import numpy as np

def blur_score(face_crop):
    """
    Computes a blur score based on the Variance of the Laplacian.
    A lower variance indicates fewer edges, meaning the image is likely blurred.
    Spoof attacks (like printed photos or screens) can sometimes be blurrier 
    than real faces due to focus issues or re-capture quality.
    
    Returns a score 0.0 to 1.0 (higher = sharper/more live-like).
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
        
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Normalize the variance to a 0-1 scale. 
    # The max_variance threshold might need tuning based on camera quality.
    max_variance = 500.0
    score = min(variance / max_variance, 1.0)
    
    print(f"[DEBUG] blur: variance={variance:.2f} -> score={score:.4f}")
    
    return score

def moire_score(face_crop):
    """
    Checks for Moiré patterns typical of screen replay attacks using 2D FFT.
    Screens have pixel grids that create high-frequency periodic patterns when recaptured.
    By analyzing the Fourier transform, we can detect anomalous high-frequency energy.
    
    Returns a score 0.0 to 1.0 (higher = less moire/more live-like).
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
        
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    
    # Compute 2D Discrete Fourier Transform
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate true magnitude (not log-scaled!)
    # Log-scaling compresses dynamic range and makes the outer pixels dominate the sum,
    # causing the hf_ratio to always be extremely high.
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    
    # Create a mask to block low frequencies (center of the spectrum)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    r = int(min(rows, cols) * 0.15) # Mask radius
    
    mask = np.ones((rows, cols), np.float32)
    cv2.circle(mask, (ccol, crow), r, 0, -1)
    
    # Apply mask to get high-frequency magnitude
    high_freq_magnitude = magnitude * mask
    
    # Calculate the ratio of high frequency energy
    total_energy = np.sum(magnitude)
    high_freq_energy = np.sum(high_freq_magnitude)
    
    hf_ratio = high_freq_energy / (total_energy + 1e-5)
    
    # Print intermediate values for debugging
    
    # CALIBRATION LOGIC (Smooth Scaling):
    # A normal live face on webcam yields an hf_ratio around 0.45.
    # Screen recaptures produce much higher hf_ratios (e.g. 0.60 to 0.80+).
    # Instead of hard-clamping everything below 0.48 to 1.0, we will map 
    # the range [min_hf_ratio, max_hf_ratio] linearly to a score [1.0, 0.0].
    
    min_hf_ratio = 0.43   # High confidence live (Score -> 1.0)
    max_hf_ratio = 0.68   # Strong moire / spoof (Score -> 0.0)
    
    hf_clamped = np.clip(hf_ratio, min_hf_ratio, max_hf_ratio)
    score = 1.0 - ((hf_clamped - min_hf_ratio) / (max_hf_ratio - min_hf_ratio))
        
    print(f"[DEBUG] moire: hf_ratio={hf_ratio:.4f} (total_energy={total_energy:.1f}, hf_energy={high_freq_energy:.1f}) -> score={score:.4f}")
    
    return score

def reflection_score(face_crop):
    """
    Detects strong specular reflections (glare) typical of glossy paper or screens.
    Natural faces have softer, graded reflections, whereas screens/photos have hard, bright spots.
    
    Returns a score 0.0 to 1.0 (higher = less glare/more live-like).
    """
    if face_crop is None or face_crop.size == 0:
        return 0.0
        
    # Convert to HSV to isolate brightness (Value)
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    
    # Threshold to find very bright regions (glare)
    # 240 is very bright (max 255)
    _, binary = cv2.threshold(v_channel, 240, 255, cv2.THRESH_BINARY)
    
    # Calculate the proportion of the image that is glaring
    glare_pixels = cv2.countNonZero(binary)
    total_pixels = face_crop.shape[0] * face_crop.shape[1]
    glare_ratio = glare_pixels / total_pixels
    
    # A high glare ratio means lots of reflection. 
    # If more than e.g. 5% of the face crop is pure white glare, it's highly suspect.
    # Note: glare_ratio is usually very small (e.g., 0.001 - 0.01 for natural skin highlights).
    max_glare_ratio = 0.05
    penalty = min(glare_ratio / max_glare_ratio, 1.0)
    score = 1.0 - penalty
    
    print(f"[DEBUG] reflection: glare_ratio={glare_ratio:.6f} ({glare_pixels}/{total_pixels} px) -> score={score:.4f}")
    
    return score
