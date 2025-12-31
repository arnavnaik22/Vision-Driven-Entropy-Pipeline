import numpy as np
import cv2
from dataclasses import dataclass
from skimage.feature import local_binary_pattern

# --- CONFIGURATION ---
MOTION_THRESHOLD = 0.05 

@dataclass
class EntropyFlags:
    use_flow: bool = True
    use_color: bool = True
    use_corners: bool = True
    use_texture: bool = True

def mean_hsv_drift(prev, curr):
    """Calculates Color Drift in HSV space (Environmental Changes)."""
    hsv1 = cv2.cvtColor(prev, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(curr, cv2.COLOR_BGR2HSV)
    return np.linalg.norm(np.array(cv2.mean(hsv2)[:3]) - np.array(cv2.mean(hsv1)[:3]))

def optical_flow_entropy(prev_gray, gray):
    """Calculates Motion Entropy via Dense Optical Flow (Farneback)."""
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag), flow

def shi_tomasi_entropy(prev_gray):
    """Calculates Topological Entropy via Corner Density (Shi-Tomasi)."""
    corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    if corners is None:
        return 0.0, None
    return len(corners) / 100.0, corners

def lbp_texture_entropy(frame):
    """Calculates Texture Entropy via Local Binary Patterns."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    return entropy

def compute_entropy_features(prev_frame, curr_frame, flags: EntropyFlags = EntropyFlags()):
    """Core CV-Signal Extraction. Returns a dictionary of un-weighted entropy proxies."""
    features = {}
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    if flags.use_flow:
        val, flow = optical_flow_entropy(prev_gray, curr_gray)
        features['flow_mag'] = val
        features['flow_field'] = flow 
    
    if flags.use_color:
        features['color_drift'] = mean_hsv_drift(prev_frame, curr_frame)
        
    if flags.use_corners:
        val, corners = shi_tomasi_entropy(prev_gray)
        features['corner_density'] = val
        features['corners'] = corners

    if flags.use_texture:
        features['texture_entropy'] = lbp_texture_entropy(curr_frame)
        
    return features

def aggregate_entropy(features):
    """
    Aggregates normalized CV features into a single scalar proxy.
    Applies Motion Gating to reject static sensor noise.
    """
    flow = features.get('flow_mag', 0.0)
    drift = features.get('color_drift', 0.0)
    
    # --- MOTION GATING ---
    if flow < MOTION_THRESHOLD and drift < MOTION_THRESHOLD:
        return 0.0
    # ---------------------

    score = 0.0
    if 'flow_mag' in features: score += features['flow_mag'] * 0.4
    if 'color_drift' in features: score += features['color_drift'] * 0.3
    if 'corner_density' in features: score += features['corner_density'] * 0.2
    if 'texture_entropy' in features: score += features['texture_entropy'] * 0.1
    return score

def aggregate_entropy_bytes(entropy_values):
    """Converts a list of scalar scores into a byte sequence."""
    if len(entropy_values) == 0: return b'\x00'
    e_min, e_max = np.min(entropy_values), np.max(entropy_values)
    if e_max == e_min: e_max += 1e-9
    normalized = np.interp(entropy_values, (e_min, e_max), (0, 255)).astype(np.uint8)
    return normalized.tobytes()