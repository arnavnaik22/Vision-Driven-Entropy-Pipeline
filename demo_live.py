import cv2
import numpy as np
import time
from collections import deque

from core.entropy import compute_entropy_features, aggregate_entropy, EntropyFlags
from core.conditioner import mix_seed_with_image
from core.kdf_aes import derive_key_iv, encrypt_bytes
from core.utils import logistic_map_permutation, image_to_bytes, visualize_cipher_bytes_as_image

# --- CONFIG ---
VIDEO_PATH = "data/lava_lamp_video.mp4"
TARGET_IMG = "data/test_image.png"

# --- TUNING PARAMETERS ---
SKIP_FRAMES = 4         # Compare Frame t vs t+4 (Simulates 4x speed)
MIN_MOTION_THRESH = 1.0 # Ignore movements smaller than 1 pixel
ARROW_SCALE = 2.0       # Visually lengthen arrows

def get_lava_mask(frame):
    """
    Creates a mask to focus ONLY on the bright/colorful lava blobs.
    Excludes the dark background and static bottle edges.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Filter for high saturation (color) and value (brightness)
    lower_bound = np.array([0, 60, 60]) 
    upper_bound = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    
    # Clean up noise
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask

def draw_dashboard(frame, flow, corners, entropy_history):
    h, w = frame.shape[:2]
    vis = frame.copy()
    
    # 1. Optical Flow Arrows
    if flow is not None:
        step = 24 
        y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        
        mag = np.sqrt(fx**2 + fy**2)
        valid_mask = mag > MIN_MOTION_THRESH
        
        lines = np.vstack([x, y, x + fx*ARROW_SCALE, y + fy*ARROW_SCALE]).T.reshape(-1, 2, 2)
        lines = lines[valid_mask]
        lines = np.int32(lines + 0.5)
        cv2.polylines(vis, lines, 0, (0, 255, 255), 2) 

    # 2. Corners (Green Crosses)
    if corners is not None:
        for i in corners:
            x_c, y_c = i.ravel()
            cv2.line(vis, (int(x_c)-4, int(y_c)), (int(x_c)+4, int(y_c)), (0, 255, 0), 1)
            cv2.line(vis, (int(x_c), int(y_c)-4), (int(x_c), int(y_c)+4), (0, 255, 0), 1)

    # 3. Entropy Graph
    graph_h = 100
    overlay = np.zeros((graph_h, w, 3), dtype=np.uint8)
    if len(entropy_history) > 1:
        points = []
        for i, val in enumerate(entropy_history):
            # Scale entropy (typically 0-3) to graph height
            y_pos = graph_h - int((val / 3.0) * (graph_h - 10)) 
            y_pos = max(0, min(graph_h-1, y_pos))
            x_pos = int((i / len(entropy_history)) * w)
            points.append((x_pos, y_pos))
        cv2.polylines(overlay, [np.array(points)], False, (0, 165, 255), 2)

    vis = np.vstack([vis, overlay])
    return vis

def run_live_demo():
    print("STARTING OPTIMIZED CV DEMO (Speedup: 4x)...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    target = cv2.imread(TARGET_IMG)
    
    entropy_history = deque(maxlen=100)
    raw_bytes = bytearray()
    
    ret, prev_frame = cap.read()
    if not ret: return
    prev_frame = cv2.resize(prev_frame, (640, 360))
    
    # Flags for standard extraction (we will override logic slightly for visual demo)
    flags = EntropyFlags() 

    while True:
        # 1. Time Lapse Loop (Skip frames for speed)
        for _ in range(SKIP_FRAMES):
            ret, _ = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                cap.read()
        
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (640, 360))
        
        # 2. Generate Mask (Focus on Lava) -- FIXED: Defined BEFORE use
        lava_mask = get_lava_mask(frame)
        
        # 3. CV Pipeline (Masked)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # A. Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate Flow Entropy ONLY inside the lava mask (The "CV Justification" Step)
        masked_flow_score = cv2.mean(mag, mask=lava_mask)[0]
        
        # B. Corners (Strictly Masked)
        corners = cv2.goodFeaturesToTrack(curr_gray, maxCorners=50, qualityLevel=0.02, minDistance=15, mask=lava_mask)
        corner_score = (len(corners)/50.0) if corners is not None else 0
        
        # 4. Aggregate Score (Visual Demo Logic)
        # We weigh flow higher here because it looks cooler on the graph
        feat_score = masked_flow_score * 0.7 + corner_score * 0.3
        
        entropy_history.append(feat_score)
        raw_bytes.append(int((feat_score * 100) % 255))
        
        # 5. Visualization
        vis = draw_dashboard(frame, flow, corners, entropy_history)
        
        status_text = f"Speed: {SKIP_FRAMES}x | Entropy: {feat_score:.3f}"
        cv2.putText(vis, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, "SPACE: Encrypt | Q: Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Optimized CV Dashboard", vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord(' '):
            print("\n📸 ENCRYPTING...")
            seed = bytes(raw_bytes[-32:]) if len(raw_bytes) > 32 else b'\x00'*32
            final_seed = mix_seed_with_image(seed, target.tobytes())
            
            key_aes, iv = derive_key_iv(final_seed)
            shuffled, _ = logistic_map_permutation(target, final_seed)
            cipher = encrypt_bytes(image_to_bytes(shuffled), key_aes, iv)
            res = visualize_cipher_bytes_as_image(cipher, target.shape)
            
            cv2.imshow("Encrypted Result", res)
            cv2.waitKey(0)
            
        prev_frame = frame
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_demo()