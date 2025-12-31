import cv2
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from core.entropy import compute_entropy_features, aggregate_entropy, EntropyFlags

VIDEO_PATH = "data/lava_lamp_video.mp4"

def run_layer_demo():
    print(" STARTING LAYER DEMO...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    hist_flow, hist_drift, hist_corners = deque(maxlen=50), deque(maxlen=50), deque(maxlen=50)
    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (320, 240))
    flags = EntropyFlags() 
    
    while True:
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        frame = cv2.resize(frame, (320, 240))
        feats = compute_entropy_features(prev_frame, frame, flags)
        
        hist_flow.append(feats.get('flow_mag', 0))
        hist_drift.append(feats.get('color_drift', 0))
        hist_corners.append(feats.get('corner_density', 0))
        
        panel_2 = np.zeros_like(frame)
        if 'flow_field' in feats:
            mag, _ = cv2.cartToPolar(feats['flow_field'][..., 0], feats['flow_field'][..., 1])
            panel_2 = cv2.applyColorMap(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET)
            
        fig, ax = plt.subplots(figsize=(3.2, 2.4), dpi=100)
        ax.plot(hist_flow, 'c', label='Flow'); ax.plot(hist_drift, 'm', label='Drift'); ax.plot(hist_corners, 'g', label='Corn')
        ax.legend(fontsize=6); ax.set_facecolor('black'); fig.patch.set_facecolor('black'); ax.grid(alpha=0.2)
        canvas = FigureCanvas(fig); canvas.draw(); plt.close(fig)
        panel_4 = cv2.cvtColor(np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(canvas.get_width_height()[::-1] + (3,)), cv2.COLOR_RGB2BGR)

        cv2.imshow("Layers", np.vstack([np.hstack([frame, panel_2]), np.hstack([np.zeros_like(frame), panel_4])]))
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        prev_frame = frame
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    run_layer_demo()