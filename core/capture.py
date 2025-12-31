import cv2
import numpy as np
import pandas as pd
from core.entropy import compute_entropy_features, aggregate_entropy, EntropyFlags

def load_frames_from_video(path, max_frames=120, step=3, skip_seconds=0, resize_scale=0.3, visualize=True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")

    # Skip Initial Seconds (Warmup)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 0 and skip_seconds > 0:
        frames_to_skip = int(fps * skip_seconds)
        print(f"Skipping first {skip_seconds}s ({frames_to_skip} frames) for stabilization...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_to_skip)

    frames = []
    entropy_scores = []
    prev_frame = None
    frame_idx = 0
    flags = EntropyFlags() 

    print("\nCapture: Extracting physical entropy (CV-Focused)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        frame = cv2.resize(frame, (int(frame.shape[1]*resize_scale), int(frame.shape[0]*resize_scale)))
        
        if prev_frame is not None:
            feats = compute_entropy_features(prev_frame, frame, flags)
            total_score = aggregate_entropy(feats)
            entropy_scores.append(total_score)

            if visualize:
                cv2.imshow("Entropy Extraction", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        prev_frame = frame
        frames.append(frame)
        frame_idx += 1
        
        if len(frames) >= max_frames: break

    cap.release()
    if visualize: cv2.destroyAllWindows()

    df = pd.DataFrame({"Frame": range(len(entropy_scores)), "Entropy": entropy_scores})
    df.to_csv("results/entropy.csv", index=False)
    
    print(f" Extracted {len(frames)} frames. Raw entropy saved to 'results/entropy.csv'.")
    return frames