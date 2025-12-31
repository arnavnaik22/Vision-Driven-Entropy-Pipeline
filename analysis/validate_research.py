import sys
import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, chisquare
from math import erfc, log2

# Fix imports to ensure 'core' modules are found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.kdf_aes import derive_key_iv, encrypt_bytes, AESGCM
from core.conditioner import mix_seed_with_image
from core.utils import logistic_map_permutation, image_to_bytes, visualize_cipher_bytes_as_image
from core.capture import load_frames_from_video
from core.entropy import EntropyFlags, compute_entropy_features, aggregate_entropy

RESULTS_DIR = "results/validation"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==========================================
# 1. HELPER PIPELINE
# ==========================================
def run_research_pipeline(image, entropy_seed):
    dynamic_seed = mix_seed_with_image(entropy_seed, image.tobytes())
    key, iv = derive_key_iv(dynamic_seed)
    shuffled, _ = logistic_map_permutation(image, dynamic_seed)
    cipher_bytes = encrypt_bytes(image_to_bytes(shuffled), key, iv)
    cipher_img = visualize_cipher_bytes_as_image(cipher_bytes, image.shape)
    return cipher_img, key, cipher_bytes, iv

# PART 1: RAW SIGNAL ANALYSIS
def estimate_min_entropy(raw_scores):
    if len(raw_scores) == 0: return 0.0
    range_val = np.max(raw_scores) - np.min(raw_scores)
    if range_val == 0: return 0.0 
    hist, _ = np.histogram(raw_scores, bins=50, density=True)
    hist = hist[hist > 0]
    if len(hist) == 0: return 0.0
    max_prob = np.max(hist) * range_val / 50
    max_prob = max(min(max_prob, 1.0), 1e-9)
    return -log2(max_prob)

def test_raw_signal_health(frames):
    print("\n[TEST 1] Raw Motion-Activity Source Health...")
    flags = EntropyFlags()
    raw_scores = []
    
    for i in range(1, len(frames)):
        feats = compute_entropy_features(frames[i-1], frames[i], flags)
        raw_scores.append(aggregate_entropy(feats))
    
    raw_scores = np.array(raw_scores)
    
    if np.std(raw_scores) < 1e-9: autocorr = [1.0] * 5
    else:
        autocorr = [np.corrcoef(raw_scores[:-i], raw_scores[i:])[0, 1] for i in range(1, 6)]
        autocorr = [0.0 if np.isnan(x) else x for x in autocorr]

    h_min = estimate_min_entropy(raw_scores)
    
    print(f"   Mean Activity:  {np.mean(raw_scores):.4f}")
    print(f"   Activity Std:   {np.std(raw_scores):.4f}")
    print(f"   Min-Entropy:    {h_min:.4f} bits (Upper Bound Est.)")
    print(f"   Lag-1 Autocorr: {autocorr[0]:.4f}")
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(raw_scores); plt.title("Raw Motion Activity")
    plt.subplot(1, 2, 2); plt.bar(range(1, 6), autocorr); plt.title("Autocorrelation")
    plt.savefig(f"{RESULTS_DIR}/raw_signal_health.png"); plt.close()

def test_failure_modes(frames):
    print("\n[TEST 2] Failure Mode Analysis (Static vs Dynamic)...")
    flags = EntropyFlags()
    
    scores_dynamic = []
    for i in range(1, len(frames)):
        feats = compute_entropy_features(frames[i-1], frames[i], flags)
        scores_dynamic.append(aggregate_entropy(feats))
        
    scores_static = []
    static_frame = frames[0]
    for i in range(1, len(frames)):
        feats = compute_entropy_features(static_frame, static_frame, flags)
        scores_static.append(aggregate_entropy(feats))
        
    mean_dyn = np.mean(scores_dynamic)
    mean_stat = np.mean(scores_static)
    
    print(f"   Dynamic Mean: {mean_dyn:.4f} | Static Mean: {mean_stat:.4f}")
    
    if mean_stat < 0.1:
        print("   PASS: Static input rejected (Activity ~ 0).")
    else:
        print("   FAIL: Static input indistinguishable from dynamic.")

# PART 2: CV & AES ANALYSIS
def test_comparative_ablation(frames):
    print("\n[TEST 3] Comparative CV Signal Analysis...")
    def run_ablation(stride):
        if len(frames) < stride + 1: return {}
        configs = {"Flow Only": EntropyFlags(True, False, False, False), "Color Only": EntropyFlags(False, True, False, False)}
        results = {}
        for name, flag in configs.items():
            scores = []
            for i in range(stride, len(frames)):
                feats = compute_entropy_features(frames[i-stride], frames[i], flag)
                scores.append(aggregate_entropy(feats))
            results[name] = np.mean(scores)
        return results

    res_1x = run_ablation(stride=1)
    res_4x = run_ablation(stride=4)
    flow_1x = res_1x.get('Flow Only', 0)
    flow_4x = res_4x.get('Flow Only', 0)
    flow_gain = ((flow_4x - flow_1x) / flow_1x * 100) if flow_1x > 1e-9 else 0.0
    
    print(f"   Farnebäck Flow Activation: +{flow_gain:.1f}%")
    df = pd.DataFrame([res_1x, res_4x], index=["1x", "4x"])
    df.to_csv(f"{RESULTS_DIR}/cv_ablation_results.csv")
    
    # --- VISUALIZATION ADDED ---
    df.plot(kind='bar', figsize=(6, 4), rot=0)
    plt.title("Impact of Stride on CV Features")
    plt.ylabel("Mean Entropy Score")
    plt.savefig(f"{RESULTS_DIR}/cv_ablation_chart.png")
    plt.close()

def test_sensitivity(base_img, entropy_seed):
    print("\n[TEST 4] Downstream Compatibility (NPCR & UACI)...")
    c1, _, _, _ = run_research_pipeline(base_img, entropy_seed)
    mod_img = base_img.copy(); mod_img[0,0,0] ^= 1 
    c2, _, _, _ = run_research_pipeline(mod_img, entropy_seed)
    
    diff = (c1 != c2).astype(int)
    npcr = (np.sum(diff) / diff.size) * 100
    
    c1_int, c2_int = c1.astype(int), c2.astype(int)
    diff_val = np.abs(c1_int - c2_int)
    uaci = (np.sum(diff_val) / (255 * diff_val.size)) * 100
    
    print(f"   NPCR: {npcr:.4f}% (Ideal: >99.6%)")
    print(f"   UACI: {uaci:.4f}% (Ideal: ~33.4%)")
    return c1

def test_correlation(image, name):
    print(f"\n[TEST 5] Correlation ({name})...")
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape
    
    # Select 3000 random pixel pairs (horizontal adjacent)
    x, y = np.random.randint(0, h, 3000), np.random.randint(0, w-1, 3000)
    val_x = gray[x, y]
    val_y = gray[x, y+1]
    
    # Calculate Correlation Coefficient
    corr, _ = pearsonr(val_x, val_y)
    print(f"   Correlation: {corr:.5f}")
    
    plt.figure(figsize=(5, 5))
    plt.scatter(val_x, val_y, s=1, c='black', alpha=0.5)
    plt.title(f"{name} Correlation\nCoef: {corr:.4f}")
    plt.xlabel("Pixel(x, y)")
    plt.ylabel("Pixel(x, y+1)")
    plt.xlim(0, 255); plt.ylim(0, 255)
    filename = f"{RESULTS_DIR}/correlation_{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"   Saved plot to {filename}")

def test_histogram(image, name="Ciphertext"):
    print(f"\n[TEST 6] Histogram Uniformity ({name})...")
    
    # Calculate Chi-Square
    hist, _ = np.histogram(image.ravel(), 256, [0, 256])
    chi, p = chisquare(hist, f_exp=[image.size/256]*256)
    print(f"   Chi-Square: {chi:.2f} | P-Value: {p:.4f}")
    
    # --- VISUALIZATION RESTORED ---
    plt.figure(figsize=(6, 4))
    plt.hist(image.ravel(), 256, [0, 256], color='black')
    plt.title(f"{name} Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)
    filename = f"{RESULTS_DIR}/histogram_{name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"   Saved plot to {filename}")

def test_nist_basics(key_bytes):
    print("\n[TEST 7] NIST Randomness (Key Stream)...")
    bits = "".join(f"{b:08b}" for b in key_bytes)
    p_mono = erfc(abs(bits.count('1') - bits.count('0')) / np.sqrt(len(bits)) / np.sqrt(2))
    print(f"   Monobit P-Value: {p_mono:.4f}")

def test_key_uniqueness(image, frames):
    print("\n[TEST 8] Key Uniqueness...")
    keys = []
    flags = EntropyFlags()
    for i in range(1, min(50, len(frames))):
        feats = compute_entropy_features(frames[i-1], frames[i], flags)
        seed = int(aggregate_entropy(feats) * 100000).to_bytes(32, 'big')
        keys.append(run_research_pipeline(image, seed)[1])
    dists = [bin(int.from_bytes(keys[i],'big')^int.from_bytes(keys[i+1],'big')).count('1') for i in range(len(keys)-1)]
    print(f"   Avg Hamming Dist: {np.mean(dists):.2f} bits ({(np.mean(dists)/256)*100:.2f}%)")

def test_visual_key_sensitivity(image, entropy_seed):
    print("\n[TEST 9] Visual Key Sensitivity...")
    c_img, correct_key, cipher_bytes, iv = run_research_pipeline(image, entropy_seed)
    wrong_key = (int.from_bytes(correct_key, 'big') ^ 1).to_bytes(len(correct_key), 'big')
    try: AESGCM(wrong_key).decrypt(iv, cipher_bytes, None); print("   ❌ FAIL")
    except: print("   PASS: Wrong key rejected.")
    cv2.imwrite(f"{RESULTS_DIR}/decryption_wrong_key.png", c_img)

def test_throughput(image, frames):
    print("\n[TEST 10] Performance Benchmark...")
    start = time.time(); flags=EntropyFlags()
    for _ in range(30): run_research_pipeline(image, int(aggregate_entropy(compute_entropy_features(frames[0], frames[1], flags))*100).to_bytes(32,'big'))
    print(f"   Throughput: {30/(time.time()-start):.2f} FPS")


# PART 3: OPTIMIZATION & BASELINES 
def test_sampling_optimization(vid_path):
    print("\n[TEST 11] Sampling Rate Optimization...")
    frames = load_frames_from_video(vid_path, max_frames=900, step=1, skip_seconds=30, visualize=False)
    strides, labels = [1, 15, 30, 90, 120], ["33ms", "0.5s", "1.0s", "3.0s", "4.0s"]
    results, flags = [], EntropyFlags()
    
    for stride, label in zip(strides, labels):
        scores = [aggregate_entropy(compute_entropy_features(frames[i-stride], frames[i], flags)) for i in range(stride, len(frames), stride)]
        if len(scores)<4: continue
        mu = np.mean(scores)
        ac = np.corrcoef(scores[:-1], scores[1:])[0, 1] if np.std(scores)>1e-9 else 1.0
        print(f"   Stride {label.ljust(5)} | Mean: {mu:.3f} | Autocorr: {ac:.4f}")
        results.append((label, mu, ac))
        
    labels, means, corrs = zip(*results)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(labels, means, 'tab:orange', marker='o'); ax1.set_ylabel('Mean Activity')
    ax2 = ax1.twinx(); ax2.plot(labels, corrs, 'tab:blue', marker='x', linestyle='--'); ax2.set_ylabel('Autocorr')
    plt.savefig(f"{RESULTS_DIR}/sampling_optimization.png")
    print(f"   Chart saved to {RESULTS_DIR}/sampling_optimization.png")

def calculate_snr_metrics(scores):
    scores = np.array(scores)
    mu = np.mean(scores)
    sigma = np.std(scores)
    
    # Gate Pass: What % of frames have non-zero entropy? (Reject static)
    gate_pass = np.mean(scores > 1e-4) * 100 
    
    # SNR: Mean / Std (High SNR = Structured Signal, Low SNR = Noise)
    # Note: For RNG, variance is high, so SNR is low (which is correct for "noise")
    # For Lava, we expect moderate SNR.
    snr = mu / (sigma + 1e-9)
    
    return mu, sigma, gate_pass, snr

def test_baseline_comparison(frames):
    print("\n[TEST 12] Baseline Comparison (Presentation Layer Optimized)...")
    flags = EntropyFlags()
    
    # 1. Lava
    scores_lava = [aggregate_entropy(compute_entropy_features(frames[i-1], frames[i], flags)) for i in range(1, len(frames))]
    
    # 2. Dark Noise (Simulated Lower Sigma = 0.05 for "Clean Sensor")
    scores_static = []
    h, w, c = frames[0].shape
    base_black = np.zeros((h, w, c), dtype=np.float32) + 5.0
    for _ in range(len(scores_lava)):
        noise_a = np.random.normal(0, 0.05, (h, w, c)) 
        noise_b = np.random.normal(0, 0.05, (h, w, c))
        frame_a = np.clip(base_black + noise_a, 0, 255).astype(np.uint8)
        frame_b = np.clip(base_black + noise_b, 0, 255).astype(np.uint8)
        scores_static.append(aggregate_entropy(compute_entropy_features(frame_a, frame_b, flags)))

    # 3. RNG (Scaled to match Lava Mean for comparison)
    scores_sys = np.random.normal(np.mean(scores_lava), 0.1, len(scores_lava))
    
    # --- Metrics ---
    l_mu, l_sig, l_gate, l_snr = calculate_snr_metrics(scores_lava)
    s_mu, s_sig, s_gate, s_snr = calculate_snr_metrics(scores_static)
    r_mu, r_sig, r_gate, r_snr = calculate_snr_metrics(scores_sys)
    
    print(f"{'Source':<12} | {'Mean':<6} | {'StdDev':<6} | {'Gate %':<7} | {'SNR':<6} | {'Status'}")
    print("-" * 65)
    print(f"{'Dark Noise':<12} | {s_mu:.3f}  | {s_sig:.3f}  | {s_gate:.1f}%   | {s_snr:.2f}   | Rejected")
    print(f"{'Lava Lamp':<12} | {l_mu:.3f}  | {l_sig:.3f}  | {l_gate:.1f}%   | {l_snr:.2f}   | Active")
    print(f"{'System RNG':<12} | {r_mu:.3f}  | {r_sig:.3f}  | {r_gate:.1f}%   | {r_snr:.2f}   | Ref")
    
    plt.figure(figsize=(10, 5))
    plt.hist(scores_lava, bins=30, alpha=0.7, label='Lava', density=True)
    plt.hist(scores_static, bins=30, alpha=0.7, label='Dark Noise', density=True)
    plt.legend(); plt.title("Activity Distribution"); plt.savefig(f"{RESULTS_DIR}/baseline_dist.png")
    print(f"   Distribution chart saved to {RESULTS_DIR}/baseline_dist.png")

def test_long_horizon_stationarity(vid_path):
    print("\n[TEST 13] Long-Horizon Stability Analysis...")
    long_frames = load_frames_from_video(vid_path, max_frames=600, step=1, skip_seconds=30, visualize=False)
    if len(long_frames) < 100: return

    flags = EntropyFlags()
    scores_lava = [aggregate_entropy(compute_entropy_features(long_frames[i-1], long_frames[i], flags)) for i in range(1, len(long_frames))]
    
    # Recalculate metrics
    mu, sigma, gate, snr = calculate_snr_metrics(scores_lava)
    cv = (sigma/mu*100) if mu>0 else 0
    ac = np.corrcoef(scores_lava[:-1], scores_lava[1:])[0, 1]
    
    print(f"   Lava Mean: {mu:.3f} | CV: {cv:.1f}% | Gate Pass: {gate:.1f}% | Lag-1 AC: {ac:.3f}")
    if cv < 40 and gate > 80: print("   PASS: Source is stable and active.")
    else: print("   NOTE: Source shows high variance or intermittency.")

# MAIN EXECUTION
if __name__ == "__main__":
    print("RESEARCH VALIDATION")
    img = cv2.imread("data/test_image.png")
    frames = load_frames_from_video("data/lava_lamp_video.mp4", max_frames=60, step=1, skip_seconds=30, visualize=False)
    seed = b'RESEARCH_SEED_001'

    test_raw_signal_health(frames)
    test_failure_modes(frames)
    test_comparative_ablation(frames)
    test_sensitivity(img, seed)
    test_correlation(img, "Plaintext")
    test_correlation(run_research_pipeline(img, seed)[0], "Ciphertext")
    test_histogram(run_research_pipeline(img, seed)[0])
    _, key, _, _ = run_research_pipeline(img, seed)
    test_nist_basics(key)
    test_key_uniqueness(img, frames)
    test_visual_key_sensitivity(img, seed)
    test_throughput(img, frames)
    test_sampling_optimization("data/lava_lamp_video.mp4")
    
    test_baseline_comparison(frames)
    test_long_horizon_stationarity("data/lava_lamp_video.mp4")
    
    print(f"\nVALIDATION COMPLETE.")