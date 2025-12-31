import argparse
import os
import cv2
import numpy as np
from core.capture import load_frames_from_video
from core.entropy import aggregate_entropy_bytes
from core.conditioner import mix_seed_with_image
from core.kdf_aes import derive_key_iv, encrypt_bytes
from core.utils import logistic_map_permutation, image_to_bytes, visualize_cipher_bytes_as_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/lava_lamp_video.mp4")
    parser.add_argument("--target", default="data/test_image.png")
    args = parser.parse_args()
    
    print("\nENCRYPTING...")
    frames = load_frames_from_video(args.video, visualize=False)
    seed = aggregate_entropy_bytes(np.loadtxt("results/entropy.csv", delimiter=",", skiprows=1, usecols=1))
    
    img = cv2.imread(args.target)
    dynamic_seed = mix_seed_with_image(seed, img.tobytes())
    key, iv = derive_key_iv(dynamic_seed)
    shuffled, _ = logistic_map_permutation(img, dynamic_seed)
    cipher = encrypt_bytes(image_to_bytes(shuffled), key, iv)
    cv2.imwrite("results/encrypted_demo.png", visualize_cipher_bytes_as_image(cipher, img.shape))
    print(" Done: results/encrypted_demo.png")

if __name__ == "__main__":
    main()