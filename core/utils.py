import numpy as np

def image_to_bytes(img):
    return img.tobytes()

def bytes_to_image(b, shape):
    return np.frombuffer(b, dtype=np.uint8).reshape(shape)

def logistic_map_permutation(img, seed_bytes):
    """
    Performs Spatial Decorrelation using a Logistic Chaotic Map.
    
    Formula: x(n+1) = r * x(n) * (1 - x(n))
    """
    h, w, c = img.shape
    N = h * w
    
    # 1. Initialize parameters from seed
    seed_val = int.from_bytes(seed_bytes[:8], 'big')
    x0 = (seed_val / 2**64)
    x0 = 0.1 + (x0 * 0.8) # Normalize to 0.1-0.9 to avoid stable points
    
    seed_r = int.from_bytes(seed_bytes[8:16], 'big')
    r = 3.99 + (seed_r / 2**64) * 0.01 # r approx 4.0 for hyper-chaos
    
    # 2. Generate chaotic sequence
    chaotic_seq = np.zeros(N, dtype=np.float64)
    x = x0
    
    # Warmup to enter attractor
    for _ in range(1000):
        x = r * x * (1 - x)
        
    for i in range(N):
        x = r * x * (1 - x)
        chaotic_seq[i] = x
        
    # 3. Create permutation indices
    perm_indices = np.argsort(chaotic_seq)
    
    # 4. Shuffle pixels
    flat = img.reshape(-1, c)
    shuffled_flat = flat[perm_indices]
    
    return shuffled_flat.reshape(h, w, c), perm_indices

def visualize_cipher_bytes_as_image(cipher, shape):
    """Visualizes encrypted bytes as a noise image for demonstration."""
    h, w, c = shape
    arr = np.frombuffer(cipher, dtype=np.uint8)
    size = h * w * c
    if len(arr) < size:
        arr = np.pad(arr, (0, size - len(arr)), 'constant')
    else:
        arr = arr[:size]
    return arr.reshape((h, w, c))