import hashlib

def sha256_digest(seed_bytes: bytes) -> bytes:
    """Standard SHA-256 hashing for entropy conditioning."""
    return hashlib.sha256(seed_bytes).digest()

def mix_seed_with_image(seed_bytes: bytes, image_bytes: bytes) -> bytes:
    """
    Mixes the entropy seed with the SHA-256 hash of the target image.
    This ensures that the final seed is dependent on both the entropy source
    and the specific image being encrypted.
    """
    image_hash = hashlib.sha256(image_bytes).digest()
    # Combine entropy + image hash
    final_seed = hashlib.sha256(seed_bytes + image_hash).digest()
    return final_seed

def sha256_digest_verbose(seed_bytes: bytes) -> bytes:
    """Verbose hashing for demonstration output."""
    print("============================================================")
    print(" [STEP 5A] CONDITIONING ENTROPY VIA SHA-256")
    print("============================================================")
    print(f"   ▪ Input seed (first 32 bytes): {seed_bytes[:32].hex()}...")
    digest = hashlib.sha256(seed_bytes).digest()
    print(f"   ▪ SHA-256 Digest (256-bit output): {digest.hex()}")
    print("============================================================\n")
    return digest