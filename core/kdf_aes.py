from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

def derive_key_iv(seed_bytes, key_len=32, iv_len=12, info=b'rme'):
    """Derives a 256-bit Key and 96-bit IV from the entropy seed."""
    hkdf = HKDF(algorithm=hashes.SHA256(), length=key_len+iv_len, salt=None, info=info)
    okm = hkdf.derive(seed_bytes)
    return okm[:key_len], okm[key_len:key_len+iv_len]

def encrypt_bytes(data, key, iv):
    """Standard AES-256-GCM Encryption."""
    return AESGCM(key).encrypt(iv, data, None)

def decrypt_bytes(cipher, key, iv):
    """Standard AES-256-GCM Decryption."""
    return AESGCM(key).decrypt(iv, cipher, None)