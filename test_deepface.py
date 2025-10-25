"""
test_deepface.py

Test DeepFace model installation and download.
Run this once before using the registration feature.
"""
import cv2
import numpy as np
print("Testing DeepFace installation...\n")

# Create a simple test image
print("[1/3] Creating test face image...")
test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

# Try to import and use DeepFace
print("[2/3] Loading DeepFace...")
try:
    from deepface import DeepFace
    print("✅ DeepFace imported successfully")
except ImportError as e:
    print(f"❌ Failed to import DeepFace: {e}")
    print("Run: pip install deepface")
    exit(1)

# Try to download/use the model
print(f"[3/3] Testing face embedding extraction...")
print("NOTE: First run will download the Facenet model (~90MB). This may take a minute...")

try:
    # This will trigger model download if not already cached
    result = DeepFace.represent(
        img_path=test_img,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend='skip'
    )
    
    if result and len(result) > 0:
        embedding = result[0]["embedding"]
        print(f"✅ Successfully extracted {len(embedding)}-dimensional embedding")
        print(f"\n{'='*60}")
        print("✅ DeepFace is ready to use!")
        print("You can now run: python register_face.py")
        print(f"{'='*60}\n")
    else:
        print("⚠️ DeepFace returned empty result")
        
except Exception as e:
    print(f"❌ Error during test: {e}")
    print("\nTroubleshooting:")
    print("1. Check your internet connection (needed for first-time model download)")
    print("2. Wait a moment and try again")
    print("3. Check if you have enough disk space (~100MB needed)")
