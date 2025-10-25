"""
face_registry.py

Manages the database of registered face embeddings.

- Loads and saves face embeddings from a JSON file.
- Computes the mean embedding for a list of face encodings.
- Finds the best match for a given face embedding from the registry.
"""
import json
import cv2
import numpy as np
from scipy.spatial.distance import cosine

REGISTRY_FILE = "face_encodings.json"
MATCH_THRESHOLD = 0.4  # FIXED: Stricter threshold for better accuracy (was 0.5)

# Lazy import InsightFace to avoid loading models until needed
_face_analyzer = None

# Find this function:
def _get_face_analyzer():
    """Lazy-load InsightFace face analyzer."""
    global _face_analyzer
    if _face_analyzer is None:
        try:
            from insightface.app import FaceAnalysis
            print("[INFO] Loading InsightFace model (first-time setup may take a moment)...")

            # --- OPTIMIZED: Prioritize CUDA (GPU), fallback to CPU ---
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"[INFO] Using providers: {providers}")
            # --- END OF OPTIMIZATION ---

            app = FaceAnalysis(name='buffalo_l', providers=providers)
            app.prepare(ctx_id=0, det_size=(640, 640))
            _face_analyzer = app
            print("[INFO] InsightFace model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load InsightFace: {e}")
            import traceback
            traceback.print_exc()
            print("[ERROR] Face recognition features will be disabled.")
            print("[HINT] Make sure insightface is properly installed: pip install insightface")
            _face_analyzer = False

    return _face_analyzer if _face_analyzer is not False else None

def load_registry():
    """Loads the face registry from the JSON file."""
    try:
        with open(REGISTRY_FILE, 'r') as f:
            registry = json.load(f)
            # Convert lists back to numpy arrays
            for name in registry:
                registry[name] = np.array(registry[name])
            print(f"[INFO] Loaded {len(registry)} registered faces.")
            return registry
    except FileNotFoundError:
        print("[INFO] No face registry found. Starting with an empty one.")
        return {}

def save_registry(registry):
    """Saves the face registry to the JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    serializable_registry = {}
    for name, embedding in registry.items():
        # Handle both numpy arrays and lists
        if isinstance(embedding, np.ndarray):
            serializable_registry[name] = embedding.tolist()
        elif isinstance(embedding, list):
            serializable_registry[name] = embedding
        else:
            print(f"[WARN] Unknown embedding type for {name}: {type(embedding)}")
            serializable_registry[name] = list(embedding)
    
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(serializable_registry, f, indent=2)
    print(f"[INFO] Face registry saved with {len(registry)} faces.")

def get_face_embedding(face_img):
    """
    Extracts face embedding from a face image using InsightFace.
    
    Args:
        face_img: Cropped face image (numpy array in BGR format)
        
    Returns:
        Face embedding as numpy array, or None if extraction fails
    """
    app = _get_face_analyzer()
    if app is None:
        print("[ERROR] InsightFace analyzer is not available (model failed to load).")
        return None
    
    # Validate input
    if face_img is None or face_img.size == 0:
        print("[WARN] Invalid face image provided")
        return None
    
    # Check minimum size
    h, w = face_img.shape[:2]
    # We need a large enough "scene" for the detector to work
    if h < 64 or w < 64:
        print(f"[WARN] Face image too small ({w}x{h}). Minimum is 64x64 pixels.")
        return None
        
    try:
        # Log the incoming patch size for debugging
        print(f"[DEBUG] Embedding input patch: {w}x{h}")

        # Primary attempt: use the patch as-is
        faces = app.get(face_img)
        
        if faces and len(faces) > 0:
            # Sort by largest face found in the patch, just in case
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            
            # Use the largest face found
            embedding = faces[0].embedding
            print(f"[INFO] Successfully extracted {len(embedding)}-dimensional embedding")
            return embedding

        # Fallback 1: upscale small patches to help detector
        min_dim = min(h, w)
        scale = 1.0
        if min_dim < 400:
            # Target minimum dimension ~640 for better detection
            scale = max(1.0, 640.0 / float(max(1, min_dim)))
        if scale > 1.01 and scale < 5.0:
            new_w, new_h = int(w * scale), int(h * scale)
            up = cv2.resize(face_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            print(f"[DEBUG] Fallback1: Upscaled to {new_w}x{new_h} (scale {scale:.2f})")
            faces = app.get(up)
            if faces and len(faces) > 0:
                faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                embedding = faces[0].embedding
                print(f"[INFO] Extracted embedding after upscaling ({len(embedding)}D)")
                return embedding

        # Fallback 2: slight brightness/contrast boost
        bright = cv2.convertScaleAbs(face_img, alpha=1.2, beta=15)
        print("[DEBUG] Fallback2: brightness/contrast boost applied")
        faces = app.get(bright)
        if faces and len(faces) > 0:
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            embedding = faces[0].embedding
            print(f"[INFO] Extracted embedding after brightness boost ({len(embedding)}D)")
            return embedding

        # Fallback 3: try RGB (defensive; typically BGR works)
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        print("[DEBUG] Fallback3: trying RGB input")
        faces = app.get(rgb)
        if faces and len(faces) > 0:
            faces = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            embedding = faces[0].embedding
            print(f"[INFO] Extracted embedding with RGB input ({len(embedding)}D)")
            return embedding

        print("[WARN] No face detected in the image by InsightFace after all fallbacks")
        return None
            
    except Exception as e:
        print(f"[ERROR] Failed to extract embedding: {e}")
        return None

def find_match(face_encoding, registry):
    """
    Finds the best match for a given face encoding in the registry.

    Args:
        face_encoding: The encoding of the face to match.
        registry: The dictionary of known face names and their encodings.

    Returns:
        A tuple of (name, distance) if a match is found, otherwise (None, None).
    """
    if not registry or face_encoding is None:
        return None, None

    best_name = None
    best_distance = float('inf')
    
    for name, stored_embedding in registry.items():
        # Calculate cosine distance
        distance = cosine(face_encoding, stored_embedding)
        
        if distance < best_distance:
            best_distance = distance
            best_name = name
    
    # Return match only if distance is below threshold
    if best_distance < MATCH_THRESHOLD:
        return best_name, best_distance
    
    return None, None
