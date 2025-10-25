"""
Anti-KYC Effectiveness Test
Tests if distorted faces can still be recognized by face recognition systems
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, '.')

def apply_anti_kyc_distortion(frame, x1, y1, x2, y2, intensity=5):
    """Apply Anti-KYC distortion."""
    face_region = frame[y1:y2, x1:x2].copy()
    h, w = face_region.shape[:2]
    
    if h < 20 or w < 20:
        return frame
    
    warp_strength = intensity * 0.3
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    eye_mask = (y_grid < h // 3).astype(np.float32)
    dx_eye = np.sin(x_grid * 0.3) * warp_strength * eye_mask
    dy_eye = np.sin(y_grid * 0.3) * warp_strength * eye_mask
    
    nose_mask = ((y_grid >= h // 3) & (y_grid < 2 * h // 3)).astype(np.float32)
    dx_nose = np.sin(x_grid * 0.2 + y_grid * 0.2) * warp_strength * 0.5 * nose_mask
    dy_nose = np.sin(y_grid * 0.2) * warp_strength * 0.5 * nose_mask
    
    mouth_mask = (y_grid >= 2 * h // 3).astype(np.float32)
    dx_mouth = np.sin(x_grid * 0.25) * warp_strength * 0.7 * mouth_mask
    dy_mouth = np.sin(y_grid * 0.25) * warp_strength * 0.7 * mouth_mask
    
    dx = dx_eye + dx_nose + dx_mouth
    dy = dy_eye + dy_nose + dy_mouth
    
    map_x = (x_grid + dx).astype(np.float32)
    map_y = (y_grid + dy).astype(np.float32)
    
    warped = cv2.remap(face_region, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    noise_strength = intensity * 0.8
    noise = np.random.normal(0, noise_strength, warped.shape).astype(np.float32)
    noisy = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    if intensity >= 5:
        noisy = np.roll(noisy, shift=1, axis=2)
    
    alpha = 1.0 + (intensity * 0.02)
    eye_region = noisy[:h//3, :]
    eye_region = np.clip(eye_region.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    noisy[:h//3, :] = eye_region
    
    mouth_region = noisy[2*h//3:, :]
    mouth_region = np.clip(mouth_region.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    noisy[2*h//3:, :] = mouth_region
    
    result = frame.copy()
    result[y1:y2, x1:x2] = noisy
    return result

print("=" * 70)
print("ANTI-KYC EFFECTIVENESS TEST")
print("=" * 70)
print("\nThis test checks if your face recognition system can match:")
print("  1. Original face → Original face (baseline)")
print("  2. Distorted face → Original face (anti-KYC test)")
print("\nExpected results:")
print("  ✓ Original-to-Original: MATCH (distance < 0.4)")
print("  ✓ Distorted-to-Original: NO MATCH (distance > 0.4)")
print("=" * 70 + "\n")

try:
    from face_registry import get_face_embedding
    RECOGNITION_AVAILABLE = True
    print("[INFO] InsightFace available - will test actual recognition")
except ImportError:
    RECOGNITION_AVAILABLE = False
    print("[WARN] InsightFace not available - will only show visual comparison")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe
import mediapipe as mp
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

print("\n" + "=" * 70)
print("INSTRUCTIONS:")
print("=" * 70)
print("1. Position your face in the camera")
print("2. Press 'C' to CAPTURE reference image (original)")
print("3. Press 'T' to TEST with different distortion levels")
print("4. Press 'Q' to quit")
print("=" * 70 + "\n")

reference_image = None
reference_embedding = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Detect faces
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    
    display_frame = frame.copy()
    
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if reference_image is None:
                cv2.putText(display_frame, "Press 'C' to capture reference", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(display_frame, "Reference captured! Press 'T' to test", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display_frame, "No face detected", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Anti-KYC Effectiveness Test", display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == 27:
        break
    
    elif key == ord('c'):
        if results.detections:
            print("\n[CAPTURE] Capturing reference image...")
            reference_image = frame.copy()
            
            if RECOGNITION_AVAILABLE:
                # Extract face region
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                
                face_crop = reference_image[y1:y2, x1:x2]
                reference_embedding = get_face_embedding(face_crop)
                
                if reference_embedding is not None:
                    print("[SUCCESS] Reference embedding generated")
                else:
                    print("[WARN] Could not generate embedding - will use visual comparison only")
            
            print("[INFO] Reference saved. Press 'T' to test with distortion")
        else:
            print("[WARN] No face detected - cannot capture")
    
    elif key == ord('t'):
        if reference_image is None:
            print("[WARN] Capture reference first (press 'C')")
            continue
        
        if not results.detections:
            print("[WARN] No face detected for testing")
            continue
        
        print("\n" + "=" * 70)
        print("TESTING ANTI-KYC EFFECTIVENESS")
        print("=" * 70)
        
        # Get current face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))
        
        # Test multiple intensities
        for intensity in [1, 3, 5, 7, 10]:
            distorted = apply_anti_kyc_distortion(frame.copy(), x1, y1, x2, y2, intensity)
            
            if RECOGNITION_AVAILABLE and reference_embedding is not None:
                # Get embedding of distorted face
                face_crop = distorted[y1:y2, x1:x2]
                distorted_embedding = get_face_embedding(face_crop)
                
                if distorted_embedding is not None:
                    # Calculate distance
                    distance = np.linalg.norm(reference_embedding - distorted_embedding)
                    
                    if distance < 0.4:
                        status = "❌ MATCHED (Anti-KYC FAILED)"
                        color = "RED"
                    else:
                        status = "✓ NO MATCH (Anti-KYC SUCCESS)"
                        color = "GREEN"
                    
                    print(f"Intensity {intensity:2d}: Distance = {distance:.4f} → {status}")
                else:
                    print(f"Intensity {intensity:2d}: Could not generate embedding")
            else:
                print(f"Intensity {intensity:2d}: Distortion applied (visual only)")
        
        print("=" * 70)
        print("[INFO] Test complete!")
        print("[INFO] Ideal result: Higher intensities should have distance > 0.4")
        print("=" * 70 + "\n")

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Test session ended")
