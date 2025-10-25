"""
Visual comparison test for Anti-KYC distortion
Shows original, blurred, and distorted versions side-by-side
"""

import cv2
import numpy as np
import sys

# Import distortion function from face_privacy_pro
sys.path.insert(0, '.')

def apply_anti_kyc_distortion(frame, x1, y1, x2, y2, intensity=5):
    """Apply subtle distortion to defeat V-KYC without obvious visual changes."""
    face_region = frame[y1:y2, x1:x2].copy()
    h, w = face_region.shape[:2]
    
    if h < 20 or w < 20:
        return frame
    
    # 1. GEOMETRIC WARPING
    warp_strength = intensity * 0.3
    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    # Eye region
    eye_mask = (y_grid < h // 3).astype(np.float32)
    dx_eye = np.sin(x_grid * 0.3) * warp_strength * eye_mask
    dy_eye = np.sin(y_grid * 0.3) * warp_strength * eye_mask
    
    # Nose region
    nose_mask = ((y_grid >= h // 3) & (y_grid < 2 * h // 3)).astype(np.float32)
    dx_nose = np.sin(x_grid * 0.2 + y_grid * 0.2) * warp_strength * 0.5 * nose_mask
    dy_nose = np.sin(y_grid * 0.2) * warp_strength * 0.5 * nose_mask
    
    # Mouth region
    mouth_mask = (y_grid >= 2 * h // 3).astype(np.float32)
    dx_mouth = np.sin(x_grid * 0.25) * warp_strength * 0.7 * mouth_mask
    dy_mouth = np.sin(y_grid * 0.25) * warp_strength * 0.7 * mouth_mask
    
    dx = dx_eye + dx_nose + dx_mouth
    dy = dy_eye + dy_nose + dy_mouth
    
    map_x = (x_grid + dx).astype(np.float32)
    map_y = (y_grid + dy).astype(np.float32)
    
    warped = cv2.remap(face_region, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # 2. MICRO-TEXTURE NOISE
    noise_strength = intensity * 0.8
    noise = np.random.normal(0, noise_strength, warped.shape).astype(np.float32)
    noisy = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # 3. COLOR CHANNEL PERTURBATION
    if intensity >= 5:
        noisy = np.roll(noisy, shift=1, axis=2)
    
    # 4. CONTRAST ADJUSTMENT
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

def apply_pixelate_blur(frame, x1, y1, x2, y2, intensity=15):
    """Apply pixelate blur effect."""
    face_region = frame[y1:y2, x1:x2]
    h, w = face_region.shape[:2]
    
    block_size = max(1, intensity)
    temp_h = max(1, h // block_size)
    temp_w = max(1, w // block_size)
    
    temp = cv2.resize(face_region, (temp_w, temp_h), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    
    result = frame.copy()
    result[y1:y2, x1:x2] = blurred
    return result

print("=" * 70)
print("ANTI-KYC VISUAL COMPARISON TEST")
print("=" * 70)
print("\nThis test will show you:")
print("  1. ORIGINAL face (no processing)")
print("  2. BLURRED face (standard privacy)")
print("  3. DISTORTED face (Anti-KYC mode)")
print("\nPress 'Q' to quit, '1-9' to change distortion intensity")
print("=" * 70 + "\n")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open camera")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe face detection
import mediapipe as mp
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

distortion_intensity = 5

print("[INFO] Camera opened successfully")
print("[INFO] Press keys 1-9 to adjust distortion intensity")
print("[INFO] Current intensity: 5 (moderate)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    
    # Detect faces
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)
    
    if results.detections:
        # Get first face
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        
        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)
        
        # Ensure valid coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            # Create three versions
            original = frame.copy()
            blurred = apply_pixelate_blur(frame.copy(), x1, y1, x2, y2, intensity=15)
            distorted = apply_anti_kyc_distortion(frame.copy(), x1, y1, x2, y2, distortion_intensity)
            
            # Add rectangles
            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(blurred, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.rectangle(distorted, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Add labels
            cv2.putText(original, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(blurred, "BLURRED", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(distorted, f"DISTORTED (Intensity: {distortion_intensity})", (10, 30), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
            
            # Stack horizontally
            if w * 3 <= 1920:  # Fit on screen
                comparison = np.hstack([original, blurred, distorted])
            else:
                # Resize to fit
                scale = 1920 / (w * 3)
                new_w = int(w * scale)
                new_h = int(h * scale)
                original = cv2.resize(original, (new_w, new_h))
                blurred = cv2.resize(blurred, (new_w, new_h))
                distorted = cv2.resize(distorted, (new_w, new_h))
                comparison = np.hstack([original, blurred, distorted])
            
            cv2.imshow("Anti-KYC Comparison: Original | Blurred | Distorted", comparison)
    else:
        # No face detected
        cv2.putText(frame, "No face detected - please position face in frame", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Anti-KYC Comparison: Original | Blurred | Distorted", frame)
    
    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    elif ord('1') <= key <= ord('9'):
        distortion_intensity = key - ord('0')
        print(f"[INFO] Distortion intensity changed to: {distortion_intensity}")

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Test completed")
