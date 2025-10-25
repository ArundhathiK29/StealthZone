"""
face_privacy_pro.py - StealthZone Pro Edition

Features:
✅ Facial Registration & Recognition (InsightFace)
✅ Only Registered Faces Stay Unblurred
✅ GUI-Based Name Entry
✅ Enhanced Accuracy with Outlier Rejection
✅ Optimized Performance (Frame Skipping)
✅ Privacy Confidence HUD
✅ Interactive Controls
✅ Snapshot Capture

Controls:
- R: Register new face (enter name, capture, save)
- F: Toggle face recognition ON/OFF
- B: Cycle blur intensity (1-10)
- C: Capture snapshot
- T: Toggle settings overlay
- Q/ESC: Quit

Registration Process:
1. Press R to start
2. Type name in GUI and press ENTER
3. Press C multiple times to capture face (5+ captures needed)
4. Press S (or ENTER) to save registration
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque
import os
from datetime import datetime
from scipy.spatial.distance import cosine

# Import face recognition if available
try:
    from face_registry import load_registry, find_match, get_face_embedding, save_registry
    FACE_REC_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Face recognition not available: {e}")
    FACE_REC_AVAILABLE = False

# ----------------------------
# CONFIG
# ----------------------------
USE_OPENCV_DNN = False  # OPTIMIZED: Disable redundant detector for better FPS
DNN_PROTO = "deploy.prototxt.txt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

FRAME_WIDTH = 960  # Reduced for better FPS
FRAME_HEIGHT = 540
DETECT_W = 240  # Even smaller detection for speed
DETECT_H = 180

PROCESS_EVERY_N_FRAMES = 3  # FIXED: More frequent detection for better tracking accuracy
FACE_REC_EVERY_N_FRAMES = 5  # Recognition runs more frequently for better accuracy
ROTATION_ANGLES = [0]  # Single angle for maximum speed - MediaPipe is already robust
SMOOTHING_ALPHA = 0.7  # FIXED: Less smoothing for more responsive tracking
FACE_MIN_SIZE = 15
BLUR_MAX_KSIZE = 71  # Reduced max for faster blur
BLUR_MIN_KSIZE = 11
KEEP_TRACK_FRAMES = 15  # FIXED: Shorter timeout so ghost boxes disappear faster
MAX_HISTORY = 30

# Recognition settings
RECOGNITION_CONFIDENCE_THRESHOLD = 0.6  # Increased for better accuracy
MIN_CAPTURES_FOR_REGISTRATION = 5

cv2.setUseOptimized(True)

# ----------------------------
# APP STATE
# ----------------------------
class AppState:
    def __init__(self):
        self.blur_type = "pixelate"  # OPTIMIZED: pixelate is much faster than gaussian
        self.blur_intensity = 5  # 1-10
        self.show_settings = True
        self.face_recognition_enabled = FACE_REC_AVAILABLE
        self.capture_mode = False  # Registration mode
        self.name_input_mode = False  # GUI name input mode
        self.capture_name = ""
        self.name_buffer = ""  # Buffer for typing name
        self.capture_images = []
        self.registry = {}
        if FACE_REC_AVAILABLE:
            try:
                self.registry = load_registry()
                print(f"[INFO] Loaded {len(self.registry)} registered faces")
            except:
                self.registry = {}

state = AppState()

# ----------------------------
# HELPERS
# ----------------------------
def clamp_odd(x):
    x = int(x)
    if x % 2 == 0:
        x += 1
    return max(1, x)

def rect_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def enhance_contrast_clahe(frame):
    """Applies CLAHE for better detection in low-light."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def merge_boxes(boxes, iou_thresh=0.3):
    """Merge overlapping detections."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    merged = []
    used = [False] * len(boxes)

    def iou(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = rect_area(a[:4])
        area_b = rect_area(b[:4])
        denom = area_a + area_b - inter
        return inter / denom if denom > 0 else 0

    for i, a in enumerate(boxes):
        if used[i]:
            continue
        group = [a]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if iou(a[:4], boxes[j][:4]) > iou_thresh:
                group.append(boxes[j])
                used[j] = True
        xs1 = [g[0] for g in group]
        ys1 = [g[1] for g in group]
        xs2 = [g[2] for g in group]
        ys2 = [g[3] for g in group]
        scores = [g[4] for g in group]
        merged.append(((min(xs1), min(ys1), max(xs2), max(ys2)), max(scores)))
    return [(int(b[0][0]), int(b[0][1]), int(b[0][2]), int(b[0][3]), b[1]) for b in merged]

def apply_blur(frame, x1, y1, x2, y2, blur_type, intensity):
    """Apply selected blur type to ROI - optimized version."""
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return frame
    
    face_w = max(1, x2 - x1)
    
    try:
        if blur_type == "pixelate":
            # Fast pixelation
            pixel_scale = max(2, int(20 - intensity * 1.5))
            small = cv2.resize(roi, (max(1, roi.shape[1] // pixel_scale),
                                    max(1, roi.shape[0] // pixel_scale)), 
                             interpolation=cv2.INTER_LINEAR)
            blurred_roi = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        elif blur_type == "strong":
            # Pixelate then blur - optimized
            pixel_scale = max(2, int(16 - intensity))
            small = cv2.resize(roi, (max(1, roi.shape[1] // pixel_scale),
                                    max(1, roi.shape[0] // pixel_scale)),
                             interpolation=cv2.INTER_LINEAR)
            pixelated = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            k = 15 + intensity * 2
            k = clamp_odd(k)
            blurred_roi = cv2.GaussianBlur(pixelated, (k, k), 0)
        else:  # gaussian - optimized
            k = 11 + intensity * 4
            k = max(BLUR_MIN_KSIZE, min(BLUR_MAX_KSIZE, k))
            k = clamp_odd(k)
            blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)
        
        frame[y1:y2, x1:x2] = blurred_roi
    except Exception as e:
        # Fallback to simple blur
        try:
            blurred_roi = cv2.GaussianBlur(roi, (21, 21), 0)
            frame[y1:y2, x1:x2] = blurred_roi
        except:
            pass
    
    return frame

def draw_settings_overlay(frame, state, fps, total_faces, registered_count):
    """Draw settings and info overlay."""
    if not state.show_settings:
        return
    
    h, w = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = 35
    cv2.putText(frame, "STEALTHZONE PRO", (20, y_offset), 
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
    
    y_offset += 30
    cv2.putText(frame, f"FPS: {fps:.1f} | Faces: {total_faces} | Registered: {registered_count}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    y_offset += 25
    cv2.putText(frame, f"Blur: {state.blur_type.upper()} (intensity {state.blur_intensity})", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    y_offset += 25
    rec_status = "ON" if state.face_recognition_enabled else "OFF"
    rec_color = (0, 255, 0) if state.face_recognition_enabled else (0, 0, 255)
    cv2.putText(frame, f"Recognition: {rec_status} ({len(state.registry)} enrolled)", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 1)
    
    y_offset += 30
    cv2.putText(frame, "CONTROLS:", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    
    controls = [
        "R: Register new face",
        "F: Toggle recognition",
        "B: Blur intensity (curr: {})".format(state.blur_intensity),
        "M: Blur mode (curr: {})".format(state.blur_type),
        "T: Toggle settings",
        "C: Capture photo",
        "Q/ESC: Quit"
    ]
    
    for i, ctrl in enumerate(controls):
        y_offset += 18
        cv2.putText(frame, ctrl, (25, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

def draw_registration_overlay(frame, state):
    """Draw registration capture mode overlay."""
    h, w = frame.shape[:2]
    
    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - 320, 20), (w//2 + 320, 170), (0, 100, 200), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    cv2.putText(frame, "REGISTRATION MODE", (w//2 - 200, 55), 
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Name: {state.capture_name}", (w//2 - 180, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Progress bar
    progress = min(len(state.capture_images) / MIN_CAPTURES_FOR_REGISTRATION, 1.0)
    bar_width = 400
    bar_filled = int(bar_width * progress)
    cv2.rectangle(frame, (w//2 - 200, 105), (w//2 + 200, 125), (100, 100, 100), -1)
    cv2.rectangle(frame, (w//2 - 200, 105), (w//2 - 200 + bar_filled, 125), (0, 255, 0), -1)
    
    cv2.putText(frame, f"Captures: {len(state.capture_images)}/{MIN_CAPTURES_FOR_REGISTRATION}", 
                (w//2 - 100, 142), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if len(state.capture_images) >= MIN_CAPTURES_FOR_REGISTRATION:
        cv2.putText(frame, "Press S (or ENTER) to SAVE | N to change name | ESC to cancel", 
                    (w//2 - 300, 162), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        cv2.putText(frame, "Press C to capture | N to change name | ESC to cancel", 
                    (w//2 - 280, 162), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def draw_name_input_overlay(frame, state):
    """Draw GUI name input overlay."""
    h, w = frame.shape[:2]
    
    # Large semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - 350, h//2 - 100), (w//2 + 350, h//2 + 100), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
    
    # Border
    cv2.rectangle(frame, (w//2 - 350, h//2 - 100), (w//2 + 350, h//2 + 100), (0, 200, 255), 3)
    
    # Title
    cv2.putText(frame, "Enter Name for Registration", (w//2 - 220, h//2 - 60), 
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    
    # Input box
    input_box_x = w//2 - 300
    input_box_y = h//2 - 20
    cv2.rectangle(frame, (input_box_x, input_box_y), (input_box_x + 600, input_box_y + 50), (255, 255, 255), -1)
    cv2.rectangle(frame, (input_box_x, input_box_y), (input_box_x + 600, input_box_y + 50), (0, 200, 255), 2)
    
    # Display typed name with cursor
    display_text = state.name_buffer + "_"
    cv2.putText(frame, display_text, (input_box_x + 10, input_box_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Instructions
    cv2.putText(frame, "Type name and press ENTER | BACKSPACE to delete | ESC to cancel", 
                (w//2 - 320, h//2 + 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

# ----------------------------
# LOAD DETECTORS
# ----------------------------
print("[INFO] Initializing detectors...")
mp_face = mp.solutions.face_detection
# OPTIMIZED: Use model_selection=0 for faster short-range detection (2m instead of 5m)
mp_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.4)

dnn_net = None
if USE_OPENCV_DNN:
    try:
        dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[INFO] OpenCV DNN model loaded successfully")
    except Exception as e:
        print(f"[WARN] Could not load OpenCV DNN: {e}")
        dnn_net = None

# ----------------------------
# TRACKING STATE
# ----------------------------
next_face_id = 0
faces_state = {}

# ----------------------------
# VIDEO CAPTURE
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

print("[INFO] StealthZone Pro started successfully!")
print(f"[INFO] Face recognition: {'ENABLED' if FACE_REC_AVAILABLE else 'DISABLED'}")

frame_idx = 0
t_last = time.time()
fps = 30.0

# ----------------------------
# MAIN LOOP
# ----------------------------
try:
    while True:
        ret, frame_orig = cap.read()
        if not ret:
            print("[WARN] Failed to read frame")
            break

        frame_idx += 1
        frame = cv2.flip(frame_orig, 1)  # Selfie view
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        # Keep a clean copy WITHOUT any drawings/overlays for captures
        clean_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # ----------------------------
        # FACE DETECTION
        # ----------------------------
        detections = []
        
        if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
            # Resize for faster detection
            detect_frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
            scale_x = w / DETECT_W
            scale_y = h / DETECT_H
            
            rotated_boxes = []
            
            # Single-angle detection for maximum speed
            for angle in ROTATION_ANGLES:
                img = detect_frame

                # MediaPipe detection (primary - fast and accurate)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = mp_detector.process(img_rgb)
                
                if results.detections:
                    for d in results.detections:
                        bboxC = d.location_data.relative_bounding_box
                        x1 = int(bboxC.xmin * DETECT_W)
                        y1 = int(bboxC.ymin * DETECT_H)
                        x2 = int((bboxC.xmin + bboxC.width) * DETECT_W)
                        y2 = int((bboxC.ymin + bboxC.height) * DETECT_H)
                        
                        # Scale back to original resolution
                        x1 = int(x1 * scale_x)
                        y1 = int(y1 * scale_y)
                        x2 = int(x2 * scale_x)
                        y2 = int(y2 * scale_y)
                        
                        score = d.score[0] if d.score else 0.5
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        
                        if (x2 - x1) >= FACE_MIN_SIZE and (y2 - y1) >= FACE_MIN_SIZE:
                            rotated_boxes.append((x1, y1, x2, y2, float(score)))

                # OpenCV DNN fallback (runs less frequently for speed)
                if dnn_net is not None and frame_idx % 15 == 0 and angle == 0:
                    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    dnn_net.setInput(blob)
                    detections_net = dnn_net.forward()
                    
                    for i in range(detections_net.shape[2]):
                        conf = float(detections_net[0, 0, i, 2])
                        if conf > 0.4:
                            box = detections_net[0, 0, i, 3:7] * np.array([DETECT_W, DETECT_H, DETECT_W, DETECT_H])
                            x1, y1, x2, y2 = box.astype("int")
                            
                            x1 = int(x1 * scale_x)
                            y1 = int(y1 * scale_y)
                            x2 = int(x2 * scale_x)
                            y2 = int(y2 * scale_y)
                            
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w - 1, x2), min(h - 1, y2)
                            
                            if (x2 - x1) >= FACE_MIN_SIZE and (y2 - y1) >= FACE_MIN_SIZE:
                                rotated_boxes.append((x1, y1, x2, y2, float(conf)))

            # Merge overlapping detections
            detections = merge_boxes(rotated_boxes, iou_thresh=0.3)

            # ----------------------------
            # FACE TRACKING
            # ----------------------------
            used_ids = set()
            new_faces_state = {}

            for det in detections:
                x1, y1, x2, y2, score = det
                d_center = box_center((x1, y1, x2, y2))
                best_id = None
                best_dist = 1e9

                # Match with existing tracks
                for fid, st in faces_state.items():
                    sb = st.get("smooth_box", st["box"])
                    sc = box_center(sb)
                    dist = math.hypot(d_center[0] - sc[0], d_center[1] - sc[1])
                    if dist < best_dist and fid not in used_ids:
                        best_dist = dist
                        best_id = fid

                # More generous matching threshold for better tracking
                match_threshold = max(w, h) * 0.25
                
                if best_id is not None and best_dist < match_threshold:
                    # Update existing track
                    st = faces_state[best_id]
                    prev_smooth = st.get("smooth_box", st["box"])
                    nsmooth = (
                        SMOOTHING_ALPHA * np.array((x1, y1, x2, y2)) +
                        (1 - SMOOTHING_ALPHA) * np.array(prev_smooth)
                    )
                    st.update({
                        "box": (x1, y1, x2, y2),
                        "score": score,
                        "smooth_box": tuple(nsmooth.tolist()),
                        "last_seen_frame": frame_idx,
                        "missed": 0
                    })
                    new_faces_state[best_id] = st
                    used_ids.add(best_id)
                # [New Code]
                else:
                    # Create new track
                    fid = next_face_id
                    next_face_id += 1
                    
                    st = {
                        "id": fid,
                        "box": (x1, y1, x2, y2),
                        "smooth_box": (x1, y1, x2, y2),
                        "score": score,
                        "last_seen_frame": frame_idx,
                        "missed": 0,
                        "history": deque(maxlen=MAX_HISTORY), # You already have this
                        
                        # --- NEW RECOGNITION LOGIC ---
                        "registered_name": None,        # The final, confirmed name
                        "recognition_history": deque(maxlen=10), # Our new "voting" history
                        "needs_recognition": True,      # Flag to run recognition
                        "last_recognition_attempt": 0   # Frame index of last try
                        # --- END NEW LOGIC ---
                    }
                    new_faces_state[fid] = st
                    used_ids.add(fid)

            # Keep lost tracks for a while
            for fid, st in faces_state.items():
                if fid not in used_ids:
                    st["missed"] = st.get("missed", 0) + 1
                    if st["missed"] <= KEEP_TRACK_FRAMES:
                        new_faces_state[fid] = st

            faces_state = new_faces_state

        # ----------------------------
        # FACE RECOGNITION (OPTIMIZED: Throttled for better FPS)
        # ----------------------------
        if (state.face_recognition_enabled and FACE_REC_AVAILABLE and state.registry):
            
            # 1. Create a list of faces that need recognition
            faces_to_recognize = []
            for fid, st in faces_state.items():
                if st.get("needs_recognition", False) and (frame_idx - st.get("last_recognition_attempt", 0) > FACE_REC_EVERY_N_FRAMES):
                    faces_to_recognize.append(st)
            
            # --- OPTIMIZED: Process ONLY ONE face per frame to prevent lag spikes ---
            # 2. This amortizes the recognition cost across multiple frames
            if faces_to_recognize:
                # Sort by largest face first (optional, but gives priority to prominent faces)
                faces_to_recognize.sort(key=lambda s: rect_area(s.get("smooth_box", s["box"])), reverse=True)
                st = faces_to_recognize[0]  # Process only the most prominent face
                # --- END OPTIMIZATION ---
                
                st["last_recognition_attempt"] = frame_idx
                
                x1, y1, x2, y2 = map(int, st.get("smooth_box", st["box"]))
                
                # Use a larger margin for better recognition
                face_w = x2 - x1
                margin = int(face_w * 0.3)
                x1m = max(0, x1 - margin)
                y1m = max(0, y1 - margin)
                x2m = min(w, x2 + margin)
                y2m = min(h, y2 + margin)
                
                face_img = frame[y1m:y2m, x1m:x2m]
                
                match_name = None # Default to no match
                if face_img.size > 0 and face_img.shape[0] > 40 and face_img.shape[1] > 40:
                    try:
                        embedding = get_face_embedding(face_img)
                        if embedding is not None:
                            match_name, distance = find_match(embedding, state.registry)
                    except Exception as e:
                        print(f"[WARN] Embedding extraction failed: {e}")
                        pass

                # --- VOTING SYSTEM ---
                st["recognition_history"].append(match_name)
                history = list(st["recognition_history"])
                
                if history:
                    names_in_history = [name for name in history if name is not None]
                    
                    if names_in_history:
                        most_common_name = max(set(names_in_history), key=names_in_history.count)
                        confidence_count = names_in_history.count(most_common_name)
                        
                        if confidence_count >= 3: 
                            if st["registered_name"] != most_common_name:
                                print(f"[RECOGNITION] ✓✓ Confirmed {most_common_name} for face {st['id']}")
                            st["registered_name"] = most_common_name
                            st["needs_recognition"] = False 
                        else:
                            st["registered_name"] = None
                            st["needs_recognition"] = True
                    else:
                        st["registered_name"] = None
                        st["needs_recognition"] = True
                else:
                    st["needs_recognition"] = True

        # ----------------------------
        # DETERMINE PRIMARY FACES (Only registered faces)
        # ----------------------------
        # [New Code]
        # ----------------------------
        # DETERMINE PRIMARY FACES (Only registered faces)
        # ----------------------------
        face_list = []
        for fid, st in faces_state.items():
            sb = st.get("smooth_box", st["box"])
            area = rect_area(sb)
            
            # --- MODIFIED ---
            # Check our new "confirmed" name
            is_registered = st.get("registered_name") is not None
            
            face_list.append((fid, st, area, is_registered))

        face_list_sorted = sorted(face_list, key=lambda x: x[2], reverse=True)
        
        # This part is no longer needed, we just use 'is_registered' in the drawing loop
        # primary_ids = set() ...
        # --- ADD THIS LINE ---
        registered_count = sum(1 for _, _, _, is_registered in face_list_sorted if is_registered)
        # --- END OF ADDITION ---

        # ----------------------------
        ## ----------------------------
        # APPLY BLUR & DRAW
        # ----------------------------
        blurred_count = 0

        # --- ADD THIS NEW LOGIC ---
        # Find the face being registered (the largest one)
        registration_target_id = None
        if state.capture_mode and face_list_sorted:
            # Get the ID of the largest face, which is the registration target
            # face_list_sorted[0] is (fid, st, area, is_registered)
            registration_target_id = face_list_sorted[0][0] 
        # --- END NEW LOGIC ---

        for idx, (fid, st, _, is_registered) in enumerate(face_list_sorted):
            x1, y1, x2, y2 = map(int, st.get("smooth_box", st["box"]))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            
            # Check if this face is the one we are registering
            is_registration_target = (fid == registration_target_id)

            if is_registered or is_registration_target:
                # This face is either recognized OR it's the one we're registering
                
                if is_registered:
                    # --- IT'S A RECOGNIZED FACE ---
                    color = (0, 255, 0)  # Bright green for recognized
                    # Get name from the face's state dictionary
                    label = f"✓ {st.get('registered_name')}"
                else:
                    # --- IT'S THE REGISTRATION TARGET ---
                    color = (255, 200, 0) # Bright blue for "Registering"
                    label = "REGISTERING..."

                # Draw the box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw label with background for better visibility
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 12), 
                            (x1 + label_size[0] + 10, y1), color, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 6),
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 2)
            
            else:
                # --- IT'S AN UNKNOWN FACE, BLUR IT ---
                frame = apply_blur(frame, x1, y1, x2, y2, state.blur_type, state.blur_intensity)
                blurred_count += 1
                
                color = (0, 120, 255)
                label = "Unknown"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Small label
                cv2.putText(frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            

        # ----------------------------
        # REGISTRATION CAPTURE MODE
        # ----------------------------
        if state.name_input_mode:
            draw_name_input_overlay(frame, state)
        elif state.capture_mode:
            draw_registration_overlay(frame, state)

        # ----------------------------
        # FPS CALCULATION
        # ----------------------------
        t_now = time.time()
        dt = t_now - t_last
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        t_last = t_now

        # ----------------------------
        # DRAW OVERLAYS
        # ----------------------------
        total_faces = len(face_list_sorted)
        draw_settings_overlay(frame, state, fps, total_faces, registered_count)

        # Privacy confidence indicator
        privacy_conf = blurred_count / max(1, total_faces) if total_faces > 0 else 1.0
        privacy_pct = int(privacy_conf * 100)
        
        # Top-right indicator
        indicator_color = (0, 255, 0) if privacy_pct > 60 else (0, 165, 255) if privacy_pct > 30 else (0, 0, 255)
        cv2.rectangle(frame, (w - 250, 10), (w - 10, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Privacy: {privacy_pct}%", (w - 240, 35),
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, indicator_color, 2)

        # ----------------------------
        # DISPLAY
        # ----------------------------
        cv2.imshow("StealthZone Pro - Press Q to quit", frame)
        
        # ----------------------------
        # KEYBOARD CONTROLS
        # ----------------------------
        key = cv2.waitKey(1) & 0xFF
        
        # Handle name input mode separately
        if state.name_input_mode:
            handled = False
            if key == 13:  # Enter key
                if state.name_buffer.strip():
                    state.capture_name = state.name_buffer.strip()
                    state.name_input_mode = False
                    state.name_buffer = ""
                    # Automatically start capture mode
                    if not state.capture_mode:
                        state.capture_mode = True
                        state.capture_images = []
                    print(f"[INFO] ✓ Name set to: '{state.capture_name}'")
                    print(f"[INFO] Now capturing - Press C to capture faces ({MIN_CAPTURES_FOR_REGISTRATION}+ needed)")
                    # Don't set handled=True, allow Enter to potentially trigger save if captures are ready
                else:
                    print("[WARN] Name cannot be empty!")
                    handled = True
            elif key == 27:  # ESC - cancel
                state.name_input_mode = False
                state.name_buffer = ""
                if state.capture_mode and not state.capture_name:
                    # Cancel entire registration if no name
                    state.capture_mode = False
                    state.capture_images = []
                    print("[INFO] Registration cancelled")
                elif state.capture_mode:
                    # Just cancel name change, keep existing name
                    print(f"[INFO] Keeping current name: {state.capture_name}")
                handled = True
            elif key == 8:  # Backspace
                state.name_buffer = state.name_buffer[:-1]
                handled = True
            elif key != 255 and 32 <= key <= 126:  # Printable characters
                if len(state.name_buffer) < 20:  # Max 20 characters
                    state.name_buffer += chr(key)
                handled = True
            
            # Skip other key processing if we handled it in name input mode
            if handled:
                continue
        
        # Normal key processing
        if key == ord('q') or key == 27:  # Q or ESC
            if state.capture_mode:
                # Cancel capture mode
                state.capture_mode = False
                state.capture_images = []
                state.capture_name = ""
                print("[INFO] Registration cancelled")
            else:
                break
        
        # --- SAVE REGISTRATION: Both ENTER and 'S' key work ---
        elif (key == 13 or key == ord('s')) and state.capture_mode and not state.name_input_mode:  # ENTER or S key
            # Save registration
            if len(state.capture_images) >= MIN_CAPTURES_FOR_REGISTRATION:
                print(f"[INFO] Processing {len(state.capture_images)} captures for '{state.capture_name}'...")
                embeddings = []
                successful_captures = 0
                
                for idx, img in enumerate(state.capture_images):
                    try:
                        emb = get_face_embedding(img)
                        if emb is not None:
                            embeddings.append(emb)
                            successful_captures += 1
                            print(f"[PROCESSING] ✓ Embedding {successful_captures}/{len(state.capture_images)}")
                        else:
                            print(f"[PROCESSING] ✗ Failed to extract embedding from image {idx+1}")
                    except Exception as e:
                        print(f"[PROCESSING] ✗ Error on image {idx+1}: {str(e)}")
                
                if embeddings and successful_captures >= 3:  # Need at least 3 valid embeddings
                    # --- OUTLIER REJECTION LOGIC (from before) ---
                    if successful_captures > 3: 
                        print(f"[INFO] Calculating mean embedding from {len(embeddings)} captures...")
                        prelim_mean = np.mean(embeddings, axis=0)
                        distances = [cosine(emb, prelim_mean) for emb in embeddings]
                        mean_dist = np.mean(distances)
                        std_dist = np.std(distances)
                        outlier_threshold = mean_dist + (1.0 * std_dist)
                        print(f"[INFO] Outlier distance threshold: {outlier_threshold:.4f}")
                        
                        final_embeddings = []
                        for i, dist in enumerate(distances):
                            if dist < outlier_threshold:
                                final_embeddings.append(embeddings[i])
                            else:
                                print(f"[WARN] Discarding capture {i+1} as outlier (distance: {dist:.4f})")
                        
                        if len(final_embeddings) < 3:
                            print(f"[WARN] Too many outliers! Using all {successful_captures} captures.")
                            final_embeddings = embeddings
                    else:
                        final_embeddings = embeddings
                    
                    print(f"[INFO] Using {len(final_embeddings)} final embeddings for average.")
                    mean_embedding = np.mean(final_embeddings, axis=0)
                    # --- END OUTLIER REJECTION ---

                    state.registry[state.capture_name] = mean_embedding.tolist()
                    
                    try:
                        save_registry(state.registry)
                        print(f"\n{'='*60}")
                        print(f"[SUCCESS] ✓✓✓ {state.capture_name} REGISTERED! ✓✓✓")
                        print(f"{'='*60}")
                        print(f"✓ Saved with {successful_captures}/{len(state.capture_images)} valid embeddings")
                        print(f"✓ {state.capture_name} will ALWAYS stay unblurred")
                        print(f"✓ Total registered: {len(state.registry)} people")
                        print(f"{'='*60}\n")
                        
                        state.capture_mode = False
                        state.capture_images = []
                        
                        if not state.face_recognition_enabled:
                            state.face_recognition_enabled = True
                            print("[INFO] Face recognition auto-enabled")
                        
                        # IMMEDIATELY try to recognize all current faces
                        print(f"[INFO] Scanning for {state.capture_name}...")
                        for fid, st in faces_state.items():
                            st["needs_recognition"] = True # Flag all faces for re-check
                        
                    except Exception as e:
                        print(f"[ERROR] Failed to save registry: {str(e)}")
                        import traceback
                        traceback.print_exc()
                elif successful_captures > 0:
                    print(f"[ERROR] Only {successful_captures} valid embeddings. Need at least 3.")
                    print("[INFO] Try again with better lighting and face positioning.")
                else:
                    print("[ERROR] Could not generate any embeddings. Try again with:")
                    print("  - Better lighting (bright, even illumination)")
                    print("  - Face directly facing camera")
                    print("  - Face filling 30-40% of frame")
                    print("  - No extreme angles or occlusions")
            else:
                print(f"[WARN] Need at least {MIN_CAPTURES_FOR_REGISTRATION} captures (you have {len(state.capture_images)})")
        
        elif key == ord('t'):  # Toggle settings (changed from 's' to 't')
            state.show_settings = not state.show_settings
        
        elif key == ord('r'):  # Registration mode
            if FACE_REC_AVAILABLE:
                if not state.capture_mode:
                    # Enter name input mode first
                    state.name_input_mode = True
                    state.name_buffer = ""
                    state.capture_name = ""
                    print(f"\n{'='*60}")
                    print(f"[REGISTRATION MODE ACTIVATED]")
                    print(f"{'='*60}")
                    print(f"Step 1: Enter name in the GUI window (type and press ENTER)")
                    print(f"Step 2: Press C to capture face images ({MIN_CAPTURES_FOR_REGISTRATION}+ needed)")
                    print(f"Step 3: Press S (or ENTER) to save registration")
                    print(f"{'='*60}\n")
            else:
                print("[WARN] Face recognition not available")
        
        elif key == ord('n') and state.capture_mode:  # Change name during registration
            state.name_input_mode = True
            state.name_buffer = state.capture_name  # Pre-fill with current name
            print("[INFO] Enter new name in GUI")
        
        elif key == ord('c'):  # Capture
            if state.capture_mode:
                # Capture face for registration
                # [New Code to Replace it With]

                if face_list_sorted:
                    fid, st, _, is_registered = face_list_sorted[0]  # Capture largest face
                    x1_box, y1_box, x2_box, y2_box = map(int, st.get("smooth_box", st["box"]))
                    
                    # Calculate face size
                    face_w = x2_box - x1_box
                    face_h = y2_box - y1_box
                    
                    # Check if face is large enough
                    if face_w < 80 or face_h < 80:
                        print(f"[WARN] Face too small ({face_w}x{face_h}). Move closer to camera (need 80x80+)")
                        continue
                    
                    # --- NEW CAPTURE LOGIC ---
                    # We will capture a larger, squarer region around the face
                    # This gives the InsightFace model more context to find the face.
                    
                    # [New Code - This saves a "tight crop"]
                    
                    # [New Code - This saves a "tight crop"]
                    
                    # --- NEW CAPTURE LOGIC (FIXED) ---
                    # We save a tight crop + a 30% margin.
                    # This is what app.rec_get_embedding() expects.
                    
                    # [This is the correct "large patch" logic]

                    # --- NEW CAPTURE LOGIC (FIXED) ---
                    # We will capture a larger, squarer region around the face
                    # This gives the InsightFace detector (in app.get()) a "scene" to work on.
                    
                    # 1. Find the center of the face
                    center_x = (x1_box + x2_box) // 2
                    center_y = (y1_box + y2_box) // 2
                    
                    # 2. Determine the size of our crop (e.g., 3.0x the face size)
                    # This makes it large enough to be a "scene"
                    crop_size = int(max(face_w, face_h) * 3.0) 
                    
                    # 3. Calculate new x1, y1, x2, y2 for the *crop*
                    x1 = max(0, center_x - crop_size // 2)
                    y1 = max(0, center_y - crop_size // 2)
                    x2 = min(w, center_x + crop_size // 2)
                    y2 = min(h, center_y + crop_size // 2)
                    
                    # 4. Grab the image from the CLEAN frame (no drawings/overlays)
                    face_img = clean_frame[y1:y2, x1:x2]
                    print(f"[DEBUG] Capture crop: box=({x1_box},{y1_box},{x2_box},{y2_box}) center=({center_x},{center_y}) crop_size={crop_size} -> patch={face_img.shape[1]}x{face_img.shape[0]}")
                    # --- END NEW CAPTURE LOGIC (FIXED) ---
                    # --- END NEW CAPTURE LOGIC (FIXED) ---

                    
                    if face_img.size > 0 and face_img.shape[0] > 40 and face_img.shape[1] > 40:
                        # --- MODIFICATION ---
                        # We removed the failing pre-check.
                        # The "Outlier Rejection" on Save (S key) will handle bad captures.
                        
                        state.capture_images.append(face_img.copy())
                        print(f"[CAPTURE] ✓ Image {len(state.capture_images)}/{MIN_CAPTURES_FOR_REGISTRATION}+ captured!")
                        
                        if len(state.capture_images) >= MIN_CAPTURES_FOR_REGISTRATION:
                            print(f"[INFO] ✓ Ready to save! Press S (or ENTER) to complete registration")
                        # --- END MODIFICATION ---
                    else:
                        print(f"[WARN] Captured image too small. Move closer to camera.")
                else:
                    print("[WARN] No face detected. Make sure:")
                    print("  - Face is clearly visible")
                    print("  - Good lighting (bright, even)")
                    print("  - Face fills 30-40% of frame")
            else:
                # Regular snapshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"stealthzone_pro_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"[SNAPSHOT] ✓ Saved: {filename}")
        
        elif key == ord('f'):  # Toggle face recognition
            state.face_recognition_enabled = not state.face_recognition_enabled
            status = "ENABLED" if state.face_recognition_enabled else "DISABLED"
            print(f"[INFO] Face recognition {status}")
        
        elif key == ord('b'):  # Blur intensity
            state.blur_intensity = (state.blur_intensity % 10) + 1
            print(f"[INFO] Blur intensity: {state.blur_intensity}")
        
        elif key == ord('m'):  # Toggle blur type (moved from 't' to 'm' for 'mode')
            types = ["pixelate", "gaussian", "strong"]
            current_idx = types.index(state.blur_type) if state.blur_type in types else 0
            state.blur_type = types[(current_idx + 1) % len(types)]
            print(f"[INFO] Blur type: {state.blur_type.upper()}")

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")
except Exception as e:
    print(f"[ERROR] Runtime error: {e}")
    import traceback
    traceback.print_exc()

# ----------------------------
# CLEANUP
# ----------------------------
cap.release()
cv2.destroyAllWindows()
print("[INFO] StealthZone Pro stopped")
