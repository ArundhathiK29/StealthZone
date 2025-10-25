"""
face_privacy_realtime_enhanced.py - StealthZone Layer 2 Enhanced

Features:
- Low-Light Enhancement (CLAHE) before detection for better accuracy in dark conditions.
- Natural “selfie-view” (flipped camera input).
- Dual Detector (MediaPipe + OpenCV DNN fallback).
- All original features (Rotation, EMA smoothing, Adaptive Blur, Privacy Confidence HUD).
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
from collections import deque

# ----------------------------
# Config / hyperparameters
# ----------------------------
USE_OPENCV_DNN = True
DNN_PROTO = "deploy.prototxt.txt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 1
ROTATION_ANGLES = [0, -12, 12]
SMOOTHING_ALPHA = 0.6
FACE_MIN_SIZE = 20
BLUR_MAX_KSIZE = 99
BLUR_MIN_KSIZE = 11
KEEP_TRACK_FRAMES = 12
MAX_HISTORY = 30

# ----------------------------
# Helpers
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
    """Applies CLAHE to enhance contrast in dark areas for low-light detection."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def merge_boxes(boxes, iou_thresh=0.25):
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

# ----------------------------
# Load detectors
# ----------------------------
mp_face = mp.solutions.face_detection
mp_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.45)

dnn_net = None
if USE_OPENCV_DNN:
    try:
        dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        print("[INFO] Loaded OpenCV DNN model.")
    except Exception as e:
        print(f"[WARN] Could not load OpenCV DNN model. Continuing without it. Error: {e}")
        dnn_net = None

# ----------------------------
# Tracking & smoothing state
# ----------------------------
next_face_id = 0
faces_state = {}

# ----------------------------
# Video capture
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap.isOpened():
    print("[FATAL ERROR] Could not open webcam (check index or if camera is in use).")
    exit()

frame_idx = 0
t_last = time.time()
fps = 0.0

try:
    while True:
        ret, frame_orig = cap.read()
        if not ret:
            print("[WARN] Failed to read frame from camera.")
            break

        frame_idx += 1
        frame_orig = cv2.flip(frame_orig, 1)  # Selfie-view
        frame = cv2.resize(frame_orig, (FRAME_WIDTH, FRAME_HEIGHT))
        h, w = frame.shape[:2]
        detections = []

        if frame_idx % PROCESS_EVERY_N_FRAMES == 0:
            frame_enhanced = enhance_contrast_clahe(frame)
            rotated_boxes = []

            for angle in ROTATION_ANGLES:
                img = frame_enhanced
                if angle != 0:
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                    img = cv2.warpAffine(frame_enhanced, M, (w, h))

                # --- MediaPipe detection ---
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = mp_detector.process(img_rgb)
                if results.detections:
                    for d in results.detections:
                        bboxC = d.location_data.relative_bounding_box
                        x1 = int(bboxC.xmin * w)
                        y1 = int(bboxC.ymin * h)
                        x2 = int((bboxC.xmin + bboxC.width) * w)
                        y2 = int((bboxC.ymin + bboxC.height) * h)
                        if angle != 0:
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            M_inv = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
                            pts = np.array([[cx, cy, 1]]).T
                            new = np.dot(M_inv, pts).flatten()
                            ncx, ncy = new[0], new[1]
                            bw, bh = x2 - x1, y2 - y1
                            x1 = int(ncx - bw / 2)
                            y1 = int(ncy - bh / 2)
                            x2 = int(ncx + bw / 2)
                            y2 = int(ncy + bh / 2)
                        score = d.score[0] if d.score else 0.5
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        if (x2 - x1) >= FACE_MIN_SIZE and (y2 - y1) >= FACE_MIN_SIZE:
                            rotated_boxes.append((x1, y1, x2, y2, float(score)))

                # --- OpenCV DNN fallback ---
                if dnn_net is not None:
                    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
                    dnn_net.setInput(blob)
                    detections_net = dnn_net.forward()
                    for i in range(detections_net.shape[2]):
                        conf = float(detections_net[0, 0, i, 2])
                        if conf > 0.45:
                            box = detections_net[0, 0, i, 3:7] * np.array([w, h, w, h])
                            x1, y1, x2, y2 = box.astype("int")
                            if angle != 0:
                                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                                M_inv = cv2.getRotationMatrix2D((w / 2, h / 2), -angle, 1.0)
                                pts = np.array([[cx, cy, 1]]).T
                                new = np.dot(M_inv, pts).flatten()
                                ncx, ncy = new[0], new[1]
                                bw, bh = x2 - x1, y2 - y1
                                x1 = int(ncx - bw / 2)
                                y1 = int(ncy - bh / 2)
                                x2 = int(ncx + bw / 2)
                                y2 = int(ncy + bh / 2)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w - 1, x2), min(h - 1, y2)
                            if (x2 - x1) >= FACE_MIN_SIZE and (y2 - y1) >= FACE_MIN_SIZE:
                                rotated_boxes.append((x1, y1, x2, y2, float(conf)))

            # --- Merge overlapping detections ---
            detections = merge_boxes(rotated_boxes, iou_thresh=0.25)

            # --- Face tracking ---
            used_ids = set()
            new_faces_state = {}

            for det in detections:
                x1, y1, x2, y2, score = det
                d_center = box_center((x1, y1, x2, y2))
                best_id = None
                best_dist = 1e9

                for fid, st in faces_state.items():
                    sb = st.get("smooth_box", st["box"])
                    sc = box_center(sb)
                    dist = math.hypot(d_center[0] - sc[0], d_center[1] - sc[1])
                    if dist < best_dist and fid not in used_ids:
                        best_dist = dist
                        best_id = fid

                if best_id is not None and best_dist < max(w, h) * 0.25:
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
                else:
                    fid = next_face_id
                    next_face_id += 1
                    st = {
                        "id": fid,
                        "box": (x1, y1, x2, y2),
                        "smooth_box": (x1, y1, x2, y2),
                        "score": score,
                        "last_seen_frame": frame_idx,
                        "missed": 0,
                        "history": deque(maxlen=MAX_HISTORY)
                    }
                    new_faces_state[fid] = st
                    used_ids.add(fid)

            for fid, st in faces_state.items():
                if fid not in used_ids:
                    st["missed"] = st.get("missed", 0) + 1
                    if st["missed"] <= KEEP_TRACK_FRAMES:
                        new_faces_state[fid] = st

            faces_state = new_faces_state

        # --- Decide primary face ---
        face_list = []
        for fid, st in faces_state.items():
            sb = st.get("smooth_box", st["box"])
            area = rect_area(sb)
            cx, cy = box_center(sb)
            center_dist = math.hypot(cx - w / 2, cy - h / 2)
            priority = area - (center_dist * 50)
            face_list.append((fid, st, priority))
        face_list_sorted = sorted(face_list, key=lambda x: x[2], reverse=True)
        primary_id = face_list_sorted[0][0] if face_list_sorted else None

        # --- Blur non-primary faces ---
        blurred_info = []
        total_faces = len(face_list_sorted)
        for fid, st, _ in face_list_sorted:
            x1, y1, x2, y2 = map(int, st.get("smooth_box", st["box"]))
            score = st.get("score", 0.5)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if fid == primary_id:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"Primary id:{fid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)
            else:
                face_w = max(1, x2 - x1)
                k = int((w / face_w) * 4)
                k = max(BLUR_MIN_KSIZE, min(BLUR_MAX_KSIZE, k))
                k = clamp_odd(k)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                pixel_scale = max(1, int(face_w / 24))
                try:
                    small = cv2.resize(roi, (max(1, roi.shape[1] // pixel_scale),
                                             max(1, roi.shape[0] // pixel_scale)))
                    pixelated = cv2.resize(small, (roi.shape[1], roi.shape[0]),
                                           interpolation=cv2.INTER_NEAREST)
                    blurred_roi = cv2.GaussianBlur(pixelated, (k, k), 0)
                except Exception:
                    blurred_roi = cv2.GaussianBlur(roi, (k, k), 0)

                frame[y1:y2, x1:x2] = blurred_roi
                blurred_info.append((fid, (x1, y1, x2, y2), score))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 120, 255), 2)
                cv2.putText(frame, f"Blurred id:{fid} ({score:.2f})", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 2)

        # --- Privacy Confidence ---
        if total_faces == 0:
            privacy_conf = 0.0
        else:
            blurred_sum = sum([s for _, _, s in blurred_info]) if blurred_info else 0.0
            privacy_conf = min(1.0, blurred_sum / max(1, total_faces))
        privacy_pct = int(privacy_conf * 100)

        # --- FPS estimation ---
        t_now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (t_now - t_last)) if t_now != t_last else fps
        t_last = t_now

        # --- HUD Overlay ---
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces detected: {total_faces}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Privacy Confidence: {privacy_pct}%", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255 if privacy_pct > 60 else 0, 255 if privacy_pct > 60 else 0), 2)

        cv2.rectangle(frame, (w - 230, 10), (w - 10, 110), (0, 0, 0), -1)
        cv2.putText(frame, "Legend:", (w - 220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Green: Primary (kept)", (w - 220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        cv2.putText(frame, "Orange: Blurred bystander", (w - 220, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 120, 255), 1)

        cv2.imshow("StealthZone Face Privacy (Press Q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

except KeyboardInterrupt:
    pass
except Exception as e:
    print(f"[ERROR] Runtime error: {e}")

cap.release()
cv2.destroyAllWindows()