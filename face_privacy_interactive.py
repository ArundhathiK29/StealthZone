# face_privacy_interactive.py
# StealthZone — Interactive real-time face privacy manager
# - MediaPipe face detection + OpenCV DNN fallback
# - Real-time face recognition (optional)
# - Dynamic primary face selection
# - Interactive settings overlay

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import threading
from collections import deque
from face_registry import load_registry, find_match, get_face_embedding

# ----------------------------
# CONFIG
# ----------------------------
USE_OPENCV_DNN = True
DNN_PROTO = "deploy.prototxt.txt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DETECT_W = 320
DETECT_H = 240

CLAHE_EVERY_N_FRAMES = 10
DNN_RUN_EVERY_N_FRAMES = 8
FACE_REC_EVERY_N_FRAMES = 5 # Run face recognition less frequently
ROTATION_ANGLES = [0, -8, 8]   # small-angles for side faces
SMOOTHING_ALPHA = 0.6
FACE_MIN_SIZE = 16
BLUR_MAX_KSIZE = 99
BLUR_MIN_KSIZE = 11
KEEP_TRACK_FRAMES = 10
MAX_HISTORY = 30
MAX_PRIMARY_ALLOW = 8

cv2.setUseOptimized(True)

# ----------------------------
# STATE
# ----------------------------
class AppState:
    def __init__(self):
        self.blur_type = "gaussian"
        self.blur_intensity = 5  # 1-10
        self.primary_detection_mode = "auto" # auto, manual
        self.show_settings = False
        self.show_hud = True
        self.manual_primary_id = None
        self.face_registry = load_registry()

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
    x1,y1,x2,y2 = box
    return max(0, x2-x1) * max(0, y2-y1)

def box_center(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def enhance_contrast_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def merge_boxes(boxes, iou_thresh=0.25):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    merged=[]; used=[False]*len(boxes)
    def iou(a,b):
        x1=max(a[0],b[0]); y1=max(a[1],b[1])
        x2=min(a[2],b[2]); y2=min(a[3],b[3])
        inter = max(0, x2-x1)*max(0, y2-y1)
        area_a = rect_area(a[:4]); area_b = rect_area(b[:4])
        denom = area_a + area_b - inter
        return inter/denom if denom>0 else 0
    for i,a in enumerate(boxes):
        if used[i]: continue
        group=[a]; used[i]=True
        for j in range(i+1,len(boxes)):
            if used[j]: continue
            if iou(a[:4], boxes[j][:4]) > iou_thresh:
                group.append(boxes[j]); used[j]=True
        xs1=[g[0] for g in group]; ys1=[g[1] for g in group]
        xs2=[g[2] for g in group]; ys2=[g[3] for g in group]
        scores=[g[4] for g in group]
        merged.append(((min(xs1), min(ys1), max(xs2), max(ys2)), max(scores)))
    return [(int(b[0][0]), int(b[0][1]), int(b[0][2]), int(b[0][3]), b[1]) for b in merged]

# ----------------------------
# BLURS
# ----------------------------
def apply_gaussian(roi, intensity):
    k = BLUR_MIN_KSIZE + (intensity-1) * ((BLUR_MAX_KSIZE - BLUR_MIN_KSIZE)//9)
    k = clamp_odd(k)
    return cv2.GaussianBlur(roi, (k,k), 0)

def apply_pixelate(roi, intensity):
    pixel_scale = max(2, intensity*2)
    h,w = roi.shape[:2]
    small_w = max(1, w//pixel_scale); small_h = max(1, h//pixel_scale)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w,h), interpolation=cv2.INTER_NEAREST)

def apply_mosaic(roi, intensity):
    block = max(4, intensity*6)
    h,w = roi.shape[:2]; out = roi.copy()
    for y in range(0,h,block):
        for x in range(0,w,block):
            y2=min(h, y+block); x2=min(w, x+block)
            patch = roi[y:y2, x:x2]
            if patch.size==0: continue
            color = patch.mean(axis=(0,1)).astype(np.uint8)
            out[y:y2, x:x2] = color
    return out

BLUR_FUNCS = {"gaussian": apply_gaussian, "pixelate": apply_pixelate, "mosaic": apply_mosaic}

# ----------------------------
# DETECTORS
# ----------------------------
mp_face = mp.solutions.face_detection
mp_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.45)

dnn_net = None
if USE_OPENCV_DNN:
    try:
        dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
        print("[INFO] Loaded OpenCV DNN model.")
    except Exception as e:
        print(f"[WARN] Could not load DNN model: {e}")
        dnn_net = None

# ----------------------------
# ASYNC DNN WORKER
# ----------------------------
dnn_lock = threading.Lock()
dnn_request_frame = None     # small-space frame for worker
dnn_result = []
dnn_stop = False

def dnn_worker():
    global dnn_request_frame, dnn_result, dnn_stop
    while not dnn_stop:
        frame = None
        with dnn_lock:
            if dnn_request_frame is not None:
                frame = dnn_request_frame.copy()
                dnn_request_frame = None
        if frame is None:
            time.sleep(0.004)
            continue
        try:
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0,177.0,123.0))
            dnn_net.setInput(blob)
            out = dnn_net.forward()
            res=[]
            H,W = frame.shape[:2]
            for i in range(out.shape[2]):
                conf = float(out[0,0,i,2])
                if conf>0.45:
                    box = out[0,0,i,3:7] * np.array([W,H,W,H])
                    x1,y1,x2,y2 = box.astype(int)
                    x1,y1 = max(0,x1), max(0,y1)
                    x2,y2 = min(W-1,x2), min(H-1,y2)
                    if x2-x1 >= FACE_MIN_SIZE and y2-y1 >= FACE_MIN_SIZE:
                        res.append((x1,y1,x2,y2,conf))
            with dnn_lock:
                dnn_result = res
        except Exception:
            with dnn_lock:
                dnn_result = []
        time.sleep(0.002)

dnn_thread = None
if dnn_net is not None:
    dnn_thread = threading.Thread(target=dnn_worker, daemon=True)
    dnn_thread.start()

# ----------------------------
# STATE
# ----------------------------
next_face_id = 0
faces_state = {}

# ----------------------------
# CAPTURE / WINDOW
# ----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    print("[FATAL] Cannot open camera")
    exit()

cv2.namedWindow("StealthZone", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("StealthZone", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ----------------------------
# SETTINGS OVERLAY (non-blocking key loop)
# ----------------------------
def settings_overlay_loop(num_primary=1, blur_style="gaussian", intensity=6):
    blur_styles = list(BLUR_FUNCS.keys())
    blur_idx = blur_styles.index(blur_style) if blur_style in blur_styles else 0
    hint = "UP/DOWN primaries   LEFT/RIGHT intensity   b:cycle blur   c:confirm   q:quit"
    last_draw = 0
    # non-blocking: loop until confirm; camera continues to be captured underneath
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        overlay = frame.copy()
        cv2.rectangle(overlay, (50,50), (FRAME_WIDTH-50, FRAME_HEIGHT-50), (0,0,0), -1)
        frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

        x,y,gap = 80,120,40
        cv2.putText(frame, "StealthZone — Interactive Setup", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220,220,220), 2)
        y += int(gap*1.5)
        cv2.putText(frame, f"Primary faces: {num_primary}   (UP/DOWN)", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
        y += gap
        cv2.putText(frame, f"Blur style: {blur_styles[blur_idx]}   (press 'b' to cycle)", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
        y += gap
        cv2.putText(frame, f"Blur intensity: {intensity} (1–10)   (LEFT/RIGHT)", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)

        # centered hint
        hint_size = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        hint_x = int((FRAME_WIDTH - hint_size[0]) / 2)
        hint_y = FRAME_HEIGHT - 80
        cv2.putText(frame, hint, (hint_x, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

        cv2.imshow("StealthZone", frame)
        key = cv2.waitKey(20) & 0xFF

        if key == 82:   # UP
            num_primary = min(MAX_PRIMARY_ALLOW, num_primary + 1)
        elif key == 84: # DOWN
            num_primary = max(1, num_primary - 1)
        elif key == 81: # LEFT
            intensity = max(1, intensity - 1)
        elif key == 83: # RIGHT
            intensity = min(10, intensity + 1)
        elif key in (ord('b'), ord('B')):
            blur_idx = (blur_idx + 1) % len(blur_styles)
        elif key in (ord('c'), 13, 10):  # confirm
            return num_primary, blur_styles[blur_idx], intensity
        elif key in (ord('q'), 27):
            cap.release(); cv2.destroyAllWindows(); exit()
        # loop continues non-blocking so camera remains live

# initial settings (can be changed in overlay)
num_primary_faces, blur_style, blur_intensity = settings_overlay_loop()

# ----------------------------
# MAIN LOOP (real-time)
# ----------------------------
frame_idx = 0
t_last = time.time()
fps = 0.0
primary_pointer = 0

try:
    while True:
        ret, full_frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame_idx += 1
        full_frame = cv2.flip(full_frame, 1)
        full_h, full_w = full_frame.shape[:2]

        # create small detection frame
        detect_frame = cv2.resize(full_frame, (DETECT_W, DETECT_H))
        if frame_idx % CLAHE_EVERY_N_FRAMES == 0:
            detect_for_detector = enhance_contrast_clahe(detect_frame)
        else:
            detect_for_detector = detect_frame

        detections_small = []

        # MediaPipe on small frame (always run)
        img_rgb = cv2.cvtColor(detect_for_detector, cv2.COLOR_BGR2RGB)
        results = mp_detector.process(img_rgb)
        if results.detections:
            for d in results.detections:
                bbox = d.location_data.relative_bounding_box
                x1 = int(bbox.xmin * DETECT_W); y1 = int(bbox.ymin * DETECT_H)
                x2 = int((bbox.xmin + bbox.width) * DETECT_W)
                y2 = int((bbox.ymin + bbox.height) * DETECT_H)
                score = d.score[0] if d.score else 0.5
                x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(DETECT_W-1,x2), min(DETECT_H-1,y2)
                if x2-x1 >= FACE_MIN_SIZE and y2-y1 >= FACE_MIN_SIZE:
                    detections_small.append((x1,y1,x2,y2,float(score)))

        # DNN fallback async trigger (submit frame occasionally / when none found)
        want_dnn = False
        if dnn_net is not None:
            if len(detections_small) == 0 or (frame_idx % DNN_RUN_EVERY_N_FRAMES == 0):
                want_dnn = True
        if want_dnn and dnn_net is not None:
            with dnn_lock:
                if dnn_request_frame is None:
                    dnn_request_frame = detect_for_detector.copy()
            # copy latest dnn results (may be from earlier invocation)
            with dnn_lock:
                if dnn_result:
                    detections_small.extend(list(dnn_result))

        # small-angle rotations if still empty (fast exit on first found)
        if len(detections_small) == 0 and len(ROTATION_ANGLES) > 1:
            for angle in ROTATION_ANGLES:
                if angle == 0: continue
                M = cv2.getRotationMatrix2D((DETECT_W/2, DETECT_H/2), angle, 1.0)
                rot = cv2.warpAffine(detect_for_detector, M, (DETECT_W, DETECT_H))
                r_rgb = cv2.cvtColor(rot, cv2.COLOR_BGR2RGB)
                rres = mp_detector.process(r_rgb)
                if rres.detections:
                    for d in rres.detections:
                        bbox = d.location_data.relative_bounding_box
                        x1 = int(bbox.xmin * DETECT_W); y1 = int(bbox.ymin * DETECT_H)
                        x2 = int((bbox.xmin + bbox.width) * DETECT_W)
                        y2 = int((bbox.ymin + bbox.height) * DETECT_H)
                        cx,cy = (x1 + x2)/2.0, (y1 + y2)/2.0
                        M_inv = cv2.getRotationMatrix2D((DETECT_W/2, DETECT_H/2), -angle, 1.0)
                        pts = np.array([[cx,cy,1]]).T
                        new = np.dot(M_inv, pts).flatten()
                        ncx, ncy = new[0], new[1]
                        bw, bh = x2-x1, y2-y1
                        nx1 = int(ncx - bw/2); ny1 = int(ncy - bh/2)
                        nx2 = int(ncx + bw/2); ny2 = int(ncy + bh/2)
                        score = d.score[0] if d.score else 0.45
                        nx1,ny1 = max(0,nx1), max(0,ny1)
                        nx2,ny2 = min(DETECT_W-1,nx2), min(DETECT_H-1,ny2)
                        if nx2-nx1 >= FACE_MIN_SIZE and ny2-ny1 >= FACE_MIN_SIZE:
                            detections_small.append((nx1,ny1,nx2,ny2,float(score)))
                    if detections_small:
                        break

        # merge & scale up to full frame
        detections_small = merge_boxes(detections_small, iou_thresh=0.2)
        scale_x = full_w / float(DETECT_W)
        scale_y = full_h / float(DETECT_H)
        detections = []
        for (x1,y1,x2,y2,score) in detections_small:
            X1 = int(x1 * scale_x); Y1 = int(y1 * scale_y)
            X2 = int(x2 * scale_x); Y2 = int(y2 * scale_y)
            X1,Y1 = max(0,X1), max(0,Y1); X2,Y2 = min(full_w-1,X2), min(full_h-1,Y2)
            if X2-X1 >= FACE_MIN_SIZE and Y2-Y1 >= FACE_MIN_SIZE:
                detections.append((X1,Y1,X2,Y2,score))

        # assign IDs & smoothing
        used_ids = set(); new_faces_state = {}
        for det in detections:
            x1,y1,x2,y2,score = det
            d_center = box_center((x1,y1,x2,y2))
            best_id = None; best_dist = 1e9
            for fid, st in faces_state.items():
                sb = st.get("smooth_box", st["box"])
                sc = box_center(sb)
                dist = math.hypot(d_center[0]-sc[0], d_center[1]-sc[1])
                if dist < best_dist and fid not in used_ids:
                    best_dist = dist; best_id = fid
            if best_id is not None and best_dist < max(full_w, full_h) * 0.25:
                st = faces_state[best_id]
                prev = st.get("smooth_box", st["box"])
                ns = SMOOTHING_ALPHA * np.array((x1,y1,x2,y2)) + (1-SMOOTHING_ALPHA) * np.array(prev)
                st.update({"box":(x1,y1,x2,y2), "score":score, "smooth_box":tuple(ns.tolist()), "last_seen_frame":frame_idx, "missed":0})
                new_faces_state[best_id] = st; used_ids.add(best_id)
            else:
                fid = next_face_id
                next_face_id += 1
                st = {"id":fid,"box":(x1,y1,x2,y2),"smooth_box":(x1,y1,x2,y2),"score":score,"last_seen_frame":frame_idx,"missed":0,"history":deque(maxlen=MAX_HISTORY)}
                new_faces_state[fid] = st; used_ids.add(fid)
        for fid, st in faces_state.items():
            if fid not in used_ids:
                st["missed"] = st.get("missed",0) + 1
                if st["missed"] <= KEEP_TRACK_FRAMES:
                    new_faces_state[fid] = st
        faces_state = new_faces_state

        # Face recognition for registered users (every few frames)
        if frame_idx % FACE_REC_EVERY_N_FRAMES == 0 and state.face_registry:
            for fid, st in faces_state.items():
                # Only check if not already recognized or periodically re-check
                if st.get('registered_name') is None or frame_idx % (FACE_REC_EVERY_N_FRAMES * 10) == 0:
                    x1, y1, x2, y2 = map(int, st.get("smooth_box", st["box"]))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(full_w, x2), min(full_h, y2)
                    
                    if (x2 - x1) > FACE_MIN_SIZE and (y2 - y1) > FACE_MIN_SIZE:
                        face_crop = full_frame[y1:y2, x1:x2]
                        
                        # Get embedding using DeepFace
                        embedding = get_face_embedding(face_crop)
                        
                        if embedding is not None:
                            name, distance = find_match(embedding, state.face_registry)
                            st['registered_name'] = name  # Will be None if no match
                            if name:
                                print(f"[INFO] Recognized registered face: {name} (distance: {distance:.3f})")

        # primary selection (largest/center + registered faces)
        face_list=[]
        for fid, st in faces_state.items():
            sb = tuple(map(int, st.get("smooth_box", st["box"])))
            area = rect_area(sb)
            cx,cy = box_center(sb)
            center_dist = math.hypot(cx - full_w/2.0, cy - full_h/2.0)
            priority = area - (center_dist * 50.0)
            
            # Boost priority for registered faces
            if st.get('registered_name') is not None:
                priority += 1e6  # Ensure registered faces are always primary
            
            face_list.append((fid, st, priority))
        face_list_sorted = sorted(face_list, key=lambda x: x[2], reverse=True)
        ids_ordered = [fid for fid, st, _ in face_list_sorted]
        total_faces = len(ids_ordered)
        primary_ids = set()
        if total_faces > 0:
            primary_pointer = primary_pointer % max(1, total_faces)
            for i in range(min(num_primary_faces, total_faces)):
                idx = (primary_pointer + i) % total_faces
                primary_ids.add(ids_ordered[idx])

        # blur non-primaries
        blurred_info=[]
        for fid, st, _ in face_list_sorted:
            x1,y1,x2,y2 = map(int, st.get("smooth_box", st["box"]))
            score = st.get("score", 0.5)
            x1,y1 = max(0,x1), max(0,y1); x2,y2 = min(full_w-1,x2), min(full_h-1,y2)
            if fid in primary_ids:
                cv2.rectangle(full_frame, (x1,y1), (x2,y2), (0,200,0), 2)
                # Show registered name if available
                label = st.get('registered_name') or f"P id:{fid}"
                cv2.putText(full_frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0),2)
            else:
                roi = full_frame[y1:y2, x1:x2]
                if roi.size == 0: continue
                try:
                    if blur_style=="gaussian":
                        out_roi = apply_gaussian(roi, blur_intensity)
                    elif blur_style=="pixelate":
                        out_roi = apply_pixelate(roi, blur_intensity)
                    else:
                        out_roi = apply_mosaic(roi, blur_intensity)
                except Exception:
                    out_roi = cv2.GaussianBlur(roi, (clamp_odd(15), clamp_odd(15)), 0)
                full_frame[y1:y2, x1:x2] = out_roi
                blurred_info.append((fid, (x1,y1,x2,y2), score))
                cv2.rectangle(full_frame, (x1,y1), (x2,y2), (0,120,255), 2)
                cv2.putText(full_frame, f"B id:{fid}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,120,255),1)

        # privacy & fps
        if total_faces == 0:
            privacy_conf = 0.0
        else:
            blurred_sum = sum([s for _,_,s in blurred_info]) if blurred_info else 0.0
            privacy_conf = min(1.0, blurred_sum / max(1, total_faces))
        privacy_pct = int(privacy_conf * 100)

        t_now = time.time()
        fps = 0.9*fps + 0.1*(1.0/(t_now - t_last)) if t_now != t_last else fps
        t_last = t_now

        # HUD
        cv2.putText(full_frame, f"FPS:{fps:.1f}", (18,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)
        cv2.putText(full_frame, f"Faces:{total_faces}", (18,58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.putText(full_frame, f"Privacy:{privacy_pct}%", (18,88), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255 if privacy_pct>60 else 255,200),2)
        cv2.putText(full_frame, f"Primaries:{num_primary_faces}  Blur:{blur_style}({blur_intensity})", (full_w-520, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220),1)
        cv2.putText(full_frame, "n:next p:prev s:settings space:snap q:quit", (full_w-520, full_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)

        cv2.imshow("StealthZone", full_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('n'):
            if total_faces > 0:
                primary_pointer = (primary_pointer + 1) % max(1, total_faces)
        elif key == ord('p'):
            if total_faces > 0:
                primary_pointer = (primary_pointer - 1) % max(1, total_faces)
        elif key == ord('s'):
            num_primary_faces, blur_style, blur_intensity = settings_overlay_loop(num_primary_faces, blur_style, blur_intensity)
            primary_pointer = 0
        elif key == 32:
            fname = f"stealth_snap_{int(time.time())}.jpg"
            cv2.imwrite(fname, full_frame)
            print(f"[INFO] saved {fname}")

except KeyboardInterrupt:
    pass
finally:
    dnn_stop = True
    if dnn_thread is not None:
        dnn_thread.join(timeout=0.2)
    cap.release()
    cv2.destroyAllWindows()
