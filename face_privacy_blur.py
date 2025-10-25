# face_privacy_blur_fixed.py
import cv2
import mediapipe as mp
import math
import time

mp_face = mp.solutions.face_detection

# ===== CONFIG =====
DETECTION_CONFIDENCE = 0.5
PRIMARY_ALPHA = 2.0           # Center weighting
BLUR_TYPE = "gaussian"        # "gaussian" or "pixel"
DETECT_EVERY_N_FRAMES = 2     # Run detection every N frames to boost FPS
# ===================

def box_to_px(box, w, h):
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    width = int(box.width * w)
    height = int(box.height * h)
    return x, y, width, height

def rect_area(box, w, h):
    return (box.width * w) * (box.height * h)

def center_distance(box, w, h):
    cx_face = (box.xmin + box.width / 2) * w
    cy_face = (box.ymin + box.height / 2) * h
    cx, cy = w / 2, h / 2
    return math.hypot(cx_face - cx, cy_face - cy)

def blur_region(frame, x, y, w, h, blur_type="gaussian"):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0:
        return frame
    if blur_type == "gaussian":
        k = (w // 15) * 2 + 1
        roi = cv2.GaussianBlur(roi, (k, k), 0)
    else:  # pixelation
        small = cv2.resize(roi, (max(1, w//10), max(1, h//10)), interpolation=cv2.INTER_LINEAR)
        roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = roi
    return frame

def main():
    global BLUR_TYPE  # ✅ declare once at top of function

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    with mp_face.FaceDetection(model_selection=0,
                               min_detection_confidence=DETECTION_CONFIDENCE) as fd:
        prev_time = time.time()
        frame_count = 0
        faces = []  # cache faces for frame-skipping

        print("🎥 Face Privacy Layer running... Press 'p' to toggle blur, 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No frame received.")
                break

            h, w = frame.shape[:2]
            frame_count += 1

            # Run detection only every N frames for efficiency
            if frame_count % DETECT_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = fd.process(rgb)
                faces = []
                if results.detections:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        faces.append({
                            "box": bbox,
                            "score": det.score[0] if det.score else 0.0
                        })
                faces = [f for f in faces if f["score"] >= DETECTION_CONFIDENCE]

            # Select primary face
            best_idx, best_score = -1, -1e9
            for i, f in enumerate(faces):
                area = rect_area(f["box"], w, h)
                dist = center_distance(f["box"], w, h)
                score = area - PRIMARY_ALPHA * dist
                if score > best_score:
                    best_score, best_idx = score, i

            # Blur or mark faces
            for i, f in enumerate(faces):
                x, y, fw, fh = box_to_px(f["box"], w, h)
                if i != best_idx:
                    frame = blur_region(frame, x, y, fw, fh, BLUR_TYPE)
                else:
                    cv2.rectangle(frame, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
                    cv2.putText(frame, "Primary", (x, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # HUD info
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS:{int(fps)}  Faces:{len(faces)}  Mode:{BLUR_TYPE}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Face Privacy Layer", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [27, ord('q')]:
                break
            elif key == ord('p'):
                BLUR_TYPE = "pixel" if BLUR_TYPE == "gaussian" else "gaussian"
                print(f"🔄 Blur mode switched to: {BLUR_TYPE}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
