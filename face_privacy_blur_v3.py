import cv2
import mediapipe as mp
import math
import time

mp_face = mp.solutions.face_detection
DETECTION_CONFIDENCE = 0.3
PRIMARY_ALPHA = 2.0
BLUR_TYPE = "gaussian"
DEBOUNCE_TIME = 0.4

def box_to_px(box, w, h):
    x, y = int(box.xmin * w), int(box.ymin * h)
    return x, y, int(box.width * w), int(box.height * h)

def rect_area(box, w, h):
    return (box.width * w) * (box.height * h)

def center_distance(box, w, h):
    cx_f, cy_f = (box.xmin + box.width/2)*w, (box.ymin + box.height/2)*h
    return math.hypot(cx_f - w/2, cy_f - h/2)

def blur_region(frame, x, y, w, h, mode, intensity_factor=1.0):
    roi = frame[y:y+h, x:x+w]
    if roi.size == 0: return frame
    base_k = max(1, (w // 7) * 2 + 1)
    k = int(base_k * (2.0 - intensity_factor))
    k = k if k % 2 == 1 else k + 1
    if mode == "gaussian":
        roi = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        small = cv2.resize(roi, (max(1,w//10), max(1,h//10)))
        roi = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    frame[y:y+h, x:x+w] = roi
    return frame

def main():
    global BLUR_TYPE
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ Webcam unavailable."); return

    last_toggle = 0
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=DETECTION_CONFIDENCE) as fd:
        print("Running Fast Face Privacy – press P to toggle blur, Q to quit")
        prev_t = time.time()

        while True:
            ok, frame = cap.read()
            if not ok: break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fd.process(rgb)

            faces = []
            if results.detections:
                for det in results.detections:
                    bbox = det.location_data.relative_bounding_box
                    faces.append({"box": bbox, "score": det.score[0]})

            # Find primary face (largest/most centered)
            best, best_idx = -1e9, -1
            for i, f in enumerate(faces):
                s = rect_area(f["box"], w, h) - PRIMARY_ALPHA * center_distance(f["box"], w, h)
                if s > best: best, best_idx = s, i

            # Blur each face except primary face
            for i, f in enumerate(faces):
                x, y, fw, fh = box_to_px(f["box"], w, h)
                area_ratio = rect_area(f["box"], w, h) / (w * h)
                intensity = max(0.4, 1.0 - area_ratio)  # Distant=Stronger Blur
                if i == best_idx:
                    cv2.rectangle(frame, (x,y), (x+fw,y+fh), (0,255,0), 2)
                    cv2.putText(frame, "Primary", (x, y-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                else:
                    frame = blur_region(frame, x, y, fw, fh, BLUR_TYPE, intensity)

            fps = 1 / (time.time() - prev_t)
            prev_t = time.time()
            cv2.putText(frame, f"FPS:{int(fps)} | Mode:{BLUR_TYPE}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.imshow("Fast Face Privacy", frame)

            k = cv2.waitKey(1) & 0xFF
            if k in [27, ord('q')]: break
            elif k == ord('p') and time.time()-last_toggle > DEBOUNCE_TIME:
                BLUR_TYPE = "pixel" if BLUR_TYPE=="gaussian" else "gaussian"
                last_toggle = time.time()
                print("🔄 Blur mode:", BLUR_TYPE)

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
