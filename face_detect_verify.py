# face_detect_verify.py
import cv2
import mediapipe as mp
import time
import math

mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Tunable params
DETECTION_CONFIDENCE = 0.5   # min detection score to keep a face
PRIMARY_ALPHA = 2.0          # weight for center-distance penalty in primary scoring
DISPLAY_SCALE = 1.0          # if you want to scale the displayed frame

def rect_area(box, frame_w, frame_h):
    # box -> [xmin, ymin, width, height] in normalized coords
    w = box.xmax - box.xmin
    h = box.ymax - box.ymin
    return (w * frame_w) * (h * frame_h)

def box_to_px(box, frame_w, frame_h):
    # returns integer pixel rect (x, y, w, h)
    x = int(box.xmin * frame_w)
    y = int(box.ymin * frame_h)
    w = int((box.xmax - box.xmin) * frame_w)
    h = int((box.ymax - box.ymin) * frame_h)
    return x, y, w, h

def center_distance(box, frame_w, frame_h):
    x, y, w, h = box_to_px(box, frame_w, frame_h)
    cx_face = x + w / 2.0
    cy_face = y + h / 2.0
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    return math.hypot(cx_face - cx, cy_face - cy)

def draw_label(img, text, x, y, bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(img, (x, y-th-4), (x+tw+4, y+4), bg_color, -1)
    cv2.putText(img, text, (x+2, y-2), font, scale, (255,255,255), thickness, cv2.LINE_AA)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam. Exiting.")
        return

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=DETECTION_CONFIDENCE) as detector:
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ No frame received from webcam — check camera index.")
                break


            frame_h, frame_w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            faces = []
            if results.detections:
                for det in results.detections:
                    # bounding box in normalized coords
                    bbox = det.location_data.relative_bounding_box
                    # MediaPipe's relative_bounding_box fields: xmin, ymin, width, height
                    # Convert to an object with xmin,xmax,ymin,ymax to reuse functions
                    class Box:
                        pass
                    b = Box()
                    b.xmin = max(0.0, bbox.xmin)
                    b.ymin = max(0.0, bbox.ymin)
                    b.xmax = min(1.0, bbox.xmin + bbox.width)
                    b.ymax = min(1.0, bbox.ymin + bbox.height)

                    score = det.score[0] if det.score else 0.0
                    faces.append({"box": b, "score": score, "detection": det})

            # Filter by detection confidence
            faces = [f for f in faces if f["score"] >= DETECTION_CONFIDENCE]

            # Compute primary face score
            best_idx = None
            best_score = -1e9
            for i, f in enumerate(faces):
                area = rect_area(f["box"], frame_w, frame_h)
                dist = center_distance(f["box"], frame_w, frame_h)
                combined = area - PRIMARY_ALPHA * dist
                faces[i]["area"] = area
                faces[i]["dist"] = dist
                faces[i]["combined_score"] = combined
                if combined > best_score:
                    best_score = combined
                    best_idx = i

            # Draw results
            for i, f in enumerate(faces):
                x, y, w, h = box_to_px(f["box"], frame_w, frame_h)
                label = f"{f['score']:.2f} a={int(f['area'])} d={int(f['dist'])}"
                if i == best_idx:
                    # primary face — green box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                    draw_label(frame, "Primary: "+label, x, y, (0,200,0))
                else:
                    # secondary faces — red box
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
                    draw_label(frame, "Other: "+label, x, y, (50,50,200))

            # FPS counter
            curr_time = time.time()
            fps = 1.0/(curr_time - prev_time) if curr_time != prev_time else 0.0
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Show frame
            if DISPLAY_SCALE != 1.0:
                frame = cv2.resize(frame, (int(frame_w*DISPLAY_SCALE), int(frame_h*DISPLAY_SCALE)))
            cv2.imshow("Face Detection + Primary Identification", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('s'):
                # press 's' to save a sample frame + debug data
                cv2.imwrite("sample_frame_debug.jpg", frame)
                print("Saved sample_frame_debug.jpg")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
