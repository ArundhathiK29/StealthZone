"""
register_face.py

A utility to register a new face for the StealthZone application.

Instructions:
1. Run the script from your terminal: `python register_face.py`
2. Enter the name of the person you are registering.
3. A webcam window will open. Position the person's face in the frame.
4. Press the 'c' key to capture an image. Take 5-10 photos with slight variations
   in expression and angle for best results.
5. A green rectangle will confirm a face was found and captured.
6. Once you have enough captures, press 's' to save the aggregated face embedding.
7. Press 'q' to quit at any time.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
from face_registry import load_registry, save_registry, get_face_embedding

# --- CONFIG ---
CAPTURES_NEEDED = 5  # Number of captures to average for a new registration
OUTPUT_DIR = "registered_faces"
mp_face = mp.solutions.face_detection
# --------------

def main():
    # Get the name for the new registration
    name = input(f"Enter the name of the person to register and press Enter: ")
    if not name:
        print("Name cannot be empty. Exiting.")
        return

    # Create directory for saving captures
    person_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    print(f"Saving captures for '{name}' in '{person_dir}'")

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    captured_encodings = []
    capture_count = 0
    
    detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Flip the frame for a "selfie" view
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()
        h, w = frame.shape[:2]

        # Find all face locations in the current frame using MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        face_boxes = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # Clamp to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                area = (x2 - x1) * (y2 - y1)
                face_boxes.append((x1, y1, x2, y2, area))

        if face_boxes:
            # Use the largest face found
            x1, y1, x2, y2, _ = max(face_boxes, key=lambda x: x[4])
            
            # Draw a rectangle around the face
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display instructions
            text = f"Captures: {capture_count}/{CAPTURES_NEEDED}. Press 'c' to capture."
            cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if capture_count >= CAPTURES_NEEDED:
                cv2.putText(display_frame, "Ready! Press 's' to save.", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(display_frame, "Press 'q' to quit.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(f"Register Face: {name}", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        
        elif key == ord('c'):
            # Capture the face encoding
            if face_boxes:
                # Get the encoding for the largest face
                x1, y1, x2, y2, _ = max(face_boxes, key=lambda x: x[4])
                
                # Add margin to face crop for better embedding extraction
                margin = 20
                x1_crop = max(0, x1 - margin)
                y1_crop = max(0, y1 - margin)
                x2_crop = min(w, x2 + margin)
                y2_crop = min(h, y2 + margin)
                
                # Extract face region with margin
                face_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
                
                # Check if face crop is large enough
                crop_h, crop_w = face_crop.shape[:2]
                if crop_w < 50 or crop_h < 50:
                    print(f"Warning: Face too small ({crop_w}x{crop_h}). Move closer to camera.")
                    continue
                
                print(f"Extracting embedding from {crop_w}x{crop_h} face crop...")
                
                # Get embedding using DeepFace
                embedding = get_face_embedding(face_crop)

                if embedding is not None:
                    captured_encodings.append(embedding)
                    capture_count += 1
                    
                    # Save the captured image file
                    img_path = os.path.join(person_dir, f"{name}_{capture_count}.jpg")
                    cv2.imwrite(img_path, face_crop)
                    
                    print(f"✅ Capture {capture_count}/{CAPTURES_NEEDED} successful. Saved to {img_path}")
                else:
                    print("❌ Warning: Could not generate embedding for the detected face.")
                    print("   Possible reasons:")
                    print("   - Face quality too low (try better lighting)")
                    print("   - First-time model download (wait a few seconds and try again)")
                    print("   - Face angle too extreme (face camera more directly)")
            else:
                print("Warning: No face detected in the frame to capture.")

        elif key == ord('s'):
            if capture_count >= CAPTURES_NEEDED:
                print("Aggregating face encodings...")
                # Calculate the mean of the captured encodings
                mean_encoding = np.mean(captured_encodings, axis=0)

                # Load the existing registry, add/update the new person, and save
                registry = load_registry()
                registry[name] = mean_encoding
                save_registry(registry)
                
                print(f"\nSuccessfully registered '{name}'!")
                break
            else:
                print(f"Not enough captures yet. Please provide at least {CAPTURES_NEEDED} captures.")

    # Cleanup
    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
