import cv2, mediapipe as mp, os, time, shutil, numpy as np, threading
import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk

# ---------------- CONFIG -----------------
DB_PATH = "lbph_face_db"
MODEL_PATH = os.path.join(DB_PATH, "lbph_model.yml")
LABELS_PATH = os.path.join(DB_PATH, "labels.txt")
NUM_SAMPLES = 10
SAMPLE_DELAY = 1.2
BLUR_KERNEL = (35, 35)
CONF_THRESHOLD = 65

os.makedirs(DB_PATH, exist_ok=True)
mp_face = mp.solutions.face_detection
recognizer = cv2.face.LBPHFaceRecognizer_create()
label_to_name, name_to_label = {}, {}

# ---------------- LOAD MODEL -----------------
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH) as f:
        for line in f:
            if "," in line:
                l, n = line.strip().split(",")
                label_to_name[int(l)] = n
                name_to_label[n] = int(l)

def save_labels():
    with open(LABELS_PATH, "w") as f:
        for l, n in label_to_name.items():
            f.write(f"{l},{n}\n")

def retrain():
    faces, labels = [], []
    for n, l in name_to_label.items():
        p = os.path.join(DB_PATH, n)
        for img in os.listdir(p):
            g = cv2.imread(os.path.join(p, img), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                faces.append(g)
                labels.append(l)
    if faces:
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_PATH)
        save_labels()
        print("✅ Model retrained.")


# ---------------- CAMERA HANDLER -----------------
running = False
blur_mode = True
mode = "idle"
pending_name = None
samples = 0
last_capture = 0

def camera_loop():
    global running, blur_mode, mode, pending_name, samples, last_capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Camera Error", "Could not open webcam.")
        return

    detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

    while running:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = detector.process(rgb)
        faces = []

        if res.detections:
            for d in res.detections:
                b = d.location_data.relative_bounding_box
                x1, y1 = int(b.xmin * w), int(b.ymin * h)
                x2, y2 = int(x1 + b.width * w), int(y1 + b.height * h)
                if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                    faces.append((x1, y1, x2, y2))

        disp = frame.copy()

        # Registration mode
        if mode == "register" and faces:
            (x1, y1, x2, y2) = faces[0]
            now = time.time()
            if now - last_capture > SAMPLE_DELAY:
                person_dir = os.path.join(DB_PATH, pending_name)
                os.makedirs(person_dir, exist_ok=True)
                face_img = cv2.resize(gray[y1:y2, x1:x2], (200, 200))
                filename = os.path.join(person_dir, f"{samples+1}.jpg")
                cv2.imwrite(filename, face_img)
                samples += 1
                last_capture = now
                if samples >= NUM_SAMPLES:
                    new_label = max(label_to_name.keys(), default=-1) + 1
                    label_to_name[new_label] = pending_name
                    name_to_label[pending_name] = new_label
                    retrain()
                    mode = "idle"
                    samples = 0
                    pending_name = None
                    messagebox.showinfo("Done", "Registration complete!")

            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(disp, f"Capturing {samples}/{NUM_SAMPLES}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Normal detection
        elif mode == "idle" and os.path.exists(MODEL_PATH):
            for (x1, y1, x2, y2) in faces:
                face = cv2.resize(gray[y1:y2, x1:x2], (200, 200))
                try:
                    label_pred, conf = recognizer.predict(face)
                    known = conf < CONF_THRESHOLD
                    name = label_to_name.get(label_pred, "Unknown") if known else "Unregistered"
                    color = (0, 255, 0) if known else (0, 0, 255)
                except:
                    known, name, color = False, "Unregistered", (0, 0, 255)

                if blur_mode and not known:
                    roi = disp[y1:y2, x1:x2]
                    disp[y1:y2, x1:x2] = cv2.GaussianBlur(roi, BLUR_KERNEL, 0)

                cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
                cv2.putText(disp, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Overlay mode info
        text = f"Privacy: {'ON' if blur_mode else 'OFF'}"
        cv2.putText(disp, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Convert for Tkinter
        img = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    cap.release()
    print("🟢 Camera closed.")


# ---------------- BUTTON HANDLERS -----------------
def start_camera():
    global running
    if not running:
        running = True
        threading.Thread(target=camera_loop, daemon=True).start()
        status_label.config(text="🟢 Camera Active", fg="green")

def stop_camera():
    global running
    running = False
    status_label.config(text="🔴 Camera Stopped", fg="red")

def toggle_privacy():
    global blur_mode
    blur_mode = not blur_mode
    toggle_btn.config(text=f"Privacy Mode: {'ON 🔒' if blur_mode else 'OFF 🔓'}",
                      bg="#4CAF50" if blur_mode else "#E53935")

def register_person():
    global mode, pending_name, samples
    if not running:
        messagebox.showinfo("Info", "Start camera first.")
        return
    name = simpledialog.askstring("Register Face", "Enter name:")
    if not name:
        return
    if name in name_to_label:
        messagebox.showerror("Error", "Name already exists.")
        return
    pending_name = name
    samples = 0
    mode = "register"

def delete_last():
    if not label_to_name:
        messagebox.showinfo("Info", "No entries.")
        return
    last_label = max(label_to_name.keys())
    last_name = label_to_name[last_label]
    shutil.rmtree(os.path.join(DB_PATH, last_name), ignore_errors=True)
    del label_to_name[last_label]
    del name_to_label[last_name]
    retrain()
    messagebox.showinfo("Deleted", f"Removed {last_name}")

def quit_app():
    global running
    running = False
    root.destroy()


# ---------------- GUI -----------------
root = tk.Tk()
root.title("StealthZone – Unified Privacy GUI")
root.geometry("820x700")
root.configure(bg="#1E1E1E")

tk.Label(root, text="🧠 StealthZone – Face Privacy System", font=("Helvetica", 18, "bold"),
         bg="#1E1E1E", fg="white").pack(pady=10)

video_label = tk.Label(root, bg="black")
video_label.pack(pady=10)

status_label = tk.Label(root, text="🔴 Camera Stopped", font=("Arial", 12), fg="red", bg="#1E1E1E")
status_label.pack(pady=5)

btn_frame = tk.Frame(root, bg="#1E1E1E")
btn_frame.pack(pady=15)

tk.Button(btn_frame, text="▶ Start Camera", font=("Arial", 12), bg="#4CAF50", fg="white",
          width=20, command=start_camera).grid(row=0, column=0, padx=10, pady=5)
tk.Button(btn_frame, text="⏹ Stop Camera", font=("Arial", 12), bg="#F44336", fg="white",
          width=20, command=stop_camera).grid(row=0, column=1, padx=10, pady=5)

toggle_btn = tk.Button(btn_frame, text="Privacy Mode: ON 🔒", font=("Arial", 12),
                       bg="#4CAF50", fg="white", width=20, command=toggle_privacy)
toggle_btn.grid(row=1, column=0, padx=10, pady=5)

tk.Button(btn_frame, text="🧍 Register Face", font=("Arial", 12),
          bg="#2196F3", fg="white", width=20, command=register_person).grid(row=1, column=1, padx=10, pady=5)

tk.Button(btn_frame, text="🗑 Delete Last", font=("Arial", 12),
          bg="#FF9800", fg="white", width=20, command=delete_last).grid(row=2, column=0, padx=10, pady=5)

tk.Button(btn_frame, text="❌ Quit", font=("Arial", 12),
          bg="grey", fg="white", width=20, command=quit_app).grid(row=2, column=1, padx=10, pady=5)

root.protocol("WM_DELETE_WINDOW", quit_app)
root.mainloop()
