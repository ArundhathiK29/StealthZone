import cv2, mediapipe as mp, os, time, shutil, numpy as np

DB_PATH="lbph_face_db"
MODEL_PATH=os.path.join(DB_PATH,"lbph_model.yml")
LABELS_PATH=os.path.join(DB_PATH,"labels.txt")
NUM_SAMPLES=10
SAMPLE_DELAY=1.2
BLUR_KERNEL=(35,35)

os.makedirs(DB_PATH,exist_ok=True)
mp_face=mp.solutions.face_detection
recognizer=cv2.face.LBPHFaceRecognizer_create()
label_to_name,name_to_label={},{}

# ---- load existing model ----
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH) as f:
        for line in f:
            if "," in line:
                l,n=line.strip().split(",")
                label_to_name[int(l)]=n
                name_to_label[n]=int(l)

def save_labels():
    with open(LABELS_PATH,"w") as f:
        for l,n in label_to_name.items():
            f.write(f"{l},{n}\n")

def retrain():
    faces,labels=[],[]
    for n,l in name_to_label.items():
        p=os.path.join(DB_PATH,n)
        for img in os.listdir(p):
            g=cv2.imread(os.path.join(p,img),cv2.IMREAD_GRAYSCALE)
            if g is not None:
                faces.append(g); labels.append(l)
    if faces:
        recognizer.train(faces,np.array(labels))
        recognizer.save(MODEL_PATH); save_labels()

cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found."); exit()

state="idle"
person_dir=None; pending_name=None; label=None; samples=0
last_capture=0; blur_mode=False
detector=mp_face.FaceDetection(model_selection=1,min_detection_confidence=0.5)

while True:
    ok,frame=cap.read()
    if not ok: break
    h,w=frame.shape[:2]
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    res=detector.process(rgb)
    faces=[]
    if res.detections:
        for d in res.detections:
            b=d.location_data.relative_bounding_box
            x1=int(b.xmin*w); y1=int(b.ymin*h)
            x2=int(x1+b.width*w); y2=int(y1+b.height*h)
            if 0<=x1<x2<=w and 0<=y1<y2<=h:
                faces.append((x1,y1,x2,y2))

    disp=frame.copy()

    # ---------- live detection & blur ----------
    if state=="idle" and os.path.exists(MODEL_PATH):
        for (x1,y1,x2,y2) in faces:
            face=cv2.resize(gray[y1:y2,x1:x2],(200,200))
            try:
                label_pred,conf=recognizer.predict(face)
                if conf<65:
                    name=label_to_name.get(label_pred,"Unknown")
                    color=(0,255,0)
                    known=True
                else:
                    name="Unregistered"; color=(0,0,255); known=False
            except:
                name="Unregistered"; color=(0,0,255); known=False

            if blur_mode and not known:
                roi=disp[y1:y2,x1:x2]
                disp[y1:y2,x1:x2]=cv2.GaussianBlur(roi,BLUR_KERNEL,0)
            cv2.rectangle(disp,(x1,y1),(x2,y2),color,2)
            cv2.putText(disp,name,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # ---------- registration capture ----------
    if state=="registering":
        now=time.time()
        if res.detections and now-last_capture>SAMPLE_DELAY:
            (x1,y1,x2,y2)=faces[0]
            crop=cv2.resize(gray[y1:y2,x1:x2],(200,200))
            samples+=1
            cv2.imwrite(os.path.join(person_dir,f"{samples}.jpg"),crop)
            last_capture=now
        cv2.putText(disp,f"Capturing {samples}/{NUM_SAMPLES}",
                    (30,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
        if samples>=NUM_SAMPLES:
            state="confirm"

    # ---------- confirmation prompt ----------
    if state=="confirm":
        cv2.putText(disp,f"Save {pending_name}? (Y/N)",
                    (30,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)

    # ---------- overlay controls ----------
    mode_text="Privacy: ON" if blur_mode else "Privacy: OFF"
    cv2.putText(disp,f"{mode_text}",(20,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
    cv2.putText(disp,"[r] register  [d] delete last  [b] toggle privacy  [q] quit",
                (20,h-20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.imshow("StealthZone – Face Privacy",disp)
    k=cv2.waitKey(1)&0xFF

    # ---------- keys ----------
    if k==ord('r') and state=="idle":
        pid=len(name_to_label)+1
        pending_name=f"person_{pid}"
        label=len(name_to_label)
        name_to_label[pending_name]=label
        label_to_name[label]=pending_name
        person_dir=os.path.join(DB_PATH,pending_name)
        os.makedirs(person_dir,exist_ok=True)
        samples=0; last_capture=time.time()
        state="registering"
        print(f"▶ Starting capture for {pending_name}")

    elif state=="confirm":
        if k==ord('y'):
            retrain()
            print(f"✔ Registered {pending_name}")
            state="idle"
        elif k==ord('n'):
            shutil.rmtree(person_dir,ignore_errors=True)
            del name_to_label[pending_name]; del label_to_name[label]
            print("✖ Discarded"); state="idle"

    elif k==ord('d') and state=="idle":
        if name_to_label:
            last=sorted(name_to_label.keys())[-1]
            print(f"🗑 Deleting {last}")
            shutil.rmtree(os.path.join(DB_PATH,last),ignore_errors=True)
            del label_to_name[name_to_label[last]]
            del name_to_label[last]
            retrain()
        else:
            print("No entries.")

    elif k==ord('b') and state=="idle":
        blur_mode=not blur_mode
        print(f"🔁 Privacy mode: {'ON' if blur_mode else 'OFF'}")

    elif k==ord('q'):
        break

cap.release(); cv2.destroyAllWindows()
print("Camera closed safely.")
