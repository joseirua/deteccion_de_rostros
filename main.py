import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os
import time
import json
from utils_capture import (
    load_face_detector, 
    detect_faces, 
    CentroidTracker, 
    next_available_uid, 
    make_user_folder_when_starting, 
    save_face_image, 
    DATASET_DIR
)
from utils_train import train_model, save_metadata

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial Unificado")
        self.root.geometry("900x700")
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both')
        
        self.tab_capture = ttk.Frame(self.notebook)
        self.tab_train = ttk.Frame(self.notebook)
        self.tab_recognize = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_capture, text='Captura')
        self.notebook.add(self.tab_train, text='Entrenar')
        self.notebook.add(self.tab_recognize, text='Reconocimiento')
        
        self.cap = None
        self.is_running = False
        
        self.setup_capture_tab()
        self.setup_train_tab()
        self.setup_recognize_tab()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_capture_tab(self):
        panel = tk.Frame(self.tab_capture)
        panel.pack(side='left', fill='y', padx=10, pady=10)
        
        tk.Label(panel, text="Control de Captura", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.btn_start_capture = tk.Button(panel, text="Iniciar Cámara", command=self.start_camera_capture, bg="#2196F3", fg="white", width=20)
        self.btn_start_capture.pack(pady=5)
        
        self.btn_stop_capture = tk.Button(panel, text="Detener Cámara", command=self.stop_camera, bg="#f44336", fg="white", width=20, state='disabled')
        self.btn_stop_capture.pack(pady=5)
        
        self.btn_capture_faces = tk.Button(panel, text="Empezar a Guardar (p)", command=self.toggle_saving_faces, bg="#4CAF50", fg="white", width=20, state='disabled')
        self.btn_capture_faces.pack(pady=20)
        
        self.lbl_status = tk.Label(panel, text="Estado: Inactivo", fg="gray")
        self.lbl_status.pack(pady=5)
        
        self.video_label_capture = tk.Label(self.tab_capture, bg="black")
        self.video_label_capture.pack(side='right', expand=True, fill='both', padx=10, pady=10)
        
        self.saving_faces = False
        self.current_user_folder = None
        self.current_user_id = None
        self.saved_count = 0
        self.next_uid = 1
        
        self.detector = None
        self.tracker = None

    def setup_train_tab(self):
        frame = tk.Frame(self.tab_train)
        frame.pack(expand=True, padx=20, pady=20)
        
        tk.Label(frame, text="Entrenamiento de Modelo", font=("Arial", 16, "bold")).pack(pady=20)
        
        tk.Label(frame, text="Seleccionar Usuario:").pack(anchor='w')
        self.combo_users = ttk.Combobox(frame, state="readonly", width=40)
        self.combo_users.pack(pady=5)
        
        self.btn_refresh = tk.Button(frame, text="Actualizar Lista", command=self.refresh_user_list)
        self.btn_refresh.pack(pady=5)
        
        tk.Label(frame, text="Datos del Usuario:", font=("Arial", 12, "bold")).pack(pady=(20, 10), anchor='w')
        
        self.entries = {}
        fields = [("Nombre", "name"), ("Apellido", "lastname"), ("ID/Matrícula", "id"), ("Edad", "age"), ("Descripción", "desc")]
        
        for label, key in fields:
            row = tk.Frame(frame)
            row.pack(fill='x', pady=2)
            tk.Label(row, text=f"{label}:", width=15, anchor='w').pack(side='left')
            entry = tk.Entry(row)
            entry.pack(side='right', expand=True, fill='x')
            self.entries[key] = entry
            
        self.btn_train = tk.Button(frame, text="Guardar y Entrenar", command=self.train, bg="#4CAF50", fg="white", font=("Arial", 12))
        self.btn_train.pack(pady=30, fill='x')
        
        self.refresh_user_list()

    def setup_recognize_tab(self):
        panel = tk.Frame(self.tab_recognize)
        panel.pack(side='left', fill='y', padx=10, pady=10)
        
        tk.Label(panel, text="Reconocimiento", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.btn_start_recog = tk.Button(panel, text="Iniciar Reconocimiento", command=self.start_camera_recognition, bg="#FF9800", fg="white", width=20)
        self.btn_start_recog.pack(pady=5)
        
        self.btn_stop_recog = tk.Button(panel, text="Detener", command=self.stop_camera, bg="#f44336", fg="white", width=20, state='disabled')
        self.btn_stop_recog.pack(pady=5)
        
        self.video_label_recog = tk.Label(self.tab_recognize, bg="black")
        self.video_label_recog.pack(side='right', expand=True, fill='both', padx=10, pady=10)
        
        self.recognizer = None
        self.labels_map = {}

    def refresh_user_list(self):
        if not os.path.exists(DATASET_DIR):
            return
        users = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
        self.combo_users['values'] = users

    def start_camera_capture(self):
        if self.is_running:
            return
            
        proto = "models/deploy.prototxt"   
        model = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.detector = load_face_detector(proto, model)
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)
        self.next_uid = next_available_uid()
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara")
            return
            
        self.is_running = True
        self.btn_start_capture.config(state='disabled')
        self.btn_stop_capture.config(state='normal')
        self.btn_capture_faces.config(state='normal')
        self.video_loop_capture()

    def toggle_saving_faces(self):
        if not self.saving_faces:
            self.current_user_folder = make_user_folder_when_starting(self.next_uid)
            self.current_user_id = None
            self.saved_count = 0
            self.saving_faces = True
            self.lbl_status.config(text=f"Capturando para nuevo usuario...", fg="green")
            self.btn_capture_faces.config(text="Detener Guardado (p)")
        else:
            self.saving_faces = False
            self.lbl_status.config(text="Captura detenida", fg="red")
            self.btn_capture_faces.config(text="Empezar a Guardar (p)")
            if self.saved_count > 0:
                self.next_uid += 1
                messagebox.showinfo("Info", f"Usuario guardado en {self.current_user_folder}")
            else:
                try:
                    os.rmdir(self.current_user_folder)
                except:
                    pass
            self.current_user_folder = None
            self.current_user_id = None

    def video_loop_capture(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            rects = detect_faces(self.detector, frame)
            assigned = self.tracker.update(rects)
            
            for oid, rect in assigned.items():
                if rect is None:
                    continue
                x, y, w, h, conf = rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{oid}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if self.saving_faces:
                    if self.current_user_id is None:
                         self.current_user_id = oid
                         
                    if self.current_user_id == oid:
                        face = frame[y:y+h, x:x+w].copy()
                        if save_face_image(self.current_user_folder, self.saved_count, face):
                            self.saved_count += 1
                        else:
                            self.saving_faces = False
                            self.root.after(0, lambda: messagebox.showinfo("Completo", "Captura completada (300 imágenes)"))
                            self.root.after(0, self.toggle_saving_faces)
            
            if self.saving_faces:
                 cv2.putText(frame, f"GUARDANDO: {self.saved_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label_capture.imgtk = imgtk
            self.video_label_capture.configure(image=imgtk)
        
        self.root.after(10, self.video_loop_capture)

    def train(self):
        selected_user = self.combo_users.get()
        if not selected_user:
            messagebox.showerror("Error", "Selecciona un usuario")
            return
            
        user_data = {
            "username": selected_user,
            "name": self.entries["name"].get(),
            "lastname": self.entries["lastname"].get(),
            "id": self.entries["id"].get(),
            "age": self.entries["age"].get(),
            "desc": self.entries["desc"].get(),
        }
        
        if not user_data["name"]:
            messagebox.showerror("Error", "Nombre es obligatorio")
            return
            
        save_metadata(user_data)
        ok, msg = train_model()
        if ok:
            messagebox.showinfo("Éxito", msg)
        else:
            messagebox.showerror("Error", msg)

    def start_camera_recognition(self):
        if self.is_running:
            return
            
        model_path = "embeddings/modelo_lbph.yml"
        labels_path = "embeddings/labels.json"
        
        if not os.path.exists(model_path) or not os.path.exists(labels_path):
             messagebox.showerror("Error", "Modelo no encontrado. Entrena primero.")
             return

        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        except:
            self.recognizer = cv2.face.createLBPHFaceRecognizer()
            
        self.recognizer.read(model_path)
        
        with open(labels_path, "r") as f:
            raw = json.load(f)
            self.labels_map = {int(k): v for k,v in raw.items()}
            
        proto = "models/deploy.prototxt"
        model = "models/res10_300x300_ssd_iter_140000.caffemodel"
        self.detector = load_face_detector(proto, model)
        
        self.cap = cv2.VideoCapture(0)
        self.is_running = True
        self.btn_start_recog.config(state='disabled')
        self.btn_stop_recog.config(state='normal')
        
        self.video_loop_recog()

    def video_loop_recog(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            rects = detect_faces(self.detector, frame)
            
            for (x, y, w, h, conf) in rects:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[y:y+h, x:x+w]
                
                try:
                    label, confidence = self.recognizer.predict(roi_gray)
                    if confidence < 70:
                        name = self.labels_map.get(label, "Usuario")
                        color = (0, 255, 0)
                    else:
                        name = "Desconocido"
                        color = (0, 0, 255)
                    
                    text = f"{name} ({confidence:.0f})"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except:
                    pass

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label_recog.imgtk = imgtk
            self.video_label_recog.configure(image=imgtk)
            
        self.root.after(10, self.video_loop_recog)

    def stop_camera(self):
        self.is_running = False
        if self.cap:
             self.cap.release()
        self.btn_start_capture.config(state='normal')
        self.btn_stop_capture.config(state='disabled')
        self.btn_capture_faces.config(state='disabled')
        self.btn_start_recog.config(state='normal')
        self.btn_stop_recog.config(state='disabled')
        self.video_label_capture.configure(image='')
        self.video_label_recog.configure(image='')

    def on_close(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
