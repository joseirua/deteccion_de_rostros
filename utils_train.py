import os
import json
import numpy as np
import cv2

def train_model(dataset_path="dataset", save_dir="embeddings"):
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        recognizer = cv2.face.createLBPHFaceRecognizer()

    print("[INFO] Iniciando entrenamiento con LBPH...")
    
    faces = []
    ids = []
    
    label_map = {}

    metadata_path = os.path.join(save_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except:
            print("[WARN] No se pudo leer metadata.json")

    if not os.path.exists(dataset_path):
        return False, "No existe el directorio dataset."

    user_folders = os.listdir(dataset_path)
    
    for folder_name in user_folders:
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        if not folder_name.startswith("usuario"):
            continue
            
        try:
            user_id = int(folder_name.replace("usuario", ""))
        except ValueError:
            continue
            
        real_name = folder_name
        if folder_name in metadata:
            data = metadata[folder_name]
            if "name" in data and data["name"].strip():
                real_name = data["name"]
                if "lastname" in data and data["lastname"].strip():
                    real_name += " " + data["lastname"]
        
        label_map[user_id] = real_name
        
        image_files = os.listdir(folder_path)
        for img_name in image_files:
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(folder_path, img_name)
            
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
                
            img_numpy = np.array(img, 'uint8')
            
            faces.append(img_numpy)
            ids.append(user_id)

    if len(faces) == 0:
        return False, "No se encontraron rostros para entrenar."

    print(f"[INFO] Entrenando con {len(faces)} rostros de {len(label_map)} usuarios...")
    
    recognizer.train(faces, np.array(ids))
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "modelo_lbph.yml")
    recognizer.write(model_path)
    
    labels_path = os.path.join(save_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(label_map, f)
        
    print(f"[INFO] Modelo guardado en {model_path}")
    
    return True, "Modelo LBPH entrenado correctamente."

def save_metadata(user_data, save_path="embeddings/metadata.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                metadata = json.load(f)
            except:
                metadata = {}
    else:
        metadata = {}

    metadata[user_data["username"]] = user_data

    with open(save_path, "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
