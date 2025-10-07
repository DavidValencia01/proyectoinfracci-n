from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path

def preprocess_dataset(dataset_path, output_path):
    """
    Preprocesa las imágenes del dataset aplicando binarización
    """
    # Crear directorio de salida si no existe
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Procesar imágenes
    img_paths = list(Path(dataset_path).rglob('*.jpg')) + list(Path(dataset_path).rglob('*.png'))
    total_images = len(img_paths)
    print(f"Procesando {total_images} imágenes...")
    
    for i, img_path in enumerate(img_paths, 1):
        if i % 100 == 0:
            print(f"Progreso: {i}/{total_images} imágenes ({(i/total_images)*100:.1f}%)")
            
        # Leer imagen
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"No se pudo leer la imagen: {img_path}")
            continue
            
        # Convertir a escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Aplicar binarización adaptativa
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Tamaño del vecindario
            2    # Constante de sustracción
        )
        
        # Convertir de nuevo a BGR para mantener compatibilidad
        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        # Guardar imagen procesada
        rel_path = img_path.relative_to(dataset_path)
        output_file = output_path / rel_path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_file), binary_bgr)

# Rutas
yaml_path = os.path.join('vehicle-detection.v3i.yolov8', 'data.yaml')
dataset_path = 'vehicle-detection.v3i.yolov8/train'  # Ajusta esto a tu ruta de entrenamiento
processed_path = 'vehicle-detection.v3i.yolov8/train_processed'

# Preprocesar dataset
print("Preprocesando imágenes...")
preprocess_dataset(dataset_path, processed_path)

# Inicializar el modelo YOLOv8n
model = YOLO('yolov8n.pt')

# Entrenar el modelo
results = model.train(
    data=yaml_path,
    epochs=5,
    imgsz=640,
    batch=8,             # Reducido para CPU
    name='vehicle_detection_model_binary',
    verbose=True,
    patience=5,
    device='cpu',        # Usar CPU
    workers=4,           # Reducido para CPU
    cache=True,          # Cachear imágenes en RAM
)

# Evaluar el modelo
results = model.val()

print("Entrenamiento y validación completados.") 