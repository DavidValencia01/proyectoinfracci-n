from ultralytics import YOLO
import torch
import os
from pathlib import Path
import shutil

def main():
    # Configurar rutas
    DATASET_PATH = 'PeruPlateNumbers.v3i.yolov8'  
    yaml_path = os.path.join(DATASET_PATH, 'data.yaml')

    # Verificar que el archivo data.yaml existe
    if not os.path.exists(yaml_path):
        print(f"Error: No se encontró el archivo {yaml_path}")
        return

    # Verificar si hay GPU disponible
    gpu_available = torch.cuda.is_available()
    print(f"GPU disponible: {gpu_available}")
    if gpu_available:
        print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        device = 'cpu'

    # Inicializar el modelo YOLOv8n
    model = YOLO('yolov8n.pt')

    # Entrenar el modelo
    results = model.train(
        data=yaml_path,
        epochs=15,              # Número de épocas
        imgsz=640,              # Tamaño de imagen
        batch=8,                # Tamaño de batch
        name='license_plate_detector_v1',  # Nombre del experimento
        exist_ok=True,          # sobrescribe la carpeta si ya existe
        verbose=True,           # Mostrar información detallada
        patience=7,             # Early stopping si no hay mejora en 7 épocas
        device=device,          # Usar GPU si está disponible
        workers=4,              # Número de workers para carga de datos
        cache=True,             # Cachear imágenes en RAM
        pretrained=True,        # Usar pesos pre-entrenados
        optimizer='Adam',       # Optimizador
        lr0=0.001,              # Learning rate inicial
        weight_decay=0.0005,    # Weight decay para regularización
        warmup_epochs=1,        # Épocas de calentamiento
        cos_lr=True,            # Learning rate con decaimiento coseno
        seed=42                 # Semilla para reproducibilidad
    )

    # Evaluar el modelo
    results = model.val()

    # Exportar el modelo entrenado
    model.export(format='onnx')

    # Mover los resultados a la raíz del proyecto si existen
    # src_dir = 'runs/detect/license_plate_detector_v1'
    # dst_dir = '../runs/detect/license_plate_detector_v1'
    # if os.path.exists(src_dir):
    #     if os.path.exists(dst_dir):
    #         shutil.rmtree(dst_dir)
    #     shutil.move(src_dir, dst_dir)

    print("Entrenamiento, validación y exportación completados.")

if __name__ == "__main__":
    main() 