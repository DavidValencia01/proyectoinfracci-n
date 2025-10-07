from ultralytics import YOLO
import os
from pathlib import Path
import shutil

def main():
    # Configurar rutas
    DATASET_PATH = 'traffic-light detection.v9i.yolov8'
    yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
    
    # Verificar que el archivo data.yaml existe
    if not os.path.exists(yaml_path):
        print(f"Error: No se encontró el archivo {yaml_path}")
        return
    
    # Inicializar el modelo YOLOv8n
    model = YOLO('yolov8n.pt')
    
    # Entrenar el modelo
    results = model.train(
        data=yaml_path,
        epochs=10,          # Número de épocas
        imgsz=640,         # Tamaño de imagen
        batch=16,          # Tamaño de batch
        name='traffic_lights_detector_v2',  # Nombre del experimento
        verbose=True,      # Mostrar información detallada
        patience=5,        # Early stopping si no hay mejora en 5 épocas
        device='cpu',      # Usar CPU (cambiar a 0 si tienes GPU disponible)
        workers=4,         # Número de workers para carga de datos
        cache=True,        # Cachear imágenes en RAM
        pretrained=True,   # Usar pesos pre-entrenados
        optimizer='Adam',  # Optimizador
        lr0=0.001,        # Learning rate inicial
        weight_decay=0.0005,  # Weight decay para regularización
        warmup_epochs=1,   # Épocas de calentamiento
        cos_lr=True,      # Learning rate con decaimiento coseno
        seed=42           # Semilla para reproducibilidad
    )

    # Evaluar el modelo
    results = model.val()

    # Mover los resultados al directorio principal
    src_dir = 'runs/detect/traffic_lights_detector_v2'
    dst_dir = '../runs/detect/traffic_lights_detector_v2'
    if os.path.exists(src_dir):
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        shutil.move(src_dir, dst_dir)

    print("Entrenamiento y validación completados.")

if __name__ == "__main__":
    main()