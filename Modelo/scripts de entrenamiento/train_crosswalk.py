from ultralytics import YOLO
import torch

# Verificar si hay GPU disponible
print(f"GPU disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")

# Cargar el modelo base YOLOv8
model = YOLO('yolov8n.pt')

# Configurar los parámetros de entrenamiento
config = {
    'data': 'Crosswalk.v6i.yolov8/data.yaml',  # Ruta al archivo de configuración del dataset
    'epochs': 15,  # Número de épocas
    'patience': 20,  # Early stopping patience
    'batch': 16,  # Tamaño del batch
    'imgsz': 640,  # Tamaño de las imágenes
    'device': 0 if torch.cuda.is_available() else 'cpu',  # Usar GPU si está disponible
    'workers': 8,  # Número de workers para carga de datos
    'name': 'crosswalk_detector_v1'  # Nombre del experimento
}

# Entrenar el modelo
results = model.train(**config)

# Guardar el modelo entrenado
model.export(format='onnx')  # Exportar a ONNX
print("Entrenamiento completado y modelo exportado.") 