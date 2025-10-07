from ultralytics import YOLO
import os
import cv2
import numpy as np
import random

# Cargar el modelo entrenado
model_path = os.path.join('runs', 'detect', 'vehicle_detection_model', 'weights', 'best.pt')
model = YOLO(model_path)

# Obtener algunas imágenes de prueba del conjunto de datos
test_images_dir = os.path.join('vehicle-detection.v3i.yolov8', 'test', 'images')
if not os.path.exists(test_images_dir):
    # Si no hay carpeta de prueba, usar algunas imágenes de entrenamiento
    test_images_dir = os.path.join('vehicle-detection.v3i.yolov8', 'train', 'images')

# Obtener lista de imágenes
image_files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Seleccionar algunas imágenes al azar para probar
num_test_images = min(5, len(image_files))
test_images = random.sample(image_files, num_test_images)

# Nombres de las clases
class_names = ['bus', 'car', 'microbus', 'motorbike', 'pickup-van', 'truck']

# Crear directorio para guardar resultados
results_dir = 'detection_results'
os.makedirs(results_dir, exist_ok=True)

# Realizar detección en las imágenes de prueba
for img_file in test_images:
    img_path = os.path.join(test_images_dir, img_file)
    
    # Realizar predicción
    results = model(img_path)
    
    # Procesar resultados
    for result in results:
        # Cargar imagen para visualización
        img = cv2.imread(img_path)
        
        # Obtener detecciones
        boxes = result.boxes.xyxy.cpu().numpy()  # coordenadas (x1, y1, x2, y2)
        confidences = result.boxes.conf.cpu().numpy()  # confianza
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # IDs de clase
        
        # Dibujar detecciones
        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            # Obtener nombre de la clase
            class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
            
            # Color basado en la clase (para distinguir visualmente)
            color = (0, 255, 0)  # Verde por defecto
            
            # Dibujar rectángulo
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Añadir etiqueta
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Guardar imagen con detecciones
        output_path = os.path.join(results_dir, f"detection_{img_file}")
        cv2.imwrite(output_path, img)
        
        print(f"Detecciones guardadas en {output_path}")

print(f"Resultados guardados en la carpeta: {results_dir}") 