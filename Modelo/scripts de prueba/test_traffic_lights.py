from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def test_on_image(model, image_path):
    """
    Prueba el modelo en una imagen y muestra los resultados
    """
    # Leer la imagen
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"No se pudo leer la imagen: {image_path}")
        return
    
    # Realizar la predicción
    results = model.predict(img, conf=0.25)  # confidence threshold 0.25
    
    # Obtener la primera predicción
    result = results[0]
    
    # Dibujar las detecciones
    annotated_frame = result.plot()
    
    # Mostrar información de las detecciones
    for box in result.boxes:
        class_id = box.cls[0].item()
        conf = box.conf[0].item()
        class_name = model.names[int(class_id)]
        print(f"Detectado {class_name} con confianza {conf:.2f}")
    
    # Guardar la imagen con las detecciones
    output_path = Path("runs/detect/test_results_v2")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"result_{image_path.name}"
    cv2.imwrite(str(output_file), annotated_frame)
    print(f"Resultado guardado en: {output_file}")

def main():
    # Cargar el modelo entrenado
    model_path = "runs/detect/traffic_lights_detector_v2/weights/best.pt"  # Ruta actualizada
    if not Path(model_path).exists():
        print(f"No se encontró el modelo en {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Directorio de imágenes de prueba
    test_dir = Path("traffic-light detection.v9i.yolov8/test/images")
    if not test_dir.exists():
        print(f"No se encontró el directorio de pruebas: {test_dir}")
        return
    
    # Procesar todas las imágenes de prueba
    image_files = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not image_files:
        print("No se encontraron imágenes de prueba")
        return
    
    print(f"Procesando {len(image_files)} imágenes...")
    
    # Variables para estadísticas
    total_detections = 0
    detections_by_class = {'Green': 0, 'Red': 0, 'Yellow': 0}
    confidence_by_class = {'Green': [], 'Red': [], 'Yellow': []}
    
    for img_path in image_files:
        print(f"\nProcesando {img_path.name}...")
        
        # Realizar predicción
        img = cv2.imread(str(img_path))
        results = model.predict(img, conf=0.25)[0]
        
        # Dibujar detecciones
        annotated_frame = results.plot()
        
        # Actualizar estadísticas
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = model.names[class_id]
            
            total_detections += 1
            detections_by_class[class_name] += 1
            confidence_by_class[class_name].append(conf)
            
            print(f"Detectado {class_name} con confianza {conf:.2f}")
        
        # Guardar imagen con detecciones
        output_path = Path("runs/detect/test_results_v2")
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"result_{img_path.name}"
        cv2.imwrite(str(output_file), annotated_frame)
    
    # Imprimir estadísticas
    print("\n=== Estadísticas de Detección ===")
    print(f"Total de detecciones: {total_detections}")
    for class_name in detections_by_class:
        detections = detections_by_class[class_name]
        confidences = confidence_by_class[class_name]
        avg_conf = np.mean(confidences) if confidences else 0
        print(f"\n{class_name}:")
        print(f"  - Número de detecciones: {detections}")
        print(f"  - Confianza promedio: {avg_conf:.2f}")
        if confidences:
            print(f"  - Confianza mín/máx: {min(confidences):.2f}/{max(confidences):.2f}")

if __name__ == "__main__":
    main() 