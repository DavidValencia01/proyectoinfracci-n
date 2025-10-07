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
    output_path = Path("runs/detect/test_results_crosswalk")
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"result_{image_path.name}"
    cv2.imwrite(str(output_file), annotated_frame)
    print(f"Resultado guardado en: {output_file}")

def main():
    # Cargar el modelo entrenado
    model_path = "runs/detect/crosswalk_detector_v1/weights/best.pt"
    if not Path(model_path).exists():
        print(f"No se encontró el modelo en {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Directorio de imágenes de prueba
    test_dir = Path("Crosswalk.v6i.yolov8/valid/images")
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
    confidence_scores = []
    
    for img_path in image_files:
        print(f"\nProcesando {img_path.name}...")
        
        # Realizar predicción
        img = cv2.imread(str(img_path))
        results = model.predict(img, conf=0.25)[0]
        
        # Dibujar detecciones
        annotated_frame = results.plot()
        
        # Actualizar estadísticas
        for box in results.boxes:
            conf = box.conf[0].item()
            total_detections += 1
            confidence_scores.append(conf)
            print(f"Detectado cruce peatonal con confianza {conf:.2f}")
        
        # Guardar imagen con detecciones
        output_path = Path("runs/detect/test_results_crosswalk")
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"result_{img_path.name}"
        cv2.imwrite(str(output_file), annotated_frame)
    
    # Imprimir estadísticas
    print("\n=== Estadísticas de Detección ===")
    print(f"Total de detecciones: {total_detections}")
    if confidence_scores:
        avg_conf = np.mean(confidence_scores)
        min_conf = np.min(confidence_scores)
        max_conf = np.max(confidence_scores)
        print(f"Confianza promedio: {avg_conf:.2f}")
        print(f"Confianza mín/máx: {min_conf:.2f}/{max_conf:.2f}")

if __name__ == "__main__":
    main() 