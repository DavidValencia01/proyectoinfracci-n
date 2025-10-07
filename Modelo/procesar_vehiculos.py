from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import time
import sys

# Mapeo de clases originales a nuevas etiquetas en español y colores
CLASE_MAPEO = {
    'car': ('Carro', (0, 0, 255)),      # Rojo
    'bus': ('Autobus', (0, 255, 0)),    # Verde
    'microbus': ('Combi', (255, 0, 0)), # Azul
    'motorbike': ('Motocicleta', (0, 255, 255)), # Amarillo
    'truck': ('Camion', (0, 165, 255)), # Naranja
    'pickup-van': ('Camioneta', (0, 0, 0))  # Negro
}

def obtener_etiqueta_y_color(class_name):
    """
    Obtiene la etiqueta en español y el color para una clase dada
    """
    # Si la clase ya está en el mapeo, devolver directamente
    if class_name in CLASE_MAPEO:
        return CLASE_MAPEO[class_name]
    
    # Si es una etiqueta en español, buscar la clase original
    for original_class, (etiqueta_espanol, color) in CLASE_MAPEO.items():
        if etiqueta_espanol == class_name:
            return (etiqueta_espanol, color)
    
    # Si no se encuentra, devolver la clase original con color blanco
    return (class_name, (255, 255, 255))

def procesar_video(model, video_path, output_path=None, conf_thresholds=None):
    """
    Procesa un video para detectar y seguir vehículos, y guarda el resultado.
    """
    if conf_thresholds is None:
        conf_thresholds = {}

    # Abrir el video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar el nombre del archivo de salida si no se proporciona
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"videos_salida/vehiculos/{video_name}_procesado.mp4"
    
    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Variables para estadísticas
    start_time = time.time()
    frames_processed = 0
    total_detections = 0
    detections_by_class = {}
    
    print(f"Procesando video con seguimiento de objetos: {video_path}")
    print(f"Resolución: {width}x{height}, FPS: {fps:.2f}")
    
    # Inicializar el diccionario de detecciones por clase
    for class_id in model.names:
        class_name = model.names[class_id]
        etiqueta_espanol, _ = obtener_etiqueta_y_color(class_name)
        detections_by_class[etiqueta_espanol] = 0
    
    # Procesar cada frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Realizar seguimiento con YOLOv8. `persist=True` mantiene los tracks entre frames.
        # Se usa un umbral bajo para que el tracker vea todo y luego filtramos nosotros.
        results = model.track(frame, persist=True, conf=0.1)[0]
        
        # Crear una copia del frame para dibujar las detecciones personalizadas
        annotated_frame = frame.copy()
        
        # Contar detecciones por clase para este frame
        frame_detections = {}
        for class_name in model.names.values():
            etiqueta_espanol, _ = obtener_etiqueta_y_color(class_name)
            frame_detections[etiqueta_espanol] = 0
        
        # Extraer los IDs de seguimiento si existen
        track_ids = results.boxes.id
        
        # Dibujar las detecciones con etiquetas personalizadas
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].item())
            
            # Obtener el umbral específico para esta clase
            class_threshold = conf_thresholds.get(class_name, 0.25)
            
            # Omitir detecciones por debajo del umbral de su clase
            if confidence < class_threshold:
                continue

            # Obtener el ID de seguimiento
            track_id = int(track_ids[i].item()) if track_ids is not None else -1
            
            # Obtener coordenadas del bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Obtener etiqueta en español y color
            etiqueta_espanol, color = obtener_etiqueta_y_color(class_name)
            
            # Dibujar el bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Crear texto con ID, etiqueta y confianza
            texto = f"{etiqueta_espanol} {confidence:.2f}"
            if track_id != -1:
                texto = f"ID {track_id}: {texto}"
            
            # Calcular tamaño del texto
            (text_width, text_height), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Dibujar fondo para el texto
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # Dibujar el texto
            cv2.putText(annotated_frame, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Actualizar contadores
            total_detections += 1
            detections_by_class[etiqueta_espanol] += 1
            frame_detections[etiqueta_espanol] += 1
        
        # Escribir información en el frame
        y_pos = 30
        cv2.putText(annotated_frame, f"Total Vehículos: {sum(frame_detections.values())}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_pos += 30
        for etiqueta_espanol, count in frame_detections.items():
            if count > 0:
                _, color = obtener_etiqueta_y_color(etiqueta_espanol)
                cv2.putText(annotated_frame, f"{etiqueta_espanol}: {count}", 
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                y_pos += 25
        
        # Guardar el frame en el video de salida
        out.write(annotated_frame)
        
        frames_processed += 1
        
        # Mostrar progreso cada 30 frames
        if frames_processed % 30 == 0:
            progress = (frames_processed / total_frames) * 100
            elapsed_time = time.time() - start_time
            fps_processing = frames_processed / elapsed_time if elapsed_time > 0 else 0
            print(f"Progreso: {progress:.1f}% - {frames_processed}/{total_frames} frames - {fps_processing:.1f} FPS")
    
    # Liberar recursos
    cap.release()
    out.release()
    
    # Mostrar estadísticas finales
    elapsed_time = time.time() - start_time
    print("\n=== Estadísticas de Procesamiento ===")
    print(f"Video procesado: {video_path}")
    print(f"Resultado guardado en: {output_path}")
    print(f"Frames procesados: {frames_processed}")
    print(f"Tiempo total: {elapsed_time:.2f} segundos")
    print(f"Velocidad de procesamiento: {frames_processed / elapsed_time:.2f} FPS")
    print(f"Total de vehículos detectados: {total_detections}")
    
    if total_detections > 0:
        print(f"Detecciones por clase:")
        for clase, detecciones in sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True):
            if detecciones > 0:
                print(f"  - {clase}: {detecciones} ({detecciones / total_detections * 100:.1f}% del total)")
    
    return output_path

def procesar_frame(model, frame, conf_thresholds=None):
    if conf_thresholds is None:
        conf_thresholds = {}
    annotated_frame = frame.copy()
    results = model.track(frame, persist=True, conf=0.1)[0]
    boxes_vehiculo = []
    if results.boxes is not None:
        track_ids = results.boxes.id
        for i, box in enumerate(results.boxes):
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            confidence = float(box.conf[0].item())
            class_threshold = conf_thresholds.get(class_name, 0.25)
            if confidence < class_threshold:
                continue
            track_id = int(track_ids[i].item()) if track_ids is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            boxes_vehiculo.append([x1, y1, x2, y2])
            etiqueta_espanol, color = obtener_etiqueta_y_color(class_name)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            texto = f"{etiqueta_espanol} {confidence:.2f}"
            if track_id != -1:
                texto = f"ID {track_id}: {texto}"
            (text_width, text_height), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_frame, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated_frame, boxes_vehiculo

def main():
    # Si se pasan argumentos, procesar solo ese video
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        model_path = "runs/detect/vehicle_detection_model_binary2/weights/best.pt"
        if not Path(model_path).exists():
            print(f"Error: No se encontró el modelo en {model_path}")
            return
        model = YOLO(model_path)
        CONF_THRESHOLDS = {
            'car': 0.4,
            'bus': 0.4,
            'microbus': 0.9,
            'motorbike': 0.3,
            'truck': 1,
            'pickup-van': 0.7
        }
        procesar_video(model, input_path, output_path, conf_thresholds=CONF_THRESHOLDS)
        print(f"\nVideo procesado correctamente: {input_path}")
        print("-" * 50)
        return
    # Cargar el modelo entrenado (usando el modelo específico solicitado)
    model_path = "runs/detect/vehicle_detection_model_binary2/weights/best.pt"
    if not Path(model_path).exists():
        print(f"Error: No se encontró el modelo en {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Definir umbrales de confianza por clase para un control más fino
    CONF_THRESHOLDS = {
        'car': 0.4,         # Umbral más bajo para no perder carros
        'bus': 0.4,         # Umbral alto para evitar falsos positivos
        'microbus': 0.9,   # Umbral intermedio para carros grandes/confusiones
        'motorbike': 0.3,
        'truck': 1,
        'pickup-van': 0.7
    }
    
    # Verificar la carpeta de videos de entrada
    videos_dir = Path("videos_entrada")
    if not videos_dir.exists():
        print(f"Error: No se encontró la carpeta de videos de entrada: {videos_dir}")
        return
    
    # Verificar y crear la carpeta de salida si no existe
    output_dir = Path("videos_salida/vehiculos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Buscar archivos de video
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(videos_dir.glob(f"*{ext}")))
    
    if not video_files:
        print("No se encontraron archivos de video en la carpeta de entrada")
        return
    
    print(f"Se encontraron {len(video_files)} videos para procesar")
    
    # Procesar cada video
    for video_path in video_files:
        video_name = video_path.stem
        output_path = output_dir / f"{video_name}_procesado.mp4"
        
        try:
            # Usamos los umbrales de confianza específicos por clase
            procesar_video(model, video_path, str(output_path), conf_thresholds=CONF_THRESHOLDS)
            print(f"\nVideo procesado correctamente: {video_name}")
            print("-" * 50)
        except Exception as e:
            print(f"\nError al procesar el video {video_name}: {str(e)}")

if __name__ == "__main__":
    main() 