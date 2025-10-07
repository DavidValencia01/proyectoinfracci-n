from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import time
import sys

def procesar_video(model, video_path, output_path=None, conf_threshold=0.30):
    """
    Procesa un video para detectar cruces peatonales y guarda el resultado
    """
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
        output_path = f"videos_salida/cruces_peatonales/{video_name}_procesado.mp4"
    
    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Variables para estadísticas
    start_time = time.time()
    frames_processed = 0
    total_detections = 0
    
    print(f"Procesando video: {video_path}")
    print(f"Resolución: {width}x{height}, FPS: {fps:.2f}")
    
    # Procesar cada frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Realizar predicción con YOLOv8
        results = model.predict(frame, conf=conf_threshold)[0]
        
        # Crear una copia del frame para dibujar las detecciones personalizadas
        annotated_frame = frame.copy()
        
        # Color morado para las detecciones (BGR format)
        purple_color = (128, 0, 128)  # BGR: (128, 0, 128) = Morado
        
        # Dibujar las detecciones personalizadas
        if results.boxes is not None:
            for box in results.boxes:
                # Obtener coordenadas del bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Obtener confianza
                conf = float(box.conf[0])
                
                # Dibujar el bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), purple_color, 2)
                
                # Crear el texto de la etiqueta
                label = f"cruce peatonal {conf:.2f}"
                
                # Calcular el tamaño del texto
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Dibujar el fondo del texto
                cv2.rectangle(annotated_frame, 
                            (x1, y1 - text_height - baseline - 5), 
                            (x1 + text_width, y1), 
                            purple_color, -1)
                
                # Dibujar el texto
                cv2.putText(annotated_frame, label, 
                           (x1, y1 - baseline - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Contar detecciones
        num_detections = len(results.boxes)
        total_detections += num_detections
        
        # Escribir información en el frame
        cv2.putText(annotated_frame, f"Cruces Peatonales: {num_detections}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
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
    print(f"Total de cruces peatonales detectados: {total_detections}")
    print(f"Promedio de detecciones por frame: {total_detections / frames_processed:.2f}")
    
    return output_path

def procesar_frame(model, frame, conf_threshold=0.30):
    annotated_frame = frame.copy()
    results = model.predict(frame, conf=conf_threshold)[0]
    boxes_cruce = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            boxes_cruce.append([x1, y1, x2, y2])
            purple_color = (128, 0, 128)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), purple_color, 2)
            label = f"cruce peatonal {float(box.conf[0]):.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - baseline - 5), (x1 + text_width, y1), purple_color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated_frame, boxes_cruce

def main():
    # Si se pasan argumentos, procesar solo ese video
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        # ... cargar modelo y procesar solo ese video ...
        model_path = "runs/detect/crosswalk_detector_v1/weights/best.pt"
        if not Path(model_path).exists():
            print(f"Error: No se encontró el modelo en {model_path}")
            return
        model = YOLO(model_path)
        procesar_video(model, input_path, output_path)
        print(f"\nVideo procesado correctamente: {input_path}")
        print("-" * 50)
        return
    # ... resto del main original ...
    # Cargar el modelo entrenado
    model_path = "runs/detect/crosswalk_detector_v1/weights/best.pt"
    if not Path(model_path).exists():
        print(f"Error: No se encontró el modelo en {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Verificar la carpeta de videos de entrada
    videos_dir = Path("videos_entrada")
    if not videos_dir.exists():
        print(f"Error: No se encontró la carpeta de videos de entrada: {videos_dir}")
        return
    
    # Verificar y crear la carpeta de salida si no existe
    output_dir = Path("videos_salida/cruces_peatonales")
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
            procesar_video(model, video_path, str(output_path))
            print(f"\nVideo procesado correctamente: {video_name}")
            print("-" * 50)
        except Exception as e:
            print(f"\nError al procesar el video {video_name}: {str(e)}")

if __name__ == "__main__":
    main() 