from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import time
import collections
import sys

# --- MODO DE DIAGNÓSTICO ---
# Poner en True para que el script imprima los valores de HUE en la consola.
DEBUG_MODE = False
DEBUG_WINDOW_SHOWN = False

class SemaphoreColorDetector:
    """
    Clase para detectar el color de un semáforo (rojo, amarillo, verde)
    dentro de una región de interés (ROI) utilizando el espacio de color HSV.
    """
    def __init__(self, min_pixel_threshold=50):
        # Rangos de color en HSV. Se restauran a valores estándar y robustos.
        self.color_ranges = {
            "Rojo": [
                # Rango de Hue ligeramente más estricto para no invadir el amarillo.
                (np.array([0, 100, 100]), np.array([8, 255, 255])),
                (np.array([172, 100, 100]), np.array([180, 255, 255]))
            ],
            "Amarillo": [
                # Rango de Hue ampliado para capturar tonos más anaranjados.
                (np.array([9, 100, 100]), np.array([35, 255, 255]))
            ],
            "Verde": [
                (np.array([40, 70, 70]), np.array([95, 255, 255]))
            ]
        }
        self.min_pixel_threshold = min_pixel_threshold

    def get_dominant_color(self, hsv_roi, original_roi):
        """
        Determina el color dominante con un método robusto de filtrado en dos etapas.
        """
        # --- Etapa 1: Máscara de Color Amplia ---
        # Crear una máscara que incluya CUALQUIER píxel que pueda ser rojo, amarillo o verde.
        # Esto elimina eficazmente el fondo (cielo, postes, etc.).
        masks = []
        for color_name, ranges in self.color_ranges.items():
            for lower, upper in ranges:
                masks.append(cv2.inRange(hsv_roi, lower, upper))
        
        # Combinar todas las máscaras de color en una sola
        any_color_mask = masks[0]
        for i in range(1, len(masks)):
            any_color_mask = cv2.bitwise_or(any_color_mask, masks[i])

        # --- Etapa 2: Máscara de Brillo Enfocada ---
        # Ahora, buscar los píxeles más brillantes DENTRO de las áreas de color.
        gray_roi = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)
        focused_gray = cv2.bitwise_and(gray_roi, gray_roi, mask=any_color_mask)

        # El método de Otsu calculará el umbral de brillo óptimo para la región de color.
        # Esto es más robusto que un umbral fijo para luces de diferente intensidad.
        _, light_mask = cv2.threshold(focused_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = np.ones((3,3), np.uint8)
        light_mask = cv2.morphologyEx(light_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        if cv2.countNonZero(light_mask) < self.min_pixel_threshold:
            return "Desconocido"

        # --- Etapa 3: Análisis Final sobre la Máscara Limpia ---
        final_analysis_roi = cv2.bitwise_and(hsv_roi, hsv_roi, mask=light_mask)

        pixel_counts = collections.Counter()
        for color_name, ranges in self.color_ranges.items():
            mask_total = cv2.inRange(final_analysis_roi, ranges[0][0], ranges[0][1])
            if len(ranges) > 1:
                for i in range(1, len(ranges)):
                    mask = cv2.inRange(final_analysis_roi, ranges[i][0], ranges[i][1])
                    mask_total = cv2.bitwise_or(mask_total, mask)
            pixel_counts[color_name] = cv2.countNonZero(mask_total)

        if not pixel_counts or max(pixel_counts.values()) == 0:
            return "Desconocido"
            
        return pixel_counts.most_common(1)[0][0]

def procesar_video(model, video_path, output_path=None, conf_threshold=0.4, iou_threshold=0.5):
    """
    Procesa un video para detectar semáforos y guarda el resultado
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
        output_path = f"videos_salida/semaforos/{video_name}_procesado.mp4"
    
    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Instanciar el detector de color con un umbral de píxeles más bajo
    color_detector = SemaphoreColorDetector(min_pixel_threshold=50)

    # Variables para estadísticas
    start_time = time.time()
    frames_processed = 0
    total_detections = 0
    detections_by_class = {'Verde': 0, 'Rojo': 0, 'Amarillo': 0, 'Desconocido': 0}
    
    print(f"Procesando video: {video_path}")
    print(f"Resolución: {width}x{height}, FPS: {fps:.2f}")
    
    # Procesar cada frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frames_processed += 1
            
        # Realizar predicción con YOLOv8
        results = model.predict(frame, conf=conf_threshold, iou=iou_threshold)[0]
        
        # Dibujar las detecciones en el frame
        annotated_frame = frame.copy()
        
        # Contar detecciones por clase
        frame_detections = {'Verde': 0, 'Rojo': 0, 'Amarillo': 0, 'Desconocido': 0}

        # Colores para cada clase (BGR)
        colors = {
            'Verde': (0, 255, 0),
            'Rojo': (0, 0, 255),
            'Amarillo': (0, 255, 255),
            'Desconocido': (255, 255, 255) # Blanco para desconocido
        }
        
        # Mapeo de nombres de clases de inglés a español (para fallback)
        class_name_map = {
            'Green': 'Verde',
            'Red': 'Rojo',
            'Yellow': 'Amarillo'
        }

        for box in results.boxes:
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- LÓGICA DE POST-PROCESAMIENTO CON CLASE DEDICADA ---
            
            # 1. Recortar la región de interés (ROI)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            # 2. Convertir ROI a espacio de color HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # 3. Obtener el color dominante usando el detector
            class_name = color_detector.get_dominant_color(hsv_roi, roi)
            
            # --- FIN DE LA LÓGICA DE POST-PROCESAMIENTO ---

            # Obtener color para el cuadro delimitador
            color = colors.get(class_name, (255, 255, 255))

            # Dibujar el rectángulo
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Preparar y dibujar la etiqueta
            label = f"{class_name} {conf:.2f}"
            font_scale = 0.5
            font_thickness = 1
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 4), (x1 + text_size[0], y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            
            total_detections += 1
            if class_name in detections_by_class:
                detections_by_class[class_name] += 1
                frame_detections[class_name] += 1
        
        # Escribir información en el frame
        y_pos = 30
        cv2.putText(annotated_frame, f"Semaforos en Verde: {frame_detections['Verde']}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
        cv2.putText(annotated_frame, f"Semaforos en Rojo: {frame_detections['Rojo']}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_pos += 30
        cv2.putText(annotated_frame, f"Semaforos en Amarillo: {frame_detections['Amarillo']}", 
                    (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Guardar el frame en el video de salida
        out.write(annotated_frame)
        
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
    print(f"Total de semáforos detectados: {total_detections}")
    print(f"Detecciones por clase:")
    for clase, detecciones in detections_by_class.items():
        print(f"  - {clase}: {detecciones} ({detecciones / total_detections * 100:.1f}% del total)")
    
    return output_path

def main():
    # Si se pasan argumentos, procesar solo ese video
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        # ... cargar modelo y procesar solo ese video ...
        model_path = "runs/detect/traffic_lights_detector_v2/weights/best.pt"
        if not Path(model_path).exists():
            print(f"Error: No se encontró el modelo en {model_path}")
            return
        model = YOLO(model_path)
        procesar_video(model, input_path, output_path)
        print(f"\nVideo procesado correctamente: {input_path}")
        print("-" * 50)
        return
    # ... resto del main original ...

if __name__ == "__main__":
    main() 