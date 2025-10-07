from ultralytics import YOLO
import cv2
import os
import numpy as np
from pathlib import Path
import time
import re
import sys

ZONAS_REGISTRALES = set("ABCDFHLMPSTUVWXYZ")

def validar_formato_placa_peruana(matricula):
    """
    Valida si la matrícula sigue el formato de placas peruanas
    """
    if not matricula or len(matricula) > 6:
        return False, "Longitud inválida"
    
    # Patrones para placas peruanas
    patrones = [
        # Vehículos menores: AB-1234 o 1234-AB
        r'^[A-Z]{2}-\d{4}$',
        r'^\d{4}-[A-Z]{2}$',
        
        # Vehículos livianos/pesados: A1B-234, ABC-123, A12-345
        r'^[A-Z][A-Z0-9]{2}-\d{3}$',
        
        # Placas especiales: EPA-123, EUA-456
        r'^E[A-Z]{2}-\d{3}$',
        
        # Formato sin guión (para casos donde no se detecta el guión)
        r'^[A-Z]{2}\d{4}$',
        r'^\d{4}[A-Z]{2}$',
        r'^[A-Z][A-Z0-9]{2}\d{3}$',
        r'^E[A-Z]{2}\d{3}$'
    ]
    
    for patron in patrones:
        if re.match(patron, matricula):
            return True, "Formato válido"
    
    return False, "Formato no reconocido"

def filtrar_caracteres_validos(caracteres_ordenados, conf_threshold=0.5):
    """
    Filtra caracteres con baja confianza y aplica reglas de placas peruanas
    """
    caracteres_filtrados = []
    
    for det in caracteres_ordenados:
        char = det['char']
        conf = det['conf']
        
        # Solo incluir caracteres con confianza suficiente
        if conf >= conf_threshold:
            # Validar que el carácter sea válido para placas peruanas
            if char.isalnum() and char in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                caracteres_filtrados.append(det)
    
    return caracteres_filtrados

def ordenar_caracteres_izquierda_derecha(detections):
    """
    Ordena las detecciones de caracteres de izquierda a derecha basándose en la coordenada x
    """
    if not detections:
        return []
    
    # Extraer coordenadas x del centro de cada bounding box
    caracteres_ordenados = []
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        conf = detection['conf']
        char = detection['char']
        center_x = (x1 + x2) / 2
        caracteres_ordenados.append({
            'char': char,
            'conf': conf,
            'center_x': center_x,
            'bbox': (x1, y1, x2, y2)
        })
    
    # Ordenar por coordenada x (izquierda a derecha)
    caracteres_ordenados.sort(key=lambda x: x['center_x'])
    
    return caracteres_ordenados

def ajustar_matricula_por_zona(caracteres_ordenados):
    """
    Reordena los caracteres para que la matrícula comience con la letra de zona registral si está presente.
    """
    for idx, det in enumerate(caracteres_ordenados):
        if det['char'] in ZONAS_REGISTRALES:
            # Poner la letra de zona al inicio y tomar los siguientes 5 caracteres
            nueva_orden = [det] + caracteres_ordenados[:idx] + caracteres_ordenados[idx+1:]
            return nueva_orden[:6]
    return caracteres_ordenados[:6]

def reconocer_caracteres_placa(model_caracteres, placa_img, conf_threshold=0.25):
    """
    Reconoce los caracteres en una imagen de placa con validación peruana y priorización de letra de zona.
    """
    results = model_caracteres.predict(placa_img, conf=conf_threshold)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = float(box.conf[0])
        class_id = int(box.cls[0])
        char = model_caracteres.names[class_id]
        detections.append({'char': char, 'conf': conf, 'bbox': (x1, y1, x2, y2)})
    caracteres_ordenados = ordenar_caracteres_izquierda_derecha(detections)
    caracteres_filtrados = filtrar_caracteres_validos(caracteres_ordenados, conf_threshold=0.5)
    # --- Priorizar letra de zona ---
    caracteres_final = ajustar_matricula_por_zona(caracteres_filtrados)
    matricula = ''.join([det['char'] for det in caracteres_final])
    es_valida, mensaje = validar_formato_placa_peruana(matricula)
    return matricula, caracteres_final, es_valida, mensaje

def procesar_video(model_placas, model_caracteres, video_path, output_path=None, 
                   conf_placas=0.40, conf_caracteres=0.25):
    """
    Procesa un video para detectar placas y reconocer sus caracteres
    """
    # Abrir el video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_path}")
        return

    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cv2.CAP_PROP_FPS
    fps = cap.get(fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Configurar el nombre del archivo de salida si no se proporciona
    if output_path is None:
        video_name = Path(video_path).stem
        output_path = f"videos_salida/caracteres_placas/{video_name}_procesado.mp4"
    
    # Configurar el escritor de video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Variables para estadísticas
    start_time = time.time()
    frames_processed = 0
    total_placas_detectadas = 0
    total_matriculas_reconocidas = 0
    total_matriculas_validas = 0
    
    print(f"Procesando video: {video_path}")
    print(f"Resolución: {width}x{height}, FPS: {fps:.2f}")
    
    # Procesar cada frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Realizar predicción de placas
        results_placas = model_placas.predict(frame, conf=conf_placas)[0]
        
        # Crear una copia del frame para dibujar las detecciones
        annotated_frame = frame.copy()
        
        # Colores para las detecciones (BGR format)
        color_placa = (203, 192, 255)  # Rosado para placas
        color_matricula_valida = (0, 255, 0)   # Verde para matrículas válidas
        color_matricula_invalida = (0, 0, 255) # Rojo para matrículas inválidas
        
        placas_en_frame = 0
        matriculas_en_frame = 0
        
        # Procesar cada placa detectada
        if results_placas.boxes is not None:
            for box in results_placas.boxes:
                # Obtener coordenadas del bounding box de la placa
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Obtener confianza de la placa
                conf_placa = float(box.conf[0])
                
                # Recortar la región de la placa
                placa_roi = frame[y1:y2, x1:x2]
                
                if placa_roi.size > 0:  # Verificar que el recorte no esté vacío
                    # Reconocer caracteres en la placa
                    matricula, caracteres, es_valida, mensaje = reconocer_caracteres_placa(
                        model_caracteres, placa_roi, conf_caracteres
                    )
                    
                    # Dibujar el bounding box de la placa
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_placa, 2)
                    
                    # Crear etiqueta de la placa
                    label_placa = f"Placa {conf_placa:.2f}"
                    
                    # Calcular el tamaño del texto de la placa
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label_placa, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # Dibujar el fondo del texto de la placa
                    cv2.rectangle(annotated_frame, 
                                (x1, y1 - text_height - baseline - 5), 
                                (x1 + text_width, y1), 
                                color_placa, -1)
                    
                    # Dibujar el texto de la placa
                    cv2.putText(annotated_frame, label_placa, 
                               (x1, y1 - baseline - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Mostrar la matrícula reconocida
                    if matricula:
                        # Elegir color según validez
                        color_matricula = color_matricula_valida if es_valida else color_matricula_invalida
                        
                        # Calcular el tamaño del texto de la matrícula
                        (mat_width, mat_height), mat_baseline = cv2.getTextSize(
                            matricula, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                        )
                        
                        # Dibujar el fondo del texto de la matrícula
                        cv2.rectangle(annotated_frame, 
                                    (x1, y2), 
                                    (x1 + mat_width, y2 + mat_height + mat_baseline + 5), 
                                    color_matricula, -1)
                        
                        # Dibujar el texto de la matrícula
                        cv2.putText(annotated_frame, matricula, 
                                   (x1, y2 + mat_height + mat_baseline), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        
                        matriculas_en_frame += 1
                        total_matriculas_reconocidas += 1
                        
                        if es_valida:
                            total_matriculas_validas += 1
                    
                    placas_en_frame += 1
                    total_placas_detectadas += 1
        
        # Escribir información en el frame
        cv2.putText(annotated_frame, f"Placas: {placas_en_frame} | Matriculas: {matriculas_en_frame}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
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
    print(f"Total de placas detectadas: {total_placas_detectadas}")
    print(f"Total de matrículas reconocidas: {total_matriculas_reconocidas}")
    print(f"Total de matrículas válidas: {total_matriculas_validas}")
    print(f"Promedio de placas por frame: {total_placas_detectadas / frames_processed:.2f}")
    print(f"Promedio de matrículas por frame: {total_matriculas_reconocidas / frames_processed:.2f}")
    print(f"Tasa de matrículas válidas: {(total_matriculas_validas/total_matriculas_reconocidas*100):.1f}%" if total_matriculas_reconocidas > 0 else "Tasa de matrículas válidas: 0%")
    
    return output_path

def procesar_imagen(model_placas, model_caracteres, imagen_path, output_path=None,
                   conf_placas=0.40, conf_caracteres=0.25):
    """
    Procesa una imagen para detectar placas y reconocer sus caracteres
    """
    # Leer la imagen
    imagen = cv2.imread(str(imagen_path))
    if imagen is None:
        print(f"Error: No se pudo leer la imagen {imagen_path}")
        return
    
    # Realizar predicción de placas
    results_placas = model_placas.predict(imagen, conf=conf_placas)[0]
    
    # Crear una copia de la imagen para dibujar las detecciones
    annotated_image = imagen.copy()
    
    # Colores para las detecciones (BGR format)
    color_placa = (203, 192, 255)  # Rosado para placas
    color_matricula_valida = (0, 255, 0)   # Verde para matrículas válidas
    color_matricula_invalida = (0, 0, 255) # Rojo para matrículas inválidas
    
    placas_detectadas = 0
    matriculas_reconocidas = 0
    matriculas_validas = 0
    
    print(f"Procesando imagen: {imagen_path}")
    
    # Procesar cada placa detectada
    if results_placas.boxes is not None:
        for box in results_placas.boxes:
            # Obtener coordenadas del bounding box de la placa
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Obtener confianza de la placa
            conf_placa = float(box.conf[0])
            
            # Recortar la región de la placa
            placa_roi = imagen[y1:y2, x1:x2]
            
            if placa_roi.size > 0:  # Verificar que el recorte no esté vacío
                # Reconocer caracteres en la placa
                matricula, caracteres, es_valida, mensaje = reconocer_caracteres_placa(
                    model_caracteres, placa_roi, conf_caracteres
                )
                
                # Dibujar el bounding box de la placa
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color_placa, 2)
                
                # Crear etiqueta de la placa
                label_placa = f"Placa {conf_placa:.2f}"
                
                # Calcular el tamaño del texto de la placa
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_placa, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                # Dibujar el fondo del texto de la placa
                cv2.rectangle(annotated_image, 
                            (x1, y1 - text_height - baseline - 5), 
                            (x1 + text_width, y1), 
                            color_placa, -1)
                
                # Dibujar el texto de la placa
                cv2.putText(annotated_image, label_placa, 
                           (x1, y1 - baseline - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Mostrar la matrícula reconocida
                if matricula:
                    print(f"Matrícula reconocida: {matricula} - {'VÁLIDA' if es_valida else 'INVÁLIDA'} - {mensaje}")
                    
                    # Elegir color según validez
                    color_matricula = color_matricula_valida if es_valida else color_matricula_invalida
                    
                    # Calcular el tamaño del texto de la matrícula
                    (mat_width, mat_height), mat_baseline = cv2.getTextSize(
                        matricula, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    
                    # Dibujar el fondo del texto de la matrícula
                    cv2.rectangle(annotated_image, 
                                (x1, y2), 
                                (x1 + mat_width, y2 + mat_height + mat_baseline + 5), 
                                color_matricula, -1)
                    
                    # Dibujar el texto de la matrícula
                    cv2.putText(annotated_image, matricula, 
                               (x1, y2 + mat_height + mat_baseline), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    matriculas_reconocidas += 1
                    if es_valida:
                        matriculas_validas += 1
                
                placas_detectadas += 1
    
    # Configurar el nombre del archivo de salida si no se proporciona
    if output_path is None:
        imagen_name = Path(imagen_path).stem
        output_path = f"detection_results/caracteres_placas_{imagen_name}.jpg"
    
    # Crear directorio de salida si no existe
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar la imagen procesada
    cv2.imwrite(output_path, annotated_image)
    
    print(f"\n=== Resultados de la Imagen ===")
    print(f"Imagen procesada: {imagen_path}")
    print(f"Resultado guardado en: {output_path}")
    print(f"Placas detectadas: {placas_detectadas}")
    print(f"Matrículas reconocidas: {matriculas_reconocidas}")
    print(f"Matrículas válidas: {matriculas_validas}")
    if matriculas_reconocidas > 0:
        print(f"Tasa de matrículas válidas: {(matriculas_validas/matriculas_reconocidas*100):.1f}%")
    
    return output_path

def main():
    # Si se pasan argumentos, procesar solo ese video
    if len(sys.argv) >= 3:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        # ... cargar modelos y procesar solo ese video ...
        model_placas_path = "runs/detect/license_plate_detector_v1/weights/best.pt"
        model_caracteres_path = "runs/detect/plate_characters_detector_v1/weights/best.pt"
        if not Path(model_placas_path).exists() or not Path(model_caracteres_path).exists():
            print(f"Error: No se encontró el modelo de placas o caracteres")
            return
        from ultralytics import YOLO
        model_placas = YOLO(model_placas_path)
        model_caracteres = YOLO(model_caracteres_path)
        procesar_video(model_placas, model_caracteres, input_path, output_path)
        print(f"\nVideo procesado correctamente: {input_path}")
        print("-" * 50)
        return
    # ... resto del main original ...

if __name__ == "__main__":
    main() 