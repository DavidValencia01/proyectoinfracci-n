import cv2
import os
import numpy as np
from pathlib import Path
import time
import json
import glob
import torch
import numbers
import re
from scipy.spatial import distance

from procesar_caracteres_en_placas import reconocer_caracteres_placa, validar_formato_placa_peruana, ZONAS_REGISTRALES
from ultralytics import YOLO
from procesar_vehiculos import procesar_frame as procesar_vehiculos_frame, obtener_etiqueta_y_color, CLASE_MAPEO
from procesar_cruces_peatonales import procesar_frame as procesar_cruces_frame
from procesar_placas_vehiculares import procesar_frame as procesar_placas_frame
from procesar_semaforos import SemaphoreColorDetector


# Rutas de los modelos (ajustar según corresponda)
RUTA_MODELO_VEHICULOS = 'runs/detect/vehicle_detection_model/weights/best.pt'
RUTA_MODELO_CRUCES = 'runs/detect/crosswalk_detector_v1/weights/best.pt'
RUTA_MODELO_PLACAS = 'runs/detect/license_plate_detector_v1/weights/best.pt'
RUTA_MODELO_CARACTERES = 'runs/detect/plate_characters_detector_v1/weights/best.pt'
RUTA_MODELO_SEMAFOROS = 'runs/detect/traffic_lights_detector_v2/weights/best.pt'

# Configuración de tiempo para infracciones
TIEMPO_TOLERANCIA_INFRACCION = 3.0  # 3 segundos de tolerancia
FPS_VIDEO = 30  # FPS típico, se ajustará automáticamente

# Nota: DIR_FRAMES_INFRACCION se gestiona desde la aplicación (app.py/infracciones.py)

# Umbrales de confianza por clase para vehículos (ajustar según tu experimentación)
CONF_THRESHOLDS_VEHICULOS = {
    'car': 0.4,
    'bus': 0.4,
    'microbus': 0.9,
    'motorbike': 0.3,
    'truck': 1,
    'pickup-van': 0.7
}

# Umbral de confianza para cruces peatonales
def get_conf_cruces():
    return 0.10

# Umbral de confianza para placas vehiculares
def get_conf_placas():
    return 0.25

# Umbral de confianza para caracteres de placa
def get_conf_caracteres():
    return 0.15

# Umbral de confianza para semáforos
def get_conf_semaforos():
    return 0.3

# Colores
COLOR_VEHICULO = (0, 255, 255)  # Amarillo
COLOR_CRUCE = (128, 0, 128)     # Morado
COLOR_PLACA = (203, 192, 255)   # Rosado
COLOR_MATRICULA_VALIDA = (0, 255, 0)   # Verde
COLOR_MATRICULA_INVALIDA = (0, 0, 255) # Rojo
COLOR_INFRACCION = (0, 0, 255)  # Rojo fuerte
COLOR_PLACA_ASOCIADA = (255, 0, 0) # Azul

def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def recortar_imagen(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    return img[y1:y2, x1:x2]

def filtrar_placas_por_vehiculo(boxes_placa, bbox_vehiculo, conf_threshold=0.3):
    """
    Filtra placas que están dentro del vehículo y aplica criterios de calidad
    """
    vx1, vy1, vx2, vy2 = bbox_vehiculo
    v_area = (vx2 - vx1) * (vy2 - vy1)
    placas_filtradas = []
    
    for placa_box in boxes_placa:
        px1, py1, px2, py2 = placa_box
        p_area = (px2 - px1) * (py2 - py1)
        
        # Verificar que la placa esté dentro del vehículo (más permisivo)
        if px1 >= vx1-20 and py1 >= vy1-20 and px2 <= vx2+20 and py2 <= vy2+20:
            # Calcular IoU con el vehículo
            iou = calcular_iou(bbox_vehiculo, placa_box)
            
            # Criterios de calidad más permisivos:
            # 1. Tamaño mínimo de placa (al menos 0.5% del área del vehículo)
            # 2. Tamaño máximo de placa (no más del 25% del área del vehículo)
            # 3. IoU mínimo con el vehículo (más permisivo)
            if (p_area > 0.005 * v_area and 
                p_area < 0.25 * v_area and 
                iou > 0.05):
                placas_filtradas.append(placa_box)
    
    return placas_filtradas

def seleccionar_mejor_placa(placas_filtradas, frame, modelo_caracteres):
    """
    Selecciona la mejor placa basándose en la calidad del reconocimiento de caracteres
    SIGUIENDO LAS REGLAS DEL SCRIPT DE PROCESAMIENTO
    """
    if not placas_filtradas:
        return None, None, False, None
    
    mejor_placa = None
    mejor_matricula = None
    mejor_es_valida = False
    mejor_score = -1
    
    for placa_box in placas_filtradas:
        x1, y1, x2, y2 = placa_box
        placa_img = frame[y1:y2, x1:x2]
        
        if placa_img.size > 0:
            # USAR LA FUNCIÓN CORRECTA DEL SCRIPT DE PROCESAMIENTO
            # Esta función ya incluye:
            # - Ordenamiento de izquierda a derecha
            # - Filtrado de caracteres válidos
            # - Priorización de letra de zona registral
            # - Validación de formato peruano
            matricula, caracteres, es_valida, mensaje = reconocer_caracteres_placa(
                modelo_caracteres, placa_img, conf_threshold=0.25
            )
            
            # Calcular score de calidad basado en las reglas peruanas
            score = 0
            
            # Bonus por formato válido (ya validado por la función)
            if es_valida:
                score += 50
            
            # Bonus por longitud correcta (4-7 caracteres)
            if 4 <= len(matricula) <= 7:
                score += 20
            
            # Bonus por contener letra de zona registral peruana
            if any(char in ZONAS_REGISTRALES for char in matricula):
                score += 15
            
            # Bonus por contener números
            if any(char.isdigit() for char in matricula):
                score += 10
            
            # Bonus por contener letras
            if any(char.isalpha() for char in matricula):
                score += 10
            
            # Bonus por tener al menos 3 caracteres
            if len(matricula) >= 3:
                score += 5
            
            # Penalización por caracteres repetidos excesivos
            if len(set(matricula)) < len(matricula) * 0.6:
                score -= 15
            
            # Penalización por caracteres no válidos
            caracteres_invalidos = sum(1 for char in matricula if not char.isalnum() and char != '-')
            score -= caracteres_invalidos * 10
            
            # Bonus por tener mezcla de letras y números
            if any(char.isalpha() for char in matricula) and any(char.isdigit() for char in matricula):
                score += 10
            
            # Bonus por comenzar con letra de zona registral (prioridad alta)
            if matricula and matricula[0] in ZONAS_REGISTRALES:
                score += 25
            
            if score > mejor_score:
                mejor_score = score
                mejor_placa = placa_box
                mejor_matricula = matricula
                mejor_es_valida = es_valida
    
    return mejor_placa, mejor_matricula, mejor_es_valida, mejor_score

def anotar_todo(frame, modelo_vehiculos, modelo_cruces, modelo_placas):
    annotated = frame.copy()
    # Vehículos
    vehiculos_results = modelo_vehiculos.track(frame, persist=True, conf=0.1)[0]
    if vehiculos_results.boxes is not None:
        for i, box in enumerate(vehiculos_results.boxes):
            class_id = int(box.cls[0].item())
            class_name = modelo_vehiculos.names[class_id]
            confidence = float(box.conf[0].item())
            track_ids = vehiculos_results.boxes.id
            track_id = int(track_ids[i].item()) if track_ids is not None else -1
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            texto = f"Vehículo {class_name} {confidence:.2f}"
            if track_id != -1:
                texto = f"ID {track_id}: {texto}"
            (text_width, text_height), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 255), -1)
            cv2.putText(annotated, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    # Cruces
    cruces_results = modelo_cruces.predict(frame, conf=0.25)[0]
    if cruces_results.boxes is not None:
        for box in cruces_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 0, 128), 2)
            label = f"Cruce Peatonal"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), (128, 0, 128), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    # Placas
    placas_results = modelo_placas.predict(frame, conf=0.4)[0]
    if placas_results.boxes is not None:
        for box in placas_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (203, 192, 255), 2)
            label = f"Placa"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), (203, 192, 255), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return annotated

def hay_superposicion(vbox, boxes_cruce):
    vx1, vy1, vx2, vy2 = vbox
    varea = (vx2 - vx1) * (vy2 - vy1)
    for cbox in boxes_cruce:
        cx1, cy1, cx2, cy2 = cbox
        ix1, iy1 = max(vx1, cx1), max(vy1, cy1)
        ix2, iy2 = min(vx2, cx2), min(vy2, cy2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter_area = iw * ih
        if inter_area > 0.15 * varea and inter_area > 0.15 * ((cx2-cx1)*(cy2-cy1)):
            return True, (ix1, iy1, ix2, iy2)
    return False, None

def asociar_placa_a_infraccion(boxes_placa, bbox_infraccion):
    if not boxes_placa or bbox_infraccion is None:
        return None
    ix1, iy1, ix2, iy2 = bbox_infraccion
    max_iou = 0
    best_box = None
    for pbox in boxes_placa:
        px1, py1, px2, py2 = pbox
        sx1, sy1 = max(ix1, px1), max(iy1, py1)
        sx2, sy2 = min(ix2, px2), min(iy2, py2)
        sw, sh = max(0, sx2 - sx1), max(0, sy2 - sy1)
        inter_area = sw * sh
        area_p = (px2 - px1) * (py2 - py1)
        area_i = (ix2 - ix1) * (iy2 - iy1)
        union_area = area_p + area_i - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        if iou > max_iou:
            max_iou = iou
            best_box = pbox
    return best_box if max_iou > 0 else None

def seleccionar_poligono_manual(frame, max_width=900, max_height=700):
    puntos = []
    h, w = frame.shape[:2]
    scale_w = max_width / w if w > max_width else 1.0
    scale_h = max_height / h if h > max_height else 1.0
    scale = min(scale_w, scale_h)
    if scale < 1.0:
        frame_resized = cv2.resize(frame, (int(w*scale), int(h*scale)))
    else:
        frame_resized = frame.copy()
    clone = frame_resized.copy()
    def draw_poly(img, pts):
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], isClosed=True, color=(0,255,0), thickness=2)
        if len(pts) == 4:
            overlay = img.copy()
            cv2.fillPoly(overlay, [np.array(pts, np.int32)], color=(128,0,128))
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(puntos) < 4:
            puntos.append((x, y))
            img2 = clone.copy()
            draw_poly(img2, puntos)
            cv2.imshow("Seleccione 4 puntos del cruce (sentido horario o antihorario)", img2)
    img2 = clone.copy()
    cv2.imshow("Seleccione 4 puntos del cruce (sentido horario o antihorario)", img2)
    cv2.setMouseCallback("Seleccione 4 puntos del cruce (sentido horario o antihorario)", click_event)
    while len(puntos) < 4:
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    # Escalar los puntos a la resolución original si fue redimensionado
    puntos_orig = [(int(x/scale), int(y/scale)) for (x, y) in puntos]
    return np.array(puntos_orig, np.int32)

def bbox_superpone_poligono(bbox, poligono):
    poligono = poligono.astype(np.int32)
    x1, y1, x2, y2 = bbox
    # Crear puntos como tuplas simples en lugar de arrays numpy
    box_pts = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
    for pt_x, pt_y in box_pts:
        if cv2.pointPolygonTest(poligono, (pt_x, pt_y), False) >= 0:
            # Basta con que un vértice esté dentro
            return True, bbox
    return False, None

# --- NUEVO: fallback Trujillo ---
def corregir_placa_trujillo(matricula, caracteres, es_valida):
    if not es_valida and len(matricula) in [5, 6]:
        if not matricula.startswith('T'):
            posible = 'T' + matricula[1:]
            es_valida_falsa, _ = validar_formato_placa_peruana(posible)
            if es_valida_falsa:
                return posible, True
    return matricula, es_valida

def encontrar_placa_mas_cercana(boxes_placa, vbox):
    # Busca la placa detectada más cercana al centro del vehículo
    if not boxes_placa:
        return None
    vx1, vy1, vx2, vy2 = vbox
    v_cx, v_cy = (vx1 + vx2) // 2, (vy1 + vy2) // 2
    min_dist = float('inf')
    best_box = None
    for px1, py1, px2, py2 in boxes_placa:
        p_cx, p_cy = (px1 + px2) // 2, (py1 + py2) // 2
        d = distance.euclidean((v_cx, v_cy), (p_cx, p_cy))
        if d < min_dist:
            min_dist = d
            best_box = (px1, py1, px2, py2)
    return best_box

def main():
    # Buscar todos los videos en la carpeta de entrada
    video_exts = ['mp4', 'avi', 'mov', 'mkv']
    video_files = []
    for ext in video_exts:
        video_files.extend(glob.glob(os.path.join(DIR_VIDEOS_ENTRADA, f'*.{ext}')))
    if not video_files:
        print('No se encontró ningún video en la carpeta de entrada.')
        return
    print(f'Se encontraron {len(video_files)} videos para procesar.')
    for idx, VIDEO_ENTRADA in enumerate(video_files, 1):
        video_name = os.path.splitext(os.path.basename(VIDEO_ENTRADA))[0]
        VIDEO_SALIDA = os.path.join(DIR_VIDEOS_SALIDA, f'{video_name}_infraccion_semaforo.mp4')
        print(f'[{idx}/{len(video_files)}] Procesando video: {VIDEO_ENTRADA}')
        print(f'Guardando resultado en: {VIDEO_SALIDA}')
        os.makedirs(os.path.dirname(VIDEO_SALIDA), exist_ok=True)
        os.makedirs(DIR_FRAMES_INFRACCION, exist_ok=True)

        # Cargar modelos
        modelo_vehiculos = YOLO(RUTA_MODELO_VEHICULOS)
        modelo_cruces = YOLO(RUTA_MODELO_CRUCES)
        modelo_placas = YOLO(RUTA_MODELO_PLACAS)
        modelo_caracteres = YOLO(RUTA_MODELO_CARACTERES)
        modelo_semaforos = YOLO(RUTA_MODELO_SEMAFOROS)
        color_detector = SemaphoreColorDetector(min_pixel_threshold=50)

        cap = cv2.VideoCapture(VIDEO_ENTRADA)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(VIDEO_SALIDA, fourcc, fps, (width, height))

        # --- Permitir selección manual del polígono del cruce peatonal ---
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el primer frame del video.")
            return
        print("Por favor, seleccione 4 puntos del cruce peatonal en la ventana (sentido horario o antihorario)")
        poligono_cruce = seleccionar_poligono_manual(frame)
        poligono_cruce = poligono_cruce.astype(np.int32)  # Asegurar tipo correcto
        print(f"Polígono seleccionado: {poligono_cruce.tolist()}")
        # Reiniciar video para procesar desde el inicio
        cap.release()
        cap = cv2.VideoCapture(VIDEO_ENTRADA)
        frame_idx = 0
        infraccion_idx = 0
        print(f"FPS del video: {fps}")
        
        # Variables para el sistema de tiempo de infracciones
        vehiculos_sobre_cruce = {}  # track_id -> {'frame_inicio': frame_idx, 'tiempo_inicio': tiempo}
        vehiculos_infraccionados = set()
        placas_infraccionadas = set()
        placas_por_vehiculo = {}  # track_id -> placa válida
        
        # Calcular frames de tolerancia basado en el FPS del video
        frames_tolerancia = int(TIEMPO_TOLERANCIA_INFRACCION * fps)
        print(f"Frames de tolerancia para {TIEMPO_TOLERANCIA_INFRACCION}s: {frames_tolerancia}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_annotated = frame.copy()
            
            # Dibujar polígono del cruce peatonal
            cv2.polylines(frame_annotated, [poligono_cruce], isClosed=True, color=(128,0,128), thickness=2)
            overlay = frame_annotated.copy()
            cv2.fillPoly(overlay, [poligono_cruce], color=(128,0,128))
            cv2.addWeighted(overlay, 0.2, frame_annotated, 0.8, 0, frame_annotated)
            label = f"cruce peatonal fijo"
            x1c, y1c = poligono_cruce[0]
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame_annotated, (x1c, y1c - text_height - baseline - 5), (x1c + text_width, y1c), (128, 0, 128), -1)
            cv2.putText(frame_annotated, label, (x1c, y1c - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 1. DETECTAR SEMÁFORO Y DETERMINAR COLOR DOMINANTE
            semaforo_results = modelo_semaforos.predict(frame, conf=get_conf_semaforos())[0]
            semaforo_rojo_activo = False
            semaforo_amarillo_activo = False
            semaforo_detectado = False
            
            if semaforo_results.boxes is not None:
                print(f"Frame {frame_idx}: Se detectaron {len(semaforo_results.boxes)} semáforos")
                for box in semaforo_results.boxes:
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Recortar la región de interés (ROI) del semáforo
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    
                    # Convertir ROI a espacio de color HSV
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    
                    # Obtener el color dominante usando el detector
                    color_semaforo = color_detector.get_dominant_color(hsv_roi, roi)
                    semaforo_detectado = True
                    
                    # Determinar si el semáforo está en rojo o amarillo
                    if color_semaforo == "Rojo":
                        semaforo_rojo_activo = True
                        print(f"Frame {frame_idx}: SEMÁFORO ROJO DETECTADO - Conf: {conf:.3f}")
                    elif color_semaforo == "Amarillo":
                        semaforo_amarillo_activo = True
                        print(f"Frame {frame_idx}: SEMÁFORO AMARILLO DETECTADO - Conf: {conf:.3f}")
                    
                    # Dibujar el semáforo con el color correspondiente
                    color_bbox = (0, 0, 255) if color_semaforo == "Rojo" else (0, 255, 0) if color_semaforo == "Verde" else (0, 255, 255)
                    cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color_bbox, 2)
                    label_semaforo = f"Semaforo {color_semaforo} {conf:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(label_semaforo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), color_bbox, -1)
                    cv2.putText(frame_annotated, label_semaforo, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                if frame_idx % 30 == 0:  # Log cada 30 frames para no saturar
                    print(f"Frame {frame_idx}: No se detectaron semáforos")
            
            # Mostrar estado del semáforo en pantalla
            if semaforo_rojo_activo:
                estado_texto = "SEMÁFORO ROJO"
                color_estado = (0, 0, 255)
            elif semaforo_amarillo_activo:
                estado_texto = "SEMÁFORO AMARILLO"
                color_estado = (0, 255, 255)
            elif semaforo_detectado:
                estado_texto = "SEMÁFORO VERDE"
                color_estado = (0, 255, 0)
            else:
                estado_texto = "SEMÁFORO NO DETECTADO"
                color_estado = (128, 128, 128)
            cv2.putText(frame_annotated, estado_texto, (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_estado, 2)
            
            # Lógica de infracción usando el polígono - USAR UN SOLO SISTEMA
            results = modelo_vehiculos.track(frame, persist=True, conf=0.1)[0]
            track_ids = results.boxes.id if results.boxes is not None else None
            vehiculos_actuales = {}
            
            if results.boxes is not None and track_ids is not None and len(results.boxes) > 0:
                if np.isscalar(track_ids):
                    track_ids = [track_ids]
                print(f"Frame {frame_idx}: Se detectaron {len(results.boxes)} vehículos")
                for i in range(len(results.boxes)):
                    box = results.boxes[i]; tid = track_ids[i]
                    if isinstance(tid, torch.Tensor):
                        track_id = int(tid.item())
                    elif isinstance(tid, (int, float, np.integer, np.floating)):
                        track_id = int(tid)
                    else:
                        continue
                    xyxy = box.xyxy[0]
                    if hasattr(xyxy, 'cpu'):
                        x1, y1, x2, y2 = xyxy.cpu().numpy().astype(int)
                    else:
                        x1, y1, x2, y2 = np.array(xyxy).astype(int)
                    vehiculos_actuales[track_id] = [x1, y1, x2, y2]
                    class_id = int(box.cls[0].item())
                    class_name = modelo_vehiculos.names[class_id]
                    confidence = float(box.conf[0].item())
                    etiqueta_espanol, color = obtener_etiqueta_y_color(class_name)
                    cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
                    texto = f"ID {track_id}: {etiqueta_espanol}"
                    (text_width, text_height), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(frame_annotated, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    print(f"Frame {frame_idx}: Vehículo ID {track_id} - {etiqueta_espanol} - Conf: {confidence:.3f}")
            else:
                if frame_idx % 30 == 0:  # Log cada 30 frames para no saturar
                    print(f"Frame {frame_idx}: No se detectaron vehículos")
            
            # Placas: siempre dibujar y reconocer caracteres
            frame_annotated, boxes_placa = procesar_placas_frame(
                modelo_placas, frame_annotated, conf_threshold=get_conf_placas())
            
            if boxes_placa:
                print(f"Frame {frame_idx}: Se detectaron {len(boxes_placa)} placas")
            else:
                if frame_idx % 30 == 0:  # Log cada 30 frames para no saturar
                    print(f"Frame {frame_idx}: No se detectaron placas")
            
            # 2. Evitar placas repetidas por frame
            placas_asignadas_este_frame = set()
            
            for track_id, vbox in vehiculos_actuales.items():
                # Mejor filtrado: solo placas que estén al menos 60% dentro del bounding box del vehículo
                placas_candidatas = []
                vx1, vy1, vx2, vy2 = vbox
                v_area = (vx2 - vx1) * (vy2 - vy1)
                for placa_box in boxes_placa:
                    px1, py1, px2, py2 = placa_box
                    # Calcular intersección
                    ix1, iy1 = max(vx1, px1), max(vy1, py1)
                    ix2, iy2 = min(vx2, px2), min(vy2, py2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter_area = iw * ih
                    p_area = (px2 - px1) * (py2 - py1)
                    # Al menos 60% del área de la placa debe estar dentro del vehículo
                    if p_area > 0 and inter_area / p_area >= 0.6:
                        placas_candidatas.append(placa_box)
                # Buscar la placa candidata más cercana al centro del vehículo que no haya sido asignada
                mejor_placa = None
                mejor_matricula = None
                mejor_es_valida = False
                mejor_dist = float('inf')
                for placa_box in placas_candidatas:
                    if tuple(placa_box) in placas_asignadas_este_frame:
                        continue
                    x1p, y1p, x2p, y2p = placa_box
                    placa_img = frame[y1p:y2p, x1p:x2p]
                    if placa_img.size > 0:
                        matricula, _, es_valida, _ = reconocer_caracteres_placa(
                            modelo_caracteres, placa_img, conf_threshold=get_conf_caracteres())
                        if not matricula or matricula in placas_asignadas_este_frame:
                            continue
                        v_cx, v_cy = (vx1 + vx2) // 2, (vy1 + vy2) // 2
                        p_cx, p_cy = (x1p + x2p) // 2, (y1p + y2p) // 2
                        dist = np.hypot(v_cx - p_cx, v_cy - p_cy)
                        if dist < mejor_dist:
                            mejor_dist = dist
                            mejor_placa = placa_box
                            mejor_matricula = matricula
                            mejor_es_valida = es_valida
                # --- ASOCIAR PLACA SOLO UNA VEZ POR VEHÍCULO ---
                if track_id not in placas_por_vehiculo and mejor_matricula and mejor_es_valida:
                    placas_por_vehiculo[track_id] = mejor_matricula
                # Mostrar la placa asociada (si existe) o la mejor del frame
                placa_mostrar = placas_por_vehiculo.get(track_id, mejor_matricula)
                es_valida_placa_mostrar = False
                if placa_mostrar:
                    es_valida_placa_mostrar, _ = validar_formato_placa_peruana(placa_mostrar)
                if mejor_placa is not None and placa_mostrar:
                    placas_asignadas_este_frame.add(tuple(mejor_placa))
                    x1p, y1p, x2p, y2p = mejor_placa
                    color_matricula = COLOR_MATRICULA_VALIDA if es_valida_placa_mostrar else COLOR_MATRICULA_INVALIDA
                    (mat_width, mat_height), mat_baseline = cv2.getTextSize(placa_mostrar, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    cv2.rectangle(frame_annotated, (x1p, y2p), (x1p + mat_width, y2p + mat_height + mat_baseline + 5), color_matricula, -1)
                    cv2.putText(frame_annotated, placa_mostrar, (x1p, y2p + mat_height + mat_baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                
                # NUEVA LÓGICA DE INFRACCIÓN CON SISTEMA DE TIEMPO
                superpuesto, bbox_infraccion = bbox_superpone_poligono(vbox, poligono_cruce)
                
                if superpuesto and bbox_infraccion is not None:
                    # Vehículo está sobre el cruce peatonal
                    if track_id not in vehiculos_sobre_cruce:
                        # Vehículo acaba de entrar al cruce
                        vehiculos_sobre_cruce[track_id] = {
                            'frame_inicio': frame_idx,
                            'tiempo_inicio': frame_idx / fps
                        }
                        print(f"Vehículo {track_id} entró al cruce en frame {frame_idx} (tiempo: {frame_idx/fps:.2f}s)")
                    
                    # Verificar si el semáforo está en rojo o amarillo
                    if semaforo_rojo_activo or semaforo_amarillo_activo:
                        tiempo_en_cruce = frame_idx - vehiculos_sobre_cruce[track_id]['frame_inicio']
                        tiempo_segundos = tiempo_en_cruce / fps
                        
                        # Mostrar contador de tiempo para vehículos sobre el cruce
                        tiempo_texto = f"Tiempo en cruce: {tiempo_segundos:.1f}s"
                        cv2.putText(frame_annotated, tiempo_texto, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        
                        # Mostrar tiempo restante de tolerancia
                        tiempo_restante = max(0, TIEMPO_TOLERANCIA_INFRACCION - tiempo_segundos)
                        if tiempo_restante > 0:
                            tolerancia_texto = f"Tolerancia: {tiempo_restante:.1f}s"
                            cv2.putText(frame_annotated, tolerancia_texto, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            tolerancia_texto = "¡INFRACCIÓN!"
                            cv2.putText(frame_annotated, tolerancia_texto, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        
                        # Verificar si ha superado el tiempo de tolerancia
                        if tiempo_en_cruce >= frames_tolerancia:
                            placa_asociada = placas_por_vehiculo.get(track_id)
                            es_valida = False
                            if placa_asociada:
                                es_valida, _ = validar_formato_placa_peruana(placa_asociada)
                            
                            # Verificar condiciones para registrar infracción
                            if (placa_asociada and
                                es_valida and
                                len(placa_asociada) >= 6 and
                                track_id not in vehiculos_infraccionados and
                                placa_asociada not in placas_infraccionadas):
                                
                                x1, y1, x2, y2 = bbox_infraccion
                                
                                # Determinar el tipo de infracción y color
                                if semaforo_rojo_activo:
                                    tipo_infraccion = "ROJO"
                                    color_infraccion = (0, 0, 255)  # Rojo
                                else:  # semaforo_amarillo_activo
                                    tipo_infraccion = "AMARILLO"
                                    color_infraccion = (0, 255, 255)  # Amarillo
                                
                                texto = f'INFRACCION SEMAFORO {tipo_infraccion}: Vehiculo posado en cruce peatonal por {tiempo_segundos:.1f}s con luz {tipo_infraccion.lower()} | Placa: {placa_asociada}'
                                (mat_width, mat_height), mat_baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                                overlay = frame_annotated.copy()
                                cv2.rectangle(overlay, (30, 30), (30 + mat_width + 10, 30 + mat_height + mat_baseline + 10), color_infraccion, -1)
                                alpha = 0.5
                                cv2.addWeighted(overlay, alpha, frame_annotated, 1 - alpha, 0, frame_annotated)
                                cv2.putText(frame_annotated, texto, (35, 30 + mat_height), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
                                
                                # Guardar frame de infracción con nombre de placa y tipo de semáforo
                                nombre_archivo = f'infraccion_semaforo_{tipo_infraccion.lower()}_{placa_asociada}_frame{frame_idx}_veh{track_id}_tiempo{tiempo_segundos:.1f}s.jpg'
                                cv2.imwrite(os.path.join(DIR_FRAMES_INFRACCION, nombre_archivo), frame_annotated)
                                
                                placas_infraccionadas.add(placa_asociada)
                                vehiculos_infraccionados.add(track_id)
                                print(f"INFRACCION SEMAFORO {tipo_infraccion} DETECTADA: Vehiculo {track_id} - Placa {placa_asociada} - Tiempo en cruce: {tiempo_segundos:.1f}s")
                else:
                    # Vehículo no está sobre el cruce
                    if track_id in vehiculos_sobre_cruce:
                        tiempo_total = (frame_idx - vehiculos_sobre_cruce[track_id]['frame_inicio']) / fps
                        print(f"Vehículo {track_id} salió del cruce en frame {frame_idx} (tiempo total: {tiempo_total:.2f}s)")
                        del vehiculos_sobre_cruce[track_id]
            out.write(frame_annotated)
            frame_idx += 1
        cap.release()
        out.release()
        print(f'Procesamiento finalizado. Video guardado en {VIDEO_SALIDA}')
        print(f'Frames de infracción guardados en {DIR_FRAMES_INFRACCION}')

if __name__ == '__main__':
    main()