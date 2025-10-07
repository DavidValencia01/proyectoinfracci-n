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
import sys

from procesar_caracteres_en_placas import reconocer_caracteres_placa, validar_formato_placa_peruana, ZONAS_REGISTRALES
from ultralytics import YOLO
from procesar_vehiculos import procesar_frame as procesar_vehiculos_frame, obtener_etiqueta_y_color, CLASE_MAPEO
from procesar_cruces_peatonales import procesar_frame as procesar_cruces_frame
from procesar_placas_vehiculares import procesar_frame as procesar_placas_frame


# Rutas de los modelos (ajustar según corresponda)
RUTA_MODELO_VEHICULOS = 'runs/detect/vehicle_detection_model/weights/best.pt'
RUTA_MODELO_CRUCES = 'runs/detect/crosswalk_detector_v1/weights/best.pt'
RUTA_MODELO_PLACAS = 'runs/detect/license_plate_detector_v1/weights/best.pt'
RUTA_MODELO_CARACTERES = 'runs/detect/plate_characters_detector_v1/weights/best.pt'

# Nota: DIR_FRAMES_INFRACCION se gestiona desde la aplicación (app.py/infracciones.py)

# Umbrales de confianza por clase para vehículos (ajustar según tu experimentación)
CONF_THRESHOLDS_VEHICULOS = {
    'car': 0.3,
    'bus': 0.3,
    'microbus': 0.3,
    'motorbike': 0.3,
    'truck': 0.3,
    'pickup-van': 0.3
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

def seleccionar_poligono_manual(frame, max_width=900, max_height=700, auto=False):
    if auto:
        h, w = frame.shape[:2]
        # Polígono rectangular central (ajustar si se desea)
        margin_x, margin_y = int(w*0.2), int(h*0.6)
        poly = np.array([
            [margin_x, margin_y],
            [w-margin_x, margin_y],
            [w-margin_x, h-10],
            [margin_x, h-10]
        ], np.int32)
        return poly
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
    auto = '--auto' in sys.argv
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
        VIDEO_SALIDA = os.path.join(DIR_VIDEOS_SALIDA, f'{video_name}_infraccion.mp4')
        print(f'[{idx}/{len(video_files)}] Procesando video: {VIDEO_ENTRADA}')
        print(f'Guardando resultado en: {VIDEO_SALIDA}')
        os.makedirs(os.path.dirname(VIDEO_SALIDA), exist_ok=True)
        os.makedirs(DIR_FRAMES_INFRACCION, exist_ok=True)

        # Cargar modelos
        modelo_vehiculos = YOLO(RUTA_MODELO_VEHICULOS)
        modelo_cruces = YOLO(RUTA_MODELO_CRUCES)
        modelo_placas = YOLO(RUTA_MODELO_PLACAS)
        modelo_caracteres = YOLO(RUTA_MODELO_CARACTERES)

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
        if auto:
            poligono_cruce = seleccionar_poligono_manual(frame, auto=True)
            print(f"Polígono automático: {poligono_cruce.tolist()}")
        else:
            print("Por favor, seleccione 4 puntos del cruce peatonal en la ventana (sentido horario o antihorario)")
            poligono_cruce = seleccionar_poligono_manual(frame)
        poligono_cruce = poligono_cruce.astype(np.int32)
        # Reiniciar video para procesar desde el inicio
        cap.release()
        cap = cv2.VideoCapture(VIDEO_ENTRADA)
        frame_idx = 0
        infraccion_idx = 0
        print(f"FPS del video: {fps}")
        tiempo_espera_frames = 0  # Cambiado a 0 - detección inmediata
        print(f"Frames requeridos para infracción: {tiempo_espera_frames}")
        vehiculos_sobre_cruce = {}
        vehiculos_infraccionados = set()
        placas_infraccionadas = set()
        placas_por_vehiculo = {}  # track_id -> placa válida
        mejor_evidencia_por_vehiculo = {}  # track_id -> {'score': mejor_score, 'frame': frame_annotated, 'texto': texto}
        
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
            
            # Lógica de infracción usando el polígono - USAR UN SOLO SISTEMA
            results = modelo_vehiculos.track(frame, persist=True, conf=0.1)[0]
            track_ids = results.boxes.id if results.boxes is not None else None
            vehiculos_actuales = {}
            
            if results.boxes is not None and track_ids is not None and len(results.boxes) > 0:
                if np.isscalar(track_ids):
                    track_ids = [track_ids]
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
                    etiqueta_espanol, color = obtener_etiqueta_y_color(class_name)
                    cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
                    texto = f"ID {track_id}: {etiqueta_espanol}"
                    (text_width, text_height), baseline = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame_annotated, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                    cv2.putText(frame_annotated, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Placas: siempre dibujar y reconocer caracteres
            frame_annotated, boxes_placa = procesar_placas_frame(
                modelo_placas, frame_annotated, conf_threshold=get_conf_placas())
            
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
                # Lógica de infracción
                superpuesto, bbox_infraccion = bbox_superpone_poligono(vbox, poligono_cruce)
                if superpuesto and bbox_infraccion is not None:
                    if track_id not in vehiculos_sobre_cruce:
                        vehiculos_sobre_cruce[track_id] = frame_idx
                        print(f"Vehículo {track_id} entró al cruce en frame {frame_idx}")
                    tiempo = frame_idx - vehiculos_sobre_cruce[track_id]
                    tiempo_segundos = tiempo / fps
                    print(f"Vehículo {track_id} lleva {tiempo_segundos:.4f}s sobre el cruce (frame {frame_idx})")
                    placa_asociada = placas_por_vehiculo.get(track_id)
                    es_valida = False
                    mejor_score = -1
                    if placa_asociada:
                        es_valida, _ = validar_formato_placa_peruana(placa_asociada)
                        # Calcular score de calidad de la placa
                        _, _, _, score = seleccionar_mejor_placa([placa_asociada], frame, modelo_caracteres) if placa_asociada is not None else (None, None, False, -1)
                        mejor_score = score if score is not None else -1
                    if (tiempo >= tiempo_espera_frames and
                        placa_asociada and
                        es_valida and
                        len(placa_asociada) >= 6):
                        x1, y1, x2, y2 = bbox_infraccion
                        texto = f'INFRACCION: >{tiempo_espera_frames/fps:.1f}s sobre cruce ({tiempo_segundos:.1f}s) | Placa: {placa_asociada}'
                        # Guardar solo si es la mejor evidencia
                        if (track_id not in mejor_evidencia_por_vehiculo or
                            mejor_score > mejor_evidencia_por_vehiculo[track_id]['score']):
                            mejor_evidencia_por_vehiculo[track_id] = {
                                'score': mejor_score,
                                'frame': frame_annotated.copy(),
                                'texto': texto,
                                'frame_idx': frame_idx
                            }
                    # No guardar aquí, solo actualizar mejor evidencia
                else:
                    # Si el vehículo sale del cruce, guardar la mejor evidencia si corresponde
                    if track_id in vehiculos_sobre_cruce:
                        print(f"Vehículo {track_id} salió del cruce en frame {frame_idx}")
                        if (track_id in mejor_evidencia_por_vehiculo and
                            track_id not in vehiculos_infraccionados and
                            placas_por_vehiculo.get(track_id) not in placas_infraccionadas):
                            evidencia = mejor_evidencia_por_vehiculo[track_id]
                            filename = f'infraccion_frame{evidencia["frame_idx"]}_veh{track_id}.jpg'
                            cv2.imwrite(os.path.join(DIR_FRAMES_INFRACCION, filename), evidencia['frame'])
                            placas_infraccionadas.add(placas_por_vehiculo.get(track_id))
                            vehiculos_infraccionados.add(track_id)
                            print(f"INFRACCIÓN DETECTADA: Vehículo {track_id} - Placa {placas_por_vehiculo.get(track_id)} - Guardada solo mejor evidencia")
                        del vehiculos_sobre_cruce[track_id]
                        if track_id in mejor_evidencia_por_vehiculo:
                            del mejor_evidencia_por_vehiculo[track_id]
            out.write(frame_annotated)
            frame_idx += 1
        cap.release()
        out.release()
        print(f'Procesamiento finalizado. Video guardado en {VIDEO_SALIDA}')
        print(f'Frames de infracción guardados en {DIR_FRAMES_INFRACCION}')
        # Al final del procesamiento, guardar el contador real
        resultado_json = {
            "infracciones": len(vehiculos_infraccionados)
        }
        json_path = os.path.splitext(VIDEO_SALIDA)[0] + '_infracciones.json'
        with open(json_path, 'w') as f:
            json.dump(resultado_json, f)

if __name__ == '__main__':
    main()