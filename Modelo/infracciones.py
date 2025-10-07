import cv2
import os
import numpy as np
from pathlib import Path
import json
import torch

# Asegurar imports de paquetes locales si se ejecuta con Python del sistema
BASE_DIR = Path(__file__).parent
import sys
sys.path.append(str(BASE_DIR / "Lib" / "site-packages"))

from ultralytics import YOLO
from procesar_placas_vehiculares import procesar_frame as procesar_placas_frame
from procesar_caracteres_en_placas import reconocer_caracteres_placa, validar_formato_placa_peruana
from procesar_vehiculos import obtener_etiqueta_y_color
from detectar_infraccion_cruce_peatonal import seleccionar_poligono_manual, bbox_superpone_poligono, COLOR_MATRICULA_VALIDA, COLOR_MATRICULA_INVALIDA, get_conf_placas, get_conf_caracteres, RUTA_MODELO_VEHICULOS, RUTA_MODELO_PLACAS, RUTA_MODELO_CARACTERES
from procesar_semaforos import SemaphoreColorDetector
from detectar_infraccion_semaforo_rojo import get_conf_semaforos, RUTA_MODELO_SEMAFOROS


def _seleccionar_poligono_auto(frame):
    h, w = frame.shape[:2]
    margin_x, margin_y = int(w * 0.2), int(h * 0.6)
    poly = np.array([
        [margin_x, margin_y],
        [w - margin_x, margin_y],
        [w - margin_x, h - 10],
        [margin_x, h - 10]
    ], np.int32)
    return poly


def procesar_video_infraccion_cruce(video_entrada: str, video_salida: str, dir_frames_infraccion: str, auto_polygon: bool = True):
    os.makedirs(os.path.dirname(video_salida), exist_ok=True)
    os.makedirs(dir_frames_infraccion, exist_ok=True)

    modelo_vehiculos = YOLO(RUTA_MODELO_VEHICULOS)
    modelo_placas = YOLO(RUTA_MODELO_PLACAS)
    modelo_caracteres = YOLO(RUTA_MODELO_CARACTERES)

    cap = cv2.VideoCapture(video_entrada)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_entrada}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Preferir H.264 (avc1) para compatibilidad HTML5; fallback a MPEG-4 (mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(video_salida, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_salida, fourcc, fps, (width, height))

    # Polígono del cruce peatonal
    ret, frame0 = cap.read()
    if not ret:
        cap.release(); out.release()
        raise RuntimeError("No se pudo leer el primer frame para definir polígono")
    poligono_cruce = (_seleccionar_poligono_auto(frame0) if auto_polygon else seleccionar_poligono_manual(frame0)).astype(np.int32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = 0
    tiempo_espera_frames = 0
    vehiculos_sobre_cruce = {}
    vehiculos_infraccionados = set()
    placas_infraccionadas = set()
    placas_por_vehiculo = {}
    mejor_evidencia_por_vehiculo = {}
    evidencias_guardadas = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_annotated = frame.copy()

        # Dibujo del polígono
        cv2.polylines(frame_annotated, [poligono_cruce], isClosed=True, color=(128, 0, 128), thickness=2)
        overlay = frame_annotated.copy()
        cv2.fillPoly(overlay, [poligono_cruce], color=(128, 0, 128))
        cv2.addWeighted(overlay, 0.2, frame_annotated, 0.8, 0, frame_annotated)

        # Vehículos
        results = modelo_vehiculos.track(frame, persist=True, conf=0.1)[0]
        track_ids = results.boxes.id if results.boxes is not None else None
        vehiculos_actuales = {}
        if results.boxes is not None and track_ids is not None and len(results.boxes) > 0:
            if np.isscalar(track_ids):
                track_ids = [track_ids]
            for i in range(len(results.boxes)):
                box = results.boxes[i]
                tid = track_ids[i]
                track_id = int(tid.item()) if isinstance(tid, torch.Tensor) else int(tid)
                x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy().astype(int) if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0]).astype(int))
                vehiculos_actuales[track_id] = [x1, y1, x2, y2]
                class_id = int(box.cls[0].item())
                etiqueta_espanol, color = obtener_etiqueta_y_color(modelo_vehiculos.names[class_id])
                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)

        # Placas
        frame_annotated, boxes_placa = procesar_placas_frame(modelo_placas, frame_annotated, conf_threshold=get_conf_placas())
        placas_asignadas_este_frame = set()

        for track_id, vbox in vehiculos_actuales.items():
            # Candidatas dentro del vehículo
            placas_candidatas = []
            vx1, vy1, vx2, vy2 = vbox
            for placa_box in boxes_placa:
                px1, py1, px2, py2 = placa_box
                ix1, iy1 = max(vx1, px1), max(vy1, py1)
                ix2, iy2 = min(vx2, px2), min(vy2, py2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = iw * ih
                p_area = (px2 - px1) * (py2 - py1)
                if p_area > 0 and inter_area / p_area >= 0.6:
                    placas_candidatas.append(placa_box)
            mejor_placa = None; mejor_matricula = None; mejor_es_valida = False; mejor_dist = float('inf')
            for placa_box in placas_candidatas:
                if tuple(placa_box) in placas_asignadas_este_frame:
                    continue
                x1p, y1p, x2p, y2p = placa_box
                placa_img = frame[y1p:y2p, x1p:x2p]
                if placa_img.size > 0:
                    matricula, _, es_valida, _ = reconocer_caracteres_placa(modelo_caracteres, placa_img, conf_threshold=get_conf_caracteres())
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
            if track_id not in placas_por_vehiculo and mejor_matricula and mejor_es_valida:
                placas_por_vehiculo[track_id] = mejor_matricula
            placa_mostrar = placas_por_vehiculo.get(track_id, mejor_matricula)
            es_valida_placa_mostrar = False
            if placa_mostrar:
                es_valida_placa_mostrar, _ = validar_formato_placa_peruana(placa_mostrar)
            if mejor_placa is not None and placa_mostrar:
                placas_asignadas_este_frame.add(tuple(mejor_placa))
                x1p, y1p, x2p, y2p = mejor_placa
                color_matricula = COLOR_MATRICULA_VALIDA if es_valida_placa_mostrar else COLOR_MATRICULA_INVALIDA
                (mw, mh), mb = cv2.getTextSize(placa_mostrar, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame_annotated, (x1p, y2p), (x1p + mw, y2p + mh + mb + 5), color_matricula, -1)
                cv2.putText(frame_annotated, placa_mostrar, (x1p, y2p + mh + mb), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Infracción por polígono
            superpuesto, bbox_infraccion = bbox_superpone_poligono(vbox, poligono_cruce)
            if superpuesto and bbox_infraccion is not None:
                if track_id not in vehiculos_sobre_cruce:
                    vehiculos_sobre_cruce[track_id] = frame_idx
                tiempo = frame_idx - vehiculos_sobre_cruce[track_id]
                placa_asociada = placas_por_vehiculo.get(track_id)
                es_valida = False
                if placa_asociada:
                    es_valida, _ = validar_formato_placa_peruana(placa_asociada)
                if tiempo >= tiempo_espera_frames and placa_asociada and es_valida and len(placa_asociada) >= 6:
                    evidencia = {
                        'frame': frame_annotated.copy(),
                        'frame_idx': frame_idx,
                        'filename': f'infraccion_frame{frame_idx}_veh{track_id}.jpg'
                    }
                    mejor_evidencia_por_vehiculo[track_id] = evidencia
            else:
                if track_id in vehiculos_sobre_cruce:
                    if track_id in mejor_evidencia_por_vehiculo and track_id not in vehiculos_infraccionados and placas_por_vehiculo.get(track_id) not in placas_infraccionadas:
                        evidencia = mejor_evidencia_por_vehiculo[track_id]
                        out_path = os.path.join(dir_frames_infraccion, evidencia['filename'])
                        cv2.imwrite(out_path, evidencia['frame'])
                        evidencias_guardadas.append(os.path.basename(out_path))
                        placas_infraccionadas.add(placas_por_vehiculo.get(track_id))
                        vehiculos_infraccionados.add(track_id)
                    del vehiculos_sobre_cruce[track_id]
                    if track_id in mejor_evidencia_por_vehiculo:
                        del mejor_evidencia_por_vehiculo[track_id]

        out.write(frame_annotated)
        frame_idx += 1

    cap.release(); out.release()

    resultado_json = {"infracciones": len(vehiculos_infraccionados), "evidencias": evidencias_guardadas}
    with open(os.path.splitext(video_salida)[0] + '_infracciones.json', 'w', encoding='utf-8') as f:
        json.dump(resultado_json, f, ensure_ascii=False)
    return resultado_json


def procesar_video_infraccion_semaforo(video_entrada: str, video_salida: str, dir_frames_infraccion: str, tiempo_tolerancia_segundos: float = 3.0):
    os.makedirs(os.path.dirname(video_salida), exist_ok=True)
    os.makedirs(dir_frames_infraccion, exist_ok=True)

    modelo_vehiculos = YOLO(RUTA_MODELO_VEHICULOS)
    modelo_placas = YOLO(RUTA_MODELO_PLACAS)
    modelo_caracteres = YOLO(RUTA_MODELO_CARACTERES)
    modelo_semaforos = YOLO(RUTA_MODELO_SEMAFOROS)
    color_detector = SemaphoreColorDetector(min_pixel_threshold=50)

    cap = cv2.VideoCapture(video_entrada)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_entrada}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Preferir H.264 (avc1) para compatibilidad HTML5; fallback a MPEG-4 (mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(video_salida, fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_salida, fourcc, fps, (width, height))

    # Polígono automático del cruce
    ret, frame0 = cap.read()
    if not ret:
        cap.release(); out.release()
        raise RuntimeError("No se pudo leer el primer frame para definir polígono")
    poligono_cruce = _seleccionar_poligono_auto(frame0).astype(np.int32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    vehiculos_sobre_cruce = {}
    vehiculos_infraccionados = set()
    placas_infraccionadas = set()
    placas_por_vehiculo = {}
    evidencias_guardadas = []
    frames_tolerancia = int(tiempo_tolerancia_segundos * fps)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_annotated = frame.copy()

        # Dibujar polígono
        cv2.polylines(frame_annotated, [poligono_cruce], isClosed=True, color=(128, 0, 128), thickness=2)
        overlay = frame_annotated.copy()
        cv2.fillPoly(overlay, [poligono_cruce], color=(128, 0, 128))
        cv2.addWeighted(overlay, 0.2, frame_annotated, 0.8, 0, frame_annotated)

        # Overlay de texto para mostrar la tolerancia fija de semáforo
        texto_tolerancia = f"Tolerancia semáforo: {tiempo_tolerancia_segundos:.1f}s"
        (tw, th), tb = cv2.getTextSize(texto_tolerancia, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        x, y = 15, 30
        overlay2 = frame_annotated.copy()
        cv2.rectangle(overlay2, (x-10, y-25), (x-10 + tw + 20, y-25 + th + tb + 20), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.35, frame_annotated, 0.65, 0, frame_annotated)
        cv2.putText(frame_annotated, texto_tolerancia, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


        # Semáforos y color
        semaforo_results = modelo_semaforos.predict(frame, conf=get_conf_semaforos())[0]
        semaforo_rojo_activo = False
        semaforo_amarillo_activo = False
        semaforo_detectado = False
        if semaforo_results.boxes is not None:
            for box in semaforo_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                color_semaforo = color_detector.get_dominant_color(hsv_roi, roi)
                semaforo_detectado = True
                if color_semaforo == "Rojo":
                    semaforo_rojo_activo = True
                elif color_semaforo == "Amarillo":
                    semaforo_amarillo_activo = True
                color_bbox = (0, 0, 255) if color_semaforo == "Rojo" else (0, 255, 0) if color_semaforo == "Verde" else (0, 255, 255)
                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color_bbox, 2)

        # Vehículos
        results = modelo_vehiculos.track(frame, persist=True, conf=0.1)[0]
        track_ids = results.boxes.id if results.boxes is not None else None
        vehiculos_actuales = {}
        if results.boxes is not None and track_ids is not None and len(results.boxes) > 0:
            if np.isscalar(track_ids):
                track_ids = [track_ids]
            for i in range(len(results.boxes)):
                box = results.boxes[i]
                tid = track_ids[i]
                track_id = int(tid.item()) if isinstance(tid, torch.Tensor) else int(tid)
                x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy().astype(int) if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0]).astype(int))
                vehiculos_actuales[track_id] = [x1, y1, x2, y2]
                class_id = int(box.cls[0].item())
                etiqueta_espanol, color = obtener_etiqueta_y_color(modelo_vehiculos.names[class_id])
                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)

        # Placas
        frame_annotated, boxes_placa = procesar_placas_frame(modelo_placas, frame_annotated, conf_threshold=get_conf_placas())
        placas_asignadas_este_frame = set()

        for track_id, vbox in vehiculos_actuales.items():
            # Placas candidatas
            placas_candidatas = []
            vx1, vy1, vx2, vy2 = vbox
            for placa_box in boxes_placa:
                px1, py1, px2, py2 = placa_box
                ix1, iy1 = max(vx1, px1), max(vy1, py1)
                ix2, iy2 = min(vx2, px2), min(vy2, py2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = iw * ih
                p_area = (px2 - px1) * (py2 - py1)
                if p_area > 0 and inter_area / p_area >= 0.6:
                    placas_candidatas.append(placa_box)
            mejor_placa = None; mejor_matricula = None; mejor_es_valida = False; mejor_dist = float('inf')
            for placa_box in placas_candidatas:
                if tuple(placa_box) in placas_asignadas_este_frame:
                    continue
                x1p, y1p, x2p, y2p = placa_box
                placa_img = frame[y1p:y2p, x1p:x2p]
                if placa_img.size > 0:
                    matricula, _, es_valida, _ = reconocer_caracteres_placa(modelo_caracteres, placa_img, conf_threshold=get_conf_caracteres())
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
            if track_id not in placas_por_vehiculo and mejor_matricula and mejor_es_valida:
                placas_por_vehiculo[track_id] = mejor_matricula
            placa_mostrar = placas_por_vehiculo.get(track_id, mejor_matricula)
            es_valida_placa_mostrar = False
            if placa_mostrar:
                es_valida_placa_mostrar, _ = validar_formato_placa_peruana(placa_mostrar)
            if mejor_placa is not None and placa_mostrar:
                placas_asignadas_este_frame.add(tuple(mejor_placa))
                x1p, y1p, x2p, y2p = mejor_placa
                color_matricula = COLOR_MATRICULA_VALIDA if es_valida_placa_mostrar else COLOR_MATRICULA_INVALIDA
                (mw, mh), mb = cv2.getTextSize(placa_mostrar, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame_annotated, (x1p, y2p), (x1p + mw, y2p + mh + mb + 5), color_matricula, -1)
                cv2.putText(frame_annotated, placa_mostrar, (x1p, y2p + mh + mb), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Infracción condicionada por semáforo
            superpuesto, bbox_infraccion = bbox_superpone_poligono(vbox, poligono_cruce)
            if superpuesto and bbox_infraccion is not None and (semaforo_rojo_activo or semaforo_amarillo_activo):
                if track_id not in vehiculos_sobre_cruce:
                    vehiculos_sobre_cruce[track_id] = frame_idx
                tiempo_en_cruce = frame_idx - vehiculos_sobre_cruce[track_id]
                if tiempo_en_cruce >= frames_tolerancia:
                    placa_asociada = placas_por_vehiculo.get(track_id)
                    es_valida = False
                    if placa_asociada:
                        es_valida, _ = validar_formato_placa_peruana(placa_asociada)
                    if placa_asociada and es_valida and len(placa_asociada) >= 6 and track_id not in vehiculos_infraccionados and placa_asociada not in placas_infraccionadas:
                        tipo = 'rojo' if semaforo_rojo_activo else 'amarillo'
                        filename = f'infraccion_semaforo_{tipo}_{placa_asociada}_frame{frame_idx}_veh{track_id}.jpg'
                        out_path = os.path.join(dir_frames_infraccion, filename)
                        cv2.imwrite(out_path, frame_annotated)
                        evidencias_guardadas.append(os.path.basename(out_path))
                        placas_infraccionadas.add(placa_asociada)
                        vehiculos_infraccionados.add(track_id)
            else:
                if track_id in vehiculos_sobre_cruce:
                    del vehiculos_sobre_cruce[track_id]

        out.write(frame_annotated)
        frame_idx += 1

    cap.release(); out.release()

    resultado_json = {"infracciones": len(vehiculos_infraccionados), "evidencias": evidencias_guardadas}
    with open(os.path.splitext(video_salida)[0] + '_infracciones.json', 'w', encoding='utf-8') as f:
        json.dump(resultado_json, f, ensure_ascii=False)
    return resultado_json