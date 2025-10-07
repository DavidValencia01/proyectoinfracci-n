import sys
import os
import cv2
import subprocess
import numpy as np
import json
from pathlib import Path
from ultralytics import YOLO
import torch
from procesar_caracteres_en_placas import reconocer_caracteres_placa, validar_formato_placa_peruana
from procesar_vehiculos import obtener_etiqueta_y_color
from procesar_placas_vehiculares import procesar_frame as procesar_placas_frame
from procesar_semaforos import SemaphoreColorDetector

# Rutas de los modelos
RUTA_MODELO_VEHICULOS = 'runs/detect/vehicle_detection_model/weights/best.pt'
RUTA_MODELO_CRUCES = 'runs/detect/crosswalk_detector_v1/weights/best.pt'
RUTA_MODELO_PLACAS = 'runs/detect/license_plate_detector_v1/weights/best.pt'
RUTA_MODELO_CARACTERES = 'runs/detect/plate_characters_detector_v1/weights/best.pt'
RUTA_MODELO_SEMAFOROS = 'runs/detect/traffic_lights_detector_v2/weights/best.pt'

# Colores
COLOR_CRUCE = (128, 0, 128)
COLOR_INFRACCION = (0, 0, 255)
COLOR_INFRACCION_SEMAFORO = (0, 0, 255)
COLOR_INFRACCION_CRUCE = (0, 128, 255)

# Configuración
TIEMPO_TOLERANCIA_INFRACCION = 1.0  # segundos


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
    puntos_orig = [(int(x/scale), int(y/scale)) for (x, y) in puntos]
    return np.array(puntos_orig, np.int32)

def bbox_superpone_poligono(bbox, poligono):
    poligono = poligono.astype(np.int32)
    x1, y1, x2, y2 = bbox
    box_pts = [(int(x1), int(y1)), (int(x2), int(y1)), (int(x2), int(y2)), (int(x1), int(y2))]
    for pt_x, pt_y in box_pts:
        if cv2.pointPolygonTest(poligono, (pt_x, pt_y), False) >= 0:
            return True, bbox
    return False, None

def procesar_video_multi_infraccion(input_path, output_path, infracciones):
    modelos = {}
    if 'cruce' in infracciones or 'semaforo' in infracciones:
        modelos['vehiculos'] = YOLO(RUTA_MODELO_VEHICULOS)
        modelos['cruces'] = YOLO(RUTA_MODELO_CRUCES)
        modelos['placas'] = YOLO(RUTA_MODELO_PLACAS)
        modelos['caracteres'] = YOLO(RUTA_MODELO_CARACTERES)
    if 'semaforo' in infracciones:
        modelos['semaforos'] = YOLO(RUTA_MODELO_SEMAFOROS)
        color_detector = SemaphoreColorDetector(min_pixel_threshold=50)
    else:
        color_detector = None
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {input_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    temp_output = str(Path(output_path).with_suffix('.temp.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    # Selección de polígono
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el primer frame del video.")
        return
    print("Por favor, seleccione 4 puntos del cruce peatonal en la ventana (sentido horario o antihorario)")
    poligono_cruce = seleccionar_poligono_manual(frame)
    poligono_cruce = poligono_cruce.astype(np.int32)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    # Contadores
    contador = {k: 0 for k in infracciones}
    vehiculos_sobre_cruce = {}
    vehiculos_infraccionados = set()
    placas_infraccionadas = set()
    placas_por_vehiculo = {}
    infracciones_list = []
    frame_idx = 0
    frames_tolerancia = int(TIEMPO_TOLERANCIA_INFRACCION * fps)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_annotated = frame.copy()
        # Dibujar polígono
        cv2.polylines(frame_annotated, [poligono_cruce], isClosed=True, color=COLOR_CRUCE, thickness=2)
        overlay = frame_annotated.copy()
        cv2.fillPoly(overlay, [poligono_cruce], color=COLOR_CRUCE)
        cv2.addWeighted(overlay, 0.2, frame_annotated, 0.8, 0, frame_annotated)
        # SEMÁFORO
        semaforo_rojo_activo = False
        semaforo_amarillo_activo = False
        semaforo_detectado = False
        if 'semaforo' in infracciones and 'semaforos' in modelos:
            semaforo_results = modelos['semaforos'].predict(frame, conf=0.3)[0]
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
                    label_semaforo = f"Semaforo {color_semaforo}"
                    cv2.putText(frame_annotated, label_semaforo, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bbox, 2)
        # VEHÍCULOS
        results = modelos['vehiculos'].track(frame, persist=True, conf=0.1)[0]
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
                class_name = modelos['vehiculos'].names[class_id]
                etiqueta_espanol, color = obtener_etiqueta_y_color(class_name)
                cv2.rectangle(frame_annotated, (x1, y1), (x2, y2), color, 2)
                texto = f"ID {track_id}: {etiqueta_espanol}"
                cv2.putText(frame_annotated, texto, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        # PLACAS
        frame_annotated, boxes_placa = procesar_placas_frame(
            modelos['placas'], frame_annotated, conf_threshold=0.25)
        placas_asignadas_este_frame = set()
        for track_id, vbox in vehiculos_actuales.items():
            # Placas asociadas
            placas_candidatas = []
            vx1, vy1, vx2, vy2 = vbox
            v_area = (vx2 - vx1) * (vy2 - vy1)
            for placa_box in boxes_placa:
                px1, py1, px2, py2 = placa_box
                ix1, iy1 = max(vx1, px1), max(vy1, py1)
                ix2, iy2 = min(vx2, px2), min(vy2, py2)
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = iw * ih
                p_area = (px2 - px1) * (py2 - py1)
                if p_area > 0 and inter_area / p_area >= 0.6:
                    placas_candidatas.append(placa_box)
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
                        modelos['caracteres'], placa_img, conf_threshold=0.15)
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
                color_matricula = (0, 255, 0) if es_valida_placa_mostrar else (0, 0, 255)
                cv2.rectangle(frame_annotated, (x1p, y2p), (x1p + 120, y2p + 30), color_matricula, -1)
                cv2.putText(frame_annotated, placa_mostrar, (x1p, y2p + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            # --- Lógica de infracción combinada ---
            superpuesto, bbox_infraccion = bbox_superpone_poligono(vbox, poligono_cruce)
            if superpuesto and bbox_infraccion is not None:
                if track_id not in vehiculos_sobre_cruce:
                    vehiculos_sobre_cruce[track_id] = frame_idx
                tiempo = frame_idx - vehiculos_sobre_cruce[track_id]
                tiempo_segundos = tiempo / fps
                placa_asociada = placas_por_vehiculo.get(track_id)
                es_valida = False
                if placa_asociada:
                    es_valida, _ = validar_formato_placa_peruana(placa_asociada)
                # Infracción por cruce peatonal
                if 'cruce' in infracciones:
                    if (tiempo >= frames_tolerancia and placa_asociada and es_valida and len(placa_asociada) >= 6 and track_id not in vehiculos_infraccionados and placa_asociada not in placas_infraccionadas):
                        contador['cruce'] += 1
                        vehiculos_infraccionados.add(track_id)
                        placas_infraccionadas.add(placa_asociada)
                        texto = f'INFRACCION CRUCE: >{TIEMPO_TOLERANCIA_INFRACCION:.1f}s sobre cruce | Placa: {placa_asociada}'
                        cv2.putText(frame_annotated, texto, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_INFRACCION_CRUCE, 2)
                        nombre_archivo = f'infraccion_cruce_{placa_asociada}_frame{frame_idx}_veh{track_id}.jpg'
                        cv2.imwrite(str(Path(output_path).parent / nombre_archivo), frame_annotated)
                        # Agregar a la lista de infracciones
                        infracciones_list.append({
                            'placa': placa_asociada,
                            'vehiculo': etiqueta_espanol if 'etiqueta_espanol' in locals() else 'Desconocido',
                            'tipo': 'Cruce',
                            'monto': '58.48'
                        })
                # Infracción por semáforo
                if 'semaforo' in infracciones and (semaforo_rojo_activo or semaforo_amarillo_activo):
                    if (tiempo >= frames_tolerancia and placa_asociada and es_valida and len(placa_asociada) >= 6 and track_id not in vehiculos_infraccionados and placa_asociada not in placas_infraccionadas):
                        contador['semaforo'] += 1
                        vehiculos_infraccionados.add(track_id)
                        placas_infraccionadas.add(placa_asociada)
                        tipo = 'ROJO' if semaforo_rojo_activo else 'AMARILLO'
                        texto = f'INFRACCION SEMAFORO {tipo}: >{TIEMPO_TOLERANCIA_INFRACCION:.1f}s sobre cruce con luz {tipo.lower()} | Placa: {placa_asociada}'
                        cv2.putText(frame_annotated, texto, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_INFRACCION_SEMAFORO, 2)
                        nombre_archivo = f'infraccion_semaforo_{tipo.lower()}_{placa_asociada}_frame{frame_idx}_veh{track_id}.jpg'
                        cv2.imwrite(str(Path(output_path).parent / nombre_archivo), frame_annotated)
                        # Agregar a la lista de infracciones
                        infracciones_list.append({
                            'placa': placa_asociada,
                            'vehiculo': etiqueta_espanol if 'etiqueta_espanol' in locals() else 'Desconocido',
                            'tipo': f'Semáforo {tipo.capitalize()}',
                            'monto': '58.48'
                        })
            else:
                if track_id in vehiculos_sobre_cruce:
                    del vehiculos_sobre_cruce[track_id]
        out.write(frame_annotated)
        frame_idx += 1
    cap.release()
    out.release()
    # Guardar contador en .json
    json_path = str(Path(output_path).with_suffix('.json'))
    with open(json_path, 'w') as f:
        json.dump({'contador': contador, 'infracciones': infracciones_list}, f)
    # Convertir a H.264 para compatibilidad web
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_output,
        '-vcodec', 'libx264', '-acodec', 'aac', output_path
    ])
    os.remove(temp_output)
    print(f"Video final guardado en: {output_path}")
    print(f"Conteo de infracciones: {contador}")

def main():
    if len(sys.argv) < 4:
        print("Uso: python procesar_multi_infraccion.py <input_video> <output_video> <infracciones separadas por coma>")
        print("Ejemplo: python procesar_multi_infraccion.py entrada.mp4 salida.mp4 cruce,semaforo")
        return
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    infracciones = sys.argv[3].split(',')
    procesar_video_multi_infraccion(input_path, output_path, infracciones)

if __name__ == "__main__":
    main() 