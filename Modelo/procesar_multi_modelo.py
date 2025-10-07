import sys
import os
import cv2
import subprocess
from pathlib import Path
from ultralytics import YOLO
import json

def cargar_modelos(opciones):
    modelos = {}
    if 'vehiculos' in opciones:
        modelos['vehiculos'] = YOLO('runs/detect/vehicle_detection_model_binary2/weights/best.pt')
    if 'semaforo' in opciones:
        modelos['semaforo'] = YOLO('runs/detect/traffic_lights_detector_v2/weights/best.pt')
    if 'cruces' in opciones:
        modelos['cruces'] = YOLO('runs/detect/crosswalk_detector_v1/weights/best.pt')
    if 'placas' in opciones:
        modelos['placas'] = YOLO('runs/detect/license_plate_detector_v1/weights/best.pt')
    if 'caracteres' in opciones:
        modelos['caracteres'] = YOLO('runs/detect/plate_characters_detector_v1/weights/best.pt')
    return modelos

def procesar_video_multi(input_path, output_path, opciones):
    modelos = cargar_modelos(opciones)
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
    # Contadores
    contador = {k: 0 for k in opciones}
    # Frame skipping: procesar solo 1 de cada 5 frames
    SKIP_N = 5
    frame_idx = 0
    processed_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        annotated = frame.copy()
        if frame_idx % SKIP_N == 0:
            # Vehículos
            if 'vehiculos' in modelos:
                results = modelos['vehiculos'].track(frame, persist=True, conf=0.1)[0]
                if results.boxes is not None:
                    contador['vehiculos'] += len(results.boxes)
                    for i, box in enumerate(results.boxes):
                        class_id = int(box.cls[0].item())
                        class_name = modelos['vehiculos'].names[class_id]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(annotated, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Semáforos
            if 'semaforo' in modelos:
                results = modelos['semaforo'].predict(frame, conf=0.25)[0]
                if results.boxes is not None:
                    contador['semaforo'] += len(results.boxes)
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(annotated, 'Semaforo', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # Cruces peatonales
            if 'cruces' in modelos:
                results = modelos['cruces'].predict(frame, conf=0.25)[0]
                if results.boxes is not None:
                    contador['cruces'] += len(results.boxes)
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (128, 0, 128), 2)
                        cv2.putText(annotated, 'Cruce', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
            # Placas
            if 'placas' in modelos:
                results = modelos['placas'].predict(frame, conf=0.4)[0]
                if results.boxes is not None:
                    contador['placas'] += len(results.boxes)
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (203, 192, 255), 2)
                        cv2.putText(annotated, 'Placa', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (203, 192, 255), 2)
            processed_frames += 1
        # Escribir frame
        out.write(annotated)
        frame_idx += 1
    cap.release()
    out.release()
    # Guardar contador en .json + meta
    json_path = str(Path(output_path).with_suffix('.json'))
    contador['_meta'] = {
        'frames_total': frame_idx,
        'frames_procesados': processed_frames,
        'procesar_cada': SKIP_N
    }
    with open(json_path, 'w') as f:
        json.dump(contador, f)
    # Convertir a H.264 para compatibilidad web
    subprocess.run([
        'ffmpeg', '-y', '-i', temp_output,
        '-vcodec', 'libx264', '-acodec', 'aac', output_path
    ])
    os.remove(temp_output)
    print(f"Video final guardado en: {output_path}")

def main():
    if len(sys.argv) < 4:
        print("Uso: python procesar_multi_modelo.py <input_video> <output_video> <opciones separadas por coma>")
        print("Ejemplo: python procesar_multi_modelo.py entrada.mp4 salida.mp4 vehiculos,semaforo,cruces")
        return
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    opciones = sys.argv[3].split(',')
    procesar_video_multi(input_path, output_path, opciones)

if __name__ == "__main__":
    main()