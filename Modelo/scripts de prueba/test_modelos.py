import cv2
import numpy as np
from ultralytics import YOLO
import os

# Rutas de los modelos
RUTA_MODELO_VEHICULOS = 'runs/detect/vehicle_detection_model/weights/best.pt'
RUTA_MODELO_CRUCES = 'runs/detect/crosswalk_detector_v1/weights/best.pt'
RUTA_MODELO_PLACAS = 'runs/detect/license_plate_detector_v1/weights/best.pt'
RUTA_MODELO_CARACTERES = 'runs/detect/plate_characters_detector_v1/weights/best.pt'
RUTA_MODELO_SEMAFOROS = 'runs/detect/traffic_lights_detector_v2/weights/best.pt'

def test_modelo(ruta_modelo, nombre_modelo):
    print(f"\n=== Probando {nombre_modelo} ===")
    if not os.path.exists(ruta_modelo):
        print(f"❌ ERROR: No se encuentra el modelo en {ruta_modelo}")
        return False
    
    try:
        modelo = YOLO(ruta_modelo)
        print(f"✅ Modelo cargado correctamente: {ruta_modelo}")
        
        # Crear una imagen de prueba (negro 640x640)
        img_test = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Hacer una predicción
        results = modelo.predict(img_test, conf=0.1)
        print(f"✅ Predicción exitosa - {len(results)} resultados")
        
        return True
    except Exception as e:
        print(f"❌ ERROR al cargar modelo: {str(e)}")
        return False

def main():
    print("=== PRUEBA DE MODELOS ===")
    
    # Probar cada modelo
    modelos = [
        (RUTA_MODELO_VEHICULOS, "Modelo de Vehículos"),
        (RUTA_MODELO_CRUCES, "Modelo de Cruces Peatonales"),
        (RUTA_MODELO_PLACAS, "Modelo de Placas"),
        (RUTA_MODELO_CARACTERES, "Modelo de Caracteres"),
        (RUTA_MODELO_SEMAFOROS, "Modelo de Semáforos")
    ]
    
    for ruta, nombre in modelos:
        test_modelo(ruta, nombre)
    
    print("\n=== FIN DE PRUEBAS ===")

if __name__ == '__main__':
    main() 