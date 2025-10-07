from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from typing import List, Optional
import shutil
import time
import uuid
import os
import sys
import asyncio

# Ajuste de Event Loop en Windows para evitar errores de Proactor (WinError 10054) al cerrar conexiones
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# Añadir site-packages local para importar cv2/ultralytics cuando se ejecuta con Python del sistema
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / "Lib" / "site-packages"))

# Importar lógica existente de procesamiento
from procesar_multi_modelo import procesar_video_multi
from infracciones import procesar_video_infraccion_cruce, procesar_video_infraccion_semaforo
import mimetypes

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
TEMPLATES_DIR = BASE_DIR / "templates"

# Crear directorios necesarios
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Detección de Infracciones de Tránsito - YOLOv8")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Modelos disponibles
AVAILABLE_MODELS = [
    {"key": "vehiculos", "label": "Vehículos"},
    {"key": "semaforo", "label": "Semáforos"},
    {"key": "cruces", "label": "Cruces peatonales"},
    {"key": "placas", "label": "Placas vehiculares"},
    {"key": "caracteres", "label": "Caracteres de placas"}
]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return RedirectResponse("/deteccion", status_code=302)

@app.get("/deteccion", response_class=HTMLResponse)
async def deteccion_page(request: Request):
    return templates.TemplateResponse(
        "deteccion.html",
        {
            "request": request,
            "available_models": AVAILABLE_MODELS,
            "error": None,
            "result_video_path": None,
            "counters": None,
        }
    )

@app.get("/infraccion", response_class=HTMLResponse)
async def infraccion_page(request: Request):
    return templates.TemplateResponse(
        "infraccion.html",
        {
            "request": request,
            "available_models": AVAILABLE_MODELS,
            "error": None,
            "infraction_result_video_path": None,
            "infraction_counters": None,
            "infraction_evidences": None,
        }
    )

@app.post("/process", response_class=HTMLResponse)
async def process_video(
    request: Request,
    file: UploadFile = File(...),
    opciones: Optional[List[str]] = Form(None)
):
    # Validar opciones
    opciones = opciones or []
    opciones = [opt for opt in opciones if opt in [m["key"] for m in AVAILABLE_MODELS]]
    if not opciones:
        return templates.TemplateResponse(
            "deteccion.html",
            {
                "request": request,
                "available_models": AVAILABLE_MODELS,
                "error": "Selecciona al menos un modelo para procesar.",
                "result_video_path": None,
                "counters": None
            }
        )

    # Guardar archivo subido
    unique_id = uuid.uuid4().hex
    input_ext = Path(file.filename).suffix or ".mp4"
    input_name = f"input_{unique_id}{input_ext}"
    output_name = f"output_{unique_id}.mp4"
    input_path = UPLOAD_DIR / input_name
    output_path = OUTPUT_DIR / output_name

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Procesar video con los modelos seleccionados
    try:
        procesar_video_multi(str(input_path), str(output_path), opciones)
    except Exception as e:
        # Si ocurre un error, mostrarlo al usuario
        return templates.TemplateResponse(
            "deteccion.html",
            {
                "request": request,
                "available_models": AVAILABLE_MODELS,
                "error": f"Error al procesar el video: {e}",
                "result_video_path": None,
                "counters": None
            }
        )

    # Leer contadores del archivo JSON generado
    counters = None
    json_path = output_path.with_suffix('.json')
    if json_path.exists():
        try:
            import json
            with json_path.open('r', encoding='utf-8') as f:
                counters = json.load(f)
        except Exception:
            counters = None

    # Renderizar resultado
    return templates.TemplateResponse(
        "deteccion.html",
        {
            "request": request,
            "available_models": AVAILABLE_MODELS,
            "result_video_path": f"/files/{output_name}",
            "counters": counters,
            "infraction_result_video_path": None,
            "infraction_counters": None,
            "infraction_evidences": None
        }
    )

@app.post("/procesar_infraccion", response_class=HTMLResponse)
async def procesar_infraccion(
    request: Request,
    file: UploadFile = File(...),
    tipo: str = Form(...),
    tolerancia: Optional[str] = Form(None),
    poligono_modo: Optional[str] = Form(None)
):
    tipo = (tipo or "").lower()
    if tipo not in ["cruce", "semaforo"]:
        return templates.TemplateResponse(
            "infraccion.html",
            {
                "request": request,
                "available_models": AVAILABLE_MODELS,
                "error": "Selecciona un tipo de infracción válido (cruce o semáforo).",
                "result_video_path": None,
                "counters": None,
                "infraction_result_video_path": None,
                "infraction_counters": None,
                "infraction_evidences": None
            }
        )

    unique_id = uuid.uuid4().hex
    input_ext = Path(file.filename).suffix or ".mp4"
    input_name = f"infr_{tipo}_input_{unique_id}{input_ext}"
    output_name = f"infr_{tipo}_output_{unique_id}.mp4"
    input_path = UPLOAD_DIR / input_name
    output_path = OUTPUT_DIR / output_name

    evid_dir_rel = Path("evidencias") / unique_id
    evid_dir = OUTPUT_DIR / evid_dir_rel
    evid_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        if tipo == "cruce":
            auto_flag = True if (poligono_modo or "auto").lower() == "auto" else False
            res = procesar_video_infraccion_cruce(str(input_path), str(output_path), str(evid_dir), auto_polygon=auto_flag)
        else:
            tol = 3.0
            res = procesar_video_infraccion_semaforo(str(input_path), str(output_path), str(evid_dir), tiempo_tolerancia_segundos=tol)
    except Exception as e:
        return templates.TemplateResponse(
            "infraccion.html",
            {
                "request": request,
                "available_models": AVAILABLE_MODELS,
                "error": f"Error al procesar infracción: {e}",
                "result_video_path": None,
                "counters": None,
                "infraction_result_video_path": None,
                "infraction_counters": None,
                "infraction_evidences": None
            }
        )

    return templates.TemplateResponse(
        "infraccion.html",
        {
            "request": request,
            "available_models": AVAILABLE_MODELS,
            "result_video_path": None,
            "counters": None,
            "infraction_result_video_path": f"/files/{output_name}",
            "infraction_counters": res,
            "infraction_evidences": [str(evid_dir_rel / e) for e in res.get("evidencias", [])]
        }
    )

@app.get("/files/{filepath:path}")
async def get_file(filepath: str):
    path = OUTPUT_DIR / filepath
    if not path.exists():
        return RedirectResponse("/", status_code=302)
    # Asegurar Content-Type correcto incluso si mimetypes falla en Windows
    ext = path.suffix.lower()
    default_mime = "application/octet-stream"
    if ext == ".mp4":
        default_mime = "video/mp4"
    elif ext in (".jpg", ".jpeg"):
        default_mime = "image/jpeg"
    elif ext == ".png":
        default_mime = "image/png"
    elif ext == ".gif":
        default_mime = "image/gif"
    elif ext == ".webm":
        default_mime = "video/webm"
    elif ext == ".json":
        default_mime = "application/json"
    mime, _ = mimetypes.guess_type(str(path))
    return FileResponse(str(path), media_type=mime or default_mime)