from pathlib import Path
import hashlib
import threading
import time
import uuid

from fastapi import Body, FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Agente EC")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


DOCS_DIR = Path("app/docs/transcripciones")
CSV_DIR = Path("app/docs/csv")
ALLOWED_DOC_EXTENSIONS = {".doc", ".docx", ".pdf"}

WORKFLOW_LOCK = threading.Lock()
WORKFLOW_STATE = {
    "job_id": None,
    "running": False,
    "status": "idle",
    "message": "Sin codificación activa",
    "started_at": None,
    "finished_at": None,
    "summary": {
        "total": 0,
        "processed": 0,
        "completed": 0,
        "errors": 0,
        "processing": 0,
        "pending": 0,
    },
    "files": [],
    "logs": [],
}


def _clean_filename_stem(filename: str, max_len: int = 50) -> str:
    stem = Path(filename).stem
    clean_name = "".join(c for c in stem if c.isalnum() or c in (" ", "-", "_"))
    clean_name = clean_name.strip()[:max_len]
    return clean_name or "archivo"


def _list_files(directory: Path, allowed_extensions: set[str]) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []

    for file_path in directory.iterdir():
        if (
            not file_path.is_file()
            or file_path.suffix.lower() not in allowed_extensions
        ):
            continue

        stats = file_path.stat()
        items.append(
            {
                "name": file_path.name,
                "size": stats.st_size,
                "modified_at": stats.st_mtime,
            }
        )

    items.sort(key=lambda x: x["modified_at"], reverse=True)
    return items


def _snapshot_workflow_state() -> dict:
    with WORKFLOW_LOCK:
        return {
            "success": True,
            "job_id": WORKFLOW_STATE["job_id"],
            "running": WORKFLOW_STATE["running"],
            "status": WORKFLOW_STATE["status"],
            "message": WORKFLOW_STATE["message"],
            "started_at": WORKFLOW_STATE["started_at"],
            "finished_at": WORKFLOW_STATE["finished_at"],
            "summary": dict(WORKFLOW_STATE["summary"]),
            "files": [dict(item) for item in WORKFLOW_STATE["files"]],
            "logs": [dict(item) for item in WORKFLOW_STATE["logs"]],
        }


def _append_workflow_log(message: str, level: str = "info") -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": timestamp, "level": level, "message": message}
    print(f"[workflow][{timestamp}][{level}] {message}")

    with WORKFLOW_LOCK:
        WORKFLOW_STATE["logs"].append(entry)
        if len(WORKFLOW_STATE["logs"]) > 500:
            WORKFLOW_STATE["logs"] = WORKFLOW_STATE["logs"][-500:]


def _read_latest_csv_text() -> str:
    csv_files = _list_files(CSV_DIR, {".csv"})
    if not csv_files:
        raise FileNotFoundError("No se encontró archivo CSV para codificar.")

    latest_csv_name = csv_files[0]["name"]
    csv_path = CSV_DIR / latest_csv_name
    return csv_path.read_text(encoding="utf-8", errors="ignore")


def _set_summary_from_files(file_items: list[dict]) -> dict:
    total = len(file_items)
    completed = sum(1 for f in file_items if f["status"] == "completed")
    errors = sum(1 for f in file_items if f["status"] == "error")
    processing = sum(1 for f in file_items if f["status"] == "processing")
    pending = sum(1 for f in file_items if f["status"] == "pending")
    processed = completed + errors

    return {
        "total": total,
        "processed": processed,
        "completed": completed,
        "errors": errors,
        "processing": processing,
        "pending": pending,
    }


def _run_workflow_job(job_id: str, contexto_estudio: str) -> None:
    try:
        from app.workflow.agentic_workflow import workflow_app

        with WORKFLOW_LOCK:
            if WORKFLOW_STATE["job_id"] != job_id:
                return
            WORKFLOW_STATE["status"] = "running"
            WORKFLOW_STATE["message"] = "Iniciando codificación por archivo..."
        _append_workflow_log("Job iniciado. Preparando lectura de CSV y transcripciones.")

        csv_text = _read_latest_csv_text()
        _append_workflow_log("CSV leído correctamente.")
        docs = _list_files(DOCS_DIR, ALLOWED_DOC_EXTENSIONS)
        _append_workflow_log(f"Se detectaron {len(docs)} transcripción(es) para procesar.")

        with WORKFLOW_LOCK:
            if WORKFLOW_STATE["job_id"] != job_id:
                return

            WORKFLOW_STATE["files"] = [
                {"name": doc["name"], "status": "pending", "error": None}
                for doc in docs
            ]
            WORKFLOW_STATE["summary"] = _set_summary_from_files(WORKFLOW_STATE["files"])
            WORKFLOW_STATE["message"] = "Iniciando codificación por archivo..."

        if not docs:
            with WORKFLOW_LOCK:
                if WORKFLOW_STATE["job_id"] != job_id:
                    return
                WORKFLOW_STATE["running"] = False
                WORKFLOW_STATE["status"] = "failed"
                WORKFLOW_STATE["message"] = "No hay transcripciones para codificar."
                WORKFLOW_STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _append_workflow_log("No hay transcripciones para codificar.", "error")
            return

        for index, doc in enumerate(docs):
            doc_name = doc["name"]
            doc_path = DOCS_DIR / doc_name

            with WORKFLOW_LOCK:
                if WORKFLOW_STATE["job_id"] != job_id:
                    return
                WORKFLOW_STATE["files"][index]["status"] = "processing"
                WORKFLOW_STATE["summary"] = _set_summary_from_files(
                    WORKFLOW_STATE["files"]
                )
                WORKFLOW_STATE["message"] = f"Codificando {doc_name}..."
            _append_workflow_log(f"Iniciando archivo: {doc_name}")

            try:
                workflow_app.invoke(
                    {
                        "csv_text": csv_text,
                        "md_text": "",
                        "document_path": str(doc_path),
                        "contexto_estudio": contexto_estudio,
                        "matriz_vaciado_final": [],
                        "revision_matriz": {},
                    }
                )

                with WORKFLOW_LOCK:
                    if WORKFLOW_STATE["job_id"] != job_id:
                        return
                    WORKFLOW_STATE["files"][index]["status"] = "completed"
                    WORKFLOW_STATE["files"][index]["error"] = None
                    WORKFLOW_STATE["summary"] = _set_summary_from_files(
                        WORKFLOW_STATE["files"]
                    )
                _append_workflow_log(f"Archivo completado: {doc_name}", "success")

            except Exception as workflow_error:
                with WORKFLOW_LOCK:
                    if WORKFLOW_STATE["job_id"] != job_id:
                        return
                    WORKFLOW_STATE["files"][index]["status"] = "error"
                    WORKFLOW_STATE["files"][index]["error"] = str(workflow_error)
                    WORKFLOW_STATE["summary"] = _set_summary_from_files(
                        WORKFLOW_STATE["files"]
                    )
                _append_workflow_log(
                    f"Error en archivo {doc_name}: {str(workflow_error)}", "error"
                )

        with WORKFLOW_LOCK:
            if WORKFLOW_STATE["job_id"] != job_id:
                return

            summary = WORKFLOW_STATE["summary"]
            WORKFLOW_STATE["running"] = False
            WORKFLOW_STATE["status"] = (
                "completed_with_errors" if summary["errors"] > 0 else "completed"
            )
            WORKFLOW_STATE["message"] = (
                f"Codificación finalizada: {summary['completed']} completados, {summary['errors']} con error."
            )
            WORKFLOW_STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _append_workflow_log(
            f"Job finalizado. Completados={summary['completed']} Errores={summary['errors']}.",
            "success" if summary["errors"] == 0 else "warning",
        )

    except Exception as e:
        with WORKFLOW_LOCK:
            if WORKFLOW_STATE["job_id"] != job_id:
                return
            WORKFLOW_STATE["running"] = False
            WORKFLOW_STATE["status"] = "failed"
            WORKFLOW_STATE["message"] = f"Error al ejecutar workflow: {str(e)}"
            WORKFLOW_STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _append_workflow_log(f"Error general del workflow: {str(e)}", "error")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "home.html",
        {"request": request, "titulo_page": "Workflow EC - Inicio"},
    )


@app.get("/config", response_class=HTMLResponse)
async def config_page(request: Request):
    return templates.TemplateResponse(
        "config.html",
        {"request": request, "titulo_page": "Workflow EC - Configuración"},
    )


@app.get("/workflow", response_class=HTMLResponse)
async def workflow_page(request: Request):
    return templates.TemplateResponse(
        "workflow.html",
        {"request": request, "titulo_page": "Workflow EC - Flujo"},
    )


@app.get("/upload/state")
async def upload_state():
    docs = _list_files(DOCS_DIR, ALLOWED_DOC_EXTENSIONS)
    csv_files = _list_files(CSV_DIR, {".csv"})

    return {
        "success": True,
        "documents": docs,
        "csv": csv_files[0] if csv_files else None,
    }


@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in ALLOWED_DOC_EXTENSIONS:
        return {
            "success": False,
            "message": f"Formato no permitido. Solo se aceptan: {', '.join(sorted(ALLOWED_DOC_EXTENSIONS))}",
        }

    try:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        contents = await file.read()

        file_hash = hashlib.sha256(contents).hexdigest()
        hash_short = file_hash[:8]

        existing_files = list(DOCS_DIR.glob(f"*_{hash_short}{file_extension}"))
        if existing_files:
            return {
                "success": False,
                "message": f"⚠️ Archivo duplicado detectado. Ya existe: {existing_files[0].name}",
                "duplicate": True,
                "existing_file": existing_files[0].name,
            }

        clean_name = _clean_filename_stem(file.filename)
        safe_filename = f"{clean_name}_{hash_short}{file_extension}"
        file_path = DOCS_DIR / safe_filename

        with open(file_path, "wb") as f:
            f.write(contents)

        return {
            "success": True,
            "message": f"Archivo '{file.filename}' guardado exitosamente",
            "filename": safe_filename,
            "original_filename": file.filename,
            "size": len(contents),
            "path": str(file_path),
            "hash": file_hash,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error al guardar el archivo: {str(e)}",
        }


@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        return {"success": False, "message": "Solo se aceptan archivos CSV"}

    try:
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        contents = await file.read()

        file_hash = hashlib.sha256(contents).hexdigest()
        hash_short = file_hash[:8]

        existing_files = list(CSV_DIR.glob(f"*_{hash_short}.csv"))
        if existing_files:
            return {
                "success": False,
                "message": f"⚠️ Archivo CSV duplicado. Ya existe: {existing_files[0].name}",
                "duplicate": True,
                "existing_file": existing_files[0].name,
            }

        clean_name = _clean_filename_stem(file.filename)
        safe_filename = f"{clean_name}_{hash_short}.csv"
        file_path = CSV_DIR / safe_filename

        with open(file_path, "wb") as f:
            f.write(contents)

        return {
            "success": True,
            "message": f"Archivo CSV '{file.filename}' guardado exitosamente",
            "filename": safe_filename,
            "original_filename": file.filename,
            "size": len(contents),
            "path": str(file_path),
            "hash": file_hash,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error al guardar el archivo CSV: {str(e)}",
        }


@app.post("/workflow/start")
async def workflow_start(payload: dict = Body(default=None)):
    payload = payload or {}
    contexto_estudio = str(payload.get("contexto_estudio", "")).strip()

    if not contexto_estudio:
        return {
            "success": False,
            "message": "Debes ingresar el contexto del estudio antes de codificar.",
        }

    docs = _list_files(DOCS_DIR, ALLOWED_DOC_EXTENSIONS)
    csv_files = _list_files(CSV_DIR, {".csv"})

    if not docs:
        return {
            "success": False,
            "message": "No hay transcripciones cargadas para codificar.",
        }
    if not csv_files:
        return {"success": False, "message": "No hay CSV cargado para codificar."}

    with WORKFLOW_LOCK:
        if WORKFLOW_STATE["running"]:
            return {
                "success": False,
                "running": True,
                "message": "Ya existe una codificación en curso.",
            }

        job_id = str(uuid.uuid4())
        WORKFLOW_STATE["job_id"] = job_id
        WORKFLOW_STATE["running"] = True
        WORKFLOW_STATE["status"] = "starting"
        WORKFLOW_STATE["message"] = "Iniciando codificación..."
        WORKFLOW_STATE["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        WORKFLOW_STATE["finished_at"] = None
        WORKFLOW_STATE["files"] = []
        WORKFLOW_STATE["logs"] = []
        WORKFLOW_STATE["summary"] = {
            "total": 0,
            "processed": 0,
            "completed": 0,
            "errors": 0,
            "processing": 0,
            "pending": 0,
        }
    _append_workflow_log("Solicitud de codificación recibida desde la interfaz.")

    threading.Thread(
        target=_run_workflow_job,
        args=(job_id, contexto_estudio),
        daemon=True,
    ).start()

    return {
        "success": True,
        "job_id": job_id,
        "message": "Codificación iniciada.",
    }


@app.get("/workflow/status")
async def workflow_status():
    return _snapshot_workflow_state()
