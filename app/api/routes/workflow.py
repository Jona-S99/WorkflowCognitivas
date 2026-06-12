import threading
import time
import uuid

from fastapi import APIRouter, Body

from app.api.runtime import (
    ALLOWED_DOC_EXTENSIONS,
    CSV_DIR,
    DOCS_DIR,
    WORKFLOW_LOCK,
    WORKFLOW_STATE,
    append_workflow_log,
    list_files,
    remove_stale_markdown_files,
    run_workflow_job,
    snapshot_workflow_state,
)


router = APIRouter(prefix="/workflow")


@router.post("/start")
async def workflow_start(payload: dict = Body(default=None)):
    payload = payload or {}
    contexto_estudio = str(payload.get("contexto_estudio", "")).strip()

    if not contexto_estudio:
        return {
            "success": False,
            "message": "Debes ingresar el contexto del estudio antes de codificar.",
        }

    # Los Markdown son derivados de las transcripciones. Si una carga se borró,
    # el workflow no debe arrancar con restos de conversiones anteriores.
    removed_markdown = remove_stale_markdown_files()
    if removed_markdown:
        append_workflow_log(
            f"Se eliminaron {removed_markdown} Markdown huérfano(s) antes de codificar.",
            "warning",
        )

    docs = list_files(DOCS_DIR, ALLOWED_DOC_EXTENSIONS)
    csv_files = list_files(CSV_DIR, {".csv"})

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
    append_workflow_log("Solicitud de codificación recibida desde la interfaz.")

    threading.Thread(
        target=run_workflow_job,
        args=(job_id, contexto_estudio),
        daemon=True,
    ).start()

    return {
        "success": True,
        "job_id": job_id,
        "message": "Codificación iniciada.",
    }


@router.get("/status")
async def workflow_status():
    return snapshot_workflow_state()
