from pathlib import Path
import threading
import time

from fastapi.templating import Jinja2Templates


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


def clean_filename_stem(filename: str, max_len: int = 50) -> str:
    stem = Path(filename).stem
    clean_name = "".join(c for c in stem if c.isalnum() or c in (" ", "-", "_"))
    clean_name = clean_name.strip()[:max_len]
    return clean_name or "archivo"


def list_files(directory: Path, allowed_extensions: set[str]) -> list[dict]:
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


def snapshot_workflow_state() -> dict:
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


def append_workflow_log(message: str, level: str = "info") -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = {"timestamp": timestamp, "level": level, "message": message}
    print(f"[workflow][{timestamp}][{level}] {message}")

    with WORKFLOW_LOCK:
        WORKFLOW_STATE["logs"].append(entry)
        if len(WORKFLOW_STATE["logs"]) > 500:
            WORKFLOW_STATE["logs"] = WORKFLOW_STATE["logs"][-500:]


def read_latest_csv_text() -> str:
    csv_files = list_files(CSV_DIR, {".csv"})
    if not csv_files:
        raise FileNotFoundError("No se encontró archivo CSV para codificar.")

    latest_csv_name = csv_files[0]["name"]
    csv_path = CSV_DIR / latest_csv_name
    return csv_path.read_text(encoding="utf-8", errors="ignore")


def set_summary_from_files(file_items: list[dict]) -> dict:
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


def run_workflow_job(job_id: str, contexto_estudio: str) -> None:
    try:
        from app.workflow.agentic_workflow import workflow_app

        with WORKFLOW_LOCK:
            if WORKFLOW_STATE["job_id"] != job_id:
                return
            WORKFLOW_STATE["status"] = "running"
            WORKFLOW_STATE["message"] = "Iniciando codificación por archivo..."
        append_workflow_log("Job iniciado. Preparando lectura de CSV y transcripciones.")

        csv_text = read_latest_csv_text()
        append_workflow_log("CSV leído correctamente.")
        docs = list_files(DOCS_DIR, ALLOWED_DOC_EXTENSIONS)
        append_workflow_log(f"Se detectaron {len(docs)} transcripción(es) para procesar.")

        with WORKFLOW_LOCK:
            if WORKFLOW_STATE["job_id"] != job_id:
                return

            WORKFLOW_STATE["files"] = [
                {"name": doc["name"], "status": "pending", "error": None}
                for doc in docs
            ]
            WORKFLOW_STATE["summary"] = set_summary_from_files(
                WORKFLOW_STATE["files"]
            )
            WORKFLOW_STATE["message"] = "Iniciando codificación por archivo..."

        if not docs:
            with WORKFLOW_LOCK:
                if WORKFLOW_STATE["job_id"] != job_id:
                    return
                WORKFLOW_STATE["running"] = False
                WORKFLOW_STATE["status"] = "failed"
                WORKFLOW_STATE["message"] = "No hay transcripciones para codificar."
                WORKFLOW_STATE["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            append_workflow_log("No hay transcripciones para codificar.", "error")
            return

        for index, doc in enumerate(docs):
            doc_name = doc["name"]
            doc_path = DOCS_DIR / doc_name

            with WORKFLOW_LOCK:
                if WORKFLOW_STATE["job_id"] != job_id:
                    return
                WORKFLOW_STATE["files"][index]["status"] = "processing"
                WORKFLOW_STATE["summary"] = set_summary_from_files(
                    WORKFLOW_STATE["files"]
                )
                WORKFLOW_STATE["message"] = f"Codificando {doc_name}..."
            append_workflow_log(f"Iniciando archivo: {doc_name}")

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
                    WORKFLOW_STATE["summary"] = set_summary_from_files(
                        WORKFLOW_STATE["files"]
                    )
                append_workflow_log(f"Archivo completado: {doc_name}", "success")

            except Exception as workflow_error:
                with WORKFLOW_LOCK:
                    if WORKFLOW_STATE["job_id"] != job_id:
                        return
                    WORKFLOW_STATE["files"][index]["status"] = "error"
                    WORKFLOW_STATE["files"][index]["error"] = str(workflow_error)
                    WORKFLOW_STATE["summary"] = set_summary_from_files(
                        WORKFLOW_STATE["files"]
                    )
                append_workflow_log(
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
        append_workflow_log(
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
        append_workflow_log(f"Error general del workflow: {str(e)}", "error")
