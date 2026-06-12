from pathlib import Path
import hashlib

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.runtime import (
    ALLOWED_DOC_EXTENSIONS,
    CSV_DIR,
    DOCS_DIR,
    MARKDOWN_DIR,
    clean_filename_stem,
    list_files,
)


router = APIRouter(prefix="/upload")


def _safe_uploaded_path(directory: Path, filename: str) -> Path:
    """Resuelve un archivo subido sin permitir escapes fuera del directorio."""
    directory.mkdir(parents=True, exist_ok=True)

    # El nombre llega desde la URL, por eso se descartan separadores o rutas
    # completas. Así una petición maliciosa no puede borrar archivos externos.
    clean_name = Path(filename).name
    candidate = (directory / clean_name).resolve()
    directory_root = directory.resolve()

    if candidate.parent != directory_root:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido.")

    return candidate


def _delete_file_if_allowed(
    directory: Path,
    filename: str,
    allowed_extensions: set[str],
) -> dict:
    """Borra un archivo persistido solo si pertenece al tipo esperado."""
    file_path = _safe_uploaded_path(directory, filename)

    if file_path.suffix.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Formato de archivo inválido.")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")

    file_path.unlink()
    return {"success": True, "message": f"Archivo '{file_path.name}' eliminado."}


def _delete_files_in_directory(directory: Path, allowed_extensions: set[str]) -> int:
    """Borra archivos de trabajo conocidos y deja intacto cualquier otro tipo."""
    deleted = 0
    directory.mkdir(parents=True, exist_ok=True)

    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            file_path.unlink()
            deleted += 1

    return deleted


def _delete_markdown_for_document(document_path: Path) -> bool:
    """Borra el Markdown derivado de una transcripción si ya existe."""
    markdown_path = MARKDOWN_DIR / f"{document_path.stem}.md"

    if markdown_path.exists() and markdown_path.is_file():
        markdown_path.unlink()
        return True

    return False


@router.get("/state")
async def upload_state():
    docs = list_files(DOCS_DIR, ALLOWED_DOC_EXTENSIONS)
    csv_files = list_files(CSV_DIR, {".csv"})

    return {
        "success": True,
        "documents": docs,
        "csv": csv_files[0] if csv_files else None,
    }


@router.post("/document")
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

        if not contents:
            return {
                "success": False,
                "message": "El archivo está vacío y no se puede cargar.",
            }

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

        clean_name = clean_filename_stem(file.filename)
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


@router.post("/csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        return {"success": False, "message": "Solo se aceptan archivos CSV"}

    try:
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        contents = await file.read()

        if not contents:
            return {
                "success": False,
                "message": "El archivo CSV está vacío y no se puede cargar.",
            }

        file_hash = hashlib.sha256(contents).hexdigest()
        hash_short = file_hash[:8]

        clean_name = clean_filename_stem(file.filename)
        safe_filename = f"{clean_name}_{hash_short}.csv"
        file_path = CSV_DIR / safe_filename

        # Regla de negocio: la pauta es única. Recién después de validar el CSV
        # se reemplazan pautas previas para evitar perder la actual por error.
        replaced_count = _delete_files_in_directory(CSV_DIR, {".csv"})

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
            "replaced_count": replaced_count,
        }

    except Exception as e:
        return {
            "success": False,
            "message": f"Error al guardar el archivo CSV: {str(e)}",
        }


@router.delete("/document/{filename}")
async def delete_document(filename: str):
    file_path = _safe_uploaded_path(DOCS_DIR, filename)
    result = _delete_file_if_allowed(DOCS_DIR, filename, ALLOWED_DOC_EXTENSIONS)
    result["deleted_markdown"] = _delete_markdown_for_document(file_path)
    return result


@router.delete("/documents")
async def delete_documents():
    deleted_count = _delete_files_in_directory(DOCS_DIR, ALLOWED_DOC_EXTENSIONS)
    deleted_markdown_count = _delete_files_in_directory(MARKDOWN_DIR, {".md"})
    return {
        "success": True,
        "message": f"Se eliminaron {deleted_count} transcripción(es).",
        "deleted_count": deleted_count,
        "deleted_markdown_count": deleted_markdown_count,
    }


@router.delete("/csv")
async def delete_csv():
    deleted_count = _delete_files_in_directory(CSV_DIR, {".csv"})
    return {
        "success": True,
        "message": f"Se eliminaron {deleted_count} pauta(s) CSV.",
        "deleted_count": deleted_count,
    }
