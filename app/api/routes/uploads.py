from pathlib import Path
import hashlib

from fastapi import APIRouter, File, UploadFile

from app.api.runtime import (
    ALLOWED_DOC_EXTENSIONS,
    CSV_DIR,
    DOCS_DIR,
    clean_filename_stem,
    list_files,
)


router = APIRouter(prefix="/upload")


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

        clean_name = clean_filename_stem(file.filename)
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
