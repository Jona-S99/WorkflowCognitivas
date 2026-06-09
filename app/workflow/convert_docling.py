# Script para convertir documentos a formato markdown usando la librería docling

# Libreias
from pathlib import Path
from docling.document_converter import DocumentConverter


# Funcion para listar docx y pdf
def listar_documentos(directorio):
    """Función para listar archivos con extensiones específicas en un directorio."""
    # 1. Convertimos el texto del directorio en un Objeto Path
    ruta_base = Path(directorio)

    # 2. Definimos un Set (conjunto) con las extensiones
    # Usamos {} porque la búsqueda en un set es más rápida que en una lista
    extensiones = {".pdf", ".doc", ".docx"}

    # 3. Comprensión de lista (List Comprehension)
    # Para cada archivo en el directorio, si su sufijo está en mis
    # extensiones, dame su ruta absoluta
    archivos = [
        str(f.absolute())  # (C) Qué guardamos: La ruta absoluta como texto
        for f in ruta_base.iterdir()  # (A) De dónde: Iteramos sobre cada elemento del directorio
        if f.suffix.lower()
        in extensiones  # (B) Condición: Solo si el sufijo está en nuestro set
    ]

    return archivos


# Defino una funcion para poder llamarla desde el main.py
def convertir_entrevistas():
    """Función principal para convertir documentos a formato markdown."""
    # Instanciamos el convertidor de documentos
    converter = DocumentConverter()

    # Listamos los documentos que sube el usuario
    docs_raw = listar_documentos("app/docs/transcripciones")

    # Convertimos los documentos a formato markdown
    docs_converted = converter.convert_all(docs_raw)

    # Guardamos los documentos convertidos en formato markdown
    for result in docs_converted:
        # 1. Obtenemos el nombre original del archivo sin la extensión (.pdf o .docx)
        # `result.input.file` es un objeto Path que se guarda en el resultado de la
        # conversión, y `stem` nos da el nombre sin la extensión
        nombre_base = Path(result.input.file).stem

        # 2. Definimos la ruta de salida
        ruta_destino = Path("app/docs/markdown") / f"{nombre_base}.md"

        # 3. Exportamos y escribimos el archivo
        contenido_md = result.document.export_to_markdown()

        with open(ruta_destino, "w", encoding="utf-8") as f:
            f.write(contenido_md)

        print("#" * 80)
        print(f"\nArchivo almacenado en: {ruta_destino}\n")


def convertir_documento(documento_path: str) -> str:
    """Convierte un único documento a markdown, lo guarda en disco y devuelve su contenido."""
    converter = DocumentConverter()
    doc_path = Path(documento_path)

    docs_converted = list(converter.convert_all([str(doc_path.absolute())]))
    if not docs_converted:
        raise ValueError(f"No se pudo convertir el documento: {doc_path}")

    result = docs_converted[0]
    nombre_base = Path(result.input.file).stem
    ruta_destino = Path("app/docs/markdown") / f"{nombre_base}.md"
    contenido_md = result.document.export_to_markdown()

    with open(ruta_destino, "w", encoding="utf-8") as f:
        f.write(contenido_md)

    print("#" * 80)
    print(f"\nArchivo almacenado en: {ruta_destino}\n")
    return contenido_md
