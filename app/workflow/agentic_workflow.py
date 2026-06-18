#############################################################################
# Script para generar el grafo de codificacion. Aqui creo el workflow
# completo, definiendo los nodos, edges y la logica de cada nodo.
# Tambien se agrega el subagente como un nodo mas del grafo principal.
#############################################################################

# ---------------------------------------------------------------------------
# Librerias
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
from pathlib import Path

from typing import TypedDict, Annotated, NotRequired
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from operator import add
import polars as pl

try:
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor
except Exception:
    register = None
    LangChainInstrumentor = None

from app.workflow.pydantic_models import PautaOrdenada
from app.workflow.sub_agent import sub_graph_compiled
from app.workflow.convert_docling import convertir_documento, convertir_entrevistas


# ---------------------------------------------------------------------------
# Variables de entorno
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Observabilidad con phoenix
# ---------------------------------------------------------------------------
if register and LangChainInstrumentor:
    try:
        tracer_provider = register(
            project_name="CodificacionCognitivas",
            endpoint="http://localhost:6006/v1/traces",
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:
        # Si observabilidad no está disponible, el workflow sigue funcionando.
        tracer_provider = None
else:
    tracer_provider = None


# ---------------------------------------------------------------------------
# Estado global del grafo
# ---------------------------------------------------------------------------
# Defino el estado global
class State(TypedDict):
    document_path: NotRequired[str]
    document_name: NotRequired[str]
    csv_text: str
    md_text: str
    bloque_preguntas: NotRequired[dict]
    contexto_estudio: str
    matriz_vaciado_final: Annotated[list[dict], add]
    revision_matriz: dict


# Inicializo el grafo con el estado global
graph = StateGraph(State)

# Mapeo código -> macrocategoría para enriquecer la matriz final.
MACROCATEGORIA_DATA = [
    ["Comprensión y comunicación", "Instrucciones inadecuadas"],
    ["Comprensión y comunicación", "Instrucciones complicadas o poco claras"],
    ["Comprensión y comunicación", "Dificultades en administración de la pregunta"],
    ["Comprensión y comunicación", "Pregunta vaga"],
    ["Comprensión y comunicación", "Conceptos complejos, técnicos o no definidos"],
    ["Comprensión y comunicación", "Traslado de temas desde la pregunta anterior"],
    ["Comprensión y comunicación", "Se necesita transición"],
    ["Comprensión y comunicación", "Orden de las preguntas"],
    ["Comprensión y comunicación", "Pregunta muy larga o fraseo complicado"],
    ["Comprensión y comunicación", "Suposición errónea"],
    ["Comprensión y comunicación", "Preguntas múltiples"],
    ["Comprensión y comunicación", "Temas trasladados de la pregunta anterior"],
    ["Comprensión y comunicación", "Periodo vago o variante"],
    ["Estimación y juicio", "Estimación compleja de realizar"],
    ["Estimación y juicio", "Potencialmente sensible o deseable socialmente"],
    ["Recuerdo", "Escasez de información"],
    ["Recuerdo", "Alto detalle requerido"],
    ["Selección de la respuesta", "Categorías complejas o no definidas"],
    ["Selección de la respuesta", "Categorías vagas"],
    ["Selección de la respuesta", "Orden de las categorías"],
    ["Selección de la respuesta", "Utiliza categorías erróneas"],
    ["Selección de la respuesta", "Categorías de respuesta muy largas"],
    ["Selección de la respuesta", "Categorías superpuestas"],
    ["Selección de la respuesta", "Categorías múltiples"],
    ["Selección de la respuesta", "Categorías faltantes"],
    ["Otros", "Otros problemas"],
    ["Otros", "Responde correctamente"],
]

MACROCATEGORIA_DF = pl.DataFrame(
    MACROCATEGORIA_DATA,
    schema=["macrocategoria", "codigo_base"],
    orient="row",
)


# ---------------------------------------------------------------------------
# Nodos del grafo
# ---------------------------------------------------------------------------


# Nodo 0: conversion de documentos a markdown
def convertir_documentos_markdown(state: State) -> dict:
    """
    Ejecuta la conversión de documentos crudos (PDF/DOC/DOCX) a markdown.
    Este nodo opera por efecto lateral escribiendo archivos en `app/docs/markdown`.
    """
    document_path = state.get("document_path")
    if document_path:
        nombre_documento = Path(document_path).stem
        contenido_md = convertir_documento(document_path)
        return {"md_text": contenido_md, "document_name": nombre_documento}

    convertir_entrevistas()
    return {}


# Nodo 1: Extracción de preguntas y sondeos desde la transcripción
def extractor_preguntas(state: State) -> dict:
    """
    LLM1:\n
    Lee la transcripción MD y la pauta CSV.
    Extrae para cada bloque: las preguntas evaluadas y sus sondeos,
    con el wording oficial y la interacción completa de la transcripción.
    """

    csv_text = state["csv_text"]
    md_text = state["md_text"]

    prompt = f"""
# Rol

Eres un extractor estricto de información desde transcripciones de entrevistas cognitivas.

# Objetivo

Lee el CSV y el MARKDOWN. Devuelve solo los bloques realmente aplicados en la transcripción.
El CSV solo define ids válidos, orden y agrupación.  
El MARKDOWN es la única fuente de verdad para decidir qué fue aplicado.

# Normalización

Para buscar ids en el Markdown, considera equivalentes:

- `EC_y29_10b_1`
- `EC\\_y29\\_10b\\_1`

Es decir, interpreta temporalmente `\\_` como `_` solo para búsqueda.
En la salida, devuelve siempre el id exactamente como aparece en el CSV.

# Formato de salida

Devuelve exclusivamente JSON válido con esta estructura:

{{
  "grupos": [
    {{
      "evaluadas": [
        {{
          "id": "...",
          "pregunta_evaluada": "...",
          "interaccion_textual": "..."
        }}
      ],
      "sondeos": [
        {{
          "id": "...",
          "pregunta_sondeo": "...",
          "interaccion_textual": "..."
        }}
      ]
    }}
  ]
}}

# Reglas obligatorias

1. Recorre el CSV fila por fila y conserva exactamente su orden.
2. Dentro de cada fila, conserva el orden de ids en `evaluadas` y `sondeos`.
3. Incluye solo ids del CSV que aparezcan realmente en el Markdown.
4. Un id se incluye solo si tiene pregunta y transcripción/interacción textual asociada.
5. Si un id está en el CSV pero no aparece aplicado en el Markdown, omítelo.
6. Si una fila completa no tiene ningún id aplicado, omite el grupo completo.
7. Si una fila está parcialmente aplicada, incluye solo los ids aplicados.
8. No inventes preguntas, sondeos ni interacciones.
9. No infieras contenido ausente.
10. No cambies ids.
11. No incluyas preguntas del Markdown que no estén en el CSV.
12. No devuelvas objetos incompletos.
13. No devuelvas campos vacíos.
14. No uses markdown ni bloques de código en la respuesta final.
15. Devuelve solo JSON válido.

# Regla crítica

Está prohibido devolver objetos con strings vacíos.

Incorrecto:

{{
  "id": "EC_e4a_fb_0",
  "pregunta_sondeo": "",
  "interaccion_textual": ""
}}

Si no puedes completar todos los campos con texto extraído del Markdown, omite ese objeto.

# Extracción

Para ids en `evaluadas`, devuelve:

- `id`
- `pregunta_evaluada`
- `interaccion_textual`

Para ids en `sondeos`, devuelve:

- `id`
- `pregunta_sondeo`
- `interaccion_textual`

# Casos especiales

Si una pregunta evaluada aparece aplicada pero sus sondeos no, devuelve el grupo con `sondeos: []`
Si un sondeo aparece aplicado pero la evaluada de esa fila no, devuelve el grupo con `evaluadas: []`.
Si `evaluadas` y `sondeos` quedarían vacíos, omite el grupo.


# CSV

{csv_text}

# MARKDOWN

{md_text}
"""

    # Configuracion del primer modelo
    llm = init_chat_model("gpt-5-mini", model_provider="openai")
    llm_estructurado = llm.with_structured_output(
        PautaOrdenada, method="function_calling", include_raw=True
    )

    # Ejecutar el modelo con gpt-5-mini
    resultado = llm_estructurado.invoke(prompt)

    # Devolver solo la parte estructurada de la salida
    return {"bloque_preguntas": resultado["parsed"].model_dump()}


# Nodo 2: subagente de codificación y revisión de codificación
def fanout_a_subagente(state: State):
    grupos = state["bloque_preguntas"]["grupos"]
    contexto_estudio = state.get("contexto_estudio", "")
    sends = []
    for bloque in grupos:
        sends.append(
            Send(
                "ciclo_calidad",
                {
                    "bloque": bloque,
                    "contexto_estudio_sub": contexto_estudio,
                    "intentos": 0,
                    "matriz_vaciado_raw": {},
                    "score": 0.0,
                    "revision": False,
                    "comentarios_revision": "",
                    "warning": False,
                    "matriz_vaciado_final": [],
                },
            )
        )
    return sends


# Nodo 3: Recolección de resultados del subagente y construcción de la matriz final
def construccion_matriz_final(state: State) -> dict:
    bloques = state.get("matriz_vaciado_final", [])

    # Columnas finales de la matriz consolidada.
    columnas = [
        "id",
        "tipo",
        "pregunta",
        "codigo",
        "comentario_pregunta",
        "extracto_1",
        "extracto_2",
        "extracto_3",
        "reflexion_codigo",
        "score_revision",
        "comentarios_revision",
        "revision_requiere_reintento",
        "warning_revision",
        "intentos_codificacion",
    ]

    registros: list[dict] = []

    for bloque in bloques:
        if not isinstance(bloque, dict):
            continue

        # Metadatos del proceso de revisión (aplican a cada fila del bloque).
        meta = {
            "score_revision": bloque.get("score_revision"),
            "comentarios_revision": bloque.get("comentarios_revision", ""),
            "revision_requiere_reintento": bloque.get(
                "revision_requiere_reintento", False
            ),
            "warning_revision": bloque.get("warning_revision", False),
            "intentos_codificacion": bloque.get("intentos_codificacion", 0),
        }

        # Formato nuevo: un bloque trae "filas" con N filas del LLM.
        filas_bloque = bloque.get("filas")
        if isinstance(filas_bloque, list):
            for fila in filas_bloque:
                if not isinstance(fila, dict):
                    continue
                registros.append(
                    {
                        "id": fila.get("id", ""),
                        "tipo": fila.get("tipo", ""),
                        "pregunta": fila.get("pregunta", ""),
                        "codigo": fila.get("codigo", ""),
                        "comentario_pregunta": fila.get("comentario_pregunta", ""),
                        "extracto_1": fila.get("extracto_1", ""),
                        "extracto_2": fila.get("extracto_2", ""),
                        "extracto_3": fila.get("extracto_3", ""),
                        "reflexion_codigo": fila.get("reflexion_codigo", ""),
                        **meta,
                    }
                )
            continue

        # Compatibilidad con formato previo: una fila directa en el bloque.
        if "id" in bloque:
            registros.append(
                {
                    "id": bloque.get("id", ""),
                    "tipo": bloque.get("tipo", ""),
                    "pregunta": bloque.get("pregunta", ""),
                    "codigo": bloque.get("codigo", ""),
                    "comentario_pregunta": bloque.get("comentario_pregunta", ""),
                    "extracto_1": bloque.get("extracto_1", ""),
                    "extracto_2": bloque.get("extracto_2", ""),
                    "extracto_3": bloque.get("extracto_3", ""),
                    "reflexion_codigo": bloque.get("reflexion_codigo", ""),
                    **meta,
                }
            )

    # Construcción del DataFrame en Polars.
    if registros:
        df = pl.DataFrame(registros).select(columnas)
    else:
        df = pl.DataFrame(
            schema={
                "id": pl.Utf8,
                "tipo": pl.Utf8,
                "pregunta": pl.Utf8,
                "codigo": pl.Utf8,
                "comentario_pregunta": pl.Utf8,
                "extracto_1": pl.Utf8,
                "extracto_2": pl.Utf8,
                "extracto_3": pl.Utf8,
                "reflexion_codigo": pl.Utf8,
                "score_revision": pl.Float64,
                "comentarios_revision": pl.Utf8,
                "revision_requiere_reintento": pl.Boolean,
                "warning_revision": pl.Boolean,
                "intentos_codificacion": pl.Int64,
            }
        )

    print("Matriz final consolidada en Polars:")
    print(df)

    # Se devuelve serializable para dejarlo en el estado del grafo.
    return {
        "revision_matriz": {
            "columns": df.columns,
            "shape": {"rows": df.height, "cols": df.width},
            "rows": df.to_dicts(),
        }
    }


# Nodo 4: Agregar macrocategoría según código
def agregar_macrocategoria(state: State) -> dict:
    revision_matriz = state.get("revision_matriz", {})
    rows = revision_matriz.get("rows", []) if isinstance(revision_matriz, dict) else []
    if rows:
        df = pl.DataFrame(rows)
    else:
        df = pl.DataFrame(
            schema={
                "id": pl.Utf8,
                "tipo": pl.Utf8,
                "pregunta": pl.Utf8,
                "codigo": pl.Utf8,
                "comentario_pregunta": pl.Utf8,
                "extracto_1": pl.Utf8,
                "extracto_2": pl.Utf8,
                "extracto_3": pl.Utf8,
                "reflexion_codigo": pl.Utf8,
                "score_revision": pl.Float64,
                "comentarios_revision": pl.Utf8,
                "revision_requiere_reintento": pl.Boolean,
                "warning_revision": pl.Boolean,
                "intentos_codificacion": pl.Int64,
            }
        )

    # Normaliza el código para mapear variantes del tipo
    # "Pregunta vaga: Ambigüedad..." -> "Pregunta vaga".
    df = df.with_columns(
        pl.col("codigo")
        .cast(pl.Utf8)
        .fill_null("")
        .str.strip_chars()
        .str.split_exact(":", 1)
        .struct.field("field_0")
        .str.strip_chars()
        .alias("codigo_base")
    )

    df_con_macro = df.join(MACROCATEGORIA_DF, on="codigo_base", how="left").drop(
        "codigo_base"
    )

    if "macrocategoria" in df_con_macro.columns:
        df_con_macro = df_con_macro.with_columns(
            pl.when(
                pl.col("codigo").cast(pl.Utf8).fill_null("").str.strip_chars() == ""
            )
            .then(pl.lit(""))
            .otherwise(pl.col("macrocategoria").fill_null("Sin clasificar"))
            .alias("macrocategoria")
        )

    print("Matriz final con macrocategoría:")
    print(df_con_macro)

    return {
        "revision_matriz": {
            "columns": df_con_macro.columns,
            "shape": {"rows": df_con_macro.height, "cols": df_con_macro.width},
            "rows": df_con_macro.to_dicts(),
        }
    }


# Nodo 5: Exportar matriz consolidada a XLSX
def exportar_matriz_xlsx(state: State) -> dict:
    revision_matriz = state.get("revision_matriz", {})
    rows = revision_matriz.get("rows", []) if isinstance(revision_matriz, dict) else []

    if rows:
        df = pl.DataFrame(rows)
    else:
        df = pl.DataFrame()

    nombre_base = state.get("document_name")
    if not nombre_base:
        document_path = state.get("document_path", "")
        nombre_base = Path(document_path).stem if document_path else "matriz_vaciado"

    carpeta_salida = Path("matriz_vaciado")
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    ruta_xlsx = carpeta_salida / f"{nombre_base}.xlsx"

    df.write_excel(ruta_xlsx)
    print(f"Matriz exportada a XLSX en: {ruta_xlsx}")

    return {"revision_matriz": revision_matriz}


# ---------------------------------------------------------------------------
# Registos de nodos en el grafo
# ---------------------------------------------------------------------------
# Para cada nodo, se registra su función en el grafo, asignandole un nombre unico.
graph.add_node("extractor_preguntas", extractor_preguntas)
graph.add_node("convertir_documentos_markdown", convertir_documentos_markdown)
graph.add_node("ciclo_calidad", sub_graph_compiled)
graph.add_node("construccion_matriz_final", construccion_matriz_final)
graph.add_node("agregar_macrocategoria", agregar_macrocategoria)
graph.add_node("exportar_matriz_xlsx", exportar_matriz_xlsx)

# ---------------------------------------------------------------------------
# Creacion de edges
# ---------------------------------------------------------------------------
# Para definir el flujo de la ejecucion, se crean edges entre los nodos,
# indicando el orden de ejecucion y el paso de informacion entre ellos.
graph.add_edge(START, "convertir_documentos_markdown")
graph.add_edge("convertir_documentos_markdown", "extractor_preguntas")
graph.add_conditional_edges(
    "extractor_preguntas",
    fanout_a_subagente,
    ["ciclo_calidad"],
)
graph.add_edge("ciclo_calidad", "construccion_matriz_final")
graph.add_edge("construccion_matriz_final", "agregar_macrocategoria")
graph.add_edge("agregar_macrocategoria", "exportar_matriz_xlsx")
graph.add_edge("exportar_matriz_xlsx", END)


# ---------------------------------------------------------------------------
# # Compilacion del grafo
# ---------------------------------------------------------------------------
workflow_app = graph.compile()
