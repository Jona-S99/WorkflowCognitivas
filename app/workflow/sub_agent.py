#############################################################################
# Script para generar el subgrafo de codificacion. Aqui creo un agente para
# la codificacion de las entrevistas cognitivas, que a su vez, se revisa con
# otro agente, generando comentarios y un score de la codificacion.
#############################################################################

# ---------------------------------------------------------------------------
# Librerias
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from app.workflow.pydantic_models import MatrizVaciadoRaw, RevisionCodificacion

load_dotenv()

# ---------------------------------------------------------------------------
# Constantes del ciclo de codificacion
# ---------------------------------------------------------------------------

## Score mínimo aceptable para aprobar una codificación
UMBRAL_SCORE = 0.75

## Reintentos máximos antes de pasar con warning
MAX_INTENTOS = 2

## Libro de codigos para la codificaion
LIBRO_CODIGOS = """
Instrucciones inadecuadas: Dificultades en entregar instrucciones correctas (falta instrucción o es errónea)  
Instrucciones complicadas o poco claras: Dudas sobre qué leer o qué instrucción dar (ej. “considere” o fraseo para encuestadores)  
Dificultades en administración de la pregunta: Problemas de administración en general (saltos, no mostrar todas las categorías, etc.)  
Pregunta vaga: Ambigüedad o múltiples interpretaciones sobre qué incluir/excluir  
Conceptos complejos, técnicos o no definidos: Conceptos muy técnicos/ambiguos o no definidos (p. ej., “teletrabajo”)  
Traslado de temas desde la pregunta anterior: Comprensión depende de la pregunta previa (p. ej., “curso” como nivel educacional)  
Se necesita transición: Falta de pregunta(s) de transición (filtros, encadenamientos)  
Orden de las preguntas: El orden dificulta la comprensión (mejor por “miembro” que por “tipo”, etc.)  
Pregunta muy larga o fraseo complicado: Sintaxis compleja, fallas gramaticales, o la persona solo retiene primera/última parte  
Suposición errónea: La pregunta asume algo incorrecto sobre la persona/su contexto (p. ej., ignora informalidad)  
Preguntas múltiples: El entrevistado entiende 2+ preguntas en una sola  
Temas trasladados de la pregunta anterior: Se arrastra el periodo de referencia previo (p. ej., piensa en 2022 vs. “últimos 7 días”)  
Periodo vago o variante: Periodos ambiguos (¿“última semana” = 7 días o lun-vie?, promedio vs. referencia exacta)
Estimación compleja de realizar: Dificultad para estimar con certeza (p. ej., edad exacta de un miembro)  
Potencialmente sensible o deseable socialmente: Sensibilidad/vergüenza/privacidad/ilegalidad; posibles cambios por presión social
Escasez de información: No se posee la información necesaria (p. ej., nivel educacional de un tercero)  
Alto detalle requerido: Exige esfuerzo de memoria o alto detalle (p. ej., número de curso de otro miembro)
Categorías complejas o no definidas: Alternativas con conceptos complejos/indefinidos (p. ej., “espacio marítimo”)  
Categorías vagas: Alternativas con conceptos vagos o de múltiple interpretación  
Orden de las categorías: El orden impide identificar fácilmente la opción adecuada  
Utiliza categorías erróneas: Unidades de respuesta siguen un objetivo distinto (confunde “espacio” con “tipo de labor”)  
Categorías de respuesta muy largas: Listas extensas que afectan la atención/lectura (p. ej., dependencia en salud)  
Categorías superpuestas: Alternativas que se traslapan  
Categorías múltiples: Caso real calza en más de una alternativa “única”  
Categorías faltantes: No existe una alternativa que represente la situación
Otros problemas: Problemas no cubiertos por los códigos anteriores  
Responde correctamente: Comprensión cabal y respuesta fácil/rápida; sin observaciones
"""


# ---------------------------------------------------------------------------
# Estado del subgrafo
# La funcion `Send` crea una instancia aislada de este estado,
# pues paralaleliza la ejecucion del subgrafo para N bloques de preguntas
# que tenga la lista obtenida del extractor del workflow prinipal.
# ---------------------------------------------------------------------------
class BlockState(TypedDict):
    # bloque es el diccionario que itera desde la lista extraida
    bloque: dict
    # contexto global del estudio definido por el usuario
    contexto_estudio_sub: str
    # Marcador de cuantas veces se ha codificado este bloque, para controlar reintentos
    intentos: int
    # este es la matriz de vaciado que genera el agente codificador
    matriz_vaciado_raw: dict
    # Score y comentarios que le asigna el agente revisor a la codificacion
    score: float
    revision: bool
    comentarios_revision: str
    # indicador booleano para marcar si el bloque pasó el proceso de codificacion-revision
    # Es un True si el bloque culmino los intentos sin pasar el umbral
    warning: bool
    # Este campo es el "puente" entre el subgrafo y el reducer del grafo principal.
    # El nodo final escribe aqui y el reducer 'add' del State principal lo acumula.
    matriz_vaciado_final: list[dict]


# ---------------------------------------------------------------------------
# SUB-GRAFO: Ciclo de Mejora de Calidad
# Este tiene que compilarse como un grafo independiente y se registra como otro
# nodo mas en el grafo principal, pero con la función `Send` que se encarga de
# pasarle cada bloque iterativamente
# ---------------------------------------------------------------------------
sub_graph = StateGraph(BlockState)


# ---------------------------------------------------------------------------
# Nodos del subgrafo
# ---------------------------------------------------------------------------


# Nodo 1 — Codificación Cognitiva
# Recibe el bloque y genera razonamiento + citas + código.
# Incrementa 'intentos' siempre, para que el conditional edge tenga info correcta.
def codificacion_cognitivas(state: BlockState) -> dict:
    bloque = state["bloque"]
    intentos = state["intentos"]

    prompt = f"""
    <rol>
    Eres un/a analista senior en entrevistas cognitivas (es-CL), experto/a en medición y diseño de cuestionarios. Trabajas con transcripciones reales y debes ceñirte estrictamente a la guía y al libro de códigos. Piensa paso a paso para codificar correctamente cada pregunta evaluada y entregar una respuesta de calidad
    </rol>

    <tarea>
    Codifica la entrevista según la pauta y usando <libro_codigos> solo para preguntas evaluadas. 
    </tarea>

    <formato_salida_obligatorio>
    Devuelve un único JSON válido con esta estructura exacta:
    {{
        "filas": [
            {{
                "id": "...",
                "tipo": "evaluada" | "sondeo",
                "pregunta": "...",
                "codigo": "...",
                "comentario_pregunta": "...",
                "extracto_1": "...",
                "extracto_2": "...",
                "extracto_3": "...",
                "reflexion_codigo": "..."
            }}
        ]
    }}
    
    Reglas obligatorias de construcción de "filas":
    1. Debes crear exactamente 1 fila por cada pregunta de entrada del bloque (todas las evaluadas + todos los sondeos).
    2. Debes preservar el orden original del bloque: primero "evaluadas" en su orden, luego "sondeos" en su orden.
    3. Si un id aparece repetido en la entrada, debe aparecer repetido en la salida.
    4. Para "evaluada": "codigo" debe ser un código exacto del <libro_codigos> (no vacío).
    5. Para "sondeo": "codigo" debe ser exactamente "" (vacío), pero "comentario_pregunta", "extracto_1", "extracto_2", "extracto_3" y "reflexion_codigo" sí deben venir completos y basados en evidencia textual.
    6. No inventes contenido. Todo debe ser trazable a la interacción textual de cada pregunta.
    7. No omitas preguntas. No agregues preguntas nuevas.
    8. Devuelve solo JSON, sin texto extra fuera del JSON.
    </formato_salida_obligatorio>

    <contexto_estudio>
    Este es el contexto del estudio y la entrevista, que debes considerar para interpretar correctamente las respuestas y asignar los códigos:
    {state["contexto_estudio_sub"]}
    </contexto_estudio>

    <libro_codigos>
    {LIBRO_CODIGOS}
    </libro_codigos>

    <bloque_a_codificar>
    {bloque}
    </bloque_a_codificar>

    <comentarios_revision>
    {state["comentarios_revision"] if intentos > 0 else "N/A"}
    </comentarios_revision>
    """

    llm = init_chat_model("gpt-5", model_provider="openai")
    llm_estructurado = llm.with_structured_output(
        MatrizVaciadoRaw, method="function_calling", include_raw=True
    )
    resultado = llm_estructurado.invoke(prompt)

    return {
        "matriz_vaciado_raw": resultado["parsed"].model_dump(),
        "intentos": intentos + 1,
    }


# Nodo 2 — Revisión de Codificación
# Recibe la matriz de vaciado sin procesar del nodo anterior, evalúa su calidad
# asigna un score y comentarios, y si el score es menor al umbral, marca el
# bloque para reintento (warning=True) o para pasar al siguiente bloque (warning=False)
def revision_codificacion(state: BlockState) -> dict:
    matriz_vaciado_raw = state["matriz_vaciado_raw"]
    intentos = state["intentos"]

    prompt = f"""
    <rol>
    Eres un/a analista senior en entrevistas cognitivas (es-CL), experto/a en medición y diseño de cuestionarios. Trabajas con transcripciones reales y debes ceñirte estrictamente a la guía y al libro de códigos. Piensa paso a paso para evaluar la calidad de la codificación de cada pregunta evaluada, asignar un score y entregar comentarios detallados que justifiquen el score.
    </rol>

    <tarea>
    Evalúa la calidad de la codificación según la pauta y usando <libro_codigos>. 
    Asigna un score entre 0 y 1 basado en qué tan bien se aplicaron los códigos del libro, la calidad de los extractos, y la profundidad de las reflexiones.
    IMPORTANTE: solo las preguntas evaluadas deben llevar código, las de sondeo no.
    Si el score es menor a {UMBRAL_SCORE} y los intentos son menores o iguales a {
        MAX_INTENTOS
    }, marca el bloque para reintento (revision = True) y entrega comentarios específicos sobre qué mejorar; 
    Si el score es menor o igual al umbral pero los intentos ya alcanzaron el máximo, marca el bloque para pasar al siguiente nodo con un warning=True y con un comentario en cada codificación para identificar qué falló
    Si el score es mayor o igual al umbral, marca el bloque como aprobado warning=False y revision=False y entrega comentarios sobre porqué pasó la codificación.
    No inventes contenido, conserva trazabilidad entre pregunta, evidencia y evaluación, y devuelve JSON con la siguiente estructura:
    {"score", "comentarios_revision", "revision", "warning"}
    </tarea>
    
    <numero_intentos>
    {intentos}
    </numero_intentos>

    <libro_codigos>
    {LIBRO_CODIGOS}
    </libro_codigos>

    <matriz_a_evaluar>
    {matriz_vaciado_raw}
    </matriz_a_evaluar>
    """

    llm = init_chat_model("gpt-5", model_provider="openai")
    llm_estructurado = llm.with_structured_output(
        RevisionCodificacion, method="function_calling", include_raw=True
    )
    resultado = llm_estructurado.invoke(prompt)
    parsed = resultado["parsed"]
    parsed_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed

    return {
        "score": parsed_dict["score"],
        "comentarios_revision": parsed_dict["comentarios_revision"],
        "revision": parsed_dict["revision"],
        "warning": parsed_dict["warning"],
    }


# Agrego una funcion auxiliar para la ruta condicional del edge de revision,
# que decide si se reintenta o se pasa al siguiente bloque dependiendo de la
# decision del agente.
def ruta_revision(state: BlockState) -> str:
    # Si debe reintentar y aún no supera máximo, vuelve a codificar
    if state["revision"] and state["intentos"] <= MAX_INTENTOS:
        return "retry"
    return "done"


# Nodo 3 — Cerrar bloque y preparar salida final
# Recibe la matriz de vaciado sin procesar, el score y comentarios de revisión,
# y prepara la matriz final que se acumula en el reducer del grafo principal.
def matriz_final(state: BlockState) -> dict:
    raw = state["matriz_vaciado_raw"]
    score = state["score"]
    revision = state["revision"]
    warning = state["warning"]
    intentos = state["intentos"]
    comentarios_revision = state["comentarios_revision"]

    # Normaliza la salida del codificador:
    # - si viene 1 fila como dict -> [dict]
    # - si ya viene como lista -> lista
    # - cualquier otro caso -> []
    if isinstance(raw, list):
        filas = raw
    elif isinstance(raw, dict):
        filas = [raw]
    else:
        filas = []

    matriz_vaciado_final: list[dict] = []
    for fila in filas:
        if not isinstance(fila, dict):
            continue
        matriz_vaciado_final.append(
            {
                **fila,
                "score_revision": score,
                "comentarios_revision": comentarios_revision,
                "revision_requiere_reintento": revision,
                "warning_revision": warning,
                "intentos_codificacion": intentos,
            }
        )

    return {"matriz_vaciado_final": matriz_vaciado_final}


# ---------------------------------------------------------------------------
# Registos de nodos del subgrafo
# ---------------------------------------------------------------------------
sub_graph.add_node("codificacion_cognitivas", codificacion_cognitivas)
sub_graph.add_node("revision_codificacion", revision_codificacion)
sub_graph.add_node("matriz_final", matriz_final)

# ---------------------------------------------------------------------------
# Creacion de edges del subgrafo
# ---------------------------------------------------------------------------
sub_graph.add_edge(START, "codificacion_cognitivas")
sub_graph.add_edge("codificacion_cognitivas", "revision_codificacion")
sub_graph.add_conditional_edges(
    "revision_codificacion",
    ruta_revision,
    {
        "retry": "codificacion_cognitivas",
        "done": "matriz_final",
    },
)
sub_graph.add_edge("matriz_final", END)

# ---------------------------------------------------------------------------
# Compilacion del subgrafo
# ---------------------------------------------------------------------------
sub_graph_compiled = sub_graph.compile()
