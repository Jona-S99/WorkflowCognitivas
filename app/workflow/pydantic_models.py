# Aqui incluire todos los modelos pydantic para gestionar
# los outputs de los LLMs

from pydantic import BaseModel, Field, model_validator
from typing import Any, List, Literal
import re


# Modelos nodo 1 ===========================================


class BloqueEvaluada(BaseModel):
    id: str = Field(description="ID exacto de la pregunta evaluada según el CSV")
    pregunta_evaluada: str = Field(description="Texto completo de la pregunta evaluada")
    interaccion_textual: str = Field(
        description="Interacción textual completa, sin resumir"
    )


class BloqueSondeo(BaseModel):
    id: str = Field(description="ID exacto del sondeo según el CSV")
    pregunta_sondeo: str = Field(description="Texto completo de la pregunta de sondeo")
    interaccion_textual: str = Field(
        description="Interacción textual completa, sin resumir"
    )


class GrupoPauta(BaseModel):
    evaluadas: List[BloqueEvaluada] = Field(
        default_factory=list, description="Lista ordenada de preguntas evaluadas"
    )
    sondeos: List[BloqueSondeo] = Field(
        default_factory=list, description="Lista ordenada de sondeos"
    )


class PautaOrdenada(BaseModel):
    grupos: List[GrupoPauta] = Field(
        min_length=1,
        description="Grupos completos de la pauta en el mismo orden del CSV",
    )

    @model_validator(mode="before")
    @classmethod
    def normalizar_grupos_mal_formados(cls, data: Any) -> Any:
        """Corrige salidas donde el LLM deja items sueltos dentro de `grupos`."""

        def _normalizar_evaluada(item: Any) -> dict[str, str] | None:
            if isinstance(item, str):
                candidate = item.strip()
                if re.fullmatch(r"[A-Za-z0-9_]+", candidate):
                    return {
                        "id": candidate,
                        "pregunta_evaluada": "",
                        "interaccion_textual": "",
                    }
                return None
            if not isinstance(item, dict):
                return None
            return {
                "id": str(item.get("id", "") or ""),
                "pregunta_evaluada": str(item.get("pregunta_evaluada", "") or ""),
                "interaccion_textual": str(item.get("interaccion_textual", "") or ""),
            }

        def _normalizar_sondeo(item: Any) -> dict[str, str] | None:
            if isinstance(item, str):
                candidate = item.strip()
                if re.fullmatch(r"[A-Za-z0-9_]+", candidate):
                    return {
                        "id": candidate,
                        "pregunta_sondeo": "",
                        "interaccion_textual": "",
                    }
                return None
            if not isinstance(item, dict):
                return None
            return {
                "id": str(item.get("id", "") or ""),
                "pregunta_sondeo": str(item.get("pregunta_sondeo", "") or ""),
                "interaccion_textual": str(item.get("interaccion_textual", "") or ""),
            }

        def _normalizar_lista(
            raw_items: Any, normalizador: Any
        ) -> list[dict[str, str]]:
            if not isinstance(raw_items, list):
                return []
            salida: list[dict[str, str]] = []
            for raw in raw_items:
                item = normalizador(raw)
                if item is not None:
                    salida.append(item)
            return salida

        if not isinstance(data, dict):
            return data

        grupos = data.get("grupos")
        if not isinstance(grupos, list):
            return data

        normalizados: list[dict[str, Any]] = []
        grupo_actual: dict[str, Any] | None = None

        for item in grupos:
            if not isinstance(item, dict):
                continue

            if "evaluadas" in item or "sondeos" in item:
                grupo_actual = {
                    "evaluadas": _normalizar_lista(
                        item.get("evaluadas"), _normalizar_evaluada
                    ),
                    "sondeos": _normalizar_lista(
                        item.get("sondeos"), _normalizar_sondeo
                    ),
                }
                normalizados.append(grupo_actual)
                continue

            if (
                "id" in item
                and "pregunta_sondeo" in item
                and "interaccion_textual" in item
            ):
                sondeo = _normalizar_sondeo(item)
                if sondeo is None:
                    continue
                if grupo_actual is None:
                    grupo_actual = {"evaluadas": [], "sondeos": []}
                    normalizados.append(grupo_actual)
                grupo_actual["sondeos"].append(sondeo)
                continue

            if (
                "id" in item
                and "pregunta_evaluada" in item
                and "interaccion_textual" in item
            ):
                evaluada = _normalizar_evaluada(item)
                if evaluada is None:
                    continue
                grupo_actual = {"evaluadas": [evaluada], "sondeos": []}
                normalizados.append(grupo_actual)

        data["grupos"] = normalizados
        return data


# Modelos sub agente nodo 1 =================================


class FilaCodificacion(BaseModel):
    id: str
    tipo: Literal["evaluada", "sondeo"]
    pregunta: str
    codigo: str
    comentario_pregunta: str
    extracto_1: str
    extracto_2: str
    extracto_3: str
    reflexion_codigo: str


class MatrizVaciadoRaw(BaseModel):
    filas: List[FilaCodificacion] = Field(min_length=1)

    @model_validator(mode="after")
    def validar_reglas(self):
        for f in self.filas:
            if f.tipo == "sondeo" and f.codigo != "":
                raise ValueError("En sondeo, codigo debe ser ''.")
            if f.tipo == "evaluada" and f.codigo == "":
                raise ValueError("En evaluada, codigo no puede estar vacío.")
        return self


# Modelos sub agente nodo 2 =================================


class RevisionCodificacion(BaseModel):
    """
    Modelo para la salida del agente revisor en el nodo de revisión de codificación.
    """

    score: float = Field(
        description="Puntuación numérica que refleja la calidad de la codificación"
    )
    revision: bool = Field(
        description="Indicador booleano que determina si se requiere revisión adicional"
    )
    comentarios_revision: str = Field(
        description="Comentarios detallados del agente revisor sobre la codificación"
    )
    warning: bool = Field(
        description="Indicador booleano que marca si el bloque debe pasar al siguiente nodo con advertencia"
    )
