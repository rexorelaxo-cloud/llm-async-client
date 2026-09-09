"""
schemas.py
==========

Esquemas de datos (Pydantic) para el cliente async de LLMs.

Definir esto primero evita el clásico "error de diccionarios anidados":
en vez de pasar dicts sueltos (`{"role": "user", "content": "..."}`) entre
funciones y confiar en que siempre tengan las claves correctas, todo pasa
por estos modelos, que Pydantic valida automáticamente.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

ProviderName = Literal["openai", "anthropic"]


class Role(str, Enum):
    """Roles válidos dentro de una conversación."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Un mensaje individual del historial de conversación."""

    role: Role
    content: str = Field(..., min_length=1, description="Contenido del mensaje, no puede estar vacío")


class ModelConfig(BaseModel):
    """Parámetros de entrada del modelo, validados.

    - temperature: 0 a 2 (igual que el rango real de las APIs de OpenAI/Anthropic)
    - max_tokens: entero positivo
    - top_p: 0 a 1
    """

    provider: ProviderName = Field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"),  # type: ignore[return-value]
        description="'openai' o 'anthropic'. Default: variable de entorno LLM_PROVIDER, o 'openai'.",
    )
    model: str = Field(..., description="p. ej. 'gpt-4o-mini' o 'claude-haiku-4-5'")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0, le=200_000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    api_key: str | None = Field(default=None, description="Si es None, se toma del .env")

    @field_validator("model")
    @classmethod
    def _modelo_no_vacio(cls, valor: str) -> str:
        if not valor.strip():
            raise ValueError("El nombre del modelo no puede estar vacío")
        return valor


class ModelResponse(BaseModel):
    """Respuesta normalizada de una llamada no-streaming, sea cual sea el proveedor."""

    content: str
    provider: str
    model: str
    finish_reason: str | None = None
