"""
clients.py
==========

Clientes asíncronos e intercambiables para OpenAI y Anthropic.

- `BaseLLMClient`: clase base abstracta. Define el contrato común
  (`generate` y `stream`) que cualquier proveedor debe cumplir.
- `OpenAIClient` / `AnthropicClient`: implementaciones concretas, usando
  los SDKs oficiales en su versión asíncrona (`AsyncOpenAI`,
  `AsyncAnthropic`). Nunca se usa el cliente síncrono dentro de una
  función `async` (eso bloquearía el event loop mientras el modelo
  "piensa"); todas las llamadas de red usan `await`.

Manejo de errores ("no dejar fugar excepciones"):
  Un error de API key inválida o de límite de cuota/rate limit NO debe
  romper el programa con un traceback crudo del SDK. Ambos clientes
  atrapan esas excepciones puntuales y las traducen a `LLMClientError`
  (o una subclase), un error propio, controlado y predecible.
"""

from __future__ import annotations

import inspect
import os
from abc import ABC, abstractmethod
from typing import AsyncGenerator

import anthropic
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from schemas import ChatMessage, ModelConfig, ModelResponse, Role


# ---------------------------------------------------------------------------
# Errores controlados
# ---------------------------------------------------------------------------

class LLMClientError(Exception):
    """Error base y controlado para cualquier falla al hablar con un LLM."""

    def __init__(self, message: str, *, provider: str, retryable: bool = False, original: Exception | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.original = original


class RateLimitExceededError(LLMClientError):
    """Límite de cuota / rate limit del proveedor. Conviene reintentar con backoff."""


class ConnectionFailedError(LLMClientError):
    """Error de red o timeout al conectar con el proveedor. Conviene reintentar."""


class InvalidAPIKeyError(LLMClientError):
    """API key inválida o faltante. No tiene sentido reintentar sin corregirla."""


def _empaquetar_error(error: Exception, provider: str) -> LLMClientError:
    """Traduce una excepción cruda del SDK a un error propio y controlado."""
    if isinstance(error, (openai.RateLimitError, anthropic.RateLimitError)):
        return RateLimitExceededError(
            f"[{provider}] Límite de cuota / rate limit excedido: {error}",
            provider=provider,
            retryable=True,
            original=error,
        )
    if isinstance(error, (openai.AuthenticationError, anthropic.AuthenticationError)):
        return InvalidAPIKeyError(
            f"[{provider}] API key inválida o faltante: {error}",
            provider=provider,
            retryable=False,
            original=error,
        )
    if isinstance(
        error,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            ConnectionError,
            TimeoutError,
        ),
    ):
        return ConnectionFailedError(
            f"[{provider}] Error de red/timeout: {error}",
            provider=provider,
            retryable=True,
            original=error,
        )
    return LLMClientError(
        f"[{provider}] Error inesperado: {error}",
        provider=provider,
        retryable=False,
        original=error,
    )


# ---------------------------------------------------------------------------
# Base asíncrona común (intercambiabilidad)
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Interfaz común que deben cumplir todos los clientes de LLM.

    Gracias a esta clase, el resto del código (por ejemplo `main.py`)
    puede tratar a `OpenAIClient` y `AnthropicClient` exactamente igual:
    los dos exponen `generate()` y `stream()` con la misma firma.
    """

    provider_name: str = "generic"

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        """Hace una llamada completa (no streaming) y devuelve la respuesta ya armada."""
        raise NotImplementedError

    @abstractmethod
    def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        """Generador asíncrono: entrega fragmentos de texto a medida que llegan."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    provider_name = "openai"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key)

    @staticmethod
    def _a_formato_openai(messages: list[ChatMessage]) -> list[dict]:
        return [{"role": m.role.value, "content": m.content} for m in messages]

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        try:
            respuesta = await self._client.chat.completions.create(
                model=self.config.model,
                messages=self._a_formato_openai(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
        except Exception as error:  # noqa: BLE001 - se traduce a un error controlado
            raise _empaquetar_error(error, self.provider_name) from error

        eleccion = respuesta.choices[0]
        return ModelResponse(
            content=eleccion.message.content or "",
            provider=self.provider_name,
            model=self.config.model,
            finish_reason=eleccion.finish_reason,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        try:
            flujo = await self._client.chat.completions.create(
                model=self.config.model,
                messages=self._a_formato_openai(messages),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                stream=True,
            )
            async for chunk in flujo:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as error:  # noqa: BLE001
            raise _empaquetar_error(error, self.provider_name) from error


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

class AnthropicClient(BaseLLMClient):
    """Cliente para la API de Anthropic.

    Nota importante: a partir de Claude Sonnet 5, la API de Anthropic dejó
    de aceptar `temperature`/`top_p` (los modelos nuevos usan "adaptive
    thinking" en vez de esos parámetros de sampling clásicos; pasarlos
    ahora da error). El SDK más nuevo ni siquiera los expone como
    argumento en `messages.create()`. Para que este cliente no se rompa
    ni con SDKs viejos ni con los nuevos, `_kwargs_soportados()` inspecciona
    en tiempo real la firma real del método instalado y sólo manda
    `temperature`/`top_p` si el SDK todavía los acepta.
    """

    provider_name = "anthropic"

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=api_key)

    @staticmethod
    def _separar_system(messages: list[ChatMessage]) -> tuple[str | None, list[dict]]:
        mensajes_sistema = [m.content for m in messages if m.role == Role.SYSTEM]
        system = "\n".join(mensajes_sistema) if mensajes_sistema else None
        mensajes_chat = [
            {"role": m.role.value, "content": m.content} for m in messages if m.role != Role.SYSTEM
        ]
        return system, mensajes_chat

    @staticmethod
    def _kwargs_soportados(metodo, candidatos: dict) -> dict:
        """Filtra `candidatos` para quedarse sólo con los que el método
        realmente acepta, según su firma actual. Evita romper el cliente
        si el SDK instalado agrega o saca parámetros (como pasó con
        `temperature`/`top_p`)."""
        parametros = inspect.signature(metodo).parameters
        acepta_kwargs_libres = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parametros.values())
        if acepta_kwargs_libres:
            return dict(candidatos)
        return {clave: valor for clave, valor in candidatos.items() if clave in parametros}

    def _kwargs_sampling(self, metodo) -> dict:
        return self._kwargs_soportados(
            metodo,
            {"temperature": self.config.temperature, "top_p": self.config.top_p},
        )

    async def generate(self, messages: list[ChatMessage]) -> ModelResponse:
        system, mensajes_chat = self._separar_system(messages)
        try:
            respuesta = await self._client.messages.create(
                model=self.config.model,
                system=system,
                messages=mensajes_chat,
                max_tokens=self.config.max_tokens,
                **self._kwargs_sampling(self._client.messages.create),
            )
        except Exception as error:  # noqa: BLE001
            raise _empaquetar_error(error, self.provider_name) from error

        texto = "".join(bloque.text for bloque in respuesta.content if bloque.type == "text")
        return ModelResponse(
            content=texto,
            provider=self.provider_name,
            model=self.config.model,
            finish_reason=respuesta.stop_reason,
        )

    async def stream(self, messages: list[ChatMessage]) -> AsyncGenerator[str, None]:
        system, mensajes_chat = self._separar_system(messages)
        try:
            async with self._client.messages.stream(
                model=self.config.model,
                system=system,
                messages=mensajes_chat,
                max_tokens=self.config.max_tokens,
                **self._kwargs_sampling(self._client.messages.stream),
            ) as flujo:
                async for texto in flujo.text_stream:
                    yield texto
        except Exception as error:  # noqa: BLE001
            raise _empaquetar_error(error, self.provider_name) from error


# ---------------------------------------------------------------------------
# Fábrica pequeña: intercambiabilidad basada en la config
# ---------------------------------------------------------------------------

_CLIENTES: dict[str, type[BaseLLMClient]] = {
    "openai": OpenAIClient,
    "anthropic": AnthropicClient,
}


def crear_cliente(config: ModelConfig) -> BaseLLMClient:
    """Instancia el cliente correcto según `config.provider`."""
    try:
        clase = _CLIENTES[config.provider]
    except KeyError as error:
        raise ValueError(f"Proveedor '{config.provider}' no soportado. Opciones: {list(_CLIENTES)}") from error
    return clase(config)
