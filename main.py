"""
main.py
=======

Script de validación: carga el .env, arma los clientes de OpenAI y
Anthropic, y les hace la misma pregunta corta en modo normal y en modo
streaming.

Uso:
    python main.py                # prueba los proveedores que tengan API key configurada
    python main.py openai         # prueba sólo OpenAI
    python main.py anthropic      # prueba sólo Anthropic

Requiere un archivo `.env` (ver `.env.example`) con:
    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

from clients import (
    AnthropicClient,
    BaseLLMClient,
    ConnectionFailedError,
    InvalidAPIKeyError,
    LLMClientError,
    OpenAIClient,
    RateLimitExceededError,
)
from schemas import ChatMessage, ModelConfig, Role

PREGUNTA = "¿Qué es la entropía?"

MODELOS_POR_DEFECTO = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


def _armar_cliente(provider: str) -> BaseLLMClient:
    config = ModelConfig(provider=provider, model=MODELOS_POR_DEFECTO[provider], max_tokens=300)  # type: ignore[arg-type]
    if provider == "openai":
        return OpenAIClient(config)
    return AnthropicClient(config)


async def probar_proveedor(provider: str) -> None:
    print(f"\n{'=' * 60}\nProveedor: {provider}\n{'=' * 60}")

    mensajes = [
        ChatMessage(role=Role.SYSTEM, content="Sos un asistente conciso."),
        ChatMessage(role=Role.USER, content=PREGUNTA),
    ]

    cliente = _armar_cliente(provider)

    try:
        # --- Modo normal ---
        print("\n[Modo normal]")
        respuesta = await cliente.generate(mensajes)
        print(respuesta.content)

        # --- Modo streaming ---
        print("\n[Modo streaming]")
        async for fragmento in cliente.stream(mensajes):
            print(fragmento, end="", flush=True)
        print()

    except InvalidAPIKeyError as error:
        print(f"\n[ERROR] API key inválida o faltante para '{provider}': {error}")
    except RateLimitExceededError as error:
        print(f"\n[ERROR] Límite de cuota/rate limit en '{provider}': {error}")
    except ConnectionFailedError as error:
        print(f"\n[ERROR] Falla de red hablando con '{provider}': {error}")
    except LLMClientError as error:
        print(f"\n[ERROR] Error controlado en '{provider}': {error}")


async def main() -> None:
    load_dotenv()

    if len(sys.argv) > 1:
        proveedores = [sys.argv[1]]
    else:
        # Si no se pasa argumento, probamos los proveedores que tengan API key en el entorno.
        proveedores = []
        if os.getenv("OPENAI_API_KEY"):
            proveedores.append("openai")
        if os.getenv("ANTHROPIC_API_KEY"):
            proveedores.append("anthropic")

        if not proveedores:
            print(
                "No encontré OPENAI_API_KEY ni ANTHROPIC_API_KEY en el entorno.\n"
                "Copiá .env.example a .env, completá al menos una API key, y volvé a correr."
            )
            return

    for provider in proveedores:
        await probar_proveedor(provider)


if __name__ == "__main__":
    asyncio.run(main())
