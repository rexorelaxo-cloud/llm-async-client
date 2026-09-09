"""
Tests de verificación (con mocks, sin API keys reales ni internet).

Ejecutar con:
    python -m pytest test_clients.py -v
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import anthropic
import httpx2
import openai
import pytest
from pydantic import ValidationError

from clients import (
    AnthropicClient,
    BaseLLMClient,
    ConnectionFailedError,
    InvalidAPIKeyError,
    OpenAIClient,
    RateLimitExceededError,
    crear_cliente,
)
from schemas import ChatMessage, ModelConfig, ModelResponse, Role

_FAKE_REQUEST = httpx2.Request("POST", "https://api.example.com/v1/chat")


def _fake_response(status_code: int) -> httpx2.Response:
    return httpx2.Response(status_code, request=_FAKE_REQUEST, json={"error": {"message": "boom"}})


# --- schemas.py --------------------------------------------------------

def test_chat_message_rechaza_contenido_vacio():
    with pytest.raises(ValidationError):
        ChatMessage(role=Role.USER, content="")


def test_model_config_valida_temperatura():
    with pytest.raises(ValidationError):
        ModelConfig(model="gpt-4o-mini", temperature=2.5)
    ModelConfig(model="gpt-4o-mini", temperature=2.0)  # no debe fallar


def test_model_config_default_provider_openai():
    assert ModelConfig(model="gpt-4o-mini").provider == "openai"


# --- intercambiabilidad -------------------------------------------------

def test_crear_cliente_openai_y_anthropic():
    c1 = crear_cliente(ModelConfig(provider="openai", model="gpt-4o-mini", api_key="x"))
    c2 = crear_cliente(ModelConfig(provider="anthropic", model="claude-haiku-4-5", api_key="x"))
    assert isinstance(c1, OpenAIClient) and isinstance(c1, BaseLLMClient)
    assert isinstance(c2, AnthropicClient) and isinstance(c2, BaseLLMClient)


# --- generate() / stream(): camino feliz --------------------------------

@pytest.mark.asyncio
async def test_openai_generate_ok():
    cliente = OpenAIClient(ModelConfig(model="gpt-4o-mini", api_key="x"))
    cliente._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="42"), finish_reason="stop")]
        )
    )
    resultado = await cliente.generate([ChatMessage(role=Role.USER, content="hola")])
    assert isinstance(resultado, ModelResponse)
    assert resultado.content == "42"


@pytest.mark.asyncio
async def test_openai_stream_ok():
    cliente = OpenAIClient(ModelConfig(model="gpt-4o-mini", api_key="x"))

    async def fake_stream():
        for t in ["Ho", "la"]:
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=t))])

    cliente._client.chat.completions.create = AsyncMock(return_value=fake_stream())
    fragmentos = [f async for f in cliente.stream([ChatMessage(role=Role.USER, content="hola")])]
    assert "".join(fragmentos) == "Hola"


@pytest.mark.asyncio
async def test_anthropic_generate_ok():
    cliente = AnthropicClient(ModelConfig(model="claude-haiku-4-5", api_key="x"))
    cliente._client.messages.create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hola desde claude")],
            stop_reason="end_turn",
        )
    )
    resultado = await cliente.generate([ChatMessage(role=Role.USER, content="hola")])
    assert resultado.content == "hola desde claude"
    assert resultado.provider == "anthropic"


def test_anthropic_omite_temperature_si_el_sdk_no_lo_acepta():
    """Reproduce el bug real encontrado en producción: los SDKs nuevos de
    Anthropic (Claude Sonnet 5 en adelante) sacaron `temperature`/`top_p`
    de `messages.create()`. El cliente no debe romperse ni mandarlos si el
    método instalado no los admite."""

    def create_sin_sampling(*, model, system, messages, max_tokens):  # sin temperature/top_p
        raise AssertionError("no debería llamarse en este test")

    cliente = AnthropicClient(ModelConfig(model="claude-sonnet-5", api_key="x"))
    kwargs = cliente._kwargs_sampling(create_sin_sampling)
    assert kwargs == {}


def test_anthropic_incluye_temperature_si_el_sdk_lo_acepta():
    def create_con_sampling(*, model, system, messages, max_tokens, temperature=None, top_p=None):
        pass

    cliente = AnthropicClient(ModelConfig(model="claude-haiku-4-5", api_key="x", temperature=0.3, top_p=0.9))
    kwargs = cliente._kwargs_sampling(create_con_sampling)
    assert kwargs == {"temperature": 0.3, "top_p": 0.9}


# --- manejo controlado de errores ---------------------------------------

@pytest.mark.asyncio
async def test_openai_rate_limit_no_rompe_el_programa():
    cliente = OpenAIClient(ModelConfig(model="gpt-4o-mini", api_key="x"))
    cliente._client.chat.completions.create = AsyncMock(
        side_effect=openai.RateLimitError("rate limited", response=_fake_response(429), body=None)
    )
    with pytest.raises(RateLimitExceededError) as exc:
        await cliente.generate([ChatMessage(role=Role.USER, content="hola")])
    assert exc.value.retryable is True


@pytest.mark.asyncio
async def test_openai_api_key_invalida_no_rompe_el_programa():
    cliente = OpenAIClient(ModelConfig(model="gpt-4o-mini", api_key="mala-key"))
    cliente._client.chat.completions.create = AsyncMock(
        side_effect=openai.AuthenticationError("invalid api key", response=_fake_response(401), body=None)
    )
    with pytest.raises(InvalidAPIKeyError):
        await cliente.generate([ChatMessage(role=Role.USER, content="hola")])


@pytest.mark.asyncio
async def test_anthropic_error_de_red_no_rompe_el_programa():
    cliente = AnthropicClient(ModelConfig(model="claude-haiku-4-5", api_key="x"))
    cliente._client.messages.create = AsyncMock(
        side_effect=anthropic.APIConnectionError(message="conexión rechazada", request=_FAKE_REQUEST)
    )
    with pytest.raises(ConnectionFailedError):
        await cliente.generate([ChatMessage(role=Role.USER, content="hola")])
