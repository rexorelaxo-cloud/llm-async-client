# TP - Cliente Async para LLMs (OpenAI + Anthropic)

Curso: AI Engineering - Coderhouse

## De qué se trata

La consigna pedía armar una capa de abstracción en Python para poder hablar
con distintos proveedores de LLM (OpenAI y Anthropic) de forma intercambiable,
100% asíncrona, con soporte de streaming y validación de datos con Pydantic.

La idea central es que el resto del código no tenga que saber con qué
proveedor está hablando: todo pasa por la misma interfaz (`BaseLLMClient`),
así que cambiar de OpenAI a Anthropic es cuestión de cambiar un string, no de
reescribir nada.

## Cómo lo resolví

- **`schemas.py`**: acá están los modelos de Pydantic. `ChatMessage` valida
  cada mensaje (rol + contenido, no deja pasar mensajes vacíos), `ModelConfig`
  valida los parámetros del modelo (`temperature` entre 0 y 2, `max_tokens`
  positivo, etc.) y `ModelResponse` es la respuesta ya normalizada, sea cual
  sea el proveedor que respondió.

- **`clients.py`**: `BaseLLMClient` es la clase base abstracta con `generate()`
  y `stream()`. `OpenAIClient` y `AnthropicClient` heredan de ahí y usan los
  clientes asíncronos oficiales de cada SDK (`AsyncOpenAI`, `AsyncAnthropic`)
  — nada de llamadas síncronas adentro de una función `async`, porque eso
  bloquea el event loop mientras el modelo procesa. El streaming lo armé con
  un generador (`yield`) que va entregando cada fragmento de texto a medida
  que llega del SDK.

  También metí manejo de errores propio: si falla la API key, si te quedaste
  sin cuota, o si hay un problema de red, el cliente no explota con un
  traceback horrible — lo atrapa y lo traduce a un error propio
  (`InvalidAPIKeyError`, `RateLimitExceededError`, `ConnectionFailedError`),
  con un flag `.retryable` para saber si tiene sentido reintentar.

- **`main.py`**: el script de prueba. Carga el `.env`, le hace la misma
  pregunta ("¿Qué es la entropía?") a los proveedores que tengan API key
  cargada, primero en modo normal y después en modo streaming.

- **`test_clients.py`**: tests con mocks, para poder probar toda la lógica
  (validación, generate, stream, manejo de errores) sin gastar API real.

## Cómo correrlo

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Después copiás el archivo de ejemplo y completás tus keys:

```bash
cp .env.example .env
```

`OPENAI_API_KEY` se genera en platform.openai.com/api-keys y
`ANTHROPIC_API_KEY` en console.anthropic.com/settings/keys (ojo: si tenés
suscripción a Claude.ai eso no te da automáticamente la key de la API, son
cosas separadas, me terminé enterando probando esto).

Correr los tests (no gastan nada, están mockeados):

```bash
python -m pytest test_clients.py -v
```

Y para probarlo con una llamada real:

```bash
python main.py              # prueba todos los proveedores con key cargada
python main.py anthropic    # solo uno en particular
```

## Algo que me pasó armando esto

Mientras probaba con una API key real, `AnthropicClient` me tiraba un error
raro: `AsyncMessages.create() got an unexpected keyword argument 'temperature'`.
Después de investigar, resulta que a partir de Claude Sonnet 5 la API de
Anthropic sacó `temperature` y `top_p` como parámetros — los modelos nuevos
usan algo llamado "adaptive thinking" en vez del sampling clásico, y el SDK
ni siquiera acepta esos argumentos. Así que en vez de hardcodear "no le mandes
temperature a Anthropic", terminé haciendo que `AnthropicClient` inspeccione
en tiempo real la firma del método (`inspect.signature`) y sólo mande
`temperature`/`top_p` si el SDK instalado los sigue aceptando. Me pareció una
solución más robusta que ir parcheando a mano cada vez que un proveedor
cambia su API.

## Estructura del repo

```
.
├── schemas.py         # Modelos Pydantic
├── clients.py         # BaseLLMClient, OpenAIClient, AnthropicClient
├── main.py            # Script de prueba (normal + streaming)
├── test_clients.py    # Tests con mocks
├── .env.example       # Plantilla de variables de entorno
├── requirements.txt
└── README.md
```
