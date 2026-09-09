# TP - Cliente Async para LLMs (OpenAI + Anthropic)

Curso: AI Engineering - Coderhouse

## De qué se trata

La consigna pedía armar un cliente en Python para hablar con distintos
proveedores de LLM (OpenAI y Anthropic) usando la misma interfaz. Todo tiene
que ser asíncrono, con soporte de streaming, y los datos de entrada se
validan con Pydantic.

La idea es que el resto del código no tenga que saber con qué proveedor está
hablando. Todo pasa por la misma clase base, así que cambiar de OpenAI a
Anthropic es solo cambiar un string.

## Cómo lo resolví

**`schemas.py`**: acá están los modelos de Pydantic. `ChatMessage` valida cada
mensaje (rol y contenido, no deja pasar mensajes vacíos). `ModelConfig` valida
los parámetros del modelo (`temperature` entre 0 y 2, `max_tokens` positivo,
etc). `ModelResponse` es la respuesta ya normalizada, sea cual sea el
proveedor que respondió.

**`clients.py`**: `BaseLLMClient` es la clase base con `generate()` y
`stream()`. `OpenAIClient` y `AnthropicClient` heredan de ahí y usan los
clientes asíncronos oficiales de cada SDK (`AsyncOpenAI`, `AsyncAnthropic`).
No uso nada síncrono adentro de una función async, porque eso bloquea el
programa entero mientras el modelo procesa. El streaming lo hice con un
generador (`yield`) que va entregando cada fragmento de texto a medida que
llega.

También agregué manejo de errores. Si falla la API key, si te quedaste sin
cuota, o si hay un problema de red, el programa no se rompe con un error
feo. Lo atrapa y muestra un mensaje claro (`InvalidAPIKeyError`,
`RateLimitExceededError`, `ConnectionFailedError`).

**`main.py`**: el script de prueba. Carga el `.env`, le hace la misma
pregunta ("¿Qué es la entropía?") a los proveedores que tengan API key
cargada, primero en modo normal y después en modo streaming.

**`test_clients.py`**: tests con mocks, para probar todo (validación,
generate, stream, manejo de errores) sin gastar API real.

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
`ANTHROPIC_API_KEY` en console.anthropic.com/settings/keys. Ojo: si tenés
suscripción a Claude.ai eso no te da automáticamente la key de la API, son
cosas separadas. Me terminé enterando probando esto.

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

Investigué y resulta que desde Claude Sonnet 5 la API de Anthropic sacó
`temperature` y `top_p` como parámetros. Los modelos nuevos usan algo llamado
"adaptive thinking" en vez del sampling clásico, y el SDK ni siquiera acepta
esos argumentos.

En vez de hardcodear "no le mandes temperature a Anthropic", hice que
`AnthropicClient` revise en tiempo real si el SDK instalado todavía acepta
esos parámetros, y sólo los manda si corresponde. Me pareció más prolijo que
ir parchando a mano cada vez que un proveedor cambia su API.

## Estructura del repo

```
.
├── schemas.py         # Modelos Pydantic
├── clients.py         # BaseLLMClient, OpenAIClient, AnthropicClient
├── main.py            # Script de prueba (normal y streaming)
├── test_clients.py    # Tests con mocks
├── .env.example       # Plantilla de variables de entorno
├── requirements.txt
└── README.md
```
