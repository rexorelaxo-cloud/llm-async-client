# Cliente async intercambiable para OpenAI y Anthropic

Cliente asíncrono, con streaming, validado con Pydantic, para hablar con OpenAI
o Anthropic bajo la misma interfaz.

## Estructura

- `schemas.py` — Modelos Pydantic: `ChatMessage` (role, content), `ModelConfig`
  (provider, model, temperature 0-2, max_tokens, top_p) y `ModelResponse`.
- `clients.py` — `BaseLLMClient` (clase base abstracta con `generate()` y
  `stream()`), `OpenAIClient` y `AnthropicClient` (implementaciones concretas
  con `AsyncOpenAI` / `AsyncAnthropic`), y los errores controlados
  (`RateLimitExceededError`, `ConnectionFailedError`, `InvalidAPIKeyError`).
- `main.py` — Script de validación: carga el `.env`, le hace la pregunta
  "¿Qué es la entropía?" a cada proveedor configurado, en modo normal y en
  modo streaming.
- `test_clients.py` — Tests automáticos con mocks (no gastan API real).
- `.env.example` — Plantilla de variables de entorno.
- `requirements.txt` — Dependencias.

## 1. Preparar el entorno

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Configurar las API keys

```bash
cp .env.example .env
```

Editá `.env` y completá al menos una de estas dos keys:

- `OPENAI_API_KEY` → se obtiene en https://platform.openai.com/api-keys
- `ANTHROPIC_API_KEY` → se obtiene en https://console.anthropic.com/settings/keys

**Importante:** si tenés una suscripción a Claude.ai (Free/Pro/Max), esa
cuenta **no** incluye automáticamente acceso a la API — son productos
separados con facturación separada. La API se paga por uso (tokens) y la
key se genera en console.anthropic.com, no en claude.ai.

## 3. Correr los tests (no gastan API real, usan mocks)

```bash
python -m pytest test_clients.py -v
```

Deberían pasar 10/10.

## 4. Probar con una llamada real

```bash
python main.py              # prueba automáticamente los proveedores que tengan key en .env
python main.py openai       # sólo OpenAI
python main.py anthropic    # sólo Anthropic
```

Vas a ver, para cada proveedor: la respuesta completa (modo normal) y después
la misma respuesta pero apareciendo palabra por palabra (modo streaming). Los
modelos por defecto son económicos (`gpt-4o-mini`, `claude-haiku-4-5`) para no
gastar de más mientras probás.

Si la API key es inválida o te quedaste sin cuota, el script **no se cae**:
imprime un error controlado y sigue.

## 5. Subir el proyecto a GitHub para entregarlo

Si nunca usaste git en esta carpeta, corré estos pasos en una terminal, parado
en la carpeta del proyecto:

```bash
git init
git add .
git commit -m "Cliente async intercambiable OpenAI/Anthropic con streaming"
```

`.env` no se sube (está en `.gitignore`), así que tu API key queda a salvo.

Después, creá el repositorio remoto. Hay dos formas:

**Opción A — con la web de GitHub:**
1. Entrá a https://github.com/new, ponele un nombre (por ejemplo
   `llm-async-client`), dejalo público o privado según lo que pida la
   consigna, y **no** marques "Add a README" (ya tenés uno).
2. GitHub te va a mostrar unos comandos parecidos a estos — copialos y
   correlos en tu terminal:

```bash
git branch -M main
git remote add origin https://github.com/TU_USUARIO/llm-async-client.git
git push -u origin main
```

**Opción B — con la CLI de GitHub (`gh`), si la tenés instalada:**

```bash
gh repo create llm-async-client --private --source=. --remote=origin --push
```

(cambiá `--private` por `--public` si la consigna pide que sea público).

Una vez subido, el link que entregás es el de tu repositorio:
`https://github.com/TU_USUARIO/llm-async-client`.

## Notas de diseño (errores comunes que evita este código)

- **No bloquea el event loop**: todas las llamadas de red usan
  `await client.chat.completions.create(...)` / `await client.messages.create(...)`
  con los clientes asíncronos oficiales (`AsyncOpenAI`, `AsyncAnthropic`),
  nunca la versión síncrona dentro de una función `async`.
- **No deja fugar excepciones**: los errores de API key inválida y de rate
  limit/cuota se atrapan dentro de `OpenAIClient`/`AnthropicClient` y se
  relanzan como `InvalidAPIKeyError` / `RateLimitExceededError` (ambas
  subclases de `LLMClientError`, con un flag `.retryable` para saber si tiene
  sentido reintentar). `main.py` las captura y muestra un mensaje claro en vez
  de un traceback crudo.
- **Diferencias entre proveedores**: a partir de Claude Sonnet 5, la API de
  Anthropic dejó de aceptar `temperature`/`top_p` (los modelos nuevos usan
  "adaptive thinking" en vez del sampling clásico). `AnthropicClient`
  detecta en tiempo real, inspeccionando la firma del SDK instalado, si esos
  parámetros siguen existiendo, y sólo los manda si corresponde — así no se
  rompe ni con SDKs viejos ni con los nuevos. OpenAI sí sigue soportando
  ambos parámetros normalmente.

## Errores esperables al probar con una API real

- `insufficient_quota` / "You have no credits remaining" en OpenAI: no es un
  bug, es que la cuenta de OpenAI no tiene crédito cargado. Se soluciona
  agregando un método de pago en https://platform.openai.com/settings/organization/billing/.
  El punto importante es que el programa **no se cae**: `main.py` atrapa el
  error y lo muestra de forma controlada.
