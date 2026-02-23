# LangCore LiteLLM

> Provider plugin for [LangCore](https://github.com/ignatg/langcore) — access 100+ language models through a single, unified interface via [LiteLLM](https://docs.litellm.ai/docs/).

[![PyPI version](https://img.shields.io/pypi/v/langcore-litellm)](https://pypi.org/project/langcore-litellm/)
[![Python](https://img.shields.io/pypi/pyversions/langcore-litellm)](https://pypi.org/project/langcore-litellm/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

**langcore-litellm** is a provider plugin for [LangCore](https://github.com/ignatg/langcore) that adds support for 100+ language models through [LiteLLM](https://docs.litellm.ai/docs/)'s unified API. Install it, prefix your model ID with `litellm/`, and every LangCore extraction call routes through LiteLLM transparently — auto-discovered via Python entry points.

---

## Features

- **100+ model support** — OpenAI, Anthropic, Google, Azure, Mistral, Groq, Cohere, HuggingFace, Ollama, vLLM, and more through a single provider
- **Native async** — uses `litellm.acompletion()` with `asyncio.Semaphore` for true non-blocking concurrent I/O (no thread pool overhead)
- **Multi-pass cache bypass** — automatic per-pass cache control ensures fresh LLM responses on repeat extraction passes while keeping the first pass cacheable
- **Token usage tracking** — captures prompt, completion, and total token counts (`UsageStats`) from every inference call
- **Concurrency control** — configurable `max_workers` semaphore limits parallel async requests to prevent rate-limit errors
- **Zero-config plugin** — auto-registered via Python entry points; no manual wiring required
- **Full parameter passthrough** — forward any LiteLLM-supported parameter (temperature, top_p, timeout, etc.) through `provider_kwargs`

---

## Installation

```bash
pip install langcore-litellm
```

Or install from source:

```bash
git clone https://github.com/JustStas/langcore-litellm
cd langcore-litellm
pip install -e .
```

---

## Quick Start

### Integration with LangCore

langcore-litellm integrates with LangCore through the **provider plugin system**. Create a model configuration, and LangCore handles the rest:

```python
import langcore as lx

# Configure the LiteLLM provider
config = lx.factory.ModelConfig(
    model_id="litellm/gpt-4o",
    provider="LiteLLMLanguageModel",
)
model = lx.factory.create_model(config)

# Use with any LangCore extraction
result = lx.extract(
    text_or_documents="Acme Corp agrees to pay $50,000 to Beta LLC by March 2025.",
    model=model,
    prompt_description="Extract parties, monetary amounts, and dates.",
    examples=[
        lx.data.ExampleData(
            text="Alpha Inc will pay $10,000 to Omega Ltd by January 2024.",
            extractions=[
                lx.data.Extraction("party", "Alpha Inc", attributes={"role": "payer"}),
                lx.data.Extraction("party", "Omega Ltd", attributes={"role": "payee"}),
                lx.data.Extraction("monetary_amount", "$10,000"),
                lx.data.Extraction("date", "January 2024", attributes={"type": "deadline"}),
            ],
        )
    ],
)

print(result)
```

---

## Usage

### Supported Models

Model IDs must be prefixed with `litellm/` (or `litellm-`) to route through this provider:

| Provider | Example Model IDs |
|----------|-------------------|
| **OpenAI** | `litellm/gpt-4o`, `litellm/gpt-4o-mini`, `litellm/gpt-4-turbo` |
| **Anthropic** | `litellm/claude-3-opus`, `litellm/claude-3.5-sonnet`, `litellm/claude-3-haiku` |
| **Google** | `litellm/gemini-2.5-pro`, `litellm/gemini-2.0-flash` |
| **Azure OpenAI** | `litellm/azure/your-deployment-name` |
| **Mistral** | `litellm/mistral-large-latest` |
| **Groq** | `litellm/groq/llama-3.1-70b` |
| **Ollama** | `litellm/ollama/llama3.1` |
| **And 100+ more** | See [LiteLLM providers](https://docs.litellm.ai/docs/providers) |

### Environment Variables

Set the appropriate API key for your provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google (Gemini)
export GEMINI_API_KEY="..."

# Azure OpenAI
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-resource.openai.azure.com/"
export AZURE_API_VERSION="2024-02-01"

# Ollama (local)
export OLLAMA_API_BASE="http://localhost:11434"
```

See the [LiteLLM documentation](https://docs.litellm.ai/docs/) for the full list of provider-specific variables.

### Async Extraction

The provider uses `litellm.acompletion()` for native async, avoiding thread overhead:

```python
import asyncio
import langcore as lx

config = lx.factory.ModelConfig(
    model_id="litellm/gpt-4o",
    provider="LiteLLMLanguageModel",
)
model = lx.factory.create_model(config)

async def main():
    result = await lx.async_extract(
        text_or_documents="Agreement between Acme Corp and Beta LLC...",
        model=model,
        prompt_description="Extract parties and obligations.",
        examples=[...],
    )
    print(result)

asyncio.run(main())
```

### Multi-Pass Extraction

When running multiple extraction passes (`extraction_passes > 1`), the provider automatically manages cache behaviour:

| Pass | Cache Behaviour |
|------|-----------------|
| 1st pass | Normal — may be served from cache |
| 2nd pass | Bypass — forces a fresh LLM response |
| 3rd+ passes | Bypass — forces a fresh LLM response |

This is fully automatic. LangCore threads a `pass_num` argument to the provider, which injects `cache={"no-cache": True}` for passes ≥ 1:

```python
result = lx.extract(
    text_or_documents="Contract text...",
    model=model,
    prompt_description="Extract all entities.",
    examples=[...],
    extraction_passes=3,  # 3 passes, only the first is cacheable
)
```

### Advanced Configuration

Forward any LiteLLM parameter through `provider_kwargs`:

```python
config = lx.factory.ModelConfig(
    model_id="litellm/gpt-4o",
    provider="LiteLLMLanguageModel",
    provider_kwargs={
        "temperature": 0.2,
        "max_tokens": 2000,
        "top_p": 0.9,
        "timeout": 60,
        "api_key": "sk-...",  # override env var
    },
)
model = lx.factory.create_model(config)
```

#### Reserved Parameters

These parameters are consumed internally and are not forwarded to LiteLLM:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_workers` | `int` | `10` | Maximum concurrent async requests (semaphore size) |
| `pass_num` | `int` | `0` | Current extraction pass index — set automatically by LangCore during multi-pass extraction |

---

## Composing with Other Plugins

langcore-litellm serves as the base provider that other LangCore plugins wrap. Stack decorators to add audit logging, output validation, or hybrid rule-based extraction:

```python
import langcore as lx
from langcore_audit import AuditLanguageModel, LoggingSink
from langcore_guardrails import GuardrailLanguageModel, SchemaValidator, OnFailAction

# Base LLM provider
config = lx.factory.ModelConfig(
    model_id="litellm/gpt-4o",
    provider="LiteLLMLanguageModel",
)
llm = lx.factory.create_model(config)

# Add output validation
guarded = GuardrailLanguageModel(
    model_id="guardrails/gpt-4o",
    inner=llm,
    validators=[SchemaValidator(MySchema, on_fail=OnFailAction.REASK)],
    max_retries=3,
)

# Add audit logging
audited = AuditLanguageModel(
    model_id="audit/gpt-4o",
    inner=guarded,
    sinks=[LoggingSink()],
)

# Use the full stack with LangCore
result = lx.extract(
    text_or_documents="Contract text...",
    model=audited,
    prompt_description="Extract entities.",
    examples=[...],
)
```

---

## Output Format

Extractions return LangCore's standard `AnnotatedDocument` with precise character intervals:

```python
AnnotatedDocument(
    extractions=[
        Extraction(
            extraction_class='party',
            extraction_text='Acme Corp',
            char_interval=CharInterval(start_pos=0, end_pos=9),
            alignment_status=<AlignmentStatus.MATCH_EXACT: 'match_exact'>,
            attributes={'role': 'payer'}
        ),
        Extraction(
            extraction_class='monetary_amount',
            extraction_text='$50,000',
            char_interval=CharInterval(start_pos=24, end_pos=31),
            alignment_status=<AlignmentStatus.MATCH_EXACT: 'match_exact'>,
            attributes={}
        ),
    ],
    text='Acme Corp agrees to pay $50,000 to Beta LLC by March 2025.'
)
```

---

## Error Handling

API failures are captured gracefully — no unhandled exceptions:

```python
ScoredOutput(score=0.0, output="LiteLLM API error: [error details]")
```

---

## Development

```bash
pip install -e .            # Install in development mode
python test_plugin.py       # Run tests
python -m build             # Build package
twine upload dist/*         # Publish to PyPI
```

## Requirements

- Python ≥ 3.12
- `langcore`
- `litellm` ≥ 1.81.13

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
