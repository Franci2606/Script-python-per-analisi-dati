# utils/llm_async.py
"""
Client LLM asincroni (dependency injection).
Ogni funzione ha firma: async chiama_llm(prompt: str) -> str

Provider supportati:
- Google Gemini  (free tier)   → GEMINI_API_KEY
- Mistral       (free tier)   → MISTRAL_API_KEY
- OpenAI (GPT)  (a pagamento)           → OPENAI_API_KEY
- Anthropic (Claude) (a pagamento)      → ANTHROPIC_API_KEY
- Ollama        (locale, illimitato)    → nessuna chiave

Le chiamate vengono loggate su file JSONL (una riga per chiamata).
"""
import os
import json
from datetime import datetime


def _log_async(path, model, prompt, risposta, provider=None):
    if not path:
        return
    record = {
        "provider": provider,
        "modello": model,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "risposta": risposta,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def crea_client_async(provider, model=None, log_file=None, timeout=300):
    """
    Restituisce una coroutine function: async chiama(prompt) -> str.

    Provider supportati:
      'gemini', 'mistral', 'openai', 'anthropic'
    (Ollama non ha ancora una versione async qui: usa la sincrona.)
    """
    from openai import AsyncOpenAI

    base_urls = {
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "mistral": "https://api.mistral.ai/v1",
        "openai": None,
    }
    env_vars = {
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    defaults = {
        "gemini": "gemini-3.5-flash-lite",
        "mistral": "ministral-3b-2512",
        "openai": "gpt-4o-mini",
    }

    provider = provider.lower().strip()

    if provider == "anthropic":
        # Anthropic ha un client async dedicato
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            timeout=timeout,
        )
        model = model or "claude-3-5-sonnet-latest"

        async def chiama(prompt: str) -> str:
            r = await client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            out = r.content[0].text
            _log_async(log_file, model, prompt, out, provider="anthropic")
            return out

        return chiama

    if provider not in base_urls:
        raise ValueError(f"Provider non supportato in async: {provider}")

    kwargs = {"api_key": os.getenv(env_vars[provider]), "timeout": timeout}
    if base_urls[provider]:
        kwargs["base_url"] = base_urls[provider]

    client = AsyncOpenAI(**kwargs)
    model = model or defaults[provider]

    async def chiama(prompt: str) -> str:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=8192, 
        )
        out = r.choices[0].message.content
        _log_async(log_file, model, prompt, out, provider=provider)
        return out

    return chiama