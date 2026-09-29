# utils/llm_clients.py
"""
Client LLM astratti (dependency injection).
Ogni funzione ha firma: chiama_llm(prompt: str) -> str

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
import time
from datetime import datetime


def _log(path, model, prompt, risposta, provider=None):
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


# ============================================================
# GOOGLE GEMINI (free tier, no carta di credito)
# ============================================================
def crea_client_gemini(
    model="gemini-3.5-flash-lite",
    temperature=0,
    log_file=None,
    response_format_json=False,
    timeout=120,
):
    """
    Restituisce una funzione chiama_llm(prompt) -> str.
    Richiede GEMINI_API_KEY nell'environment.
    Usa l'endpoint compatibile OpenAI di Google.

    Modelli consigliati (free tier):
      - gemini-3.5-flash-lite
      - gemini-3.5-flash
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        timeout=timeout,
    )

    def chiama(prompt: str) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        out = r.choices[0].message.content
        _log(log_file, model, prompt, out, provider="gemini")
        return out

    return chiama


# ============================================================
# MISTRAL (free tier, no carta di credito)
# ============================================================
def crea_client_mistral(
    model="ministral-3b-2512",
    temperature=0,
    log_file=None,
    response_format_json=False,
    timeout=120,
    pausa_tra_chiamate=1.5,
):
    """
    Restituisce una funzione chiama_llm(prompt) -> str.
    Richiede MISTRAL_API_KEY nell'environment.

    Usa l'endpoint compatibile OpenAI di Mistral.

    Modelli consigliati (free tier):
      - ministral-3b-2512        (12.5 RPS, 1.3M TPM)
      - codestral-2508           (2 RPS, 625K TPM)
      - mistral-large-2512       (1 RPS, 250K TPM)
      - mistral-small-latest     (1 RPS, 20K TPM)

    NOTA: la pausa_tra_chiamate evita errori 429 sul free tier.
    """
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1",
        timeout=timeout,
    )

    def chiama(prompt: str) -> str:
        if pausa_tra_chiamate > 0:
            time.sleep(pausa_tra_chiamate)

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        out = r.choices[0].message.content
        _log(log_file, model, prompt, out, provider="mistral")
        return out

    return chiama


# ============================================================
# OPENAI (a pagamento)
# ============================================================
def crea_client_openai(
    model="gpt-4o-mini",
    temperature=0.2,
    log_file=None,
    response_format_json=True,
    timeout=120,
):
    """
    Restituisce una funzione chiama_llm(prompt) -> str.
    Richiede OPENAI_API_KEY nell'environment.
    """
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=timeout)

    def chiama(prompt: str) -> str:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kwargs)
        out = r.choices[0].message.content
        _log(log_file, model, prompt, out, provider="openai")
        return out

    return chiama


# ============================================================
# ANTHROPIC (a pagamento)
# ============================================================
def crea_client_anthropic(
    model="claude-3-5-sonnet-latest",
    max_tokens=4096,
    log_file=None,
    timeout=120,
):
    """
    Restituisce una funzione chiama_llm(prompt) -> str.
    Richiede ANTHROPIC_API_KEY nell'environment.

    NOTA: i modelli Claude recenti non accettano più 'temperature'
    come parametro, quindi non lo passiamo.
    """
    from anthropic import Anthropic
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=timeout)

    def chiama(prompt: str) -> str:
        r = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        out = r.content[0].text
        _log(log_file, model, prompt, out, provider="anthropic")
        return out

    return chiama


# ============================================================
# OLLAMA (locale, illimitato)
# ============================================================
def crea_client_ollama(
    model="llama3.1:8b",
    temperature=0.2,
    host="http://localhost:11434",
    log_file=None,
    timeout=300,
):
    """
    Restituisce una funzione chiama_llm(prompt) -> str.
    Richiede un server Ollama attivo su host.
    Nessuna chiave API necessaria.
    """
    import requests

    def chiama(prompt: str) -> str:
        r = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=timeout,
        )
        r.raise_for_status()
        out = r.json().get("response", "")
        _log(log_file, model, prompt, out, provider="ollama")
        return out

    return chiama


# ============================================================
# FACTORY: crea client da nome provider
# ============================================================
def crea_client(provider: str, model: str = None, log_file=None, pausa=None):
    """
    Factory che restituisce una funzione chiama_llm(prompt) in base
    al nome del provider.

    Provider supportati:
      'gemini', 'mistral', 'openai', 'anthropic', 'ollama'
    """
    provider = provider.lower().strip()

    if provider == "gemini":
        return crea_client_gemini(model=model or "gemini-3.5-flash-lite", log_file=log_file)
    if provider == "mistral":
        kwargs = {}
        if pausa is not None:
            kwargs["pausa_tra_chiamate"] = pausa
        return crea_client_mistral(
            model=model or "ministral-3b-2512",
            log_file=log_file,
            **kwargs,
        )
    if provider == "openai":
        return crea_client_openai(model=model or "gpt-4o-mini", log_file=log_file)
    if provider == "anthropic":
        return crea_client_anthropic(model=model or "claude-3-5-sonnet-latest", log_file=log_file)
    if provider == "ollama":
        return crea_client_ollama(model=model or "llama3.1:8b", log_file=log_file)

    raise ValueError(f"Provider non supportato: {provider}")


# ============================================================
# CATALOGO PROVIDER (per la UI)
# ============================================================
PROVIDER_DISPONIBILI = {
    "gemini": {
        "label": "Google Gemini ",
        "models": ["gemini-3.5-flash-lite", "gemini-3.5-flash"],
        "env_var": "GEMINI_API_KEY",
        "gratis": True,
    },

    "mistral": {
        "label": "Mistral ",
        "models": [
            "ministral-3b-2512",       # consigliato: 12.5 RPS, 1.3M TPM
            "codestral-2508",           # 2 RPS, 625K TPM
            "mistral-large-2512",       # 1 RPS, 250K TPM
            "mistral-small-latest",     # 1 RPS, 20K TPM (lento)
        ],
        "env_var": "MISTRAL_API_KEY",
        "gratis": True,
    },
    "openai": {
        "label": "OpenAI ",
        "models": ["gpt-4o-mini", "gpt-4o"],
        "env_var": "OPENAI_API_KEY",
        "gratis": False,
    },
    "anthropic": {
        "label": "Anthropic Claude ",
        "models": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
        "env_var": "ANTHROPIC_API_KEY",
        "gratis": False,
    },
    "ollama": {
        "label": "Ollama ",
        "models": ["llama3.1:8b", "qwen2.5:7b", "mistral:7b"],
        "env_var": None,
        "gratis": True,
    },
}