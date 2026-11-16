"""Explicação das transições com Gemini (seção 4.11, opcional, camada gratuita).

CONTRATO (implementar mantendo as assinaturas):

- SDK `google-genai` instalado: `from google import genai; client = genai.Client(api_key=...)`;
  `client.models.generate_content(model=GEMINI_MODEL, contents=..., config=...)`.
  Pedir resposta em JSON (response_mime_type="application/json") com uma lista de frases.
- O LLM NÃO decide a ordem: só recebe a ordem final e os números e devolve uma frase curta
  (até ~20 palavras, português do Brasil, linguagem de DJ) por transição.
- Se GEMINI_API_KEY não existir, `is_available()` é False e a interface esconde o recurso.
- Em qualquer erro (cota, rede, JSON inválido, quantidade errada de frases), levantar
  ExplainError com mensagem amigável. Nunca registrar a chave em log/mensagem.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
from typing import Any

import config
from modules import util
from modules.models import Analysis, Track, Transition

logger = logging.getLogger(__name__)

GEMINI_TIMEOUT_MS = 60_000         # HttpOptions.timeout do google-genai é em milissegundos
GEMINI_TEMPERATURE = 0.7
MAX_WORDS = 20

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_NUMBERING_RE = re.compile(r"^\s*(?:\d+\s*[.)\-:]|[-*•])\s+")
_LIST_KEYS = ("frases", "phrases", "explicacoes", "explicações", "transicoes", "transições")
_PHRASE_KEYS = ("frase", "phrase", "texto", "text", "explicacao", "explicação")


class ExplainError(Exception):
    pass


def is_available() -> bool:
    """True se há GEMINI_API_KEY no .env e o SDK google-genai está instalado."""
    if not config.get_secret("GEMINI_API_KEY"):
        return False
    try:
        return importlib.util.find_spec("google.genai") is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
def _num(value: float | None, decimals: int = 1) -> str:
    if value is None:
        return "?"
    return f"{value:.{decimals}f}".replace(".", ",")


def _transition_line(i: int, tr: Transition, by_id: dict[str, Track], analyses: dict[str, Analysis]) -> str:
    a_track, b_track = by_id.get(tr.from_id), by_id.get(tr.to_id)
    a_name = a_track.display_name if a_track else tr.from_id
    b_name = b_track.display_name if b_track else tr.to_id
    a, b = analyses.get(tr.from_id), analyses.get(tr.to_id)

    bpm = (f"BPM {_num(a.effective_bpm if a else None)} → {_num(b.effective_bpm if b else None)} "
           f"(diferença {_num(tr.bpm_diff)})")
    energy = (f"energia (0 a 1) na saída de A {_num(a.energy_out if a else None, 2)}, na entrada de B "
              f"{_num(b.energy_in if b else None, 2)} (diferença {_num(tr.energy_diff, 2)})")
    cam_a = a.camelot if a and a.camelot else "?"
    cam_b = b.camelot if b and b.camelot else "?"
    dist = "desconhecida" if tr.camelot_dist is None else str(tr.camelot_dist)
    camelot = f"Camelot {cam_a} → {cam_b} (distância {dist})"
    parts = [bpm, energy, camelot]
    if a and b:
        parts.append(f"saída de A {_num(a.outro_s, 0)} s, intro de B {_num(b.intro_s, 0)} s")
    parts.append(f"voz: {tr.voice}")
    parts.append(f"classificação: {tr.label or 'sem classificação'}")
    return f"{i}. \"{a_name}\" → \"{b_name}\" | " + " | ".join(parts)


def build_prompt(tracks: list[Track], analyses: dict[str, Analysis],
                 transitions: list[Transition]) -> str:
    """Prompt em português com, por transição: nomes A→B, BPM, energia saída/entrada,
    Camelot, rótulo de voz e classificação."""
    by_id = {t.video_id: t for t in tracks}
    n = len(transitions)
    lines = [_transition_line(i, tr, by_id, analyses) for i, tr in enumerate(transitions, start=1)]
    return "\n".join([
        "Você é um DJ experiente de funk mandelão ajudando outro DJ a tocar um set.",
        "A ordem das faixas JÁ FOI DECIDIDA e não deve ser mudada nem comentada.",
        f"Abaixo estão as {n} transições do set, na ordem, com os números medidos de cada uma.",
        "",
        *lines,
        "",
        f"Para cada transição, escreva UMA frase curta (no máximo {MAX_WORDS} palavras), em português do Brasil,",
        "com linguagem de DJ e uma dica prática de mixagem (onde entrar, segurar ou cortar o grave,",
        "cuidado com a voz, ajuste de pitch). Use só os números acima; não invente dados.",
        f"Responda SOMENTE com JSON no formato {{\"frases\": [\"...\", \"...\"]}}, com exatamente {n} frases,",
        "na mesma ordem das transições.",
    ])


# ---------------------------------------------------------------------------
# Leitura da resposta
# ---------------------------------------------------------------------------
def _load_json(text: str) -> Any:
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text, strict=False)  # strict=False aceita quebra de linha dentro das frases
    except ValueError:
        pass
    # Texto em volta do JSON: tenta o trecho entre o primeiro e o último colchete/chave.
    for open_, close in (("{", "}"), ("[", "]")):
        start, end = text.find(open_), text.rfind(close)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1], strict=False)
            except ValueError:
                continue
    raise ExplainError("O Gemini devolveu uma resposta que não é JSON válido. Tente de novo.")


def _as_phrase(item: Any) -> str | None:
    if isinstance(item, dict):
        for key in _PHRASE_KEYS:
            if isinstance(item.get(key), str):
                item = item[key]
                break
        else:
            item = next((v for v in item.values() if isinstance(v, str)), None)
    if not isinstance(item, str):
        return None
    phrase = _NUMBERING_RE.sub("", re.sub(r"\s+", " ", item)).strip()
    return phrase or None


def parse_phrases(text: str | None, expected: int) -> list[str]:
    """Extrai as frases de uma resposta: lista JSON pura ou objeto {"frases": [...]}.

    Aceita cercas de código (```json) e texto em volta. Levanta ExplainError se não
    conseguir ler ou se a quantidade de frases for diferente de `expected`.
    """
    if not text or not text.strip():
        raise ExplainError("O Gemini não devolveu texto. Tente de novo.")
    data = _load_json(text)
    if isinstance(data, dict):
        found = next((data[k] for k in _LIST_KEYS if isinstance(data.get(k), list)), None)
        if found is None:
            found = next((v for v in data.values() if isinstance(v, list)), None)
        data = found
    if not isinstance(data, list):
        raise ExplainError("O Gemini não devolveu uma lista de frases. Tente de novo.")
    phrases = [_as_phrase(item) for item in data]
    if any(p is None for p in phrases):
        raise ExplainError("O Gemini devolveu frases vazias ou em formato inesperado. Tente de novo.")
    if len(phrases) != expected:
        raise ExplainError(f"O Gemini devolveu {len(phrases)} frases para {expected} transições. Tente de novo.")
    return phrases  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Chamada ao Gemini
# ---------------------------------------------------------------------------
def _friendly_error(e: BaseException, model: str) -> ExplainError:
    """Mensagem para o DJ a partir de erros do SDK; nunca inclui o texto cru (pode citar a chave)."""
    code = getattr(e, "code", None)
    status = str(getattr(e, "status", "") or "")
    message = str(getattr(e, "message", "") or "").lower()
    logger.warning("Falha ao chamar o Gemini: %s (código %s, status %s)", type(e).__name__, code, status or "-")
    if code == 429 or status == "RESOURCE_EXHAUSTED":
        return ExplainError("A cota gratuita do Gemini acabou por agora. Espere um minuto (ou até amanhã) e tente de novo.")
    if "api key" in message or "api_key" in message or status == "UNAUTHENTICATED" or code == 401:
        return ExplainError("A GEMINI_API_KEY do .env é inválida ou expirou.")
    if code == 403 or status == "PERMISSION_DENIED":
        return ExplainError("A GEMINI_API_KEY não tem permissão para usar o Gemini (confira a chave no Google AI Studio).")
    if code == 404 or status == "NOT_FOUND":
        return ExplainError(f"O modelo {model} não foi encontrado no Gemini. Ajuste GEMINI_MODEL no config.py.")
    if isinstance(code, int) and code >= 500:
        return ExplainError("O Gemini está instável agora. Tente de novo em instantes.")
    if isinstance(code, int) and code >= 400:
        return ExplainError(f"O Gemini recusou o pedido (HTTP {code}). Tente de novo.")
    name = type(e).__name__.lower()
    if isinstance(e, (ConnectionError, TimeoutError, OSError)) or any(
            w in name for w in ("connect", "timeout", "network", "transport", "ssl")):
        return ExplainError("Sem conexão com o Gemini. Verifique a internet e tente de novo.")
    return ExplainError(f"Erro inesperado ao chamar o Gemini ({type(e).__name__}).")


def _make_client(api_key: str | None):
    key = api_key or config.get_secret("GEMINI_API_KEY")
    if not key:
        raise ExplainError("GEMINI_API_KEY não configurada no .env.")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ExplainError("O pacote google-genai não está instalado (pip install google-genai).") from None
    util.setup_ssl()
    try:
        return genai.Client(api_key=key, http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS))
    except Exception as e:  # noqa: BLE001
        logger.warning("Falha ao criar o cliente do Gemini: %s", type(e).__name__)
        raise ExplainError("Não consegui iniciar o cliente do Gemini. Confira a GEMINI_API_KEY no .env.") from None


def _generation_config():
    try:
        from google.genai import types
    except ImportError:  # cliente injetado sem o SDK: dicionário equivalente
        return {"response_mime_type": "application/json", "temperature": GEMINI_TEMPERATURE}
    return types.GenerateContentConfig(response_mime_type="application/json", temperature=GEMINI_TEMPERATURE)


def _response_text(resp: Any) -> str | None:
    try:
        text = resp.text
    except Exception:  # noqa: BLE001 - resposta sem partes de texto
        text = None
    if text:
        return text
    feedback = getattr(resp, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise ExplainError("O Gemini bloqueou o pedido pelo filtro de conteúdo. Tente de novo mais tarde.")
    return None


def explain_transitions(tracks: list[Track], analyses: dict[str, Analysis],
                        transitions: list[Transition], api_key: str | None = None,
                        model: str | None = None, client=None) -> list[str]:
    """Uma frase por transição (len == len(transitions)). `client` injetável para testes."""
    if not transitions:
        return []
    # Sem modelo explícito, tenta os reservas quando o principal está sobrecarregado/sem cota.
    models = [model] if model else [config.GEMINI_MODEL, *getattr(config, "GEMINI_FALLBACK_MODELS", [])]
    client = client if client is not None else _make_client(api_key)
    prompt = build_prompt(tracks, analyses, transitions)
    for i, m in enumerate(models):
        try:
            resp = client.models.generate_content(model=m, contents=prompt, config=_generation_config())
        except ExplainError:
            raise
        except Exception as e:  # noqa: BLE001 - tudo vira mensagem amigável
            code = getattr(e, "code", None)
            if i + 1 < len(models) and isinstance(code, int) and (code >= 500 or code in (404, 429)):
                logger.info("Gemini %s falhou (%s); tentando %s.", m, code, models[i + 1])
                continue
            raise _friendly_error(e, m) from None
        return parse_phrases(_response_text(resp), len(transitions))
    raise ExplainError("O Gemini não respondeu. Tente de novo.")
