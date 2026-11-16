"""Testes da explicação das transições (modules/explain.py) com cliente Gemini falso."""
from __future__ import annotations

import json

import pytest
from google.genai import errors as genai_errors

import config
from modules import explain, util
from modules.models import Analysis, Track, Transition

KEY = "chave-secreta-gemini-xyz"


def make_set():
    tracks = [
        Track(video_id="aaaaaaaaaaa", title="Casei Com a Putaria", artists=["MC Ryan SP", "Kotim"]),
        Track(video_id="bbbbbbbbbbb", title="Fui Mlk", artists=["Nilo"]),
        Track(video_id="ccccccccccc", title="Felina", artists=["WIU"]),
    ]
    analyses = {
        "aaaaaaaaaaa": Analysis(video_id="aaaaaaaaaaa", duration_s=158, bpm=130.0, bpm_raw=65.0,
                                energy_in=0.4, energy_out=0.72, intro_s=8, outro_s=12, camelot="8A"),
        "bbbbbbbbbbb": Analysis(video_id="bbbbbbbbbbb", duration_s=175, bpm=128.0, bpm_raw=128.0,
                                bpm_manual=131.5, energy_in=0.55, energy_out=0.8, intro_s=16, outro_s=4,
                                camelot="9A"),
        # "ccccccccccc" sem análise (ainda não analisada)
    }
    transitions = [
        Transition(from_id="aaaaaaaaaaa", to_id="bbbbbbbbbbb", cost=0.2, bpm_diff=1.5, energy_diff=0.17,
                   camelot_dist=1, voice="limpa", label="suave"),
        Transition(from_id="bbbbbbbbbbb", to_id="ccccccccccc", cost=1.1, bpm_diff=6.0, energy_diff=0.4,
                   camelot_dist=None, voice="choque de voz", label="difícil"),
    ]
    return tracks, analyses, transitions


class FakeResponse:
    def __init__(self, text=None, block_reason=None):
        self._text = text
        self.prompt_feedback = type("PF", (), {"block_reason": block_reason})() if block_reason else None

    @property
    def text(self):
        return self._text


class FakeClient:
    """Imita client.models.generate_content do google-genai."""

    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []
        self.models = self

    def generate_content(self, *, model, contents, config=None):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        return self.response


@pytest.fixture(autouse=True)
def no_real_key(monkeypatch):
    monkeypatch.setattr(config, "get_secret", lambda name: None)
    monkeypatch.setattr(util, "setup_ssl", lambda: None)


# ---------------------------------------------------------------------------
# is_available e prompt
# ---------------------------------------------------------------------------
def test_is_available(monkeypatch):
    assert explain.is_available() is False
    monkeypatch.setattr(config, "get_secret", lambda name: KEY if name == "GEMINI_API_KEY" else None)
    assert explain.is_available() is True


def test_build_prompt_contains_numbers_and_rules():
    tracks, analyses, transitions = make_set()
    p = explain.build_prompt(tracks, analyses, transitions)
    assert "MC Ryan SP, Kotim - Casei Com a Putaria" in p and "Nilo - Fui Mlk" in p and "WIU - Felina" in p
    assert "BPM 130,0 → 131,5" in p          # usa o BPM corrigido pelo DJ
    assert "diferença 1,5" in p
    assert "saída de A 0,72" in p and "entrada de B 0,55" in p
    assert "Camelot 8A → 9A (distância 1)" in p
    assert "Camelot 9A → ? (distância desconhecida)" in p
    assert "voz: limpa" in p and "voz: choque de voz" in p
    assert "classificação: suave" in p and "classificação: difícil" in p
    assert "BPM 131,5 → ?" in p              # faixa sem análise não quebra o prompt
    assert "exatamente 2 frases" in p
    assert "não deve ser mudada" in p
    assert "20 palavras" in p and "português do Brasil" in p
    assert p.index("1. ") < p.index("2. ")


# ---------------------------------------------------------------------------
# parse_phrases
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    '["Entra no grave da saída.", "Corta a voz antes."]',
    '{"frases": ["Entra no grave da saída.", "Corta a voz antes."]}',
    '```json\n{"frases": ["Entra no grave da saída.", "Corta a voz antes."]}\n```',
    '```\n["Entra no grave da saída.", "Corta a voz antes."]\n```',
    'Claro! Aqui está:\n{"frases": ["Entra no grave da saída.", "Corta a voz antes."]}\nBom set!',
    '[{"frase": "1. Entra no grave da saída."}, {"frase": "2) Corta a voz antes."}]',
    '{"resultado": ["  Entra no grave\n da saída. ", "Corta a voz antes."]}',
])
def test_parse_phrases_formats(text):
    assert explain.parse_phrases(text, 2) == ["Entra no grave da saída.", "Corta a voz antes."]


@pytest.mark.parametrize("text, match", [
    ('["só uma"]', "1 frases para 2"),
    ('["a", "b", "c"]', "3 frases para 2"),
    ("isso não é json", "JSON"),
    ('{"ok": true}', "lista"),
    ('["a", ""]', "vazias"),
    ('["a", 3]', "formato"),
    ("", "não devolveu texto"),
    (None, "não devolveu texto"),
])
def test_parse_phrases_errors(text, match):
    with pytest.raises(explain.ExplainError, match=match):
        explain.parse_phrases(text, 2)


# ---------------------------------------------------------------------------
# explain_transitions
# ---------------------------------------------------------------------------
def test_explain_transitions_with_fake_client():
    tracks, analyses, transitions = make_set()
    client = FakeClient(FakeResponse(json.dumps({"frases": ["Mixa na intro de 16 s.", "Cuidado: voz em cima de voz."]})))
    out = explain.explain_transitions(tracks, analyses, transitions, client=client)
    assert out == ["Mixa na intro de 16 s.", "Cuidado: voz em cima de voz."]
    call = client.calls[0]
    assert call["model"] == config.GEMINI_MODEL
    assert call["contents"] == explain.build_prompt(tracks, analyses, transitions)
    assert call["config"].response_mime_type == "application/json"


def test_explain_transitions_model_override_and_empty():
    tracks, analyses, transitions = make_set()
    client = FakeClient(FakeResponse('["a", "b"]'))
    explain.explain_transitions(tracks, analyses, transitions, model="gemini-x", client=client)
    assert client.calls[0]["model"] == "gemini-x"
    assert explain.explain_transitions(tracks, analyses, [], client=FakeClient(error=AssertionError())) == []


def test_explain_transitions_wrong_count_raises():
    tracks, analyses, transitions = make_set()
    with pytest.raises(explain.ExplainError, match="3 frases para 2"):
        explain.explain_transitions(tracks, analyses, transitions, client=FakeClient(FakeResponse('["a","b","c"]')))


def test_explain_transitions_without_key():
    tracks, analyses, transitions = make_set()
    with pytest.raises(explain.ExplainError, match="GEMINI_API_KEY"):
        explain.explain_transitions(tracks, analyses, transitions)


def test_explain_transitions_builds_real_client_with_key(monkeypatch):
    from google import genai

    created = {}

    class FakeGenaiClient(FakeClient):
        def __init__(self, *, api_key=None, http_options=None, **kw):
            super().__init__(FakeResponse('["a", "b"]'))
            created.update(api_key=api_key, http_options=http_options)

    monkeypatch.setattr(genai, "Client", FakeGenaiClient)
    tracks, analyses, transitions = make_set()
    assert explain.explain_transitions(tracks, analyses, transitions, api_key=KEY) == ["a", "b"]
    assert created["api_key"] == KEY
    assert created["http_options"].timeout == explain.GEMINI_TIMEOUT_MS
    # Sem api_key explícita, usa a do .env.
    monkeypatch.setattr(config, "get_secret", lambda name: "do-env" if name == "GEMINI_API_KEY" else None)
    explain.explain_transitions(tracks, analyses, transitions)
    assert created["api_key"] == "do-env"


@pytest.mark.parametrize("error, match", [
    (genai_errors.ClientError(429, {"error": {"code": 429, "message": f"Quota exceeded for {KEY}",
                                              "status": "RESOURCE_EXHAUSTED"}}), "cota"),
    (genai_errors.ClientError(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key.",
                                              "status": "INVALID_ARGUMENT"}}), "inválida"),
    (genai_errors.ClientError(403, {"error": {"code": 403, "message": "denied", "status": "PERMISSION_DENIED"}}),
     "permissão"),
    (genai_errors.ClientError(404, {"error": {"code": 404, "message": "not found", "status": "NOT_FOUND"}}),
     "não foi encontrado"),
    (genai_errors.ServerError(503, {"error": {"code": 503, "message": "overloaded", "status": "UNAVAILABLE"}}),
     "instável"),
    (ConnectionError(f"falhou com key={KEY}"), "Sem conexão"),
    (TimeoutError(), "Sem conexão"),
    (RuntimeError(f"algo {KEY}"), "inesperado"),
])
def test_explain_transitions_api_errors(error, match, caplog):
    tracks, analyses, transitions = make_set()
    with caplog.at_level("DEBUG"):
        with pytest.raises(explain.ExplainError, match=match) as e:
            explain.explain_transitions(tracks, analyses, transitions, client=FakeClient(error=error))
    assert KEY not in str(e.value)
    assert KEY not in caplog.text


def test_explain_transitions_httpx_errors():
    import httpx

    tracks, analyses, transitions = make_set()
    for err in (httpx.ConnectError("x"), httpx.ReadTimeout("x")):
        with pytest.raises(explain.ExplainError, match="Sem conexão"):
            explain.explain_transitions(tracks, analyses, transitions, client=FakeClient(error=err))


def test_explain_transitions_blocked_or_empty_response():
    tracks, analyses, transitions = make_set()
    with pytest.raises(explain.ExplainError, match="bloqueou"):
        explain.explain_transitions(tracks, analyses, transitions,
                                    client=FakeClient(FakeResponse(None, block_reason="PROHIBITED_CONTENT")))
    with pytest.raises(explain.ExplainError, match="não devolveu texto"):
        explain.explain_transitions(tracks, analyses, transitions, client=FakeClient(FakeResponse(None)))


def test_explain_with_real_sdk_response_object():
    """Resposta montada com os tipos reais do SDK (formato que o generate_content devolve)."""
    from google.genai import types

    resp = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        role="model", parts=[types.Part(text='{"frases": ["Solta o grave no 1.", "Segura a voz de B."]}')]))])
    tracks, analyses, transitions = make_set()
    assert explain.explain_transitions(tracks, analyses, transitions, client=FakeClient(resp)) == \
        ["Solta o grave no 1.", "Segura a voz de B."]


@pytest.mark.network
def test_real_gemini(monkeypatch):
    """Chamada real (camada gratuita). Pula se não houver GEMINI_API_KEY no .env."""
    import os

    monkeypatch.undo()  # volta o setup_ssl real (interceptação TLS nesta máquina)
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        pytest.skip("GEMINI_API_KEY não definida")
    tracks, analyses, transitions = make_set()
    out = explain.explain_transitions(tracks, analyses, transitions, api_key=key)
    assert len(out) == 2 and all(out)
