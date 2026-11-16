"""Painel local de curadoria musical do DJ SK (Streamlit).

Rode com:  streamlit run app.py

Fluxo: 1 referências → sugestões; 2 triagem com player (aprovar/reprovar);
3 setlist (estilo, ordem, passagens, Estúdio de Vocal); 4 exportação (Rekordbox e Spotify);
5 métricas de tempo.

O Streamlit roda este arquivo de novo a cada clique. Por isso os objetos de longa
duração (fila de downloads, orçamento, cronômetro) e as decisões ficam em
`st.session_state`, e a curadoria é salva em disco (modules/state) a cada mudança.
O visual (cores, cartões, cabeçalho, linha do tempo) vem de modules/theme e modules/ui_parts.
"""
from __future__ import annotations

import logging
import math
import mimetypes
import os
import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from modules import (analyze, budget, download, explain, export, jobs, maintenance,  # noqa: E402
                     metrics, mixwindows, order, preview, spotify_import, state, stems, suggest,
                     theme, ui_parts, util, viz)
from modules import setlist_insights as si  # noqa: E402
from modules.models import Analysis, Track, Transition  # noqa: E402

logger = logging.getLogger(__name__)

APPROVED = "aprovada"
REJECTED = "reprovada"
FINAL_PREVIEW_S = 30                 # "Final" toca os últimos 30 s
PROGRESS_REFRESH_S = 1.0             # intervalo do fragmento que redesenha o progresso
METRICS_REFRESH_S = 5                # intervalo dos números ao vivo das métricas
MORE_LIKE_MAX = 10                   # quantas faixas novas o "Mais parecidas" acrescenta
SPOTIFY_RESULTS = 8                  # resultados da busca no Spotify
WEIGHT_LABELS = {"bpm": "Velocidade (BPM)", "energia": "Energia", "tom": "Tom", "voz": "Voz (MC)"}
VOICE_TEXT = {
    "limpa": "limpa ✔",
    "voz de um lado": "voz de um lado ⚠",
    "choque de voz": "choque de voz ✖",
    "sem stems": "sem stems",
}
FILTERS = ["Pendentes", "Aprovadas", "Reprovadas", "Todas"]
SORTS = ["Relevância", "BPM", "Energia"]
POSITIONS = {"inicio": "Início", "meio": "Meio", "final": "Final"}
TOP = "__topo__"                     # destino das mensagens que aparecem no alto da página
REFS = "__referencias__"             # mensagens da seção de referências

_header_slot = None                  # st.empty() do cabeçalho com o orçamento (redesenhado após gastos)
_stepper_slot = None                 # st.empty() da barra de etapas


# ---------------------------------------------------------------------------
# Utilitários de texto e formatação
# ---------------------------------------------------------------------------
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:])")
_KEY_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def _html(body: Any) -> None:
    """st.html que ignora HTML vazio (ex.: card ainda sem selos): o Streamlit exige corpo."""
    if body is not None and str(body).strip():
        st.html(body)


def _md(text: Any) -> str:
    """Escapa texto vindo de fora (títulos, artistas) para não virar Markdown/LaTeX."""
    return _MD_SPECIAL.sub(r"\\\1", str(text or ""))


def _key_part(text: Any, max_len: int = 48) -> str:
    """Pedaço seguro para usar em `key` de widget/container."""
    return _KEY_SAFE.sub("_", str(text or ""))[:max_len] or "x"


def _reload_env() -> None:
    """Relê o .env a cada execução: chaves novas valem sem reiniciar o painel."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(config.BASE_DIR / ".env", override=True)


def _no_latex(text: str) -> str:
    """Dois "$" no mesmo texto (R$ e US$) viram fórmula LaTeX no Markdown do Streamlit."""
    return text.replace("$", r"\$")


def _dec(value: Any, decimals: int = 1) -> str:
    """Número com vírgula decimal (pt-BR)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(x):
        return "–"
    return f"{x:.{decimals}f}".replace(".", ",")


def _brl_plain(value: float) -> str:
    """"10,00" (sem o "R$ ")."""
    return util.format_brl(value).replace("R$", "").strip()


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(round(max(0.0, seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    return f"{m} min {s:02d} s"


def _fmt_saved_at(text: Any) -> str:
    try:
        return datetime.fromisoformat(str(text)).strftime("%d/%m/%Y %H:%M")
    except (TypeError, ValueError):
        return str(text or "")


def _known_errors() -> tuple[type, ...]:
    names = [(suggest, "SuggestError"), (download, "DownloadError"), (stems, "StemsError"),
             (spotify_import, "SpotifyError"), (explain, "ExplainError"), (preview, "PreviewError")]
    return tuple(c for c in (getattr(m, n, None) for m, n in names) if isinstance(c, type))


def _error_message(exc: BaseException, action: str) -> str:
    """Mensagem amigável (sem stack trace) para mostrar com st.error.

    Chame dentro do `except`, para o log do terminal guardar o traceback dos erros inesperados.
    """
    if isinstance(exc, _known_errors()):
        return str(exc) or f"Não foi possível {action}."
    if isinstance(exc, NotImplementedError):
        return f"Não foi possível {action}: este recurso ainda não está disponível nesta versão do painel."
    if isinstance(exc, OSError):
        logger.warning("Erro de arquivo ao %s: %s", action, exc)
        detail = exc.strerror or str(exc)
        return (f"Não foi possível {action}: {detail}. Confira se a pasta existe e se algum arquivo "
                "não está aberto em outro programa (ex.: Rekordbox).")
    logger.exception("Erro inesperado ao %s", action)
    return f"Não foi possível {action} (erro inesperado: {type(exc).__name__}). Os detalhes estão no terminal."


def _flash(where: str, kind: str, message: str) -> None:
    """Guarda uma mensagem para a próxima renderização (callbacks rodam antes da página)."""
    st.session_state.setdefault("flash", []).append((where, kind, message))


def _show_flash(where: str) -> None:
    msgs = st.session_state.get("flash") or []
    keep = []
    for w, kind, message in msgs:
        if w == where:
            getattr(st, kind, st.info)(message)
        else:
            keep.append((w, kind, message))
    st.session_state.flash = keep


# ---------------------------------------------------------------------------
# Sessão
# ---------------------------------------------------------------------------
def _defaults() -> dict[str, Any]:
    return {
        "ref_count": 1,
        "references": [],            # list[Track] resolvidas na última busca
        "suggestions": [],           # list[Track] candidatas (lista completa ranqueada)
        "page": 0,                   # quantas páginas de N_SUGGESTIONS já foram para a triagem
        "extra_visible": 0,          # faixas acrescentadas pelo "Mais parecidas" (visíveis além das páginas)
        "similar_of": {},            # video_id -> título da faixa que trouxe esta pelo "Mais parecidas"
        "decisions": {},             # video_id -> "aprovada" | "reprovada" (ordem = ordem de decisão)
        "order": [],                 # ordem do set (video_ids) gerada/ajustada
        "weights": dict(config.DEFAULT_WEIGHTS),
        "preset": si.DEFAULT_PRESET,  # estilo do set escolhido
        "energy_mode": "suave",      # "suave" | "subindo" (ordenação)
        "arc": ui_parts.DEFAULT_ARC,  # arco de energia alvo do mapa do set
        "catalog": {},               # video_id -> Track de tudo que passou pela triagem
        "active": None,              # {"vid", "pos"}: card que está com o player
        "focus_vid": None,           # card que recebe os atalhos de teclado
        "preview": None,             # {"a", "b", "path"}: passagem tocando no setlist
        "explanations": None,        # {"order": [...], "texts": [...]} do Gemini
        "stem_results": {},          # video_id -> resumo da separação
        "stem_player": None,         # video_id cujo vocal está aberto no player
        "ref_report": [],
        "suggest_warnings": [],
        "resolved_cache": {},        # texto digitado -> Track (evita buscar de novo)
        "export_report": None,
        "spotify_export": None,      # resultado da playlist criada no Spotify
        "spotify_import_report": None,
        "sp_results": [],            # resultados da busca no Spotify
        "restored_at": None,
        "confirm_end": False,
        "reset_open": False,         # janela "Zerar painel" aberta
        "rendered_terminal": frozenset(),
        "visible_ids": [],
        "voice_lookup": None,
        "progress_timer": False,
    }


# Chaves de widgets que guardam valores da curadoria: somem junto com ela.
_CURATION_WIDGET_KEYS = ("set_preset", "energy_rising", "set_arc", "start_id",
                         *(f"w_{k}" for k in WEIGHT_LABELS))


def _make_budget():
    try:
        return budget.Budget()
    except Exception:  # noqa: BLE001
        logger.exception("Falha ao abrir o registro de orçamento")
        return None


def _make_metrics():
    try:
        return metrics.SessionMetrics()
    except Exception:  # noqa: BLE001
        logger.exception("Falha ao iniciar o cronômetro da sessão")
        return None


def _jobs():
    """Fila de downloads/análises da sessão (criada sob demanda)."""
    ss = st.session_state
    if ss.get("jobs") is None:
        try:
            ss.jobs = jobs.TrackJobs()
            ss.pop("jobs_error", None)
        except Exception as e:  # noqa: BLE001
            ss.jobs_error = _error_message(e, "iniciar a fila de downloads")
            return None
    return ss.jobs


def _init_session() -> None:
    ss = st.session_state
    util.setup_ssl()
    config.ensure_dirs()
    _reload_env()
    if ss.get("booted"):
        return
    for key, value in _defaults().items():
        ss.setdefault(key, value)
    ss.setdefault("flash", [])
    ss.budget = _make_budget()
    ss.metrics = _make_metrics()
    ss.jobs = None
    _restore_saved_session()
    ss.booted = True


def _tracks_from(items: Any) -> list[Track]:
    out = []
    for d in items or []:
        try:
            out.append(d if isinstance(d, Track) else Track.from_dict(d))
        except (TypeError, ValueError, KeyError, AttributeError):
            logger.warning("Faixa inválida ignorada na sessão salva: %r", d)
    return out


def _restore_saved_session() -> None:
    try:
        snap = state.load_session()
    except Exception as e:  # noqa: BLE001
        logger.warning("Não foi possível ler a sessão salva: %s", e)
        return
    if not isinstance(snap, dict) or not (snap.get("suggestions") or snap.get("decisions")):
        return
    ss = st.session_state
    refs = _tracks_from(snap.get("references"))
    sugg = _tracks_from(snap.get("suggestions"))
    extra = _tracks_from(snap.get("extra_tracks"))
    ss.references = refs
    ss.suggestions = sugg
    ss.catalog = {t.video_id: t for t in [*extra, *refs, *sugg]}
    try:
        ss.page = max(0, int(snap.get("page") or 0)) or (1 if sugg else 0)
    except (TypeError, ValueError):
        ss.page = 1 if sugg else 0
    try:
        ss.extra_visible = max(0, int(snap.get("extra_visible") or 0))
    except (TypeError, ValueError):
        ss.extra_visible = 0
    similar = snap.get("similar_of")
    ss.similar_of = ({str(k): str(v) for k, v in similar.items() if v} if isinstance(similar, dict) else {})
    decisions = snap.get("decisions") or {}
    ss.decisions = {str(k): v for k, v in decisions.items() if v in (APPROVED, REJECTED)}
    ss.order = [v for v in snap.get("order") or [] if isinstance(v, str)]
    weights = dict(config.DEFAULT_WEIGHTS)
    for k, v in (snap.get("weights") or {}).items():
        try:
            weights[k] = float(v)
        except (TypeError, ValueError):
            pass
    ss.weights = weights
    preset = snap.get("preset")
    ss.preset = preset if preset in si.PRESETS else si.DEFAULT_PRESET
    mode = snap.get("energy_mode")
    ss.energy_mode = mode if mode in order.ENERGY_MODES else "suave"
    arc = snap.get("set_arc")
    if arc in ui_parts.ARCS:
        ss.arc = arc
    texts = [str(t) for t in snap.get("ref_texts") or [] if t]
    texts = texts[: config.MAX_REFERENCES]
    for i, text in enumerate(texts):
        ss[f"ref_{i}"] = text
    ss.ref_count = max(1, len(texts))
    met = ss.get("metrics")
    if met is not None and hasattr(met, "restore"):
        met.restore(snap.get("metrics"))
        _sync_metric_decisions(met)
    ss.restored_at = snap.get("saved_at") or "?"
    # Reenvia para a fila: o que já está no cache volta pronto em instantes.
    approved = [t for vid, t in ss.catalog.items() if ss.decisions.get(vid) == APPROVED]
    _submit(_visible_tracks() + approved)


def _sync_metric_decisions(met) -> None:
    """Sessão salva antiga (sem as decisões no cronômetro): copia as decisões da triagem."""
    ss = st.session_state
    try:
        if (not ss.decisions or not getattr(met, "started", False) or getattr(met, "finished", False)
                or not hasattr(met, "record_decision") or met.decisions):
            return
        for vid, d in ss.decisions.items():
            met.record_decision(vid, d)
    except Exception:  # noqa: BLE001
        logger.warning("Falha ao copiar as decisões para as métricas", exc_info=True)


def _snapshot() -> dict[str, Any]:
    ss = st.session_state
    sugg_ids = {t.video_id for t in ss.suggestions}
    extra = [t for vid, t in ss.catalog.items() if vid in ss.decisions and vid not in sugg_ids]
    met = ss.get("metrics")
    return {
        "references": [t.to_dict() for t in ss.references],
        "suggestions": [t.to_dict() for t in ss.suggestions],
        "page": int(ss.page),
        "decisions": dict(ss.decisions),
        "order": list(ss.order),
        "weights": dict(ss.weights),
        "metrics": met.to_dict() if met is not None and hasattr(met, "to_dict") else {},
        # Extras do painel (além do contrato de modules/state):
        "extra_tracks": [t.to_dict() for t in extra],   # decididas que saíram da lista atual
        "ref_texts": [ss.get(f"ref_{i}", "") for i in range(ss.ref_count)],
        "extra_visible": int(ss.get("extra_visible") or 0),
        "similar_of": dict(ss.get("similar_of") or {}),
        "preset": ss.get("preset") or si.DEFAULT_PRESET,
        "energy_mode": ss.get("energy_mode") or "suave",
        "set_arc": _current_arc(),
    }


def _persist() -> None:
    try:
        state.save_session(_snapshot())
    except Exception as e:  # noqa: BLE001
        logger.warning("Não foi possível salvar a sessão em disco: %s", e)


def _shutdown_jobs() -> None:
    ss = st.session_state
    old_jobs = ss.get("jobs")
    if old_jobs is not None:
        try:
            old_jobs.shutdown(wait=False)
        except Exception:  # noqa: BLE001
            logger.warning("Falha ao encerrar a fila de downloads", exc_info=True)
    ss.jobs = None


def _reset_curation(clear_refs: bool = True) -> None:
    """Volta a curadoria ao começo (não mexe no cache nem no orçamento)."""
    ss = st.session_state
    _shutdown_jobs()
    for key, value in _defaults().items():
        if key == "ref_count" and not clear_refs:
            continue
        ss[key] = value
    if clear_refs:
        for i in range(config.MAX_REFERENCES):
            ss[f"ref_{i}"] = ""
    for key in _CURATION_WIDGET_KEYS:
        ss.pop(key, None)
    ss.metrics = _make_metrics()
    ss.pop("spotify_client", None)
    ss.pop("spotify_user", None)
    ss.pop("spotify_playlists", None)


# ---------------------------------------------------------------------------
# Faixas, fila e análises
# ---------------------------------------------------------------------------
def _loaded_count() -> int:
    """Quantas sugestões estão na triagem (páginas carregadas + as do "Mais parecidas")."""
    ss = st.session_state
    pages = max(0, int(ss.page or 0))
    if not pages:
        return 0
    return pages * config.N_SUGGESTIONS + max(0, int(ss.get("extra_visible") or 0))


def _visible_tracks() -> list[Track]:
    ss = st.session_state
    return list(ss.suggestions[: _loaded_count()])


_UNPLAYABLE = ("restrição de idade", "indisponível ou privado")


def _unplayable(vid: str) -> bool:
    """Vídeo com restrição de idade ou indisponível: nunca vai baixar, some da triagem."""
    tj = _jobs()
    if tj is None:
        return False
    stt = tj.status(vid)
    return stt.state == jobs.ERROR and any(m in (stt.error or "") for m in _UNPLAYABLE)


def _submit(tracks: list[Track]) -> None:
    if not tracks:
        return
    tj = _jobs()
    if tj is None:
        return
    try:
        tj.submit(list(tracks))
    except Exception as e:  # noqa: BLE001
        _flash(TOP, "error", _error_message(e, "enviar as faixas para download"))


def _set_candidates(tracks: list[Track]) -> None:
    """Coloca uma nova lista de candidatas na triagem (1ª página vai para a fila)."""
    ss = st.session_state
    seen: set[str] = set()
    unique = []
    for t in tracks:
        if t.video_id and t.video_id not in seen:
            seen.add(t.video_id)
            unique.append(t)
    ss.suggestions = unique
    for t in unique:
        ss.catalog[t.video_id] = t
    ss.page = 1 if unique else 0
    ss.extra_visible = 0
    ss.similar_of = {}
    ss.active = None
    ss.export_report = None
    ss.spotify_export = None
    _submit(unique[: config.N_SUGGESTIONS])
    _persist()


def _start_metrics_if_needed() -> None:
    """Marca o início da medição. Uma nova busca antes de qualquer decisão recomeça o cronômetro;
    depois que o DJ já começou a triar, o início original é mantido."""
    ss = st.session_state
    met = ss.get("metrics")
    if met is None:
        return
    try:
        if not met.started or getattr(met, "finished", False) or not ss.decisions:
            met.start()
    except Exception:  # noqa: BLE001
        logger.warning("Falha ao marcar o início da sessão", exc_info=True)


def _status(vid: str):
    tj = st.session_state.get("jobs")
    if tj is None:
        return jobs.JobStatus()
    try:
        return tj.status(vid)
    except Exception:  # noqa: BLE001
        return jobs.JobStatus()


def _analysis(vid: str) -> Analysis | None:
    tj = st.session_state.get("jobs")
    if tj is None:
        return None
    try:
        return tj.get_analysis(vid)
    except Exception:  # noqa: BLE001
        return None


def _audio_path(track: Track | None) -> Path | None:
    if track is None:
        return None
    try:
        p = download.audio_path_for(track)
    except Exception:  # noqa: BLE001
        return None
    return Path(p) if p else None


def _mime(path: Path) -> str:
    return mimetypes.guess_type(str(path))[0] or "audio/mpeg"


def _duration(track: Track, an: Analysis | None) -> float | None:
    if an is not None and an.duration_s:
        return float(an.duration_s)
    return float(track.duration_s) if track.duration_s else None


def _play_window(pos: str, duration: float | None) -> tuple[int, int | None]:
    """(start_time, end_time) do st.audio para Início / Meio / Final."""
    if not duration or duration <= 0:
        return 0, None
    if pos == "meio":
        return int(duration / 2), None
    if pos == "final":
        start = int(max(0.0, duration - FINAL_PREVIEW_S))
        return start, max(start + 1, int(math.ceil(duration)))
    return 0, None


def _terminal_ids(ids: list[str]) -> frozenset:
    """Faixas que já terminaram (prontas ou com erro): mudou → vale redesenhar os cards."""
    out = set()
    for vid in ids:
        if _status(vid).state in (jobs.READY, jobs.ERROR):
            out.add(vid)
    return frozenset(out)


def _approved_tracks() -> list[Track]:
    ss = st.session_state
    return [ss.catalog[vid] for vid, d in ss.decisions.items() if d == APPROVED and vid in ss.catalog]


def _has_stems(vid: str) -> bool:
    try:
        return bool(stems.has_stems(vid))
    except Exception:  # noqa: BLE001
        return False


def _vocals_path(vid: str) -> Path | None:
    res = st.session_state.stem_results.get(vid) or {}
    if res.get("vocals"):
        return Path(res["vocals"])
    try:
        cached = stems.load_stems(vid)
    except Exception:  # noqa: BLE001
        return None
    return cached.vocals_path if cached is not None else None


def _voice_lookup(refresh: bool = False):
    ss = st.session_state
    if refresh or ss.get("voice_lookup") is None:
        try:
            ss.voice_lookup = mixwindows.make_voice_lookup()
        except Exception as e:  # noqa: BLE001
            logger.warning("Janelas de voz indisponíveis: %s", e)
            ss.voice_lookup = None
    return ss.voice_lookup


def _record_metric_decision(vid: str, decision: str | None) -> None:
    """Leva a decisão (ou o "desfazer") para o cronômetro das métricas."""
    met = st.session_state.get("metrics")
    if met is None:
        return
    try:
        if decision is None:
            if hasattr(met, "undo_decision"):
                met.undo_decision(vid)
        elif hasattr(met, "record_decision"):
            met.record_decision(vid, decision)
    except Exception:  # noqa: BLE001
        logger.warning("Falha ao registrar a decisão nas métricas", exc_info=True)


# ---------------------------------------------------------------------------
# Callbacks (rodam antes da página ser redesenhada)
# ---------------------------------------------------------------------------
def _cb_add_ref() -> None:
    ss = st.session_state
    ss.ref_count = min(config.MAX_REFERENCES, ss.ref_count + 1)


def _cb_remove_ref() -> None:
    ss = st.session_state
    if ss.ref_count > 1:
        ss.ref_count -= 1
        ss.pop(f"ref_{ss.ref_count}", None)


def _cb_fresh_start() -> None:
    _reset_curation()
    try:
        state.clear_session()
    except Exception as e:  # noqa: BLE001
        logger.warning("Não foi possível apagar a sessão salva: %s", e)
    _flash(TOP, "info", "Começando do zero. A sessão anterior foi descartada (o cache de áudio continua).")


def _cb_dismiss_restored() -> None:
    st.session_state.restored_at = None


def _cb_play(vid: str, pos: str) -> None:
    ss = st.session_state
    ss.active = {"vid": vid, "pos": pos}
    ss.preview = None                    # um player de cada vez


def _cb_stop() -> None:
    st.session_state.active = None


def _next_pending(after_vid: str) -> str | None:
    """Próximo card pendente (com áudio disponível) depois de `after_vid`, dando a volta na lista."""
    ss = st.session_state
    ids = list(ss.visible_ids)
    if not ids:
        return None
    start = ids.index(after_vid) + 1 if after_vid in ids else 0
    for vid in ids[start:] + ids[:start]:
        if vid == after_vid or vid in ss.decisions:
            continue
        track = ss.catalog.get(vid)
        if track is not None and _audio_path(track) is not None:
            return vid
    return None


def _cb_decide(vid: str, decision: str) -> None:
    ss = st.session_state
    ss.decisions.pop(vid, None)
    ss.decisions[vid] = decision          # reinsere no fim: ordem de aprovação
    _record_metric_decision(vid, decision)
    if ss.get("quick_mode", True):
        nxt = _next_pending(vid)
        ss.active = {"vid": nxt, "pos": "meio"} if nxt else None
        if nxt:
            ss.preview = None
    elif decision == REJECTED and ss.active and ss.active.get("vid") == vid:
        ss.active = None
    ss.export_report = None
    _persist()


def _cb_undo(vid: str) -> None:
    ss = st.session_state
    ss.decisions.pop(vid, None)
    _record_metric_decision(vid, None)
    ss.export_report = None
    _persist()


def _cb_retry(vid: str) -> None:
    track = st.session_state.catalog.get(vid)
    if track is not None:
        _submit([track])


def _cb_bpm(vid: str) -> None:
    ss = st.session_state
    value = ss.get(f"bpm_{vid}")
    bpm = float(value) if value else None
    try:
        an = analyze.set_manual_bpm(vid, bpm)
    except Exception as e:  # noqa: BLE001
        _flash(vid, "error", _error_message(e, "corrigir o BPM"))
        return
    if an is None:
        _flash(vid, "warning", "A análise desta faixa ainda não está no cache; tente de novo quando ela ficar pronta.")
        return
    tj = ss.get("jobs")
    if tj is not None:
        try:
            tj.update_analysis(an)
        except Exception:  # noqa: BLE001
            logger.warning("Falha ao atualizar a análise em memória", exc_info=True)


def _cb_more() -> None:
    ss = st.session_state
    start = _loaded_count()
    if start >= len(ss.suggestions):
        return
    ss.page += 1
    _submit(ss.suggestions[start: start + config.N_SUGGESTIONS])
    _persist()


def _cb_weights() -> None:
    ss = st.session_state
    ss.weights = {k: float(ss.get(f"w_{k}", ss.weights.get(k, 0.0))) for k in WEIGHT_LABELS}
    ss.explanations = None
    ss.auto_order = True
    _persist()


def _cb_preset() -> None:
    """Novo estilo do set: aplica os pesos e o modo de energia (e reordena se já havia ordem)."""
    ss = st.session_state
    key = ss.get("set_preset") or si.DEFAULT_PRESET
    if key not in si.PRESETS:
        key = si.DEFAULT_PRESET
    ss.preset = key
    w = si.preset_weights(key)
    ss.weights = {k: float(w.get(k, 0.0)) for k in WEIGHT_LABELS}
    for k in WEIGHT_LABELS:
        ss[f"w_{k}"] = float(min(max(w.get(k, 0.0), 0.0), float(config.WEIGHT_SLIDER_MAX)))
    ss.energy_mode = si.preset_energy_mode(key)
    ss.energy_rising = ss.energy_mode == "subindo"
    ss.explanations = None
    ss.auto_order = True
    _persist()


def _cb_energy_mode() -> None:
    ss = st.session_state
    ss.energy_mode = "subindo" if ss.get("energy_rising") else "suave"
    ss.explanations = None
    ss.auto_order = True
    _persist()


def _cb_arc() -> None:
    ss = st.session_state
    if ss.get("set_arc") in ui_parts.ARCS:
        ss.arc = ss.set_arc
    _persist()


def _cb_move(vid: str, delta: int) -> None:
    ss = st.session_state
    cur = list(ss.get("cur_order") or ss.order)
    if vid not in cur:
        return
    i = cur.index(vid)
    j = i + delta
    if 0 <= j < len(cur):
        cur[i], cur[j] = cur[j], cur[i]
        ss.order = cur
        ss.explanations = None
        ss.export_report = None
        ss.preview = None
        _persist()


def _cb_stop_preview() -> None:
    st.session_state.preview = None


def _cb_toggle_vocal(vid: str) -> None:
    ss = st.session_state
    ss.stem_player = None if ss.stem_player == vid else vid


def _cb_reset_credit() -> None:
    bud = st.session_state.get("budget")
    if bud is None:
        return
    try:
        bud.reset_insufficient_credit()
        _flash(TOP, "info", "Trava de saldo liberada. Se a Replicate recusar de novo, ela volta.")
    except Exception as e:  # noqa: BLE001
        _flash(TOP, "error", _error_message(e, "liberar a trava de saldo"))


def _cb_open_folder(folder: str) -> None:
    try:
        os.startfile(folder)  # type: ignore[attr-defined]  # só existe no Windows
    except Exception as e:  # noqa: BLE001
        _flash(TOP, "error", _error_message(e, "abrir a pasta"))


def _cb_ask_end() -> None:
    st.session_state.confirm_end = True


def _cb_cancel_end() -> None:
    st.session_state.confirm_end = False


def _cb_end_session() -> None:
    ss = st.session_state
    rejected = [vid for vid, d in ss.decisions.items() if d == REJECTED]
    removed: int | None = None
    try:
        removed = download.purge(rejected)
    except Exception as e:  # noqa: BLE001
        _flash(TOP, "error", _error_message(e, "apagar as reprovadas do cache"))
    try:
        state.clear_session()
    except Exception as e:  # noqa: BLE001
        logger.warning("Não foi possível apagar a sessão salva: %s", e)
    _reset_curation()
    msg = "Sessão encerrada e sessão salva apagada."
    if removed is not None:
        msg += f" {removed} arquivo(s) de {len(rejected)} faixa(s) reprovada(s) apagado(s) do cache."
    _flash(TOP, "success", msg)


def _cb_reset_open() -> None:
    ss = st.session_state
    ss.reset_open = True
    ss.confirm_end = False
    for cat in maintenance.CATEGORIES:            # abre sempre com as marcações padrão
        ss[f"reset_cat_{cat}"] = cat in maintenance.DEFAULT_RESET
    ss.reset_confirm = False


def _cb_reset_close() -> None:
    st.session_state.reset_open = False


# ---------------------------------------------------------------------------
# Topo: cabeçalho com orçamento e barra de etapas
# ---------------------------------------------------------------------------
def _render_header() -> None:
    slot = _header_slot
    if slot is None:
        return
    ss = st.session_state
    bud = ss.get("budget")
    spent = left = total = None
    cap: float | None = float(config.BUDGET_CAP_BRL)
    blocked = False
    if bud is not None:
        try:
            spent = float(bud.spent_brl)
            cap = float(getattr(bud, "cap_brl", config.BUDGET_CAP_BRL))
            left = float(bud.remaining_credit_usd)
            total = float(getattr(bud, "credit_usd", config.REPLICATE_CREDIT_USD))
            blocked = bool(bud.insufficient_credit)
        except Exception as e:  # noqa: BLE001
            logger.warning("Não foi possível ler o orçamento: %s", e)
            spent = left = total = None
    tags: list[Any] = [("Funk mandelão · submundo", "accent", "🔥")]
    n_sugg = len(ss.get("suggestions") or [])
    if n_sugg:
        tags.append((f"{n_sugg} sugestões", "info", "🎧"))
    n_app = len(_approved_tracks())
    if n_app:
        tags.append((f"{n_app} aprovadas", "ok", "✅"))
    if ss.get("order"):
        tags.append((f"set com {len(ss.get('cur_order') or ss.order)} músicas", "good", "🎚️"))
    with slot.container():
        theme.hero("DJ SK · Curadoria",
                   "Do rádio do YouTube Music ao Rekordbox: ache, ouça, aprove e monte o set em minutos.",
                   spent, cap, left, total, blocked, kicker="Painel de curadoria · funk submundo", tags=tags)


def _steps() -> list[tuple[str, str, str]]:
    """Etapas do painel com o status de cada uma (a primeira não concluída fica ativa)."""
    ss = st.session_state
    visible = _visible_tracks()
    n_vis = len(visible)
    decided = sum(1 for t in visible if t.video_id in ss.decisions)
    n_app = len(_approved_tracks())
    exported = bool(ss.get("export_report") or ss.get("spotify_export"))
    flags = [
        bool(ss.suggestions),
        n_app > 0 and (decided >= n_vis or bool(ss.order)),
        n_app > 0 and bool(ss.order),
        exported,
        exported,
    ]
    labels = ["Referências", f"Triagem {decided}/{n_vis}" if n_vis else "Triagem",
              f"Setlist ({n_app})" if n_app else "Setlist", "Exportar", "Métricas"]
    anchors = ["#referencias", "#triagem", "#setlist", "#exportar", "#metricas"]
    out = []
    active_given = False
    for label, done, anchor in zip(labels, flags, anchors):
        if done:
            status = "done"
        elif not active_given:
            status, active_given = "active", True
        else:
            status = "todo"
        out.append((label, status, anchor))
    return out


def _render_stepper() -> None:
    slot = _stepper_slot
    if slot is None:
        return
    slot.html(theme.stepper_html(_steps()))


def _render_notices() -> None:
    ss = st.session_state
    with st.container():
        _show_flash(TOP)
        if ss.get("restored_at"):
            c1, c2, c3 = st.columns([6, 2, 1], vertical_alignment="center")
            c1.info(f"Sessão anterior restaurada (salva em {_fmt_saved_at(ss.restored_at)}): "
                    f"{len(ss.suggestions)} sugestões e {len(ss.decisions)} decisões.")
            c2.button("Começar do zero", key="btn_fresh_start", on_click=_cb_fresh_start,
                      help="Descarta a sessão salva (o cache de áudio continua).")
            c3.button("OK", key="btn_dismiss_restored", on_click=_cb_dismiss_restored)


# ---------------------------------------------------------------------------
# 1 Referências
# ---------------------------------------------------------------------------
def _render_references() -> None:
    ss = st.session_state
    pending = ss.pop("pending_refs", None)
    if pending:
        for i in range(config.MAX_REFERENCES):
            ss[f"ref_{i}"] = pending[i] if i < len(pending) else ""
        ss.ref_count = max(1, min(config.MAX_REFERENCES, len(pending)))

    theme.section_header(1, "Referências",
                         f"Escolha de 1 a {config.MAX_REFERENCES} músicas com a cara do set. "
                         "O painel busca as parecidas no rádio do YouTube Music.", anchor="referencias")
    _show_flash(REFS)
    tab_yt, tab_sp, tab_local = st.tabs(["🎵 Digitar referências", "🟢 Spotify", "📁 Pasta local"])
    with tab_yt:
        st.caption(f"Até {config.MAX_REFERENCES} músicas no formato \"Artista - Faixa\" "
                   "(também aceita link do YouTube ou do YouTube Music).")
        for i in range(ss.ref_count):
            st.text_input(f"Referência {i + 1}", key=f"ref_{i}",
                          placeholder="Artista - Faixa  (ou link do YouTube / YouTube Music)")
        with st.container(horizontal=True, vertical_alignment="center"):
            st.button("➕ Mais uma", key="btn_add_ref", on_click=_cb_add_ref,
                      disabled=ss.ref_count >= config.MAX_REFERENCES, help="Mais uma referência")
            if ss.ref_count > 1:
                st.button("➖ Tirar a última", key="btn_remove_ref", on_click=_cb_remove_ref)
    with tab_sp:
        _render_spotify()
    with tab_local:
        _render_local_folder()

    refs_now = [ss.get(f"ref_{i}") or "" for i in range(ss.ref_count)]
    _html(ui_parts.ref_chips_html(refs_now))
    with st.container(horizontal=True, vertical_alignment="center"):
        st.toggle("Filtrar slowed/reverb", value=True, key="filter_slowed",
                  help="Esconde versões slowed, reverb, sped up, 8D e nightcore.")
        search = st.button("🔎 Buscar sugestões", key="btn_search", type="primary")
    if search:
        _do_search()

    if ss.ref_report:
        lines = []
        for r in ss.ref_report:
            if r.get("track"):
                lines.append(f"✔ {_md(r['query'])} → **{_md(r['track'])}**")
            else:
                lines.append(f"✖ {_md(r['query'])}: {_md(r.get('error') or 'não encontrada')}")
        st.markdown("  \n".join(lines))
    for w in ss.suggest_warnings:
        st.warning(w)


def _do_search() -> None:
    ss = st.session_state
    queries = [(ss.get(f"ref_{i}") or "").strip() for i in range(ss.ref_count)]
    queries = [q for q in queries if q]
    if not queries:
        st.warning("Digite pelo menos uma referência no formato \"Artista - Faixa\".")
        return
    _start_metrics_if_needed()
    report: list[dict] = []
    refs: list[Track] = []
    with st.spinner("Encontrando as referências no YouTube Music…"):
        for q in queries:
            track = ss.resolved_cache.get(q)
            if track is None:
                try:
                    track = suggest.resolve_reference(q)
                except Exception as e:  # noqa: BLE001
                    report.append({"query": q, "track": None, "error": _error_message(e, f"buscar \"{q}\"")})
                    continue
            if track is None:
                report.append({"query": q, "track": None,
                               "error": "não encontrada no YouTube Music (confira a grafia ou cole o link)"})
                continue
            ss.resolved_cache[q] = track
            if track.video_id not in {r.video_id for r in refs}:
                refs.append(track)
            report.append({"query": q, "track": track.display_name, "error": None})
    ss.ref_report = report
    ss.suggest_warnings = []
    if not refs:
        st.error("Nenhuma referência foi encontrada. Confira a grafia (\"Artista - Faixa\") ou cole o link do YouTube.")
        return
    exclude = None if ss.get("filter_slowed", True) else []
    with st.spinner("Buscando sugestões no rádio do YouTube Music…"):
        try:
            result = suggest.get_suggestions(refs, exclude_keywords=exclude)
        except Exception as e:  # noqa: BLE001
            st.error(_error_message(e, "buscar sugestões"))
            return
    ss.references = refs
    ss.suggest_warnings = list(result.warnings or [])
    ss.order = [v for v in ss.order if ss.decisions.get(v) == APPROVED]
    _set_candidates(list(result.tracks))


# ---------------------------------------------------------------------------
# 1b Spotify (busca sem login, conta, importar playlist)
# ---------------------------------------------------------------------------
def _spotify_configured() -> bool:
    try:
        return bool(spotify_import.is_configured())
    except Exception:  # noqa: BLE001
        return False


def _spotify_login_state() -> str:
    try:
        return str(spotify_import.saved_login_state())
    except Exception:  # noqa: BLE001
        return "none"


def _render_spotify() -> None:
    if not _spotify_configured():
        st.info("Spotify desligado. Para usar, preencha SPOTIFY_CLIENT_ID e SPOTIFY_CLIENT_SECRET no arquivo "
                ".env (veja o README) e recarregue a página.")
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        _render_spotify_search()
    with right:
        _render_spotify_account()


def _render_spotify_search() -> None:
    ss = st.session_state
    st.markdown("**Buscar uma música no Spotify** · não precisa entrar na conta")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.text_input("Música ou artista", key="sp_query", placeholder="Ex.: MC Ryan SP Casei com a Putaria")
        go = st.button("🔎 Buscar", key="btn_sp_search")
    if go:
        q = (ss.get("sp_query") or "").strip()
        if not q:
            st.warning("Digite o nome da música ou do artista.")
        else:
            with st.spinner("Buscando no Spotify…"):
                try:
                    ss.sp_results = list(spotify_import.search_tracks(q, limit=SPOTIFY_RESULTS))
                except Exception as e:  # noqa: BLE001
                    st.error(_error_message(e, "buscar no Spotify"))
                    ss.sp_results = []
            if not ss.sp_results:
                st.info("Nada encontrado no Spotify. Tente outro nome.")
    for i, d in enumerate(ss.get("sp_results") or []):
        if not isinstance(d, dict):
            continue
        sid = _key_part(d.get("spotify_id") or d.get("uri") or i)
        with st.container(key=f"{theme.CARD_KEY_PREFIX}sp_{i}"):
            cols = st.columns([1, 4, 2], vertical_alignment="center")
            if d.get("cover_url"):
                cols[0].image(d["cover_url"], width=56)
            cols[1].html(ui_parts.spotify_result_html(d))
            use = cols[2].button("➕ Usar como referência", key=f"sp_use_{i}_{sid}",
                                 help="Acha esta música no YouTube Music e coloca nas referências.")
        if use:
            _use_spotify_ref(d)


def _use_spotify_ref(d: dict) -> None:
    ss = st.session_state
    with st.spinner("Achando a música no YouTube Music…"):
        try:
            track = spotify_import.resolve_on_ytmusic(spotify_import.to_spotify_track(d))
        except Exception as e:  # noqa: BLE001
            st.error(_error_message(e, "achar a música no YouTube Music"))
            return
    if track is None:
        st.warning("Não achei essa música no YouTube Music. Tente outra versão ou digite \"Artista - Faixa\" "
                   "na aba Digitar referências.")
        return
    text = track.display_name
    ss.resolved_cache[text] = track
    ss.catalog[track.video_id] = track
    current = [(ss.get(f"ref_{i}") or "").strip() for i in range(ss.ref_count)]
    current = [c for c in current if c]
    if text in current:
        _flash(REFS, "info", f"\"{text}\" já está nas referências.")
        st.rerun()
    replaced = len(current) >= config.MAX_REFERENCES
    if replaced:
        current = current[: config.MAX_REFERENCES - 1]
    ss.pending_refs = [*current, text]
    note = " (troquei a última, porque o limite é " f"{config.MAX_REFERENCES})" if replaced else ""
    _flash(REFS, "success", f"Referência do Spotify adicionada: {text}{note}. Agora clique em 🔎 Buscar sugestões.")
    st.rerun()


def _spotify_connected(sp) -> None:
    ss = st.session_state
    ss.spotify_client = sp
    try:
        ss.spotify_user = spotify_import.current_user_info(sp)
    except Exception as e:  # noqa: BLE001
        logger.info("Não consegui ler o nome da conta do Spotify: %s", e)
        ss.spotify_user = {}
    try:
        ss.spotify_playlists = spotify_import.list_playlists(sp)
    except Exception as e:  # noqa: BLE001
        _flash(REFS, "error", _error_message(e, "ler as playlists do Spotify"))
        ss.spotify_playlists = []


def _render_spotify_account() -> None:
    ss = st.session_state
    st.markdown("**Sua conta do Spotify**")
    sp = ss.get("spotify_client")
    if sp is None:
        login = _spotify_login_state()
        if login == "outdated":
            st.caption("O painel agora também cria playlists no Spotify: entre de novo uma vez para autorizar.")
            label = "Entrar de novo no Spotify"
        elif login == "saved":
            st.caption("Você já entrou antes neste computador: é só conectar.")
            label = "🟢 Conectar ao Spotify"
        else:
            st.caption("Entre com a conta do DJ SK para importar playlists e criar o set no Spotify. O navegador "
                       "abre o login; depois de autorizar, volte para esta aba.")
            label = "Entrar no Spotify"
        if st.button(label, key="btn_spotify_login"):
            with st.spinner("Aguardando o login no Spotify…"):
                try:
                    sp = spotify_import.get_client(open_browser=True)
                except Exception as e:  # noqa: BLE001
                    st.error(_error_message(e, "entrar no Spotify"))
                    sp = None
                else:
                    _spotify_connected(sp)
                    st.rerun()   # barra lateral e exportação também passam a mostrar a conta
        if sp is None:
            return
    user = ss.get("spotify_user") or {}
    badges = [theme.badge(f"Conectado como {user.get('name') or 'conta do Spotify'}", "ok", "🟢")]
    if user.get("product"):
        badges.append(theme.badge(str(user["product"]).capitalize(), "info"))
    _html(ui_parts.badges_html(badges))

    playlists = ss.get("spotify_playlists") or []
    readable = [p for p in playlists if p.get("readable")]
    hidden = len(playlists) - len(readable)
    if hidden:
        st.caption(f"{hidden} playlist(s) de outras pessoas não podem ser lidas no modo de "
                   "desenvolvimento do Spotify (só as suas ou colaborativas).")
    if not readable:
        st.info("Nenhuma playlist sua ou colaborativa encontrada nesta conta.")
    else:
        by_id = {p["id"]: p for p in readable}
        pid = st.selectbox("Playlist", list(by_id), key="spotify_playlist",
                           format_func=lambda i: f"{by_id[i].get('name', i)} "
                                                 f"({by_id[i].get('tracks_total', '?')} faixas)")
        chosen = by_id.get(pid, {})
        info_cols = st.columns([1, 3], vertical_alignment="center")
        if chosen.get("image_url"):
            info_cols[0].image(chosen["image_url"], width=72)
        info_cols[1].html(ui_parts.badges_html([
            theme.badge(chosen.get("name") or "Playlist", "accent", "🎶"),
            theme.badge(f"{chosen.get('tracks_total', '?')} faixas", "info"),
            theme.badge(f"de {chosen.get('owner') or '?'}", "muted"),
        ]))
        mode = st.radio("Usar as faixas como", ["Referências", "Candidatas diretas na triagem"],
                        key="spotify_mode", horizontal=True,
                        help=f"Como referências entram só as {config.MAX_REFERENCES} primeiras (rádio do "
                             "YouTube Music). Para triar a playlist inteira, use candidatas diretas.")
        total = chosen.get("tracks_total") or 0
        if mode == "Referências" and total > config.MAX_REFERENCES:
            st.caption(f"Esta playlist tem {total} faixas: como referência só entram as "
                       f"{config.MAX_REFERENCES} primeiras. Para triar todas, escolha \"Candidatas diretas\".")
        if st.button("📥 Importar", key="btn_spotify_import", type="primary"):
            _import_spotify(sp, pid, as_refs=(mode == "Referências"), name=chosen.get("name"))
        embed = ui_parts.spotify_embed_url("playlist", pid)
        if embed:
            with st.expander("▶ Ouvir a playlist aqui (player do Spotify)"):
                st.iframe(embed, height=152)
    _render_spotify_import_report()
    if st.button("Sair do Spotify", key="btn_spotify_logout"):
        for k in ("spotify_client", "spotify_user", "spotify_playlists", "spotify_export"):
            ss.pop(k, None)
        st.rerun()


def _render_spotify_import_report() -> None:
    rep = st.session_state.get("spotify_import_report")
    if not rep:
        return
    found, missing = int(rep.get("found") or 0), list(rep.get("missing") or [])
    total = found + len(missing)
    if total:
        st.progress(found / total, text=f"{found} de {total} encontradas no YouTube Music"
                                        + (f" · playlist {rep['name']}" if rep.get("name") else ""))
    if missing:
        with st.expander(f"{len(missing)} música(s) que não achei no YouTube Music"):
            _html(ui_parts.badges_html([theme.badge(m, "muted") for m in missing[:40]], compact=True))


def _import_spotify(sp, playlist_id: str, as_refs: bool, name: str | None = None) -> None:
    ss = st.session_state
    bar = st.progress(0.0, text="Lendo a playlist e encontrando as faixas no YouTube Music…")

    def _progress(done: int, total: int) -> None:
        try:
            bar.progress(min(1.0, done / max(1, total)), text=f"Achando no YouTube Music: {done}/{total}")
        except Exception:  # noqa: BLE001
            pass

    try:
        tracks, missing = spotify_import.import_playlist(
            sp, playlist_id, limit=config.MAX_REFERENCES if as_refs else None, progress=_progress)
    except Exception as e:  # noqa: BLE001
        bar.empty()
        st.error(_error_message(e, "importar a playlist"))
        return
    bar.empty()
    ss.spotify_import_report = {"found": len(tracks), "missing": list(missing), "name": name}
    if not tracks:
        st.error("Nenhuma faixa da playlist foi encontrada no YouTube Music.")
        return
    if as_refs:
        refs = tracks[: config.MAX_REFERENCES]
        for t in refs:
            ss.resolved_cache[t.display_name] = t
        ss.pending_refs = [t.display_name for t in refs]
        _flash(REFS, "success", f"{len(refs)} referência(s) importadas do Spotify. Clique em 🔎 Buscar sugestões.")
        st.rerun()
    _start_metrics_if_needed()
    _set_candidates(tracks)
    st.success(f"{len(tracks)} faixa(s) da playlist foram para a triagem.")


def _render_local_folder() -> None:
    ss = st.session_state
    st.caption("Sem internet? Use músicas que já estão no computador: elas vão direto para a triagem.")
    st.text_input("Caminho da pasta com os áudios", key="local_folder",
                  placeholder=r"C:\Users\...\Músicas\Mandelão")
    st.checkbox("Incluir subpastas", key="local_recursive")
    if st.button("📂 Carregar pasta", key="btn_local"):
        folder = (ss.get("local_folder") or "").strip().strip('"')
        if not folder:
            st.warning("Informe o caminho da pasta.")
            return
        try:
            if ss.get("local_recursive"):
                tracks = download.local_tracks_from_folder(folder, recursive=True)
            else:
                tracks = download.local_tracks_from_folder(folder)
        except Exception as e:  # noqa: BLE001
            st.error(_error_message(e, "ler a pasta"))
            return
        if not tracks:
            exts = ", ".join(config.LOCAL_AUDIO_EXTENSIONS)
            st.warning(f"Nenhum arquivo de áudio ({exts}) encontrado em {folder}.")
            return
        _start_metrics_if_needed()
        _set_candidates(tracks)
        st.success(f"{len(tracks)} arquivo(s) locais foram para a triagem "
                   "(as sugestões anteriores saíram da lista; as aprovadas continuam no setlist).")


# ---------------------------------------------------------------------------
# 2 Triagem
# ---------------------------------------------------------------------------
def _progress_fragment() -> None:
    """Só redesenha a barra de progresso (não recria os players dos cards)."""
    ss = st.session_state
    tj = ss.get("jobs")
    if tj is None:
        return
    try:
        s = tj.summary()
        done = bool(tj.all_done())
    except Exception:  # noqa: BLE001
        return
    total = int(s.get("total", 0) or 0)
    have = int(s.get("baixadas", 0) or 0)
    ready = int(s.get(jobs.READY, 0) or 0)
    errors = int(s.get(jobs.ERROR, 0) or 0)
    if total:
        text = f"baixando: {have}/{total} · analisadas: {ready}/{total}"
        if errors:
            text += f" · com erro: {errors}"
        st.progress(min(1.0, have / total), text=text)
    if ss.get("active") is not None or ss.get("preview") is not None:
        return  # algo está tocando: não interrompe a música com um rerun
    terminal = _terminal_ids(list(ss.get("visible_ids") or []))
    if terminal != ss.get("rendered_terminal"):
        ss.rendered_terminal = terminal
        st.rerun()
    if done and ss.get("progress_timer"):
        ss.progress_timer = False
        st.rerun()  # tudo pronto: o próximo run cria o fragmento sem timer


def _render_progress() -> None:
    ss = st.session_state
    tj = ss.get("jobs")
    try:
        done = tj is None or bool(tj.all_done())
    except Exception:  # noqa: BLE001
        done = True
    ss.progress_timer = not done
    st.fragment(_progress_fragment, run_every=None if done else PROGRESS_REFRESH_S)()


def _sorted_for_triage(tracks: list[Track], how: str | None) -> list[Track]:
    """Relevância = ordem do rádio; BPM = mais lentas primeiro; Energia = mais fortes primeiro."""
    if how == "BPM":
        def key_bpm(t: Track) -> tuple[int, float]:
            an = _analysis(t.video_id)
            return (0, float(an.effective_bpm)) if an is not None else (1, 0.0)
        return sorted(tracks, key=key_bpm)
    if how == "Energia":
        def key_energy(t: Track) -> tuple[int, float]:
            an = _analysis(t.video_id)
            return (0, -float(an.energy_mean)) if an is not None else (1, 0.0)
        return sorted(tracks, key=key_energy)
    return list(tracks)


def _focus_vid(shown: list[Track]) -> str | None:
    """Card que recebe os atalhos: o que está tocando; senão, o primeiro pendente com áudio."""
    ss = st.session_state
    ids = [t.video_id for t in shown]
    act = (ss.get("active") or {}).get("vid")
    if act in ids:
        return act
    for t in shown:
        if t.video_id not in ss.decisions and _audio_path(t) is not None:
            return t.video_id
    for t in shown:
        if _audio_path(t) is not None:
            return t.video_id
    return None


def _render_triage() -> None:
    ss = st.session_state
    theme.section_header(2, "Triagem", "Ouça um pedaço de cada música e decida: ✅ entra no set, ❌ fica de fora.",
                         anchor="triagem")
    tj = _jobs()
    if tj is None and ss.get("jobs_error"):
        st.error(ss.jobs_error)
    visible = _visible_tracks()
    if not visible:
        st.info("Busque sugestões, importe uma playlist ou carregue uma pasta local para começar a triagem.")
        ss.visible_ids = []
        ss.focus_vid = None
        return
    ss.visible_ids = [t.video_id for t in visible]

    n_app = sum(1 for t in visible if ss.decisions.get(t.video_id) == APPROVED)
    n_rej = sum(1 for t in visible if ss.decisions.get(t.video_id) == REJECTED)
    n_pend = len(visible) - n_app - n_rej
    c = st.columns(4)
    c[0].metric("Pendentes", n_pend)
    c[1].metric("Aprovadas", n_app)
    c[2].metric("Reprovadas", n_rej)
    c[3].metric("Na triagem", len(visible))

    ss.rendered_terminal = _terminal_ids(ss.visible_ids)
    _render_progress()

    if "triage_filter" not in ss:
        ss.triage_filter = FILTERS[0]
    if "triage_sort" not in ss:
        ss.triage_sort = SORTS[0]
    with st.container(horizontal=True, vertical_alignment="bottom"):
        flt = st.segmented_control("Mostrar", FILTERS, key="triage_filter", required=True)
        how = st.segmented_control("Ordenar por", SORTS, key="triage_sort", required=True,
                                   help="Relevância = quantas referências sugeriram e a posição no rádio.")
        st.toggle("Modo rápido", value=True, key="quick_mode",
                  help="Depois de aprovar ou reprovar, o próximo card pendente começa a tocar no meio.")
        st.button("🔄 Atualizar cards", key="btn_refresh",
                  help="Redesenha os cards com o status mais recente dos downloads.")
    _html(ui_parts.kbd_legend_html(note="valem para o card em foco (borda roxa ou rosa)"))

    def keep(t: Track) -> bool:
        d = ss.decisions.get(t.video_id)
        return {"Pendentes": d is None, "Aprovadas": d == APPROVED,
                "Reprovadas": d == REJECTED}.get(flt or "Todas", True)

    shown = _sorted_for_triage([t for t in visible if keep(t)], how)
    focus = _focus_vid(shown)
    ss.focus_vid = focus
    if not shown:
        st.caption("Nenhuma faixa neste filtro." if flt != "Pendentes"
                   else "Nenhuma faixa pendente nesta página. 🎉")
    for i in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, track in zip(cols, shown[i:i + 2]):
            with col:
                _render_card(track, focus=track.video_id == focus)

    total = len(ss.suggestions)
    loaded = len(visible)
    if total > loaded:
        n_next = min(config.N_SUGGESTIONS, total - loaded)
        st.button(f"Carregar mais {n_next} sugestões", key="btn_more", on_click=_cb_more)
        st.caption(f"Mostrando {loaded} de {total} sugestões.")
    else:
        st.caption(f"Todas as {total} sugestões estão na triagem.")


def _status_text(stt) -> str:
    if stt.state == jobs.DOWNLOADING:
        return f"⬇️ baixando {int(round((stt.progress or 0) * 100))}%"
    return {jobs.QUEUED: "⏳ na fila", jobs.ANALYZING: "🔬 analisando",
            jobs.READY: "✔ pronta", jobs.ERROR: "⚠️ erro"}.get(stt.state, str(stt.state))


_STATUS_TONE = {jobs.QUEUED: "muted", jobs.DOWNLOADING: "info", jobs.ANALYZING: "info",
                jobs.READY: "good", jobs.ERROR: "bad"}


def _decision_badge(decision: str | None, stt) -> str:
    if decision == APPROVED:
        return theme.badge("aprovada", "ok", "✅")
    if decision == REJECTED:
        return theme.badge("reprovada", "bad", "❌")
    return theme.badge(_status_text(stt), _STATUS_TONE.get(stt.state, "muted"))


def _card_badges(track: Track, an: Analysis | None, focus: bool) -> list[str]:
    ss = st.session_state
    out: list[str] = []
    if focus:
        out.append(theme.badge("em foco: atalhos ativos", "info", "⌨"))
    if an is not None:
        bpm = an.effective_bpm
        zone = ui_parts.bpm_zone(bpm)
        tone = zone[1] if zone else "muted"
        manual = " · manual" if an.bpm_manual else ""
        out.append(theme.badge(f"{ui_parts.bpm_text(bpm)} BPM{manual}", tone, "♫",
                               title="Velocidade da música (batidas por minuto)"))
        if zone:
            out.append(theme.badge(zone[0], tone))
        if an.camelot or an.key:
            out.append(theme.chip("tom", an.camelot or "?", key=True, strong_first=False))
            if an.key:
                out.append(theme.chip(an.key))
        raw = theme._float(an.bpm_raw)
        if not an.bpm_manual and raw and abs(raw - float(an.bpm)) > 1.5:
            out.append(theme.badge(f"detectado {ui_parts.bpm_text(raw)} BPM: confira", "warn", "⚠",
                                   title="O detector achou outra velocidade e o painel ajustou para a faixa do funk "
                                         f"({config.BPM_MIN:.0f}–{config.BPM_MAX:.0f} BPM). Se não bater, "
                                         "corrija no campo Corrigir BPM."))
    n_refs = len(ss.get("references") or [])
    if n_refs > 1 and (track.score or 0) > 1:
        out.append(theme.badge(f"{track.score}/{n_refs} referências", "warn", "🔥",
                               title="Quantas das suas referências sugeriram esta música"))
    if track.source == "spotify":
        out.append(theme.badge("Spotify", "ok", "🟢"))
    if track.is_local:
        out.append(theme.badge("arquivo local", "info", "📁"))
    similar = (ss.get("similar_of") or {}).get(track.video_id)
    if similar:
        out.append(theme.badge(f"parecida com {similar}", "info", "🔁"))
    return out


def _render_card(track: Track, focus: bool = False) -> None:
    ss = st.session_state
    vid = track.video_id
    an = _analysis(vid)
    stt = _status(vid)
    decision = ss.decisions.get(vid)
    active = ss.active if ss.active and ss.active.get("vid") == vid else None
    path = _audio_path(track)
    duration = _duration(track, an)
    keys = ui_parts.SHORTCUTS if focus else {}
    prefix = (theme.PLAYING_KEY_PREFIX if active is not None
              else theme.FOCUS_KEY_PREFIX if focus else theme.CARD_KEY_PREFIX)

    with st.container(key=f"{prefix}{vid}"):
        head = st.columns([1.2, 5, 2.2], vertical_alignment="center") if track.thumbnail_url \
            else st.columns([6, 2.2], vertical_alignment="center")
        if track.thumbnail_url:
            head[0].image(track.thumbnail_url, width=72)
        head[-2].markdown(f"**{_md(track.title)}**  \n{_md(track.artist_str)} · {util.format_duration(duration)}")
        head[-1].html(ui_parts.badges_html([_decision_badge(decision, stt)]))
        _html(ui_parts.badges_html(_card_badges(track, an, focus), compact=True))

        if an is not None:
            info, spark = st.columns([3, 2], vertical_alignment="center")
            info.caption(f"intro {int(round(an.intro_s))} s · saída {int(round(an.outro_s))} s · "
                         f"energia média {ui_parts.pct(an.energy_mean)}%")
            spark.markdown(viz.sparkline_svg(an.energy, width=160, height=28), unsafe_allow_html=True)
            dur = an.duration_s or duration or 0
            markers = []
            highlight = []
            if dur:
                final_at = max(0.0, dur - FINAL_PREVIEW_S) / dur
                markers = [(0.0, "Início"), (0.5, "Meio"), (final_at, "Final (últimos 30 s)")]
                if an.intro_s:
                    highlight.append((0.0, an.intro_s / dur))
                if an.outro_s:
                    highlight.append((1.0 - an.outro_s / dur, 1.0))
            st.markdown(viz.waveform_svg(an.waveform, width=600, height=56, markers=markers,
                                         highlight=highlight), unsafe_allow_html=True)
        elif stt.state != jobs.ERROR:
            st.caption("Velocidade, energia e forma de onda aparecem quando o download e a análise terminarem.")

        if stt.state == jobs.ERROR:
            st.error(stt.error or "Falha ao baixar ou analisar esta faixa.")
            st.button("Tentar de novo", key=f"retry_{vid}", on_click=_cb_retry, args=(vid,))
        _show_flash(vid)

        if active is not None:
            if path is None:
                st.caption("Aguardando o download para tocar…")
            else:
                start, end = _play_window(active.get("pos", "inicio"), duration)
                st.audio(str(path), format=_mime(path), start_time=start, end_time=end, autoplay=True)

        with st.container(horizontal=True, vertical_alignment="center"):
            for pos, label in POSITIONS.items():
                needs_duration = pos != "inicio"
                st.button(f"▶ {label}", key=f"play_{pos}_{vid}", on_click=_cb_play, args=(vid, pos),
                          disabled=path is None or (needs_duration and not duration),
                          type="primary" if active and active.get("pos") == pos else "secondary",
                          shortcut=keys.get(pos))
            if active is not None:
                st.button("⏹", key=f"stop_{vid}", on_click=_cb_stop, help="Parar o player")

        with st.container(horizontal=True, vertical_alignment="bottom"):
            st.button("✅ Aprovar", key=f"approve_{vid}", on_click=_cb_decide, args=(vid, APPROVED),
                      disabled=decision == APPROVED, shortcut=keys.get("approve"))
            st.button("❌ Reprovar", key=f"reject_{vid}", on_click=_cb_decide, args=(vid, REJECTED),
                      disabled=decision == REJECTED, shortcut=keys.get("reject"))
            if decision is not None:
                st.button("↩ Pendente", key=f"undo_{vid}", on_click=_cb_undo, args=(vid,),
                          help="Desfaz a decisão")
            similar = st.button("🔁 Mais parecidas", key=f"similar_{vid}", disabled=track.is_local,
                                help="Busca o rádio desta música e coloca as parecidas novas logo abaixo dela."
                                if not track.is_local else "Arquivo local não tem rádio no YouTube Music.")
            st.number_input("Corrigir BPM", min_value=0.0, max_value=250.0, step=0.5, format="%.1f",
                            value=float(an.bpm_manual) if an is not None and an.bpm_manual else None,
                            key=f"bpm_{vid}", on_change=_cb_bpm, args=(vid,), disabled=an is None,
                            placeholder=_dec(an.bpm, 1) if an is not None else "–",
                            help="BPM certo desta faixa (0 ou vazio volta ao BPM detectado).",
                            width=150)
        if similar:
            _more_like(track)


def _more_like(track: Track) -> None:
    """"Mais parecidas": rádio desta faixa; as novas entram logo depois dela e vão para o download."""
    ss = st.session_state
    exclude = None if ss.get("filter_slowed", True) else []
    with st.spinner(f"Buscando músicas parecidas com {track.title}…"):
        try:
            result = suggest.get_suggestions([track], exclude_keywords=exclude)
        except Exception as e:  # noqa: BLE001
            _flash(track.video_id, "error", _error_message(e, "buscar músicas parecidas"))
            st.rerun()
    known = {t.video_id for t in ss.suggestions} | set(ss.decisions) | {t.video_id for t in ss.references}
    new: list[Track] = []
    for t in result.tracks or []:
        if t.video_id and t.video_id not in known and t.video_id != track.video_id:
            known.add(t.video_id)
            new.append(t)
        if len(new) >= MORE_LIKE_MAX:
            break
    if not new:
        _flash(track.video_id, "info", "Nenhuma música nova: as parecidas com esta já estão na lista.")
        st.rerun()
    ids = [t.video_id for t in ss.suggestions]
    at = ids.index(track.video_id) + 1 if track.video_id in ids else _loaded_count()
    ss.suggestions = [*ss.suggestions[:at], *new, *ss.suggestions[at:]]
    ss.extra_visible = int(ss.get("extra_visible") or 0) + len(new)
    for t in new:
        ss.catalog[t.video_id] = t
        ss.similar_of[t.video_id] = track.title
    _submit(new)
    _persist()
    _flash(track.video_id, "success", f"{len(new)} música(s) parecida(s) entraram na triagem, logo depois desta.")
    st.rerun()


# ---------------------------------------------------------------------------
# 3 Setlist
# ---------------------------------------------------------------------------
def _current_order(analyses: dict[str, Analysis]) -> list[str]:
    """Ordem salva, sem as faixas que deixaram de estar aprovadas, mais as aprovadas novas no fim."""
    ss = st.session_state
    if not ss.order:
        return []
    base = [v for v in ss.order if v in analyses]
    new = [v for v in analyses if v not in base]
    if new:
        st.info(f"{len(new)} faixa(s) aprovadas depois da última ordenação entraram no fim. "
                "Clique em \"Gerar ordem\" para reotimizar.")
    return base + new


def _preset_label(key: str) -> str:
    p = si.PRESETS.get(key) or {}
    return f"{p.get('icon', '')} {p.get('label', key)}".strip()


def _render_style_picker() -> tuple[dict[str, float], str]:
    """Estilo do set (presets) + "Ajuste fino" com os pesos explicados. Devolve (pesos, modo de energia)."""
    ss = st.session_state
    if ss.get("set_preset") not in si.PRESETS:
        ss.set_preset = ss.get("preset") if ss.get("preset") in si.PRESETS else si.DEFAULT_PRESET
    st.pills("Estilo do set", list(si.PRESETS), key="set_preset", required=True,
             format_func=_preset_label, on_change=_cb_preset,
             help="Cada estilo muda o que o painel valoriza na hora de escolher a ordem.")
    key = ss.get("set_preset") or si.DEFAULT_PRESET
    info = si.PRESETS.get(key, si.PRESETS[si.DEFAULT_PRESET])
    preset_w = si.preset_weights(key)
    custom = (any(abs(float(ss.weights.get(k, 0.0)) - float(preset_w.get(k, 0.0))) > 1e-6 for k in WEIGHT_LABELS)
              or ss.get("energy_mode", "suave") != si.preset_energy_mode(key))
    st.caption(f"{info['icon']} **{info['label']}**: {info['description']}"
               + (" · com ajustes seus no Ajuste fino." if custom else ""))

    weights: dict[str, float] = {}
    with st.expander("🎛️ Ajuste fino: o que pesa mais na ordem", expanded=False):
        cols = st.columns(2, gap="large")
        for idx, (k, label) in enumerate(WEIGHT_LABELS.items()):
            wkey = f"w_{k}"
            if wkey not in ss:
                ss[wkey] = float(min(max(ss.weights.get(k, config.DEFAULT_WEIGHTS.get(k, 0.0)), 0.0),
                                     float(config.WEIGHT_SLIDER_MAX)))
            with cols[idx % 2]:
                weights[k] = float(st.slider(label, 0.0, float(config.WEIGHT_SLIDER_MAX), step=0.1,
                                             key=wkey, on_change=_cb_weights))
                st.caption(si.WEIGHT_HELP.get(k, ""))
        if "energy_rising" not in ss:
            ss.energy_rising = ss.get("energy_mode") == "subindo"
        st.toggle("📈 Energia sempre subindo (o set vai esquentando)", key="energy_rising",
                  on_change=_cb_energy_mode,
                  help="Ligado: o painel evita quedas de energia de uma música para a outra. "
                       "Desligado: só evita mudanças bruscas.")
    return weights, ("subindo" if ss.get("energy_rising") else "suave")


def _generate_order(analyses: dict[str, Analysis], weights: dict[str, float], start_id: str | None,
                    mode: str) -> None:
    ss = st.session_state
    try:
        res = order.order_set(list(analyses.values()), weights=weights, start_id=start_id,
                              voice_lookup=_voice_lookup(refresh=True), energy_mode=mode)
    except Exception as e:  # noqa: BLE001
        st.error(_error_message(e, "gerar a ordem"))
        return
    ss.order = list(res.order)
    ss.explanations = None
    ss.export_report = None
    ss.preview = None
    _persist()


def _render_setlist() -> dict[str, Any] | None:
    ss = st.session_state
    theme.section_header(3, "Setlist", "O painel acha a ordem em que as músicas emendam melhor. "
                                       "Escolha um estilo, gere a ordem e ouça as passagens.", anchor="setlist")
    approved = _approved_tracks()
    if not approved:
        st.info("Aprove músicas na triagem para montar o setlist.")
        ss.cur_order = []
        return None
    analyses: dict[str, Analysis] = {}
    for t in approved:
        an = _analysis(t.video_id)
        if an is not None:
            analyses[t.video_id] = an
    missing = [t for t in approved if t.video_id not in analyses]
    caption = f"{len(approved)} música(s) aprovada(s)."
    if missing:
        caption += (f" {len(missing)} ainda sem análise (aguardando download/análise); "
                    "entram na ordem quando ficarem prontas.")
    st.caption(caption)

    weights, mode = _render_style_picker()

    start_opts: list[str | None] = [None, *analyses.keys()]
    if ss.get("start_id") not in start_opts:
        ss.pop("start_id", None)
    with st.container(horizontal=True, vertical_alignment="bottom"):
        start_id = st.selectbox(
            "Música que abre o set", start_opts, key="start_id",
            format_func=lambda v: "Automático (a de menor energia)" if v is None else ss.catalog[v].display_name)
        clicked = st.button("🧮 Gerar ordem", key="btn_order", type="primary", disabled=not analyses)
    auto = ss.pop("auto_order", False)
    if analyses and (clicked or (auto and ss.order)):
        _generate_order(analyses, weights, start_id, mode)

    cur = _current_order(analyses)
    ss.cur_order = cur
    result = None
    if cur:
        try:
            result = order.evaluate_order(cur, analyses, weights=weights, voice_lookup=_voice_lookup(),
                                          energy_mode=mode)
        except Exception as e:  # noqa: BLE001
            st.error(_error_message(e, "avaliar a ordem"))
    if result is not None and result.order:
        _render_set_overview(result, analyses, weights, mode, approved)
        _render_energy_map(result, analyses)
        _render_passages(result, analyses)
        _render_manual_order(result, analyses)
    else:
        st.caption("Clique em \"Gerar ordem\". Sem ordem, a exportação usa a ordem de aprovação.")

    if result is not None and result.order:
        in_order = set(result.order)
        export_tracks = [ss.catalog[v] for v in result.order] + [t for t in approved if t.video_id not in in_order]
        export_transitions = list(result.transitions)
        ordered_ids = list(result.order)
    else:
        export_tracks = approved
        export_transitions = None
        ordered_ids = [t.video_id for t in approved]

    _render_vocal_studio([ss.catalog[v] for v in ordered_ids if v in ss.catalog]
                         + [t for t in approved if t.video_id not in set(ordered_ids)])
    return {"tracks": export_tracks, "analyses": analyses, "transitions": export_transitions,
            "ordered": export_transitions is not None}


_LABEL_WORDS = {"suave": ("passagem suave", "passagens suaves"), "boa": ("passagem boa", "passagens boas"),
                "atenção": ("pede atenção", "pedem atenção"), "difícil": ("difícil", "difíceis")}


def _current_arc() -> str:
    """Arco de energia escolhido (o widget some quando não há setlist; a escolha fica guardada)."""
    ss = st.session_state
    for value in (ss.get("set_arc"), ss.get("arc")):
        if value in ui_parts.ARCS:
            return value
    return ui_parts.DEFAULT_ARC


def _render_set_overview(result, analyses: dict[str, Analysis], weights: dict[str, float], mode: str,
                         approved: list[Track]) -> None:
    """Nota de fluidez, comparação com a ordem de aprovação, resumo e linha do tempo."""
    ss = st.session_state
    score = si.set_score(list(result.transitions), weights)
    verdict, vtone = si.score_verdict(score)
    baseline = [t.video_id for t in approved if t.video_id in analyses]
    cmp = None
    if len(baseline) >= 2 and len(result.order) >= 2:
        try:
            cmp = si.compare_orders(baseline, list(result.order), analyses, weights, _voice_lookup(),
                                    energy_mode=mode)
        except Exception:  # noqa: BLE001
            logger.warning("Falha ao comparar com a ordem de aprovação", exc_info=True)
    summary = si.set_summary(list(result.order), analyses)
    arc = _current_arc()
    adherence = ui_parts.arc_adherence([analyses[v].energy_mean for v in result.order], arc)

    rings = [theme.score_ring_svg(score, "Fluidez do set")]
    if cmp is not None:
        rings.append(theme.score_ring_svg(cmp["baseline_score"], "Na ordem em que você aprovou", tone="muted"))
    if adherence is not None:
        rings.append(theme.score_ring_svg(adherence, f"Segue o arco “{ui_parts.ARCS[arc]['label']}”", tone="info"))

    lines = []
    if result.transitions:
        lines.append("Nota de 0 a 100 para o quanto as músicas emendam bem, uma na outra.")
    else:
        lines.append("Com uma música só ainda não há passagens para avaliar.")
    ctext, _ = ui_parts.compare_text(cmp)
    if ctext:
        lines.append(ctext)
    chips = []
    if summary.get("n"):
        chips.append(theme.badge(f"≈ {ui_parts.set_duration_text(summary['duration_s'])} de set", "info", "⏱"))
        lo, hi = summary.get("bpm_min"), summary.get("bpm_max")
        if lo is not None and hi is not None:
            lo_t, hi_t = ui_parts.bpm_text(lo), ui_parts.bpm_text(hi)
            chips.append(theme.badge(f"{lo_t} BPM" if lo_t == hi_t else f"{lo_t}–{hi_t} BPM", "accent", "♫"))
        if summary.get("energy_start") is not None and summary.get("energy_end") is not None:
            chips.append(theme.badge(f"energia {ui_parts.pct(summary['energy_start'])}% → "
                                     f"{ui_parts.pct(summary['energy_end'])}%", "accent", "⚡"))
        if summary.get("peak_pos"):
            chips.append(theme.badge(f"pico na música {summary['peak_pos']:02d}", "warn", "🔥"))
        for cam, n in summary.get("keys") or []:
            chips.append(theme.chip("tom", f"{cam} ×{n}", key=True, strong_first=False))
    counts = si.label_counts(list(result.transitions))
    for label, n in counts.items():
        if n:
            one, many = _LABEL_WORDS.get(label, (label, label))
            chips.append(theme.badge(f"{n} {one if n == 1 else many}", theme.tone_for_label(label)))

    c1, c2 = st.columns([2, 3], vertical_alignment="center", gap="large")
    c1.html(theme.rings_html(rings))
    c2.html(ui_parts.verdict_html(verdict, vtone, lines, chips))

    playing = set()
    pv = ss.get("preview") or {}
    if pv.get("a") and pv.get("b"):
        playing = {pv["a"], pv["b"]}
    items, links = [], []
    for i, vid in enumerate(result.order):
        t = ss.catalog.get(vid)
        an = analyses[vid]
        items.append({"pos": i + 1, "title": t.title if t else vid, "artist": t.artist_str if t else "",
                      "bpm": an.effective_bpm, "camelot": an.camelot, "energy": an.energy_mean,
                      "thumb": t.thumbnail_url if t else None,
                      "duration": an.duration_s or (t.duration_s if t else None), "playing": vid in playing})
    for tr in result.transitions:
        head = si.transition_headline(tr, analyses.get(tr.from_id), analyses.get(tr.to_id))
        links.append({"label": tr.label, "short": ui_parts.headline_short(head)})
    theme.timeline(items, links)
    theme.legend()


def _render_energy_map(result, analyses: dict[str, Analysis]) -> None:
    ss = st.session_state
    n = len(result.order)
    if n < 2:
        return
    st.markdown("**📈 Mapa de energia do set** · energia (rosa) e velocidade (roxo) música a música")
    if ss.get("set_arc") not in ui_parts.ARCS:
        ss.set_arc = _current_arc()
    st.segmented_control("Arco de energia que você quer (linha tracejada)", list(ui_parts.ARCS), key="set_arc",
                         required=True, on_change=_cb_arc,
                         format_func=lambda k: f"{ui_parts.ARCS[k]['icon']} {ui_parts.ARCS[k]['label']}")
    arc = _current_arc()
    st.caption(ui_parts.ARCS.get(arc, {}).get("description", ""))
    energies = [float(analyses[v].energy_mean) for v in result.order]
    rows = []
    for i, vid in enumerate(result.order):
        t = ss.catalog.get(vid)
        an = analyses[vid]
        rows.append({"pos": i + 1, "faixa": t.display_name if t else vid, "bpm": float(an.effective_bpm),
                     "tom": an.camelot or "?", "energia": round(float(an.energy_mean) * 100, 1),
                     "passagem": "início" if i == 0 else (result.transitions[i - 1].label or "boa")})
    target = ui_parts.arc_target(energies, arc)
    try:
        from modules import ui_charts

        chart = ui_charts.energy_map_chart(rows, [v * 100 for v in target] if target else None)
        st.altair_chart(chart, width="stretch")
    except Exception:  # noqa: BLE001
        logger.warning("Falha ao desenhar o mapa de energia", exc_info=True)
        st.line_chart({"energia": [r["energia"] for r in rows]})


def _render_passages(result, analyses: dict[str, Analysis]) -> None:
    ss = st.session_state
    n = len(result.order)
    if not result.transitions:
        return
    width = 3 if n > 99 else 2
    expl = ss.explanations
    texts = expl["texts"] if expl and expl.get("order") == list(result.order) else None
    with st.container(horizontal=True, vertical_alignment="center"):
        st.markdown("**🎧 Passagens, uma por uma** · como cada música entra na próxima")
        if _explain_available():
            ask = st.button("💬 Explicar passagens (Gemini)", key="btn_explain",
                            help="Uma frase curta por passagem. A ordem não muda.")
        else:
            ask = False
    if ask:
        tracks = [ss.catalog[v] for v in result.order]
        with st.spinner("Pedindo as explicações ao Gemini…"):
            try:
                phrases = explain.explain_transitions(tracks, analyses, list(result.transitions))
            except Exception as e:  # noqa: BLE001
                st.error(_error_message(e, "explicar as transições"))
            else:
                ss.explanations = {"order": list(result.order), "texts": [str(p) for p in phrases]}
                st.rerun()
    if any(_audio_path(ss.catalog.get(v)) is not None for v in result.order):
        _warm_preview()

    for i, tr in enumerate(result.transitions):
        a, b = result.order[i], result.order[i + 1]
        ta, tb = ss.catalog.get(a), ss.catalog.get(b)
        if texts and i < len(texts):
            tr.explanation = texts[i]
        an_a, an_b = analyses.get(a), analyses.get(b)
        reasons = si.transition_reasons(tr, an_a, an_b)
        headline = si.transition_headline(tr, an_a, an_b)
        pv = ss.get("preview") or {}
        playing = pv.get("a") == a and pv.get("b") == b
        prefix = theme.PLAYING_KEY_PREFIX if playing else theme.CARD_KEY_PREFIX
        with st.container(key=f"{prefix}pass_{i}"):
            _html(ui_parts.passage_html(i + 1, i + 2, width, ta.display_name if ta else a,
                                          tb.display_name if tb else b, headline, tr.label, reasons,
                                          tr.explanation))
            path_a, path_b = _audio_path(ta), _audio_path(tb)
            ready = path_a is not None and path_b is not None
            with st.container(horizontal=True, vertical_alignment="center"):
                listen = st.button("▶ Ouvir passagem", key=f"listen_{a}__{b}", disabled=not ready,
                                   type="primary" if playing else "secondary",
                                   help="Toca o fim desta música emendado no começo da próxima (≈ 28 s)."
                                   if ready else "O áudio de uma das músicas ainda não foi baixado.")
                if playing:
                    st.button("⏹ Fechar", key=f"listen_stop_{i}", on_click=_cb_stop_preview)
                st.caption(f"{VOICE_TEXT.get(tr.voice, tr.voice)} · ΔBPM {_dec(tr.bpm_diff, 1)} · "
                           f"nota de dificuldade {_dec(tr.cost, 2)}")
            if listen and ready:
                _make_preview(a, b, path_a, path_b, an_a, an_b)
            _show_flash(f"pass:{a}:{b}")
            if playing and pv.get("path"):
                st.audio(str(pv["path"]), format="audio/wav", autoplay=True)


@st.cache_resource(show_spinner=False)
def _warm_preview() -> bool:
    """Prepara o librosa numa thread uma vez por processo: a 1ª "Ouvir passagem" sai rápida."""
    threading.Thread(target=lambda: preview.warm_up(), name="djsk-preview-warmup", daemon=True).start()
    return True


def _make_preview(a: str, b: str, path_a: Path, path_b: Path, an_a: Analysis | None,
                  an_b: Analysis | None) -> None:
    ss = st.session_state
    with st.spinner("Montando a passagem (fim de uma + começo da outra)…"):
        try:
            p = preview.make_transition_preview(
                path_a, path_b, a, b,
                bpm_a=float(an_a.effective_bpm) if an_a is not None else None,
                bpm_b=float(an_b.effective_bpm) if an_b is not None else None)
        except (FileNotFoundError, preview.PreviewError, ValueError) as e:
            _flash(f"pass:{a}:{b}", "warning", str(e) or "Não consegui montar esta passagem.")
            return
        except Exception as e:  # noqa: BLE001
            _flash(f"pass:{a}:{b}", "error", _error_message(e, "montar a passagem"))
            return
    ss.preview = {"a": a, "b": b, "path": str(p)}
    ss.active = None                     # para o player da triagem
    st.rerun()


def _render_manual_order(result, analyses: dict[str, Analysis]) -> None:
    ss = st.session_state
    n = len(result.order)
    width = 3 if n > 99 else 2
    with st.expander("✋ Mudar a ordem na mão", expanded=False):
        st.caption("Use ▲ e ▼ para trocar uma música de lugar. A fluidez e as passagens se atualizam na hora.")
        for i, vid in enumerate(result.order):
            track = ss.catalog.get(vid)
            an = analyses[vid]
            row = st.columns([0.6, 7, 0.6, 0.6], vertical_alignment="center")
            row[0].markdown(f"**{i + 1:0{width}d}**")
            name = _md(track.display_name if track else vid)
            details = [f"{_dec(an.effective_bpm, 1)} BPM", f"energia {ui_parts.pct(an.energy_mean)}%"]
            if an.camelot:
                details.append(f"tom {_md(an.camelot)}")
            row[1].markdown(f"{name}  \n:gray[{' · '.join(details)}]")
            row[2].button("▲", key=f"up_{vid}", on_click=_cb_move, args=(vid, -1), disabled=i == 0,
                          help="Subir uma posição")
            row[3].button("▼", key=f"down_{vid}", on_click=_cb_move, args=(vid, 1), disabled=i == n - 1,
                          help="Descer uma posição")


def _explain_available() -> bool:
    try:
        return bool(explain.is_available())
    except Exception:  # noqa: BLE001
        return False


def _render_vocal_studio(tracks: list[Track]) -> None:
    ss = st.session_state
    bud = ss.get("budget")
    est_usd = float(config.STEM_COST_ESTIMATE_USD)
    est_brl = est_usd * float(config.USD_BRL)
    with st.expander("🎙️ Estúdio de Vocal (função paga): separar a voz do MC", expanded=False):
        st.caption(_no_latex(f"Separa a faixa em 4 partes (voz, bateria, baixo, outros) com Demucs na Replicate. "
                             f"Custo estimado: US$ {_dec(est_usd, 3)} (≈ {util.format_brl(est_brl)}) por faixa. "
                             "A voz separada também ajuda a ordem a evitar dois MCs cantando juntos."))
        if not config.get_secret("REPLICATE_API_TOKEN"):
            can, reason = False, "Configure REPLICATE_API_TOKEN no arquivo .env para usar o Estúdio de Vocal."
        elif bud is None:
            can, reason = False, "O medidor de orçamento não abriu; o Estúdio de Vocal fica travado por segurança."
        else:
            try:
                can, reason = bud.can_run()
            except Exception as e:  # noqa: BLE001
                can, reason = False, _error_message(e, "consultar o orçamento")
        if not can:
            st.warning(f"🔒 {reason}")
        try:
            local_ok = bool(stems.local_demucs_available())
        except Exception:  # noqa: BLE001
            local_ok = False
        if local_ok:
            st.caption("Plano B: Demucs local disponível (sem custo, mais lento).")
        elif not can:
            st.caption("Plano B: Demucs local (sem custo). Instale o pacote `demucs` neste Python para liberar.")
        if bud is not None and getattr(bud, "insufficient_credit", False):
            st.button("Liberar trava de saldo (já adicionei crédito)", key="btn_reset_credit",
                      on_click=_cb_reset_credit)

        for track in tracks:
            vid = track.video_id
            path = _audio_path(track)
            done = _has_stems(vid)
            with st.container(key=f"{theme.CARD_KEY_PREFIX}stem_{vid}"):
                row = st.columns([5, 2, 2], vertical_alignment="center")
                row[0].markdown(_md(track.display_name))
                if path is None:
                    why = "O áudio desta faixa ainda não foi baixado."
                elif done:
                    why = "Os stems desta faixa já foram separados."
                else:
                    why = None if can else reason
                with row[1]:
                    run_remote = st.button("🎙️ Extrair Acapella", key=f"stem_{vid}",
                                           disabled=(not can) or path is None or done, help=why)
                run_local = False
                if local_ok:
                    with row[2]:
                        run_local = st.button("Separar localmente", key=f"stem_local_{vid}",
                                              disabled=path is None or done,
                                              help="Plano B: Demucs no computador, sem custo.")
                if run_remote or run_local:
                    _run_stems(track, path, local=run_local)
                _show_flash(f"stem:{vid}")
                info = ss.stem_results.get(vid)
                if done or info:
                    c1, c2 = st.columns([3, 2], vertical_alignment="center")
                    if info:
                        c1.caption(_no_latex(f"Gasto: {util.format_brl(info.get('cost_brl') or 0.0)} "
                                             f"(US$ {_dec(info.get('cost_usd') or 0.0, 4)}) · {info.get('backend')}"))
                    else:
                        c1.caption("Stems prontos (cache, sem custo).")
                    c2.button("⏹ Fechar vocal" if ss.stem_player == vid else "▶ Ouvir vocal",
                              key=f"vocal_{vid}", on_click=_cb_toggle_vocal, args=(vid,))
                    if ss.stem_player == vid:
                        vocals = _vocals_path(vid)
                        if vocals is not None and vocals.is_file():
                            st.audio(str(vocals), format=_mime(vocals))
                        else:
                            st.caption("Arquivo do vocal não encontrado em stems/.")


def _run_stems(track: Track, path: Path | None, local: bool) -> None:
    ss = st.session_state
    vid = track.video_id
    if path is None:
        return
    msg = ("Separando com o Demucs local (pode levar alguns minutos)…" if local
           else "Separando os stems na Replicate (cerca de 70 s)…")
    with st.spinner(msg):
        try:
            stems.register_name(vid, track.display_name)
            if local:
                res = stems.separate_local(vid, path)
            else:
                res = stems.separate_replicate(vid, path, ss.budget)
        except Exception as e:  # noqa: BLE001
            _flash(f"stem:{vid}", "error", _error_message(e, "extrair a acapella"))
            _render_header()
            return
    vocals = res.vocals_path
    ss.stem_results[vid] = {
        "backend": res.backend,
        "cost_brl": float(res.cost_brl or 0.0),
        "cost_usd": float(res.cost_usd or 0.0),
        "predict_time_s": res.predict_time_s,
        "vocals": str(vocals) if vocals else None,
    }
    ss.stem_player = vid
    ss.voice_lookup = None   # stems novos mudam as janelas de voz
    gpu = f", {_dec(res.predict_time_s, 0)} s de GPU" if res.predict_time_s else ""
    _flash(f"stem:{vid}", "success",
           _no_latex(f"Acapella pronta. Gasto: {util.format_brl(float(res.cost_brl or 0.0))} "
                     f"(US$ {_dec(float(res.cost_usd or 0.0), 4)}{gpu})."))
    st.rerun()   # redesenha o medidor do topo com o gasto novo


# ---------------------------------------------------------------------------
# 4 Exportar (Rekordbox e Spotify)
# ---------------------------------------------------------------------------
def _render_export_section(ctx: dict[str, Any] | None) -> None:
    theme.section_header(4, "Exportar", "Leve o set para o Rekordbox (pasta numerada + playlist .m3u8) "
                                        "e, se quiser, crie a playlist no Spotify.", anchor="exportar")
    if ctx is None:
        st.info("Aprove músicas na triagem para exportar o set.")
        return
    _render_export(ctx["tracks"], ctx["analyses"], ctx["transitions"], ctx["ordered"])


def _render_export(tracks: list[Track], analyses: dict[str, Analysis],
                   transitions: list[Transition] | None, ordered: bool) -> None:
    ss = st.session_state
    how = "na ordem do setlist" if ordered else "na ordem de aprovação (nenhuma ordem gerada)"
    name = st.text_input("Nome do set", value=f"Set {datetime.now():%d-%m %Hh%M}", key="export_name",
                         help="Cada exportação cria uma pasta nova com este nome dentro de Setlists/.")
    left, right = st.columns(2, gap="large")
    with left, st.container(key=f"{theme.CARD_KEY_PREFIX}export_rb"):
        st.markdown("**📦 Rekordbox** · pasta + playlist")
        st.caption(f"Cria a pasta {config.SETLISTS_DIR}\\<nome> com cópias numeradas, <nome>.m3u8 "
                   f"e <nome>.csv, {how} ({len(tracks)} música(s)).")
        if st.button("📦 Exportar pasta + .m3u8", key="btn_export", type="primary", disabled=not tracks):
            _do_export(tracks, analyses, transitions, name)
        _render_export_report()
    with right, st.container(key=f"{theme.CARD_KEY_PREFIX}export_sp"):
        st.markdown("**🎧 Spotify** · playlist na sua conta")
        _render_spotify_export(tracks, name)


def _do_export(tracks: list[Track], analyses: dict[str, Analysis], transitions: list[Transition] | None,
               name: str) -> None:
    ss = st.session_state
    with st.spinner("Copiando os arquivos e gerando a playlist…"):
        try:
            folder_name = util.safe_filename((name or "").strip() or "Set", max_len=80)
            out_dir, n = Path(config.SETLISTS_DIR) / folder_name, 2
            while out_dir.exists():          # nunca sobrescreve uma playlist já exportada
                out_dir = Path(config.SETLISTS_DIR) / f"{folder_name} ({n})"
                n += 1
            result = export.export_setlist(tracks, analyses, transitions,
                                           out_dir=out_dir, playlist_name=out_dir.name)
        except Exception as e:  # noqa: BLE001
            st.error(_error_message(e, "exportar o setlist"))
            return
    record = None
    met = ss.get("metrics")
    if met is not None:
        try:
            record = met.finish(len(_approved_tracks()), len(ss.decisions))
        except Exception:  # noqa: BLE001
            logger.warning("Falha ao registrar o fim da sessão", exc_info=True)
    ss.export_report = {
        "folder": str(result.folder),
        "m3u8": str(result.m3u8_path),
        "m3u8_portable": str(result.portable_m3u8_path) if getattr(result, "portable_m3u8_path", None) else None,
        "csv": str(result.csv_path),
        "files": [Path(p).name for p in result.files],
        "skipped": list(result.skipped),
        "record": record,
    }
    _persist()


def _render_export_report() -> None:
    rep = st.session_state.get("export_report")
    if not rep:
        return
    st.success(f"Setlist exportado em {rep['folder']} ({len(rep['files'])} faixa(s)). "
               f"No Rekordbox, importe {Path(rep['m3u8']).name} ou arraste a pasta numerada.")
    portable = rep.get("m3u8_portable")
    if portable:
        st.info(f"Vai levar no pen drive para outro computador? Copie a pasta inteira e, lá, importe "
                f"{Path(portable).name}: ela acha as músicas pela própria pasta, em qualquer lugar.")
    if rep["skipped"]:
        st.warning("Puladas (sem arquivo de áudio): " + "; ".join(rep["skipped"]))
    with st.expander("Arquivos exportados"):
        names = [*rep["files"], Path(rep["m3u8"]).name, *([Path(portable).name] if portable else []),
                 Path(rep["csv"]).name]
        st.code("\n".join(names), language=None)
    if hasattr(os, "startfile"):
        st.button("📂 Abrir a pasta", key="btn_open_export", on_click=_cb_open_folder, args=(rep["folder"],))
    record = rep.get("record") or {}
    if record.get("aprovadas"):
        st.caption(f"Sessão registrada nas métricas: {_fmt_elapsed(record.get('duracao_s') or 0)} para "
                   f"{record['aprovadas']} faixa(s) aprovadas ({_dec(record.get('s_por_faixa'), 0)} s por faixa).")


def _render_spotify_export(tracks: list[Track], name: str) -> None:
    ss = st.session_state
    if not _spotify_configured():
        st.caption("Spotify desligado (chaves no .env). A exportação para o Rekordbox funciona sem ele.")
        return
    sp = ss.get("spotify_client")
    if sp is None:
        st.caption("Para criar a playlist no Spotify, entre na sua conta na seção 1 (aba 🟢 Spotify).")
        return
    user = ss.get("spotify_user") or {}
    st.caption(f"Cria a playlist \"{name or spotify_import.DEFAULT_PLAYLIST_NAME}\" (privada) na conta "
               f"{user.get('name') or 'conectada'}, na mesma ordem do set.")
    if st.button("🎧 Criar playlist no Spotify", key="btn_spotify_export", disabled=not tracks):
        bar = st.progress(0.0, text="Procurando as músicas no Spotify…")

        def _progress(done: int, total: int) -> None:
            try:
                bar.progress(min(1.0, done / max(1, total)), text=f"Procurando no Spotify: {done}/{total}")
            except Exception:  # noqa: BLE001
                pass

        try:
            res = spotify_import.export_set_to_spotify(
                sp, name or spotify_import.DEFAULT_PLAYLIST_NAME, list(tracks),
                description="Set montado no painel de curadoria do DJ SK", progress=_progress)
        except Exception as e:  # noqa: BLE001
            bar.empty()
            st.error(_error_message(e, "criar a playlist no Spotify"))
        else:
            bar.empty()
            ss.spotify_export = res
    res = ss.get("spotify_export")
    if not res:
        return
    missing = list(res.get("missing") or [])
    st.success(f"Playlist criada no Spotify com {res.get('added', 0)} música(s)."
               + (f" {len(missing)} não foram achadas lá." if missing else ""))
    if res.get("playlist_url"):
        st.link_button("Abrir no Spotify", res["playlist_url"], icon="🟢")
    if missing:
        with st.expander(f"{len(missing)} música(s) que não estão no Spotify"):
            _html(ui_parts.badges_html([theme.badge(m, "muted") for m in missing[:40]], compact=True))
    embed = ui_parts.spotify_embed_url("playlist", res.get("playlist_id"))
    if embed:
        st.iframe(embed, height=152)


# ---------------------------------------------------------------------------
# 5 Métricas (seção 6 da especificação)
# ---------------------------------------------------------------------------
def _n_downloaded() -> int:
    tj = st.session_state.get("jobs")
    if tj is None:
        return 0
    try:
        return int(tj.summary().get("baixadas", 0) or 0)
    except Exception:  # noqa: BLE001
        return 0


def _live_stats() -> dict[str, Any] | None:
    ss = st.session_state
    met = ss.get("metrics")
    if met is None or not hasattr(met, "live_stats"):
        return None
    rep = ss.get("export_report")
    bud = ss.get("budget")
    try:
        spent = float(bud.spent_brl) if bud is not None else 0.0
    except Exception:  # noqa: BLE001
        spent = 0.0
    try:
        return met.live_stats(n_suggested=len(ss.suggestions), n_downloaded=_n_downloaded(),
                              n_exported=len(rep["files"]) if rep else None, spent_brl=spent)
    except Exception:  # noqa: BLE001
        logger.warning("Falha ao calcular as métricas ao vivo", exc_info=True)
        return None


def _times(x: Any) -> str:
    return f"{_dec(x, 1)}×"


def _metric_cards(s: dict[str, Any]) -> list[str]:
    lo_s, hi_s = float(s["baseline_lo_s"]), float(s["baseline_hi_s"])
    approved = int(s.get("approved") or 0)
    spa = s.get("s_per_approved")
    if approved:
        if (s.get("saved_min_hi_raw") or 0) > 0:
            saved, saved_sub = (ui_parts.fmt_range_minutes(s["saved_min_lo"], s["saved_min_hi"]),
                                f"nesta sessão, para {approved} música(s) aprovada(s)")
        else:
            saved, saved_sub = "0 min", "ainda no ritmo do manual: o ganho aparece conforme você aprova"
    else:
        saved, saved_sub = "–", "aprove músicas para começar a contar"
    cards = [theme.kpi_card("Tempo economizado", saved, saved_sub, tone="ok", icon="⏱")]
    if spa:
        tone = "ok" if spa < lo_s else "warn"
        cards.append(theme.kpi_card("Tempo por música aprovada", f"{_dec(spa, 0)} s",
                                    f"no manual: {lo_s:.0f}–{hi_s:.0f} s", tone=tone, icon="🎵"))
    else:
        cards.append(theme.kpi_card("Tempo por música aprovada", "–", f"no manual: {lo_s:.0f}–{hi_s:.0f} s",
                                    tone="muted", icon="🎵"))
    if s.get("speedup_lo"):
        cards.append(theme.kpi_card("Ganho de velocidade", f"{_times(s['speedup_lo'])} a {_times(s['speedup_hi'])}",
                                    "mais rápido que o processo manual", tone="accent", icon="🚀"))
    else:
        cards.append(theme.kpi_card("Ganho de velocidade", "–", "comparado com o processo manual",
                                    tone="muted", icon="🚀"))
    rate = s.get("approval_rate")
    sub = []
    if rate is not None:
        sub.append(f"{int(round(rate * 100))}% aprovadas")
    if s.get("avg_decision_s"):
        sub.append(f"{_dec(s['avg_decision_s'], 0)} s por decisão")
    cards.append(theme.kpi_card("Decisões na triagem", str(int(s.get("decisions") or 0)),
                                " · ".join(sub) or "aprove ou reprove para medir o ritmo", tone="info", icon="✅"))
    cpa = s.get("cost_per_approved_brl")
    cost = float(s.get("cost_brl") or 0.0)
    cards.append(theme.kpi_card("Custo por música", util.format_brl(cpa) if cpa is not None else "–",
                                f"gasto no Estúdio de Vocal: {util.format_brl(cost)}" if cost
                                else "a triagem não custa nada", tone="good", icon="💸"))
    elapsed = float(s.get("elapsed_s") or 0.0)
    cards.append(theme.kpi_card("Tempo de sessão", _fmt_elapsed(elapsed) if s.get("started") else "–",
                                "parado na exportação" if s.get("finished") else "desde a busca de sugestões",
                                tone="default", icon="⌛"))
    return cards


def _metrics_live() -> None:
    """Números ao vivo (roda de novo sozinho a cada METRICS_REFRESH_S enquanto a sessão está aberta)."""
    s = _live_stats()
    if s is None:
        st.info("As métricas aparecem quando você buscar sugestões.")
        return
    theme.kpi_grid(_metric_cards(s), columns=3)
    try:
        from modules import ui_charts
    except ImportError:  # pragma: no cover - altair/pandas vêm com o Streamlit
        ui_charts = None
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Segundos por música: manual x painel**")
        if ui_charts is not None:
            try:
                st.altair_chart(ui_charts.manual_vs_panel_chart(s.get("s_per_approved"), s["baseline_lo_s"],
                                                                s["baseline_hi_s"]), width="stretch")
            except Exception:  # noqa: BLE001
                logger.warning("Falha ao desenhar o gráfico manual x painel", exc_info=True)
        proj = s.get("set_projection") or {}
        if proj.get("panel_min"):
            _html(ui_parts.badges_html([
                theme.badge(f"Set de {proj['size']} músicas: {ui_parts.fmt_minutes(proj['panel_min'])} com o painel",
                            "accent", "🎚️"),
                theme.badge(f"{ui_parts.fmt_minutes(proj['manual_lo_min'])} a "
                            f"{ui_parts.fmt_minutes(proj['manual_hi_min'])} no manual", "muted", "🐢"),
            ]))
        else:
            st.caption(f"Projeção para um set de {proj.get('size', config.TYPICAL_SET_SIZE)} músicas: "
                       f"{ui_parts.fmt_minutes(proj.get('manual_lo_min'))} a "
                       f"{ui_parts.fmt_minutes(proj.get('manual_hi_min'))} no processo manual.")
    with c2:
        st.markdown("**Funil da curadoria** · do rádio até o set exportado")
        if ui_charts is not None:
            try:
                st.altair_chart(ui_charts.funnel_chart(s.get("funnel") or []), width="stretch")
            except Exception:  # noqa: BLE001
                logger.warning("Falha ao desenhar o funil", exc_info=True)
    if s.get("finished"):
        st.caption("Cronômetro parado na exportação. Uma nova busca de sugestões abre outra medição.")


def _render_metrics() -> None:
    ss = st.session_state
    base_min, base_max = (float(v) for v in config.BASELINE_SECONDS_PER_TRACK)
    theme.section_header(5, "Métricas", f"Quanto tempo o painel devolve ao DJ, comparado com o processo manual "
                                        f"({base_min:.0f}–{base_max:.0f} s por música).", anchor="metricas")
    met = ss.get("metrics")
    running = bool(met is not None and getattr(met, "started", False) and not getattr(met, "finished", False))
    st.fragment(_metrics_live, run_every=METRICS_REFRESH_S if running else None)()
    _render_history(met)


def _render_history(met) -> None:
    try:
        history = met.history() if met is not None else []
    except Exception:  # noqa: BLE001
        history = []
    st.markdown("**🗂️ Histórico de sessões**")
    if not history:
        st.caption("Nenhuma sessão registrada ainda: o registro acontece quando você exporta o set.")
        return
    try:
        hs = metrics.history_stats(history)
    except Exception:  # noqa: BLE001
        hs = None
    if hs:
        cards = [
            theme.kpi_card("Sessões registradas", str(hs["sessions"]), f"{hs['total_approved']} música(s) aprovadas",
                           tone="info", icon="🗂️"),
            theme.kpi_card("Tempo economizado no total",
                           ui_parts.fmt_range_minutes(hs["total_saved_min_lo"], hs["total_saved_min_hi"]),
                           f"em {ui_parts.fmt_minutes(hs['total_minutes'])} de uso do painel", tone="ok", icon="⏱"),
            theme.kpi_card("Melhor tempo por música",
                           f"{_dec(hs['best_s_per_track'], 0)} s" if hs.get("best_s_per_track") else "–",
                           f"média: {_dec(hs['avg_s_per_track'], 0)} s" if hs.get("avg_s_per_track") else None,
                           tone="accent", icon="🏁"),
        ]
        theme.kpi_grid(cards, columns=3)
        if hs.get("trend"):
            try:
                from modules import ui_charts

                lo, hi = (float(v) for v in config.BASELINE_SECONDS_PER_TRACK)
                st.altair_chart(ui_charts.trend_chart(hs["trend"], lo, hi), width="stretch")
            except Exception:  # noqa: BLE001
                logger.warning("Falha ao desenhar a evolução das sessões", exc_info=True)
    rows = []
    for h in reversed(history[-20:]):
        comp = h.get("comparacao") or {}
        rate = h.get("taxa_aprovacao")
        rows.append({
            "início": str(h.get("inicio", "")).replace("T", " "),
            "aprovadas": h.get("aprovadas", 0),
            "decisões": h.get("decisoes", h.get("revisadas", 0)),
            "taxa de aprovação": f"{int(round(float(rate) * 100))}%" if isinstance(rate, (int, float)) else "–",
            "duração": _fmt_elapsed(float(h.get("duracao_s") or 0)),
            "s por faixa": _dec(h.get("s_por_faixa"), 0),
            "s por decisão": _dec(h.get("s_por_decisao"), 0) if h.get("s_por_decisao") is not None else "–",
            "ganho": (f"{_dec(comp.get('speedup_min'), 1)}× a {_dec(comp.get('speedup_max'), 1)}×"
                      if comp.get("speedup_min") else "–"),
        })
    with st.expander(f"Tabela das últimas {len(rows)} sessão(ões)"):
        st.dataframe(rows, hide_index=True)


# ---------------------------------------------------------------------------
# Barra lateral: integrações, atalhos e sessão (encerrar / zerar painel)
# ---------------------------------------------------------------------------
def _render_sidebar() -> None:
    ss = st.session_state
    spotify_ok = _spotify_configured()
    user = ss.get("spotify_user") or {}
    if ss.get("spotify_client") is not None:
        sp_detail = f"conectado como {user.get('name') or 'sua conta'}"
    elif spotify_ok:
        sp_detail = {"saved": "login salvo: conecte na seção 1",
                     "outdated": "entre de novo para criar playlists"}.get(_spotify_login_state())
    else:
        sp_detail = None
    items = [
        {"name": "YouTube Music", "purpose": "rádio de sugestões e download", "ok": True},
        {"name": "Replicate", "purpose": "Estúdio de Vocal (separar a voz)",
         "ok": bool(config.get_secret("REPLICATE_API_TOKEN"))},
        {"name": "Spotify", "purpose": "buscar, importar e criar playlists", "ok": spotify_ok, "detail": sp_detail},
        {"name": "Gemini", "purpose": "explicar as passagens do set", "ok": _explain_available()},
        {"name": "YouTube Data API", "purpose": "plano B das sugestões",
         "ok": bool(config.get_secret("YOUTUBE_API_KEY"))},
    ]
    with st.sidebar:
        st.subheader("🔌 Integrações")
        _html(ui_parts.integrations_html(items))
        st.caption("As chaves ficam no arquivo .env (veja .env.example). Nenhuma é obrigatória para a triagem.")
        st.divider()
        st.subheader("⌨ Atalhos da triagem")
        _html(ui_parts.kbd_legend_html(note="no card em foco"))
        st.divider()
        st.subheader("🗃️ Sessão")
        st.caption(f"Cache de áudio: {config.CACHE_DIR}")
        if not ss.get("confirm_end"):
            st.button("Encerrar sessão", key="btn_end", on_click=_cb_ask_end,
                      help="Apaga do cache as faixas reprovadas e limpa a sessão salva.")
        else:
            n_rej = sum(1 for d in ss.decisions.values() if d == REJECTED)
            st.warning(f"Encerrar apaga do cache {n_rej} faixa(s) reprovada(s) e limpa a sessão salva. "
                       "As aprovadas continuam no cache e a pasta exportada não é alterada.")
            c1, c2 = st.columns(2)
            c1.button("Sim, encerrar", key="btn_end_confirm", type="primary", on_click=_cb_end_session)
            c2.button("Cancelar", key="btn_end_cancel", on_click=_cb_cancel_end)
        st.button("🧹 Zerar painel", key="btn_reset_open", on_click=_cb_reset_open,
                  help="Escolha o que apagar: músicas baixadas, sessão, histórico de métricas, stems, setlists...")


@st.dialog("🧹 Zerar painel", width="medium", on_dismiss=_cb_reset_close)
def _reset_dialog() -> None:
    ss = st.session_state
    try:
        usage = maintenance.usage()
    except Exception as e:  # noqa: BLE001
        st.error(_error_message(e, "medir o espaço usado"))
        usage = {}
    st.markdown("Marque o que apagar. **Não são apagados:** as chaves (.env), o login do Spotify, "
                "o cookies.txt e o próprio programa.")
    selected: list[str] = []
    for cat in maintenance.CATEGORIES:
        info = usage.get(cat) or {}
        key = f"reset_cat_{cat}"
        if key not in ss:
            ss[key] = cat in maintenance.DEFAULT_RESET
        size = f"{maintenance.format_bytes(info.get('bytes', 0))} · {int(info.get('files', 0))} arquivo(s)"
        safe = bool(info.get("safe", True))
        checked = st.checkbox(f"{maintenance.LABELS[cat]} ({size})", key=key, disabled=not safe,
                              help=None if safe else "Pasta fora do projeto: o painel não apaga por segurança.")
        st.caption(maintenance.DESCRIPTIONS[cat])
        if checked and safe:
            selected.append(cat)
    if "orcamento" in selected:
        st.warning("O medidor de gasto volta a R$ 0,00, mas o dinheiro gasto na Replicate foi real e não volta.")
    if "setlists" in selected:
        st.warning("As pastas de setlist exportadas vão sumir. Se você já importou no Rekordbox, "
                   "as faixas também somem de lá.")
    if "stems" in selected:
        st.warning("Separar as vozes de novo gasta crédito na Replicate.")
    sure = st.checkbox("Sim, quero apagar de vez o que está marcado", key="reset_confirm")
    c1, c2 = st.columns(2)
    go = c1.button("🧹 Apagar agora", key="btn_reset_go", type="primary", disabled=not (sure and selected))
    cancel = c2.button("Cancelar", key="btn_reset_cancel")
    if cancel:
        ss.reset_open = False
        st.rerun()
    if go and sure and selected:
        _do_reset(selected)


def _do_reset(selected: list[str]) -> None:
    """Para a fila, apaga as categorias marcadas, limpa a memória do painel e recomeça a página."""
    _shutdown_jobs()
    try:
        rep = maintenance.reset(selected)
    except Exception as e:  # noqa: BLE001
        st.error(_error_message(e, "zerar o painel"))
        return
    names = ", ".join(maintenance.LABELS.get(c, c).lower() for c in selected)
    msgs = [(TOP, "success", f"🧹 Painel zerado ({names}): {rep.summary()}")]
    msgs += [(TOP, "warning", f) for f in rep.failures]
    st.session_state.clear()
    try:
        st.cache_data.clear()
    except Exception:  # noqa: BLE001
        pass
    st.session_state.flash = msgs
    st.rerun()


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
def main() -> None:
    global _header_slot, _stepper_slot
    st.set_page_config(page_title="DJ SK · Curadoria", page_icon="🎧", layout="wide")
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    theme.inject_css()
    _init_session()
    _render_sidebar()
    _header_slot = st.empty()
    _stepper_slot = st.empty()
    _render_header()
    _render_stepper()
    _render_notices()
    if st.session_state.get("reset_open"):
        _reset_dialog()
    _render_references()
    st.divider()
    _render_triage()
    st.divider()
    ctx = _render_setlist()
    st.divider()
    _render_export_section(ctx)
    st.divider()
    _render_metrics()
    # A página pode ter mudado neste run (busca, decisões): redesenha topo e etapas.
    _render_header()
    _render_stepper()


main()
