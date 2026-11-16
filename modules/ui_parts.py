"""Peças de interface do painel (HTML puro, sem Streamlit) e contas pequenas da tela.

Tudo aqui devolve texto HTML em uma linha só (funciona em `st.html`) ou números simples,
e pode ser testado sem abrir o painel. Todo texto que vem de fora (títulos do YouTube,
nomes do Spotify) passa por `theme.esc`. As classes CSS ficam em `modules/theme.py`.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from modules import theme, util

esc = theme.esc

# ---------------------------------------------------------------------------
# Atalhos de teclado da triagem
# ---------------------------------------------------------------------------
# tecla -> (o que faz, chave interna). "C" e "R" são do próprio Streamlit e não podem ser usadas.
SHORTCUTS: dict[str, str] = {
    "approve": "A",
    "reject": "X",
    "inicio": "I",
    "meio": "M",
    "final": "F",
}
SHORTCUT_HELP: list[tuple[str, str]] = [
    ("A", "aprovar"), ("X", "reprovar"), ("I", "tocar o início"), ("M", "tocar o meio"), ("F", "tocar o final"),
]


def kbd_legend_html(items: Sequence[tuple[str, str]] = SHORTCUT_HELP, note: str | None = None) -> str:
    """Legenda dos atalhos: [A] aprovar  [X] reprovar ..."""
    keys = "".join(f'<span class="djsk-kbd-item"><kbd class="djsk-kbd">{esc(k)}</kbd>{esc(t)}</span>'
                   for k, t in items)
    note_html = f'<span class="djsk-kbd-note">{esc(note)}</span>' if note else ""
    return theme._root(f'<div class="djsk-kbd-legend"><span class="djsk-kbd-title">⌨ Atalhos</span>'
                       f'{keys}{note_html}</div>')


# ---------------------------------------------------------------------------
# Etiquetas
# ---------------------------------------------------------------------------
def badges_html(badges: Sequence[str], *, compact: bool = False) -> str:
    """Linha de etiquetas (`theme.badge`/`theme.chip` já prontos)."""
    inner = "".join(str(b) for b in badges if b)
    if not inner:
        return ""
    cls = "djsk-badges djsk-badges--compact" if compact else "djsk-badges"
    return theme._root(f'<div class="{cls}">{inner}</div>')


def reason_tone(tone: Any) -> str:
    """Tom dos motivos de `setlist_insights` -> tom do tema ("neutral" vira cinza)."""
    t = str(tone or "").strip().lower()
    return t if t in theme.TONES else "muted"


def bpm_zone(bpm: Any) -> tuple[str, str] | None:
    """Zona de velocidade do funk: (texto, tom) ou None sem BPM.

    O mandelão se concentra perto de 130 BPM; as versões aceleradas, perto de 150.
    """
    x = theme._float(bpm)
    if x is None or x <= 0:
        return None
    if 125.0 <= x <= 135.5:
        return "zona 130 · mandelão", "accent"
    if 145.0 <= x <= 155.5:
        return "zona 150 · acelerado", "warn"
    return "fora da zona do funk", "muted"


def bpm_text(bpm: Any) -> str:
    """128.0 -> "128"; 129.5 -> "129,5"."""
    return theme._fmt_bpm(bpm)


def pct(value: Any) -> int:
    """0..1 -> 0..100 (inteiro); inválido -> 0."""
    x = theme._float(value)
    return int(round(theme._clamp(x, 0.0, 1.0) * 100)) if x is not None else 0


def ref_chips_html(texts: Sequence[str]) -> str:
    """Referências escolhidas como etiquetas ("Referências: [MC X - Faixa] ...")."""
    items = [t for t in (str(x or "").strip() for x in texts) if t]
    if not items:
        chips = theme.badge("nenhuma ainda", "muted")
    else:
        chips = "".join(theme.badge(t, "accent", "🎵") for t in items)
    return theme._root(f'<div class="djsk-badges"><span class="djsk-badges-title">Referências:</span>{chips}</div>')


# ---------------------------------------------------------------------------
# Integrações (barra lateral)
# ---------------------------------------------------------------------------
def integrations_html(items: Sequence[Mapping[str, Any]]) -> str:
    """Lista de integrações: [{"name", "purpose", "ok": bool, "detail": str | None}]."""
    rows = []
    for it in items:
        ok = bool(it.get("ok"))
        status = theme.badge("ligado" if ok else "desligado", "ok" if ok else "muted")
        detail = it.get("detail")
        detail_html = f'<div class="djsk-int-detail">{esc(detail)}</div>' if detail else ""
        rows.append(f'<div class="djsk-int djsk-tone-{"ok" if ok else "muted"}"><span class="djsk-dot"></span>'
                    f'<div class="djsk-int-text"><div class="djsk-int-name">{esc(it.get("name"))}</div>'
                    f'<div class="djsk-int-purpose">{esc(it.get("purpose"))}</div>{detail_html}</div>'
                    f'{status}</div>')
    return theme._root(f'<div class="djsk-int-list">{"".join(rows)}</div>')


# ---------------------------------------------------------------------------
# Passagens do set
# ---------------------------------------------------------------------------
def passage_html(n_from: int, n_to: int, width: int, name_a: str, name_b: str, headline: str,
                 label: str, reasons: Sequence[Mapping[str, Any]], explanation: str | None = None) -> str:
    """Cabeçalho do cartão de uma passagem: "01 → 02", frase-resumo, nomes e motivos coloridos."""
    tone = theme.tone_for_label(label)
    lab = (str(label or "").strip() or "passagem")
    num = f"{int(n_from):0{width}d} → {int(n_to):0{width}d}"
    reason_badges = "".join(
        theme.badge(r.get("text", ""), reason_tone(r.get("tone")), r.get("icon")) for r in reasons or [])
    expl = (f'<div class="djsk-pass-expl">💬 {esc(explanation)}</div>' if explanation else "")
    return theme._root(
        f'<div class="djsk-pass djsk-tone-{tone}">'
        f'<div class="djsk-pass-top"><span class="djsk-pass-num">{esc(num)}</span>'
        f'{theme.badge(lab.capitalize(), tone)}<span class="djsk-pass-head">{esc(headline)}</span></div>'
        f'<div class="djsk-pass-names"><span>{esc(name_a)}</span><span class="djsk-pass-arrow">➜</span>'
        f'<span>{esc(name_b)}</span></div>'
        f'<div class="djsk-badges djsk-badges--compact">{reason_badges}</div>{expl}</div>')


def verdict_html(title: str, tone: str, lines: Sequence[str] = (), badges: Sequence[str] = ()) -> str:
    """Veredito ao lado da nota do set: título colorido, frases curtas e etiquetas."""
    t = theme._tone(tone, default="muted")
    subs = "".join(f'<div class="djsk-verdict-sub">{esc(x)}</div>' for x in lines if x)
    chips = "".join(str(b) for b in badges if b)
    chips_html = f'<div class="djsk-badges">{chips}</div>' if chips else ""
    return theme._root(f'<div class="djsk-verdict djsk-tone-{t}"><div class="djsk-verdict-title">{esc(title)}</div>'
                       f'{subs}{chips_html}</div>')


def headline_short(headline: str) -> str:
    """"Passagem suave: mesma velocidade e energia parecida" -> "mesma velocidade e energia parecida"."""
    text = str(headline or "")
    return text.split(": ", 1)[1] if ": " in text else ""


def compare_text(cmp: Mapping[str, Any] | None) -> tuple[str, str]:
    """Frase (e tom) comparando a ordem do set com a ordem em que o DJ aprovou."""
    if not cmp:
        return "", "muted"
    pct_ = theme._float(cmp.get("improvement_pct")) or 0.0
    if pct_ >= 0.5:
        return (f"{_int_pct(pct_)}% mais fácil de mixar que a ordem em que você aprovou", "ok")
    if pct_ <= -0.5:
        return (f"{_int_pct(-pct_)}% mais difícil que a ordem em que você aprovou "
                "(você mexeu na mão; clique em Gerar ordem para otimizar)", "warn")
    return "Igual à ordem em que você aprovou: ela já estava boa", "info"


def _int_pct(x: float) -> str:
    return str(int(round(x)))


# ---------------------------------------------------------------------------
# Arco de energia do set (curva alvo)
# ---------------------------------------------------------------------------
ARCS: dict[str, dict[str, str]] = {
    "livre": {"label": "Sem alvo", "icon": "➖", "description": "Só mostra a energia real do set."},
    "pico": {"label": "Aquecer → pico → descer", "icon": "⛰️",
             "description": "Começa calmo, chega no auge no meio e respira no final."},
    "subida": {"label": "Subida contínua", "icon": "📈",
               "description": "A energia sobe do começo ao fim, sem respiro."},
    "ondas": {"label": "Ondas", "icon": "🌊",
              "description": "Sobe e desce algumas vezes, para a pista respirar entre os picos."},
}
DEFAULT_ARC = "livre"


def arc_shape(kind: str, n: int) -> list[float] | None:
    """Curva alvo (0..1) para `n` faixas; None para "livre", desconhecido ou n < 2."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    if kind not in ARCS or kind == "livre" or n < 2:
        return None
    out = []
    for i in range(n):
        x = i / (n - 1)
        if kind == "pico":
            v = math.sin(math.pi * x) ** 0.8
        elif kind == "subida":
            v = x
        else:  # ondas: três subidas
            v = 0.5 - 0.5 * math.cos(2 * math.pi * 1.5 * x)
        out.append(max(0.0, min(1.0, v)))
    return out


def arc_target(energies: Sequence[Any], kind: str) -> list[float] | None:
    """Curva alvo na mesma escala da energia do set (entre a menor e a maior energia)."""
    vals = [theme._float(e) for e in energies]
    if any(v is None for v in vals):
        return None
    shape = arc_shape(kind, len(vals))
    if shape is None:
        return None
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span < 0.05:                      # set quase plano: usa uma faixa mínima em volta da média
        mid = (hi + lo) / 2
        lo, span = max(0.0, mid - 0.1), 0.2
    return [lo + s * span for s in shape]


def arc_adherence(energies: Sequence[Any], kind: str) -> int | None:
    """Quanto a energia real segue o arco escolhido (0..100); None sem alvo."""
    vals = [theme._float(e) for e in energies]
    if any(v is None for v in vals):
        return None
    shape = arc_shape(kind, len(vals))
    if shape is None:
        return None
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span < 1e-9:
        norm = [0.5] * len(vals)
    else:
        norm = [(v - lo) / span for v in vals]
    err = sum(abs(a - b) for a, b in zip(norm, shape)) / len(vals)
    return int(round(max(0.0, 1.0 - err) * 100))


# ---------------------------------------------------------------------------
# Spotify
# ---------------------------------------------------------------------------
def spotify_result_html(d: Mapping[str, Any]) -> str:
    """Resultado da busca do Spotify: título, artistas, álbum e duração."""
    title = esc(d.get("title") or "Sem título")
    artists = esc(", ".join(str(a) for a in d.get("artists") or []) or "Artista desconhecido")
    extra = []
    if d.get("album"):
        extra.append(esc(d["album"]))
    dur = theme._float(d.get("duration_s"))
    if dur:
        extra.append(esc(util.format_duration(dur)))
    sub = f'<div class="djsk-sp-sub">{" · ".join(extra)}</div>' if extra else ""
    return theme._root(f'<div class="djsk-sp"><div class="djsk-sp-title">{title}</div>'
                       f'<div class="djsk-sp-artist">{artists}</div>{sub}</div>')


def spotify_embed_url(kind: str, spotify_id: str | None) -> str | None:
    """URL do player embutido do Spotify (playlist/faixa); None para id inválido."""
    sid = str(spotify_id or "").strip()
    if kind not in ("playlist", "track") or not sid or not sid.replace("_", "").isalnum():
        return None
    return f"https://open.spotify.com/embed/{kind}/{sid}?utm_source=generator&theme=0"


# ---------------------------------------------------------------------------
# Números
# ---------------------------------------------------------------------------
def fmt_minutes(minutes: Any) -> str:
    """12.4 -> "12 min"; 75 -> "1h15"; inválido -> "–"."""
    x = theme._float(minutes)
    if x is None:
        return "–"
    m = int(round(max(0.0, x)))
    h, rest = divmod(m, 60)
    return f"{h}h{rest:02d}" if h else f"{m} min"


def fmt_range_minutes(lo: Any, hi: Any) -> str:
    """(10.2, 17.8) -> "10–18 min"."""
    a, b = theme._float(lo), theme._float(hi)
    if a is None or b is None:
        return "–"
    ra, rb = int(round(max(0.0, a))), int(round(max(0.0, b)))
    return f"{ra} min" if ra == rb else f"{ra}–{rb} min"


def set_duration_text(seconds: Any) -> str:
    """Duração do set: 2890 -> "48 min"; 4000 -> "1h07"."""
    x = theme._float(seconds)
    if x is None or x <= 0:
        return "–"
    return fmt_minutes(x / 60.0)
