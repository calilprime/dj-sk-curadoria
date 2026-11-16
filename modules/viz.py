"""Desenhos SVG dos cards da triagem: forma de onda e curva de energia.

Funções puras (sem Streamlit): recebem listas de números e devolvem o SVG como texto,
em uma linha só, pronto para `st.markdown(svg, unsafe_allow_html=True)`.

As cores funcionam no tema claro e no escuro: o traço principal usa a cor de destaque
do tema do painel e os elementos neutros (marcadores, sombreado, linha de base) usam
`currentColor`, que segue a cor do texto do tema.
"""
from __future__ import annotations

import html
import math
from collections.abc import Sequence

ACCENT = "#FF3D7F"          # mesma cor primária do .streamlit/config.toml
NEUTRAL = "currentColor"    # segue a cor do texto (clara no tema escuro, escura no claro)

_MAX_POINTS_PER_PX = 0.75   # mais pontos que isso não aparece na tela e só pesa no HTML


def _clean(values: Sequence[float] | None) -> list[float]:
    """Números finitos e não negativos (None/NaN/negativos viram 0)."""
    out: list[float] = []
    for v in values or []:
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = 0.0
        out.append(x if math.isfinite(x) and x > 0 else 0.0)
    return out


def _downsample(values: list[float], n: int, mode: str = "max") -> list[float]:
    """Reduz para `n` pontos por blocos (pico para a forma de onda, média para a energia)."""
    if n <= 0 or len(values) <= n:
        return list(values)
    out: list[float] = []
    step = len(values) / n
    for i in range(n):
        a = int(i * step)
        b = max(a + 1, int((i + 1) * step))
        chunk = values[a:b]
        out.append(max(chunk) if mode == "max" else sum(chunk) / len(chunk))
    return out


def _scale(values: list[float]) -> list[float]:
    """Mantém 0..1; se algum valor passar de 1, divide tudo pelo maior."""
    top = max(values, default=0.0)
    if top > 1.0:
        return [v / top for v in values]
    return values


def _num(x: float) -> str:
    """Número curto para o SVG (1 casa, sem "-0.0")."""
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def _clamp01(x: float) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, x))


def _open_svg(width: int, height: int, label: str) -> str:
    label = html.escape(label, quote=True)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="100%" height="{height}" preserveAspectRatio="none" role="img" '
            f'aria-label="{label}" style="display:block;overflow:visible">'
            f'<title>{label}</title>')


def waveform_svg(waveform: Sequence[float] | None, width: int = 600, height: int = 64,
                 markers: Sequence[tuple[float, str]] | None = None,
                 highlight: Sequence[tuple[float, float]] | None = None) -> str:
    """Forma de onda espelhada (envelope de pico 0..1) como SVG.

    markers: [(posição 0..1, rótulo)] desenhados como linhas verticais (ex.: Início, Meio, Final).
    highlight: [(início 0..1, fim 0..1)] trechos sombreados (ex.: intro e saída).
    Sem dados, devolve só a linha central tracejada.
    """
    width = max(10, int(width))
    height = max(8, int(height))
    mid = height / 2.0
    parts = [_open_svg(width, height, "Forma de onda")]

    for a, b in highlight or []:
        x0, x1 = sorted((_clamp01(a), _clamp01(b)))
        if x1 - x0 > 0:
            parts.append(f'<rect x="{_num(x0 * width)}" y="0" width="{_num((x1 - x0) * width)}" '
                         f'height="{height}" fill="{NEUTRAL}" fill-opacity="0.10"/>')

    values = _scale(_clean(waveform))
    if values:
        values = _downsample(values, max(2, int(width * _MAX_POINTS_PER_PX)), mode="max")
    if len(values) == 1:
        values = values * 2
    if values:
        n = len(values)
        amp = mid * 0.95
        xs = [i * width / (n - 1) for i in range(n)]
        top = " ".join(f"{_num(x)},{_num(mid - v * amp)}" for x, v in zip(xs, values))
        bottom = " ".join(f"{_num(x)},{_num(mid + v * amp)}"
                          for x, v in zip(reversed(xs), reversed(values)))
        parts.append(f'<path d="M{top} L{bottom} Z" fill="{ACCENT}" fill-opacity="0.85"/>')
    else:
        parts.append(f'<line x1="0" y1="{_num(mid)}" x2="{width}" y2="{_num(mid)}" '
                     f'stroke="{NEUTRAL}" stroke-opacity="0.35" stroke-dasharray="4 4" '
                     f'vector-effect="non-scaling-stroke"/>')

    for pos, label in markers or []:
        x = _num(_clamp01(pos) * width)
        parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{height}" stroke="{NEUTRAL}" '
                     f'stroke-opacity="0.8" stroke-width="2" vector-effect="non-scaling-stroke">'
                     f'<title>{html.escape(str(label), quote=True)}</title></line>')
    parts.append("</svg>")
    return "".join(parts)


def sparkline_svg(values: Sequence[float] | None, width: int = 160, height: int = 28) -> str:
    """Curva de energia pequena (valores 0..1) com área preenchida, como SVG."""
    width = max(10, int(width))
    height = max(6, int(height))
    pad = 1.5
    parts = [_open_svg(width, height, "Curva de energia")]
    base_y = _num(height - pad)
    parts.append(f'<line x1="0" y1="{base_y}" x2="{width}" y2="{base_y}" stroke="{NEUTRAL}" '
                 f'stroke-opacity="0.25" vector-effect="non-scaling-stroke"/>')
    vals = _scale(_clean(values))
    if vals:
        vals = _downsample(vals, max(2, int(width * _MAX_POINTS_PER_PX)), mode="mean")
    if len(vals) == 1:
        vals = vals * 2
    if vals:
        n = len(vals)
        usable = height - 2 * pad
        pts = [(i * width / (n - 1), height - pad - v * usable) for i, v in enumerate(vals)]
        line = " ".join(f"{_num(x)},{_num(y)}" for x, y in pts)
        parts.append(f'<path d="M0,{base_y} L{line} L{width},{base_y} Z" fill="{ACCENT}" '
                     f'fill-opacity="0.18"/>')
        parts.append(f'<polyline points="{line}" fill="none" stroke="{ACCENT}" stroke-width="1.5" '
                     f'stroke-linejoin="round" vector-effect="non-scaling-stroke"/>')
    parts.append("</svg>")
    return "".join(parts)


_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline_text(values: Sequence[float] | None, n: int = 12) -> str:
    """Versão em texto da curva de energia (ex.: "▁▃▆█▇▅"), para listas compactas."""
    vals = _scale(_clean(values))
    if not vals:
        return ""
    vals = _downsample(vals, max(1, int(n)), mode="mean")
    last = len(_BLOCKS) - 1
    return "".join(_BLOCKS[min(last, int(round(v * last)))] for v in vals)
