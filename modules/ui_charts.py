"""Gráficos do painel (Altair). Cada função recebe dados simples e devolve um `alt.Chart`.

Cores iguais às de `modules/theme.py`: rosa = energia/painel, roxo = velocidade (BPM),
verde/azul-esverdeado/âmbar/vermelho = qualidade da passagem.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import altair as alt
import pandas as pd

from modules import theme

P = theme.PALETTE
QUALITY_DOMAIN = ["início", "suave", "boa", "atenção", "difícil"]
QUALITY_RANGE = [P["muted"], P["ok"], P["good"], P["warn"], P["bad"]]
_AXIS = {"labelColor": P["text_soft"], "titleColor": P["text_soft"], "gridColor": "rgba(255,255,255,.06)",
         "domainColor": "rgba(255,255,255,.12)", "tickColor": "rgba(255,255,255,.12)"}


def _x_positions(n: int) -> alt.X:
    return alt.X("pos:Q", title="posição no set",
                 scale=alt.Scale(domain=[0.6, max(1, n) + 0.4], nice=False),
                 axis=alt.Axis(tickMinStep=1, format="d", values=list(range(1, max(1, n) + 1)), **_AXIS))


def energy_map_chart(rows: Sequence[Mapping[str, Any]], target: Sequence[float] | None = None,
                     height: int = 300) -> alt.LayerChart:
    """"Mapa de energia do set": energia (área + linha, pontos coloridos pela passagem de entrada)
    e velocidade (BPM) numa segunda linha com eixo próprio à direita.

    rows: [{"pos", "faixa", "bpm", "tom", "energia" (0..100), "passagem" ("início"|"suave"|...)}]
    target: curva alvo (0..100) com o mesmo tamanho de rows, desenhada tracejada (opcional).
    """
    df = pd.DataFrame(list(rows))
    n = len(df)
    if target is not None and len(target) == n:
        df["alvo"] = [float(v) for v in target]
    x = _x_positions(n)
    base = alt.Chart(df).encode(x=x)
    tooltip = [alt.Tooltip("pos:Q", title="posição"), alt.Tooltip("faixa:N", title="música"),
               alt.Tooltip("bpm:Q", title="BPM", format=".1f"), alt.Tooltip("tom:N", title="tom"),
               alt.Tooltip("energia:Q", title="energia (%)", format=".0f"),
               alt.Tooltip("passagem:N", title="passagem de entrada")]
    y_energy = alt.Y("energia:Q", title="energia (%)", scale=alt.Scale(domain=[0, 100]),
                     axis=alt.Axis(titleColor=P["accent"], **{k: v for k, v in _AXIS.items() if k != "titleColor"}))
    area = base.mark_area(opacity=0.22, color=P["accent"], interpolate="monotone").encode(y=y_energy)
    line = base.mark_line(color=P["accent"], strokeWidth=3, interpolate="monotone").encode(y="energia:Q")
    dots = base.mark_circle(size=170, opacity=1, stroke=P["bg"], strokeWidth=2).encode(
        y="energia:Q",
        color=alt.Color("passagem:N", scale=alt.Scale(domain=QUALITY_DOMAIN, range=QUALITY_RANGE),
                        legend=alt.Legend(title="passagem de entrada", orient="bottom", labelColor=P["text_soft"],
                                          titleColor=P["text_soft"])),
        tooltip=tooltip)
    layers = [area, line]
    if "alvo" in df.columns:
        layers.append(base.mark_line(color="#FFFFFF", opacity=0.55, strokeWidth=2, strokeDash=[6, 4],
                                     interpolate="monotone").encode(
            y="alvo:Q", tooltip=[alt.Tooltip("alvo:Q", title="energia alvo (%)", format=".0f")]))
    layers.append(dots)
    energy = alt.layer(*layers)

    bpms = [float(b) for b in df["bpm"]] if n else [0.0]
    lo, hi = min(bpms), max(bpms)
    pad = max(2.0, (hi - lo) * 0.25)
    y_bpm = alt.Y("bpm:Q", title="velocidade (BPM)", scale=alt.Scale(domain=[lo - pad, hi + pad], zero=False),
                  axis=alt.Axis(orient="right", grid=False, titleColor=P["accent_2"],
                                **{k: v for k, v in _AXIS.items() if k not in ("titleColor", "gridColor")}))
    bpm = base.mark_line(color=P["accent_2"], strokeWidth=2, point=alt.OverlayMarkDef(color=P["accent_2"], size=40),
                         interpolate="monotone").encode(y=y_bpm, tooltip=tooltip)
    return (alt.layer(energy, bpm).resolve_scale(y="independent")
            .properties(height=height).configure_view(strokeWidth=0))


def manual_vs_panel_chart(panel_s: float | None, manual_lo_s: float, manual_hi_s: float,
                          height: int = 150) -> alt.LayerChart:
    """Barras horizontais "segundos por música": processo manual (faixa 60–90 s) x painel."""
    rows = [{"como": "Processo manual", "inicio": float(manual_lo_s), "fim": float(manual_hi_s),
             "rotulo": f"{manual_lo_s:.0f}–{manual_hi_s:.0f} s", "cor": "manual"}]
    if panel_s:
        rows.append({"como": "Com o painel", "inicio": 0.0, "fim": float(panel_s),
                     "rotulo": f"{panel_s:.0f} s", "cor": "painel"})
    df = pd.DataFrame(rows)
    top = max(float(manual_hi_s), float(panel_s or 0)) * 1.18
    y = alt.Y("como:N", title=None, sort=["Processo manual", "Com o painel"],
              axis=alt.Axis(labelColor=P["text"], labelFontSize=13, domain=False, ticks=False))
    color = alt.Color("cor:N", scale=alt.Scale(domain=["manual", "painel"], range=[P["muted"], P["accent"]]),
                      legend=None)
    manual_base = alt.Chart(df[df["cor"] == "manual"]).encode(y=y)
    ghost = manual_base.mark_bar(height=26, cornerRadius=6, opacity=0.25, color=P["muted"]).encode(
        x=alt.X("zero:Q", title="segundos por música", scale=alt.Scale(domain=[0, top]),
                axis=alt.Axis(**_AXIS)), x2="fim:Q").transform_calculate(zero="0")
    bars = alt.Chart(df).mark_bar(height=26, cornerRadius=6).encode(
        y=y, x=alt.X("inicio:Q", scale=alt.Scale(domain=[0, top])), x2="fim:Q", color=color,
        tooltip=[alt.Tooltip("como:N", title=""), alt.Tooltip("rotulo:N", title="tempo por música")])
    text = alt.Chart(df).mark_text(align="left", dx=8, fontSize=13, fontWeight="bold", color=P["text"]).encode(
        y=y, x="fim:Q", text="rotulo:N")
    return alt.layer(ghost, bars, text).properties(height=height).configure_view(strokeWidth=0)


def funnel_chart(funnel: Sequence[tuple[str, int]], height: int = 220) -> alt.LayerChart:
    """Funil da curadoria (barras horizontais na ordem dada, com valor e % da etapa anterior)."""
    rows = []
    prev = None
    for i, (label, value) in enumerate(funnel):
        v = max(0, int(value or 0))
        share = f" · {round(100 * v / prev)}% da etapa anterior" if prev else ""
        rows.append({"etapa": str(label), "n": v, "ordem": i, "texto": f"{v}{share}" if i else str(v)})
        prev = v if v else None
    df = pd.DataFrame(rows)
    order = [r["etapa"] for r in rows]
    top = max([r["n"] for r in rows] + [1]) * 1.55
    y = alt.Y("etapa:N", title=None, sort=order,
              axis=alt.Axis(labelColor=P["text"], labelFontSize=13, domain=False, ticks=False))
    bars = alt.Chart(df).mark_bar(height=24, cornerRadius=6).encode(
        y=y, x=alt.X("n:Q", title=None, scale=alt.Scale(domain=[0, top]), axis=None),
        color=alt.Color("ordem:O", scale=alt.Scale(range=[P["accent_2"], "#9A55F2", "#C04BD0", "#E244A6", P["accent"]]),
                        legend=None),
        tooltip=[alt.Tooltip("etapa:N", title="etapa"), alt.Tooltip("n:Q", title="músicas")])
    text = alt.Chart(df).mark_text(align="left", dx=8, fontSize=12, color=P["text_soft"]).encode(
        y=y, x="n:Q", text="texto:N")
    return alt.layer(bars, text).properties(height=height).configure_view(strokeWidth=0)


def trend_chart(trend: Sequence[tuple[int, float]], manual_lo_s: float, manual_hi_s: float,
                height: int = 230) -> alt.LayerChart:
    """Evolução das sessões: segundos por música aprovada, com a faixa do processo manual ao fundo."""
    df = pd.DataFrame([{"sessao": int(n), "s": float(s)} for n, s in trend])
    n_max = int(df["sessao"].max()) if len(df) else 1
    top = max([float(manual_hi_s)] + [float(s) for s in df["s"]] if len(df) else [float(manual_hi_s)]) * 1.15
    band = alt.Chart(pd.DataFrame([{"lo": manual_lo_s, "hi": manual_hi_s}])).mark_rect(
        color=P["muted"], opacity=0.18).encode(y=alt.Y("lo:Q"), y2="hi:Q")
    band_text = alt.Chart(pd.DataFrame([{"y": manual_hi_s, "t": "processo manual"}])).mark_text(
        align="left", baseline="bottom", dx=4, dy=-3, fontSize=11, color=P["muted"]).encode(
        y="y:Q", text="t:N", x=alt.value(0))
    x = alt.X("sessao:Q", title="sessão", scale=alt.Scale(domain=[0.8, n_max + 0.2], nice=False),
              axis=alt.Axis(tickMinStep=1, format="d", **_AXIS))
    y = alt.Y("s:Q", title="segundos por música", scale=alt.Scale(domain=[0, top]), axis=alt.Axis(**_AXIS))
    line = alt.Chart(df).mark_line(color=P["accent"], strokeWidth=3,
                                   point=alt.OverlayMarkDef(color=P["accent"], size=70)).encode(
        x=x, y=y, tooltip=[alt.Tooltip("sessao:Q", title="sessão"),
                           alt.Tooltip("s:Q", title="s por música", format=".0f")])
    return alt.layer(band, band_text, line).properties(height=height).configure_view(strokeWidth=0)
