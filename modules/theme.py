"""Identidade visual do painel: tema escuro de "app de DJ" (rosa #FF3D7F + roxo #7C5CFF).

Duas camadas:

- Funções PURAS, que só montam HTML (texto) e podem ser testadas sem Streamlit:
  `esc`, `badge`, `kpi_card`, `kpi_grid_html`, `hero_html`, `stepper_html`,
  `section_header_html`, `score_ring_svg`, `energy_bar`, `timeline_html`, `legend_html`.
- Funções que DESENHAM na página (chamam `st.html`):
  `inject_css`, `kpi_grid`, `hero`, `stepper`, `section_header`, `timeline`, `legend`.

Regras:
- todo texto que vem de fora (títulos do YouTube, artistas, rótulos) passa por `esc()`;
- o HTML sai em uma linha só, sem recuo: funciona em `st.html` e também em
  `st.markdown(..., unsafe_allow_html=True)` (o Markdown não vira bloco de código);
- as classes CSS começam com "djsk-" para não colidir com as do Streamlit.
"""
from __future__ import annotations

import html
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from modules import util

# ---------------------------------------------------------------------------
# Cores
# ---------------------------------------------------------------------------
PALETTE: dict[str, str] = {
    "bg": "#0B0B12",          # fundo da página
    "bg_soft": "#11111A",     # barra lateral
    "surface": "#161622",     # cartões
    "surface_2": "#1E1E2D",   # cartão por cima de cartão, chips
    "border": "#2A2A3D",
    "text": "#F3F3F8",
    "text_soft": "#B9B9CC",
    "muted": "#8A8AA3",
    "accent": "#FF3D7F",      # rosa (cor principal)
    "accent_2": "#7C5CFF",    # roxo (segunda cor)
    "ok": "#22C55E",          # verde: passagem suave
    "good": "#14B8A6",        # verde-azulado: passagem boa
    "warn": "#F59E0B",        # âmbar: atenção
    "bad": "#EF4444",         # vermelho: difícil
    "info": "#38BDF8",        # azul: informação
}

# Tom -> cor. Tons desconhecidos caem em "muted" (ou "default" nos cartões de número).
TONES: dict[str, str] = {
    "ok": PALETTE["ok"],
    "good": PALETTE["good"],
    "warn": PALETTE["warn"],
    "bad": PALETTE["bad"],
    "info": PALETTE["info"],
    "muted": PALETTE["muted"],
    "accent": PALETTE["accent"],
}

# Qualidade da passagem (Transition.label) -> tom.
LABEL_TONE: dict[str, str] = {"suave": "ok", "boa": "good", "atenção": "warn", "difícil": "bad"}

# Explicação curta de cada qualidade, para a legenda.
LABEL_HINT: dict[str, str] = {
    "suave": "passa fácil, quase sem mexer",
    "boa": "funciona bem com um ajuste leve",
    "atenção": "dá certo, mas pede cuidado",
    "difícil": "melhor trocar a ordem ou preparar a virada",
}

# Prefixos de `key` de st.container que o CSS reconhece (o Streamlit vira a key em
# classe "st-key-<key>"): st.container(border=True, key=f"{CARD_KEY_PREFIX}{vid}").
CARD_KEY_PREFIX = "djsk_card_"
PLAYING_KEY_PREFIX = "djsk_playing_"
FOCUS_KEY_PREFIX = "djsk_focus_"     # card em foco (recebe os atalhos de teclado), sem pulsar

_STATUSES = ("done", "active", "todo")
_SLUG_RE = re.compile(r"[^a-z0-9_-]+")

# ---------------------------------------------------------------------------
# CSS global
# ---------------------------------------------------------------------------
_FONTS_URL = ("https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
              "&family=Space+Grotesk:wght@500;600;700&display=swap")

_CSS = """
@import url('__FONTS_URL__');
:root{
--djsk-bg:#0B0B12;--djsk-bg-soft:#11111A;--djsk-surface:#161622;--djsk-surface-2:#1E1E2D;
--djsk-border:rgba(255,255,255,.08);--djsk-border-strong:rgba(255,255,255,.16);
--djsk-text:#F3F3F8;--djsk-text-soft:#B9B9CC;--djsk-muted:#8A8AA3;
--djsk-accent:#FF3D7F;--djsk-accent-2:#7C5CFF;
--djsk-ok:#22C55E;--djsk-good:#14B8A6;--djsk-warn:#F59E0B;--djsk-bad:#EF4444;--djsk-info:#38BDF8;
--djsk-grad:linear-gradient(135deg,#FF3D7F 0%,#B14BFF 55%,#7C5CFF 100%);
--djsk-radius:16px;--djsk-radius-sm:10px;
--djsk-font-head:'Space Grotesk','Segoe UI',system-ui,-apple-system,Roboto,sans-serif;
--djsk-font-body:'Inter','Segoe UI',system-ui,-apple-system,Roboto,sans-serif;
--djsk-shadow:0 10px 30px -18px rgba(0,0,0,.8);
--djsk-top:3.75rem;
}
/* ---------- tons ---------- */
.djsk-tone-ok{--djsk-tone:var(--djsk-ok)}
.djsk-tone-good{--djsk-tone:var(--djsk-good)}
.djsk-tone-warn{--djsk-tone:var(--djsk-warn)}
.djsk-tone-bad{--djsk-tone:var(--djsk-bad)}
.djsk-tone-info{--djsk-tone:var(--djsk-info)}
.djsk-tone-muted{--djsk-tone:var(--djsk-muted)}
.djsk-tone-accent,.djsk-tone-default{--djsk-tone:var(--djsk-accent)}
/* ---------- página do Streamlit ---------- */
.stApp{background:
 radial-gradient(1100px 520px at 0% -8%,rgba(255,61,127,.10),transparent 60%),
 radial-gradient(900px 480px at 100% 0%,rgba(124,92,255,.11),transparent 62%),
 var(--djsk-bg);}
[data-testid="stHeader"]{background:rgba(11,11,18,.72);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);}
[data-testid="stAppViewContainer"],[data-testid="stMain"]{scroll-behavior:smooth;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#14141F 0%,#0E0E16 100%);border-right:1px solid var(--djsk-border);}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{font-family:var(--djsk-font-head);letter-spacing:-.01em;}
h1,h2,h3{letter-spacing:-.02em;}
hr{border-color:var(--djsk-border)!important;}
/* números (st.metric) como cartões */
[data-testid="stMetric"]{background:linear-gradient(180deg,var(--djsk-surface-2),var(--djsk-surface));border:1px solid var(--djsk-border);border-radius:var(--djsk-radius);padding:14px 16px;box-shadow:var(--djsk-shadow);animation:djsk-fade-up .35s ease-out both;}
[data-testid="stMetricLabel"]{color:var(--djsk-text-soft);}
[data-testid="stMetricValue"]{font-family:var(--djsk-font-head);font-weight:700;letter-spacing:-.02em;}
/* botão principal com gradiente */
[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{background-image:var(--djsk-grad);border:0;color:#fff;box-shadow:0 8px 22px -10px rgba(255,61,127,.75);transition:transform .15s ease,filter .15s ease,box-shadow .15s ease;}
[data-testid="stBaseButton-primary"]:hover:not(:disabled),[data-testid="stBaseButton-primaryFormSubmit"]:hover:not(:disabled){filter:brightness(1.08);transform:translateY(-1px);box-shadow:0 12px 26px -10px rgba(255,61,127,.9);color:#fff;}
[data-testid="stBaseButton-primary"]:disabled,[data-testid="stBaseButton-primaryFormSubmit"]:disabled{background-image:none;box-shadow:none;}
[data-testid="stBaseButton-secondary"]{transition:border-color .15s ease,color .15s ease;}
/* containers com key "djsk_card_..." / "djsk_playing_..." viram cartões (borda comum: cor/raio do config.toml) */
[class*="st-key-djsk_card"],[class*="st-key-djsk_playing"],[class*="st-key-djsk_focus"]{background:linear-gradient(180deg,rgba(30,30,45,.92),rgba(22,22,34,.92));border:1px solid var(--djsk-border);border-radius:var(--djsk-radius);padding:14px 16px;box-shadow:var(--djsk-shadow);animation:djsk-fade-up .35s ease-out both;}
[class*="st-key-djsk_playing"]{border-color:rgba(255,61,127,.65);animation:djsk-fade-up .35s ease-out both,djsk-pulse 2s ease-out .4s infinite;}
[class*="st-key-djsk_focus"]{border-color:rgba(124,92,255,.75);box-shadow:0 0 0 1px rgba(124,92,255,.35),var(--djsk-shadow);}
/* expanders, abas, alertas */
[data-testid="stExpander"] details{border-radius:var(--djsk-radius-sm);border-color:var(--djsk-border);background:rgba(22,22,34,.6);}
[data-testid="stExpander"] summary:hover{color:var(--djsk-accent);}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{color:var(--djsk-accent);}
[data-testid="stAlert"]{border-radius:var(--djsk-radius-sm);}
[data-testid="stDataFrame"]{border-radius:var(--djsk-radius-sm);overflow:hidden;}
/* barra de etapas fixa no topo */
[data-testid="stElementContainer"]:has(.djsk-stepper),.element-container:has(.djsk-stepper){position:sticky;top:var(--djsk-top);z-index:90;}
/* ---------- animações ---------- */
@keyframes djsk-fade-up{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
@keyframes djsk-pulse{0%{box-shadow:0 0 0 0 rgba(255,61,127,.55)}70%{box-shadow:0 0 0 12px rgba(255,61,127,0)}100%{box-shadow:0 0 0 0 rgba(255,61,127,0)}}
@keyframes djsk-eq{0%,100%{transform:scaleY(.35)}50%{transform:scaleY(1)}}
@media (prefers-reduced-motion:reduce){
 .djsk-root *,[class*="st-key-djsk_"],[data-testid="stMetric"]{animation:none!important;transition:none!important;}
}
/* ---------- base dos componentes ---------- */
.djsk-root{font-family:var(--djsk-font-body);color:var(--djsk-text);line-height:1.4;}
.djsk-root *{box-sizing:border-box;}
.djsk-playing{animation:djsk-pulse 2s ease-out infinite!important;border-color:rgba(255,61,127,.75)!important;}
/* badge */
.djsk-badge{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font:600 12px/1.5 var(--djsk-font-body);
 color:var(--djsk-tone,var(--djsk-muted));background:color-mix(in srgb,var(--djsk-tone,var(--djsk-muted)) 15%,transparent);
 border:1px solid color-mix(in srgb,var(--djsk-tone,var(--djsk-muted)) 35%,transparent);white-space:nowrap;vertical-align:middle;}
.djsk-badge-icon{font-size:12px;line-height:1;}
.djsk-badge.djsk-playing{border-radius:999px;}
/* chips (BPM, tom, duração) */
.djsk-chip{display:inline-flex;align-items:center;gap:4px;padding:2px 8px;border-radius:8px;font:600 11px/1.6 var(--djsk-font-body);
 color:var(--djsk-text-soft);background:var(--djsk-surface-2);border:1px solid var(--djsk-border);white-space:nowrap;}
.djsk-chip b{color:var(--djsk-text);font-weight:700;}
.djsk-chip--key{color:#CFC4FF;background:rgba(124,92,255,.16);border-color:rgba(124,92,255,.35);}
/* cartões de número */
.djsk-kpi-grid{display:grid;gap:12px;margin:2px 0 6px;
 grid-template-columns:repeat(auto-fit,minmax(max(170px,calc((100% - (var(--djsk-cols,4) - 1) * 12px) / var(--djsk-cols,4))),1fr));}
.djsk-kpi{position:relative;overflow:hidden;padding:14px 16px 13px;border-radius:var(--djsk-radius);
 background:linear-gradient(180deg,var(--djsk-surface-2),var(--djsk-surface));border:1px solid var(--djsk-border);
 box-shadow:var(--djsk-shadow);animation:djsk-fade-up .35s ease-out both;min-width:0;}
.djsk-kpi::before{content:"";position:absolute;inset:0 0 auto 0;height:3px;background:var(--djsk-tone,var(--djsk-accent));opacity:.9;}
.djsk-kpi.djsk-tone-default::before,.djsk-kpi.djsk-tone-accent::before{background:var(--djsk-grad);}
.djsk-kpi-top{display:flex;align-items:center;gap:8px;color:var(--djsk-text-soft);font:500 13px/1.3 var(--djsk-font-body);}
.djsk-kpi-icon{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:8px;font-size:14px;
 background:color-mix(in srgb,var(--djsk-tone,var(--djsk-accent)) 18%,transparent);}
.djsk-kpi-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.djsk-kpi-value{margin-top:8px;font:700 28px/1.1 var(--djsk-font-head);letter-spacing:-.02em;color:var(--djsk-text);
 overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.djsk-kpi-sub{margin-top:6px;font:500 12px/1.35 var(--djsk-font-body);color:var(--djsk-muted);}
.djsk-kpi:not(.djsk-tone-default):not(.djsk-tone-muted) .djsk-kpi-sub{color:color-mix(in srgb,var(--djsk-tone) 70%,var(--djsk-text-soft));}
/* cabeçalho */
.djsk-hero{position:relative;overflow:hidden;display:flex;flex-wrap:wrap;align-items:stretch;justify-content:space-between;gap:20px;
 padding:26px 28px;border-radius:22px;border:1px solid var(--djsk-border-strong);
 background:radial-gradient(600px 260px at 0% 0%,rgba(255,61,127,.30),transparent 70%),
 radial-gradient(520px 260px at 100% 100%,rgba(124,92,255,.34),transparent 70%),
 linear-gradient(135deg,#1A1222 0%,#141427 100%);box-shadow:0 24px 60px -30px rgba(255,61,127,.45);}
.djsk-hero-eq{display:inline-flex;gap:3px;align-items:flex-end;height:14px;opacity:.9;}
.djsk-hero-eq i{display:block;width:4px;height:100%;border-radius:2px;background:var(--djsk-grad);transform-origin:bottom;animation:djsk-eq 1.2s ease-in-out infinite;}
.djsk-hero-eq i:nth-child(2){animation-delay:-.3s}.djsk-hero-eq i:nth-child(3){animation-delay:-.6s}
.djsk-hero-eq i:nth-child(4){animation-delay:-.9s}.djsk-hero-eq i:nth-child(5){animation-delay:-.45s}
.djsk-hero-main{flex:1 1 340px;min-width:0;}
.djsk-hero-kicker{display:flex;align-items:center;gap:10px;font:600 12px/1 var(--djsk-font-body);letter-spacing:.14em;text-transform:uppercase;color:#FF8FB5;}
.djsk-hero-title{margin:10px 0 6px;font:700 clamp(30px,4.2vw,46px)/1.05 var(--djsk-font-head);letter-spacing:-.035em;
 background:linear-gradient(90deg,#FFFFFF 0%,#FFC2D6 55%,#C9B8FF 100%);-webkit-background-clip:text;background-clip:text;color:transparent;}
.djsk-hero-sub{margin:0;max-width:62ch;font:500 15px/1.5 var(--djsk-font-body);color:var(--djsk-text-soft);}
.djsk-hero-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px;}
.djsk-budget{flex:0 1 330px;min-width:240px;align-self:center;padding:14px 16px;border-radius:var(--djsk-radius);
 background:rgba(11,11,18,.55);border:1px solid var(--djsk-border);backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);}
.djsk-budget-head{display:flex;justify-content:space-between;align-items:center;font:600 12px/1 var(--djsk-font-body);
 letter-spacing:.08em;text-transform:uppercase;color:var(--djsk-muted);}
.djsk-budget-pct{color:var(--djsk-text-soft);letter-spacing:0;}
.djsk-budget-value{margin:8px 0 10px;font:700 24px/1.1 var(--djsk-font-head);color:var(--djsk-text);letter-spacing:-.02em;}
.djsk-budget-value span{font-size:15px;font-weight:500;color:var(--djsk-muted);}
.djsk-meter{height:8px;border-radius:999px;background:rgba(255,255,255,.08);overflow:hidden;}
.djsk-meter>span{display:block;height:100%;min-width:2px;border-radius:999px;background:var(--djsk-grad);transition:width .4s ease;}
.djsk-budget.djsk-tone-warn .djsk-meter>span{background:var(--djsk-warn);}
.djsk-budget.djsk-tone-bad .djsk-meter>span{background:var(--djsk-bad);}
.djsk-budget.djsk-tone-muted .djsk-meter>span{background:var(--djsk-muted);}
.djsk-budget-foot{margin-top:9px;font:500 12px/1.45 var(--djsk-font-body);color:var(--djsk-text-soft);}
.djsk-budget-foot div+div{color:var(--djsk-muted);}
.djsk-budget-alert{margin-top:8px;padding:6px 9px;border-radius:8px;font:600 12px/1.35 var(--djsk-font-body);
 color:#FFD1D1;background:rgba(239,68,68,.16);border:1px solid rgba(239,68,68,.4);}
/* barra de etapas */
.djsk-stepper{display:flex;align-items:center;gap:6px;overflow-x:auto;scrollbar-width:none;padding:8px 10px;border-radius:999px;
 background:rgba(17,17,26,.94);border:1px solid var(--djsk-border);backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
 box-shadow:0 12px 30px -20px rgba(0,0,0,.9);}
.djsk-stepper::-webkit-scrollbar{display:none;}
.djsk-step{display:inline-flex;align-items:center;gap:8px;padding:5px 12px 5px 5px;border-radius:999px;white-space:nowrap;
 text-decoration:none!important;color:var(--djsk-muted)!important;font:600 13px/1 var(--djsk-font-body);transition:background .15s ease,color .15s ease;}
a.djsk-step:hover{background:rgba(255,255,255,.06);color:var(--djsk-text)!important;}
.djsk-step-dot{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;font:700 12px/1 var(--djsk-font-head);
 color:var(--djsk-muted);background:var(--djsk-surface-2);border:1px solid var(--djsk-border-strong);}
.djsk-step--done{color:var(--djsk-text-soft)!important;}
.djsk-step--done .djsk-step-dot{color:#06120A;background:var(--djsk-ok);border-color:transparent;}
.djsk-step--active{color:var(--djsk-text)!important;background:rgba(255,61,127,.12);}
.djsk-step--active .djsk-step-dot{color:#fff;background-image:var(--djsk-grad);border-color:transparent;animation:djsk-pulse 2s ease-out infinite;}
.djsk-step-sep{flex:1 0 14px;max-width:46px;height:2px;border-radius:2px;background:var(--djsk-border-strong);}
.djsk-step-sep.is-done{background:linear-gradient(90deg,var(--djsk-ok),color-mix(in srgb,var(--djsk-ok) 40%,transparent));}
/* título de seção */
.djsk-section{display:flex;align-items:center;gap:14px;margin:6px 0 2px;}
.djsk-anchor{display:block;position:relative;scroll-margin-top:calc(var(--djsk-top) + 70px);}
.djsk-section-num{flex:0 0 auto;display:grid;place-items:center;width:40px;height:40px;border-radius:12px;
 font:700 18px/1 var(--djsk-font-head);color:#fff;background-image:var(--djsk-grad);box-shadow:0 8px 20px -10px rgba(255,61,127,.9);}
.djsk-section-text{min-width:0;}
.djsk-section-title{font:700 24px/1.15 var(--djsk-font-head);letter-spacing:-.02em;color:var(--djsk-text);}
.djsk-section-sub{margin-top:3px;font:500 14px/1.4 var(--djsk-font-body);color:var(--djsk-muted);}
/* anel de nota */
.djsk-ring{display:inline-flex;flex-direction:column;align-items:center;gap:6px;margin:0;}
.djsk-rings{display:flex;flex-wrap:wrap;gap:18px 26px;align-items:flex-start;}
.djsk-ring figcaption{font:600 12px/1.3 var(--djsk-font-body);color:var(--djsk-text-soft);text-align:center;max-width:140px;}
/* barra de energia */
.djsk-ebar{position:relative;flex:1 1 auto;min-width:36px;height:6px;border-radius:999px;background:rgba(255,255,255,.09);overflow:hidden;}
.djsk-ebar>span{display:block;height:100%;border-radius:999px;background:var(--djsk-tone,var(--djsk-accent));}
.djsk-ebar.djsk-tone-accent>span{background:var(--djsk-grad);}
/* legenda */
.djsk-legend{display:flex;flex-wrap:wrap;align-items:center;gap:8px 16px;padding:10px 14px;border-radius:12px;
 background:rgba(22,22,34,.7);border:1px solid var(--djsk-border);font:500 13px/1.35 var(--djsk-font-body);color:var(--djsk-text-soft);}
.djsk-legend-title{font-weight:700;color:var(--djsk-text);}
.djsk-legend-item{display:inline-flex;align-items:center;gap:7px;}
.djsk-legend-item b{color:var(--djsk-tone);font-weight:700;}
.djsk-dot{display:inline-block;flex:0 0 auto;width:10px;height:10px;border-radius:50%;background:var(--djsk-tone,var(--djsk-muted));
 box-shadow:0 0 0 3px color-mix(in srgb,var(--djsk-tone,var(--djsk-muted)) 22%,transparent);}
/* linha do tempo do set */
.djsk-tl{display:flex;flex-direction:column;gap:12px;}
.djsk-empty{padding:22px;border-radius:var(--djsk-radius);border:1px dashed var(--djsk-border-strong);text-align:center;
 font:500 14px/1.45 var(--djsk-font-body);color:var(--djsk-muted);background:rgba(22,22,34,.45);}
.djsk-tl-summary{display:flex;flex-wrap:wrap;align-items:center;gap:8px;}
.djsk-tl-summary-title{font:700 14px/1 var(--djsk-font-head);color:var(--djsk-text);margin-right:4px;}
.djsk-tl-map{padding:12px 14px 10px;border-radius:14px;background:rgba(22,22,34,.7);border:1px solid var(--djsk-border);}
.djsk-tl-bars{display:flex;align-items:flex-end;height:46px;}
.djsk-tl-bar{flex:1 1 0;height:100%;display:flex;align-items:flex-end;justify-content:center;min-width:0;}
.djsk-tl-bar>span{display:block;width:62%;max-width:26px;min-width:2px;border-radius:4px 4px 2px 2px;background:var(--djsk-grad);opacity:.9;}
.djsk-tl-bar.djsk-playing>span{outline:2px solid #fff;}
.djsk-tl-bar.djsk-playing{animation:none!important;}
.djsk-tl-flow{display:flex;gap:2px;margin-top:6px;height:6px;}
.djsk-tl-flow>span{flex:1 1 0;border-radius:3px;background:var(--djsk-tone,var(--djsk-muted));opacity:.95;}
.djsk-tl-map-cap{display:flex;justify-content:space-between;margin-top:7px;font:500 11px/1 var(--djsk-font-body);color:var(--djsk-muted);}
.djsk-tl-track{display:flex;align-items:center;overflow-x:auto;overflow-y:hidden;padding:8px 4px 14px;scroll-snap-type:x proximity;
 scrollbar-width:thin;scrollbar-color:rgba(255,61,127,.55) rgba(255,255,255,.05);overscroll-behavior-x:contain;}
.djsk-tl-track::-webkit-scrollbar{height:8px;}
.djsk-tl-track::-webkit-scrollbar-track{background:rgba(255,255,255,.05);border-radius:999px;}
.djsk-tl-track::-webkit-scrollbar-thumb{background:var(--djsk-grad);border-radius:999px;}
.djsk-tl-card{flex:0 0 172px;width:172px;scroll-snap-align:start;border-radius:var(--djsk-radius);overflow:hidden;
 background:linear-gradient(180deg,var(--djsk-surface-2),var(--djsk-surface));border:1px solid var(--djsk-border);
 box-shadow:var(--djsk-shadow);animation:djsk-fade-up .35s ease-out both;transition:transform .15s ease,border-color .15s ease;}
.djsk-tl-card:hover{transform:translateY(-2px);border-color:var(--djsk-border-strong);}
.djsk-tl-cover{position:relative;height:96px;background:linear-gradient(135deg,#2A1530,#1B1838);}
.djsk-tl-cover img{display:block;width:100%;height:100%;object-fit:cover;}
.djsk-tl-cover::after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,rgba(0,0,0,0) 35%,rgba(10,10,16,.85) 100%);}
.djsk-tl-ph{display:grid;place-items:center;width:100%;height:100%;font:700 34px/1 var(--djsk-font-head);color:rgba(255,255,255,.22);
 background:radial-gradient(80px 60px at 30% 30%,rgba(255,61,127,.35),transparent 70%),radial-gradient(90px 70px at 80% 80%,rgba(124,92,255,.4),transparent 70%);}
.djsk-tl-pos{position:absolute;left:10px;bottom:6px;z-index:1;font:700 30px/1 var(--djsk-font-head);letter-spacing:-.04em;color:#fff;text-shadow:0 2px 12px rgba(0,0,0,.6);}
.djsk-tl-dur{position:absolute;right:8px;bottom:9px;z-index:1;padding:2px 6px;border-radius:6px;font:600 11px/1.4 var(--djsk-font-body);color:#fff;background:rgba(0,0,0,.55);}
.djsk-tl-now{position:absolute;left:8px;top:8px;z-index:1;}
.djsk-tl-body{padding:10px 12px 12px;display:flex;flex-direction:column;gap:7px;}
.djsk-tl-title{font:700 14px/1.25 var(--djsk-font-head);color:var(--djsk-text);overflow:hidden;min-height:2.5em;
 display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;word-break:break-word;}
.djsk-tl-artist{margin-top:-4px;font:500 12px/1.3 var(--djsk-font-body);color:var(--djsk-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.djsk-tl-chips{display:flex;flex-wrap:wrap;gap:5px;}
.djsk-tl-energy{display:flex;align-items:center;gap:7px;font:600 11px/1 var(--djsk-font-body);color:var(--djsk-muted);}
.djsk-tl-energy-val{min-width:30px;text-align:right;color:var(--djsk-text-soft);}
.djsk-tl-link{flex:0 0 118px;width:118px;display:flex;flex-direction:column;align-items:center;gap:6px;padding:0 4px;text-align:center;}
.djsk-tl-link-label{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:999px;font:700 11px/1.4 var(--djsk-font-body);
 color:var(--djsk-tone);background:color-mix(in srgb,var(--djsk-tone) 16%,transparent);border:1px solid color-mix(in srgb,var(--djsk-tone) 40%,transparent);white-space:nowrap;}
.djsk-tl-link-line{position:relative;display:block;width:100%;height:3px;border-radius:3px;
 background:linear-gradient(90deg,color-mix(in srgb,var(--djsk-tone) 25%,transparent),var(--djsk-tone));}
.djsk-tl-link-line::after{content:"";position:absolute;right:-2px;top:50%;width:9px;height:9px;border-top:3px solid var(--djsk-tone);
 border-right:3px solid var(--djsk-tone);transform:translateY(-50%) rotate(45deg);border-radius:2px;}
.djsk-tl-link-short{font:500 11px/1.3 var(--djsk-font-body);color:var(--djsk-text-soft);display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden;}
/* linha de etiquetas */
.djsk-badges{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:2px 0;}
.djsk-badges--compact{gap:5px;}
.djsk-badges-title{font:600 12px/1 var(--djsk-font-body);color:var(--djsk-muted);margin-right:2px;}
/* atalhos de teclado */
.djsk-kbd-legend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 14px;padding:8px 12px;border-radius:12px;
 background:rgba(124,92,255,.08);border:1px solid rgba(124,92,255,.28);font:500 12px/1.3 var(--djsk-font-body);color:var(--djsk-text-soft);}
.djsk-kbd-title{font-weight:700;color:var(--djsk-text);}
.djsk-kbd-item{display:inline-flex;align-items:center;gap:6px;}
.djsk-kbd{display:inline-grid;place-items:center;min-width:22px;height:22px;padding:0 6px;border-radius:6px;font:700 12px/1 var(--djsk-font-head);
 color:var(--djsk-text);background:var(--djsk-surface-2);border:1px solid var(--djsk-border-strong);box-shadow:inset 0 -2px 0 rgba(0,0,0,.35);}
.djsk-kbd-note{color:var(--djsk-muted);}
/* integrações (barra lateral) */
.djsk-int-list{display:flex;flex-direction:column;gap:8px;}
.djsk-int{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:12px;background:rgba(22,22,34,.7);border:1px solid var(--djsk-border);}
.djsk-int-text{flex:1 1 auto;min-width:0;}
.djsk-int-name{font:700 13px/1.25 var(--djsk-font-head);color:var(--djsk-text);}
.djsk-int-purpose{font:500 11px/1.3 var(--djsk-font-body);color:var(--djsk-muted);}
.djsk-int-detail{margin-top:2px;font:600 11px/1.3 var(--djsk-font-body);color:var(--djsk-ok);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
/* cartão de passagem do set */
.djsk-pass{display:flex;flex-direction:column;gap:7px;padding-left:10px;border-left:3px solid var(--djsk-tone,var(--djsk-muted));}
.djsk-pass-top{display:flex;flex-wrap:wrap;align-items:center;gap:8px;}
.djsk-pass-num{font:700 15px/1 var(--djsk-font-head);letter-spacing:-.01em;color:var(--djsk-text);padding:4px 8px;border-radius:8px;background:var(--djsk-surface-2);border:1px solid var(--djsk-border);}
.djsk-pass-head{font:600 14px/1.35 var(--djsk-font-body);color:var(--djsk-text);}
.djsk-pass-names{display:flex;flex-wrap:wrap;align-items:center;gap:6px;font:500 12px/1.35 var(--djsk-font-body);color:var(--djsk-muted);}
.djsk-pass-arrow{color:var(--djsk-tone,var(--djsk-muted));font-weight:700;}
.djsk-pass-expl{font:italic 500 13px/1.4 var(--djsk-font-body);color:var(--djsk-text-soft);}
/* resultado da busca do Spotify */
.djsk-sp{min-width:0;}
.djsk-sp-title{font:700 14px/1.25 var(--djsk-font-head);color:var(--djsk-text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.djsk-sp-artist{font:500 12px/1.3 var(--djsk-font-body);color:var(--djsk-text-soft);}
.djsk-sp-sub{font:500 11px/1.3 var(--djsk-font-body);color:var(--djsk-muted);}
/* veredito ao lado da nota do set */
.djsk-verdict{display:flex;flex-direction:column;gap:6px;}
.djsk-verdict-title{font:700 20px/1.2 var(--djsk-font-head);color:var(--djsk-tone,var(--djsk-text));letter-spacing:-.01em;}
.djsk-verdict-sub{font:500 13px/1.4 var(--djsk-font-body);color:var(--djsk-text-soft);}
@media (max-width:640px){
 .djsk-hero{padding:20px;} .djsk-budget{flex-basis:100%;}
 .djsk-tl-card{flex-basis:150px;width:150px;} .djsk-tl-link{flex-basis:96px;width:96px;}
}
""".replace("__FONTS_URL__", _FONTS_URL)


def css() -> str:
    """O CSS global (texto), sem a tag <style>."""
    return _CSS


def inject_css() -> None:
    """Injeta o CSS global na página. Chame no começo de cada execução do app.

    Fica no máximo uma vez por execução (chamadas repetidas na mesma execução são
    ignoradas). Usa `st.html` só com <style>: o Streamlit aplica o estilo sem
    criar um bloco visível na página.
    """
    import streamlit as st

    session, marker = _run_marker()
    if marker is not None and _injected.get(session) is marker:
        return
    st.html(f"<style>{_CSS}</style>")
    if marker is not None:
        if len(_injected) > 64:      # sessões antigas (abas fechadas) não acumulam
            _injected.clear()
        _injected[session] = marker


# sessão -> objeto que só existe durante uma execução do script daquela sessão
_injected: dict[Any, Any] = {}


def _run_marker() -> tuple[Any, Any]:
    """(id da sessão, marcador da execução atual); (None, None) fora do Streamlit.

    O Streamlit recria `ctx.cursors` (um dict novo) no começo de cada execução; guardar
    o próprio objeto (não o id) evita confundir execuções diferentes.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:  # pragma: no cover - Streamlit sempre instalado no painel
        return None, None
    ctx = get_script_run_ctx(suppress_warning=True)
    if ctx is None:
        return None, None
    return getattr(ctx, "session_id", None), getattr(ctx, "cursors", None)


# ---------------------------------------------------------------------------
# Ajudantes puros
# ---------------------------------------------------------------------------
def esc(text: Any) -> str:
    """Escapa texto para HTML (inclui aspas). None vira texto vazio."""
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def tone_for_label(label: Any) -> str:
    """Tom da qualidade de uma passagem ("suave", "boa", "atenção", "difícil").

    Aceita maiúsculas e a forma sem acento ("atencao", "dificil"); desconhecido -> "muted".
    """
    key = util.normalize_text(str(label or ""))
    for name, tone in LABEL_TONE.items():
        if util.normalize_text(name) == key:
            return tone
    return "muted"


def tone_color(tone: Any) -> str:
    """Cor (hex) de um tom; desconhecido -> cor neutra."""
    return TONES.get(_tone(tone), PALETTE["muted"])


def _tone(tone: Any, default: str = "muted", extra: Iterable[str] = ()) -> str:
    t = str(tone).strip().lower() if tone is not None else ""
    if t in TONES or t in extra:
        return t
    return default


def _float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def _dec(value: Any, decimals: int = 1) -> str:
    """Número com vírgula decimal (pt-BR); inválido -> "–"."""
    x = _float(value)
    if x is None:
        return "–"
    return f"{x:.{decimals}f}".replace(".", ",")


def _fmt_bpm(value: Any) -> str:
    """128.0 -> "128"; 129.5 -> "129,5"; inválido -> "–"."""
    x = _float(value)
    if x is None or x <= 0:
        return "–"
    if abs(x - round(x)) < 0.05:
        return str(int(round(x)))
    return _dec(x, 1)


def _brl(value: Any) -> str:
    x = _float(value)
    return util.format_brl(x) if x is not None else "indisponível"


def _usd(value: Any) -> str:
    x = _float(value)
    return util.format_brl(x).replace("R$", "US$") if x is not None else "indisponível"


def _slug(anchor: Any) -> str:
    """Âncora segura para id/href: "#Referências" -> "referencias"."""
    text = util.normalize_text(str(anchor or "").lstrip("#")).replace(" ", "-")
    return _SLUG_RE.sub("", text)


def _safe_url(url: Any) -> str | None:
    """Só aceita http(s) (capas do YouTube); qualquer outra coisa é descartada."""
    if not url:
        return None
    text = str(url).strip()
    try:
        scheme = urlparse(text).scheme.lower()
    except ValueError:
        return None
    return text if scheme in ("http", "https") else None


def _root(inner: str, extra_class: str = "") -> str:
    cls = f"djsk-root {extra_class}".strip()
    return f'<div class="{cls}">{inner}</div>'


# ---------------------------------------------------------------------------
# Peças pequenas (HTML)
# ---------------------------------------------------------------------------
def badge(text: Any, tone: str = "muted", icon: str | None = None, *, title: str | None = None,
          pulse: bool = False) -> str:
    """Etiqueta arredondada colorida pelo tom (ok/good/warn/bad/info/muted/accent)."""
    t = _tone(tone)
    cls = f"djsk-badge djsk-tone-{t}" + (" djsk-playing" if pulse else "")
    attr = f' title="{esc(title)}"' if title else ""
    ic = f'<span class="djsk-badge-icon" aria-hidden="true">{esc(icon)}</span>' if icon else ""
    return f'<span class="{cls}"{attr}>{ic}<span>{esc(text)}</span></span>'


def chip(text: Any, strong: Any = None, *, key: bool = False, strong_first: bool = True) -> str:
    """Chip discreto: "<b>128</b> BPM" (strong antes) ou "tom <b>8A</b>" (strong_first=False)."""
    cls = "djsk-chip djsk-chip--key" if key else "djsk-chip"
    bold = f"<b>{esc(strong)}</b>" if strong not in (None, "") else ""
    plain = esc(text)
    sep = " " if bold and plain else ""
    inner = f"{bold}{sep}{plain}" if strong_first else f"{plain}{sep}{bold}"
    return f'<span class="{cls}">{inner}</span>'


def rings_html(rings: Sequence[str]) -> str:
    """Põe vários `score_ring_svg` lado a lado (quebra linha em tela estreita)."""
    return _root(f'<div class="djsk-rings">{"".join(str(r) for r in rings or [])}</div>')


def kpi_card(label: Any, value: Any, sub: Any = None, tone: str = "default",
             icon: str | None = None) -> str:
    """Cartão de número (rótulo, valor grande, linha de apoio). Use dentro de `kpi_grid`."""
    t = _tone(tone, default="default", extra=("default",))
    ic = f'<span class="djsk-kpi-icon" aria-hidden="true">{esc(icon)}</span>' if icon else ""
    sub_html = f'<div class="djsk-kpi-sub">{esc(sub)}</div>' if sub not in (None, "") else ""
    return (f'<div class="djsk-kpi djsk-tone-{t}">'
            f'<div class="djsk-kpi-top">{ic}<span class="djsk-kpi-label" title="{esc(label)}">{esc(label)}</span></div>'
            f'<div class="djsk-kpi-value" title="{esc(value)}">{esc(value)}</div>{sub_html}</div>')


def kpi_grid_html(cards: Sequence[str], columns: int = 4) -> str:
    """Grade responsiva de cartões (`kpi_card`). Em tela estreita quebra a linha sozinha."""
    try:
        cols = int(columns)
    except (TypeError, ValueError):
        cols = 4
    cols = int(_clamp(cols, 1, 8))
    inner = "".join(str(c) for c in cards or [])
    return _root(f'<div class="djsk-kpi-grid" style="--djsk-cols:{cols}">{inner}</div>')


def energy_bar(value_0_1: Any, tone: str = "accent") -> str:
    """Mini barra horizontal de energia (0..1). Fora da faixa é cortado; inválido = 0."""
    x = _float(value_0_1)
    v = _clamp(x, 0.0, 1.0) if x is not None else 0.0
    pct = round(v * 100)
    t = _tone(tone, default="accent")
    return (f'<span class="djsk-ebar djsk-tone-{t}" role="meter" aria-valuemin="0" aria-valuemax="100" '
            f'aria-valuenow="{pct}" aria-label="Energia {pct}%" title="Energia {pct}%">'
            f'<span style="width:{v * 100:.1f}%"></span></span>')


def _auto_score_tone(v: int) -> str:
    if v >= 80:
        return "ok"
    if v >= 60:
        return "good"
    if v >= 40:
        return "warn"
    return "bad"


def score_ring_svg(value_0_100: Any, label: Any, tone: str | None = None, size: int = 88) -> str:
    """Anel circular de nota ("87/100"). Valor cortado em 0..100.

    Sem `tone`, a cor sai da própria nota (80+ verde, 60+ verde-azulado, 40+ âmbar, abaixo vermelho).
    O anel é desenhado com CSS (conic-gradient), não com <svg>: o `st.html` do Streamlit
    remove SVG, e assim o anel aparece tanto em `st.html` quanto em `st.markdown`.
    Os estilos essenciais vão no próprio elemento (funciona mesmo sem `inject_css`).
    """
    x = _float(value_0_100)
    v = int(round(_clamp(x, 0.0, 100.0))) if x is not None else 0
    t = _tone(tone, default=_auto_score_tone(v)) if tone is not None else _auto_score_tone(v)
    color = TONES.get(t, PALETTE["accent"])
    try:
        px = int(_clamp(float(size), 48, 200))
    except (TypeError, ValueError):
        px = 88
    thick = max(6, round(px * 0.095))
    lab = esc(label)
    arc = (f"background:conic-gradient({color} 0 {v}%,rgba(255,255,255,.09) {v}% 100%);"
           f"-webkit-mask:radial-gradient(farthest-side,transparent calc(100% - {thick}px),#000 calc(100% - {thick - 1}px));"
           f"mask:radial-gradient(farthest-side,transparent calc(100% - {thick}px),#000 calc(100% - {thick - 1}px));"
           "position:absolute;inset:0;border-radius:50%")
    glow = f"box-shadow:0 0 22px -6px {color};" if v > 0 else ""
    return (f'<figure class="djsk-ring djsk-tone-{t}" role="img" aria-label="{lab}: nota {v} de 100" '
            f'style="display:inline-flex;flex-direction:column;align-items:center;gap:6px;margin:0">'
            f'<div class="djsk-ring-dial" style="position:relative;width:{px}px;height:{px}px;border-radius:50%;{glow}">'
            f'<div class="djsk-ring-arc" style="{arc}"></div>'
            f'<div class="djsk-ring-text" style="position:absolute;inset:0;display:flex;flex-direction:column;'
            f'align-items:center;justify-content:center;line-height:1">'
            f'<span class="djsk-ring-val" style="font:700 {round(px * 0.27)}px/1 var(--djsk-font-head,sans-serif);'
            f'color:{PALETTE["text"]};letter-spacing:-.03em">{v}</span>'
            f'<span class="djsk-ring-max" style="margin-top:3px;font:600 {max(9, round(px * 0.11))}px/1 '
            f'var(--djsk-font-body,sans-serif);color:{PALETTE["muted"]}">/100</span></div></div>'
            f'<figcaption>{lab}</figcaption></figure>')


def legend_html() -> str:
    """Legenda das cores das passagens (suave/boa/atenção/difícil)."""
    items = "".join(
        f'<span class="djsk-legend-item djsk-tone-{tone}"><span class="djsk-dot"></span>'
        f'<b>{esc(label.capitalize())}</b>{esc(LABEL_HINT.get(label, ""))}</span>'
        for label, tone in LABEL_TONE.items())
    return _root(f'<div class="djsk-legend"><span class="djsk-legend-title">Cores das passagens:</span>{items}</div>')


# ---------------------------------------------------------------------------
# Blocos de página (HTML)
# ---------------------------------------------------------------------------
def _tag_badge(tag: Any) -> str:
    """Etiqueta do cabeçalho: texto simples ou (texto, tom[, ícone])."""
    if isinstance(tag, (tuple, list)) and tag:
        text = tag[0]
        tone = tag[1] if len(tag) > 1 else "accent"
        icon = tag[2] if len(tag) > 2 else None
        return badge(text, tone, icon)
    return badge(tag, "accent")


def hero_html(title: Any, subtitle: Any, spent_brl: Any, cap_brl: Any, credit_left_usd: Any,
              credit_total_usd: Any, insufficient: bool = False, *, kicker: Any = "Painel de curadoria",
              tags: Sequence[Any] | None = None) -> str:
    """Cabeçalho com gradiente, título grande e medidor do orçamento (R$ gasto / teto).

    Valores None (orçamento indisponível) aparecem como "indisponível", sem quebrar.
    """
    spent = _float(spent_brl)
    cap = _float(cap_brl)
    frac = (spent / cap) if spent is not None and cap and cap > 0 else None
    if insufficient or (frac is not None and frac >= 1.0):
        tone = "bad"
    elif frac is None:
        tone = "muted"
    elif frac >= 0.8:
        tone = "warn"
    else:
        tone = "accent"
    width = _clamp(frac, 0.0, 1.0) * 100 if frac is not None else 0.0
    pct = f"{_dec(frac * 100, 0)}% usado" if frac is not None else "—"
    left = (cap - spent) if spent is not None and cap is not None else None
    foot = []
    if left is not None:
        foot.append(f"Sobram {esc(_brl(max(0.0, left)))}" if left >= 0 else "Teto ultrapassado")
    credit_left = _float(credit_left_usd)
    credit_total = _float(credit_total_usd)
    if credit_left is None:
        foot.append("Crédito Replicate: indisponível")
    else:
        total = f" de {esc(_usd(credit_total))}" if credit_total is not None else ""
        foot.append(f"Crédito Replicate: {esc(_usd(credit_left))}{total}")
    alert = ('<div class="djsk-budget-alert">⚠ A Replicate informou saldo insuficiente: '
             'o Estúdio de Vocal fica pausado.</div>') if insufficient else ""
    tags_html = ""
    if tags:
        tags_html = '<div class="djsk-hero-tags">' + "".join(_tag_badge(t) for t in tags) + "</div>"
    eq = '<span class="djsk-hero-eq" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>'
    kick = f'<div class="djsk-hero-kicker">{eq}<span>{esc(kicker)}</span></div>' if kicker else ""
    sub = f'<p class="djsk-hero-sub">{esc(subtitle)}</p>' if subtitle not in (None, "") else ""
    cap_txt = esc(_brl(cap)) if cap is not None else "indisponível"
    budget = (f'<div class="djsk-budget djsk-tone-{tone}" role="group" aria-label="Orçamento">'
              f'<div class="djsk-budget-head"><span>Orçamento</span><span class="djsk-budget-pct">{pct}</span></div>'
              f'<div class="djsk-budget-value">{esc(_brl(spent))} <span>/ {cap_txt}</span></div>'
              f'<div class="djsk-meter" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
              f'aria-valuenow="{width:.0f}"><span style="width:{width:.1f}%"></span></div>'
              f'<div class="djsk-budget-foot">{"".join(f"<div>{x}</div>" for x in foot)}</div>{alert}</div>')
    return _root(f'<div class="djsk-hero"><div class="djsk-hero-main">{kick}'
                 f'<div class="djsk-hero-title" role="heading" aria-level="1">{esc(title)}</div>'
                 f'{sub}{tags_html}</div>{budget}</div>')


def stepper_html(steps: Sequence[Sequence[Any]]) -> str:
    """Barra de etapas: [(rótulo, "done"|"active"|"todo", "#ancora"), ...]."""
    parts: list[str] = []
    prev_done = False
    for i, step in enumerate(steps or []):
        step = list(step) + [None, None, None]
        label, status, anchor = step[0], step[1], step[2]
        status = str(status or "todo").strip().lower()
        if status not in _STATUSES:
            status = "todo"
        if i > 0:
            parts.append(f'<span class="djsk-step-sep{" is-done" if prev_done else ""}" aria-hidden="true"></span>')
        mark = "✓" if status == "done" else str(i + 1)
        inner = (f'<span class="djsk-step-dot" aria-hidden="true">{mark}</span>'
                 f'<span class="djsk-step-label">{esc(label)}</span>')
        cls = f"djsk-step djsk-step--{status}"
        current = ' aria-current="step"' if status == "active" else ""
        slug = _slug(anchor)
        if slug:
            parts.append(f'<a class="{cls}" href="#{slug}"{current}>{inner}</a>')
        else:
            parts.append(f'<span class="{cls}"{current}>{inner}</span>')
        prev_done = status == "done"
    return _root(f'<nav class="djsk-stepper" aria-label="Etapas do painel">{"".join(parts)}</nav>')


def section_header_html(number: Any, title: Any, subtitle: Any = None, anchor: str | None = None) -> str:
    """Título de seção com número em destaque; `anchor` cria o alvo dos links da barra de etapas."""
    slug = _slug(anchor)
    a = f'<a class="djsk-anchor" id="{slug}" aria-hidden="true"></a>' if slug else ""
    sub = f'<div class="djsk-section-sub">{esc(subtitle)}</div>' if subtitle not in (None, "") else ""
    return _root(f'{a}<div class="djsk-section"><div class="djsk-section-num" aria-hidden="true">{esc(number)}</div>'
                 f'<div class="djsk-section-text"><div class="djsk-section-title" role="heading" aria-level="2">'
                 f'{esc(title)}</div>{sub}</div></div>')


# ---------------------------------------------------------------------------
# Linha do tempo do set
# ---------------------------------------------------------------------------
def _link_tone(link: Mapping[str, Any]) -> str:
    tone = link.get("tone")
    if tone is not None and str(tone).strip().lower() in TONES:
        return str(tone).strip().lower()
    return tone_for_label(link.get("label"))


def _label_text(label: Any) -> str:
    text = str(label or "").strip()
    return text[:1].upper() + text[1:] if text else "Passagem"


def _total_duration(items: Sequence[Mapping[str, Any]]) -> str | None:
    total = 0.0
    for it in items:
        d = it.get("duration")
        secs = _float(d) if isinstance(d, (int, float)) else util.parse_duration(str(d) if d else None)
        if secs is None:
            return None
        total += secs
    if total <= 0:
        return None
    minutes = int(total / 60 + 0.5)
    h, m = divmod(minutes, 60)
    return f"{h}h{m:02d}" if h else f"{m} min"


def _duration_text(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return util.format_duration(float(value)) if _float(value) is not None else ""
    return str(value or "").strip()


def _card(item: Mapping[str, Any], index: int, width: int) -> str:
    pos = item.get("pos")
    try:
        pos_n = int(pos)
    except (TypeError, ValueError):
        pos_n = index + 1
    title = str(item.get("title") or "Sem título")
    artist = str(item.get("artist") or "")
    thumb = _safe_url(item.get("thumb"))
    if thumb:
        cover = f'<img src="{esc(thumb)}" alt="" loading="lazy" referrerpolicy="no-referrer">'
    else:
        initial = (title.strip()[:1] or "♪").upper()
        cover = f'<div class="djsk-tl-ph" aria-hidden="true">{esc(initial)}</div>'
    dur = _duration_text(item.get("duration"))
    dur_html = f'<span class="djsk-tl-dur">{esc(dur)}</span>' if dur else ""
    playing = bool(item.get("playing"))
    now = (f'<span class="djsk-tl-now">{badge("tocando", "accent", "▶", pulse=True)}</span>'
           if playing else "")
    chips = [chip("BPM", _fmt_bpm(item.get("bpm")))]
    camelot = item.get("camelot")
    if camelot:
        chips.append(chip("tom", camelot, key=True, strong_first=False))
    energy = _float(item.get("energy"))
    e = _clamp(energy, 0.0, 1.0) if energy is not None else 0.0
    delay = min(index, 20) * 25
    cls = "djsk-tl-card djsk-playing" if playing else "djsk-tl-card"
    tip = f"{pos_n:0{width}d} · {artist + ' - ' if artist else ''}{title}"
    return (f'<div class="{cls}" role="listitem" title="{esc(tip)}" style="animation-delay:{delay}ms">'
            f'<div class="djsk-tl-cover">{cover}{now}<span class="djsk-tl-pos">{pos_n:0{width}d}</span>{dur_html}</div>'
            f'<div class="djsk-tl-body"><div class="djsk-tl-title">{esc(title)}</div>'
            f'<div class="djsk-tl-artist">{esc(artist) or "&nbsp;"}</div>'
            f'<div class="djsk-tl-chips">{"".join(chips)}</div>'
            f'<div class="djsk-tl-energy"><span>Energia</span>{energy_bar(e)}'
            f'<span class="djsk-tl-energy-val">{round(e * 100)}%</span></div></div></div>')


def _connector(link: Mapping[str, Any] | None, a: int, b: int, width: int) -> str:
    link = link or {}
    tone = _link_tone(link)
    label = _label_text(link.get("label"))
    short = str(link.get("short") or "").strip()
    tip = f"{a:0{width}d} → {b:0{width}d} · {label}" + (f" · {short}" if short else "")
    short_html = f'<span class="djsk-tl-link-short">{esc(short)}</span>' if short else ""
    return (f'<div class="djsk-tl-link djsk-tone-{tone}" title="{esc(tip)}" aria-label="{esc(tip)}">'
            f'<span class="djsk-tl-link-label"><span class="djsk-dot"></span>{esc(label)}</span>'
            f'<span class="djsk-tl-link-line" aria-hidden="true"></span>{short_html}</div>')


def _summary(items: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any] | None]) -> str:
    counts: dict[str, int] = {}
    for link in links:
        lab = str((link or {}).get("label") or "").strip().lower()
        tone = _link_tone(link or {})
        key = next((k for k, t in LABEL_TONE.items() if t == tone), lab or "outra")
        counts[key] = counts.get(key, 0) + 1
    plural = {"suave": "suaves", "boa": "boas", "atenção": "de atenção", "difícil": "difíceis"}
    parts = [f'<span class="djsk-tl-summary-title">{len(items)} faixa{"s" if len(items) != 1 else ""}</span>']
    total = _total_duration(items)
    if total:
        parts.append(badge(f"≈ {total} de set", "info", "⏱"))
    bpms = [b for b in (_float(it.get("bpm")) for it in items) if b and b > 0]
    if bpms:
        lo, hi = min(bpms), max(bpms)
        txt = f"{_fmt_bpm(lo)} BPM" if _fmt_bpm(lo) == _fmt_bpm(hi) else f"{_fmt_bpm(lo)}–{_fmt_bpm(hi)} BPM"
        parts.append(badge(txt, "accent", "♫"))
    for name, tone in LABEL_TONE.items():
        n = counts.get(name, 0)
        if n:
            word = name if n == 1 else plural[name]
            parts.append(badge(f"{n} {word}", tone))
    return f'<div class="djsk-tl-summary">{"".join(parts)}</div>'


def _map(items: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any] | None], width: int) -> str:
    n = len(items)
    bars = []
    for i, it in enumerate(items):
        e = _float(it.get("energy"))
        e = _clamp(e, 0.0, 1.0) if e is not None else 0.0
        h = 8 + e * 92
        title = str(it.get("title") or "")
        cls = "djsk-tl-bar djsk-playing" if it.get("playing") else "djsk-tl-bar"
        bars.append(f'<div class="{cls}" title="{i + 1:0{width}d} · {esc(title)} · energia {round(e * 100)}%">'
                    f'<span style="height:{h:.0f}%"></span></div>')
    flow = []
    for i in range(n - 1):
        link = links[i] if i < len(links) else None
        tone = _link_tone(link or {})
        label = _label_text((link or {}).get("label"))
        flow.append(f'<span class="djsk-tone-{tone}" title="{i + 1:0{width}d} → {i + 2:0{width}d} · {esc(label)}"></span>')
    side = 50.0 / n
    return (f'<div class="djsk-tl-map" aria-hidden="true"><div class="djsk-tl-bars">{"".join(bars)}</div>'
            f'<div class="djsk-tl-flow" style="margin:6px {side:.3f}% 0">{"".join(flow)}</div>'
            f'<div class="djsk-tl-map-cap"><span>início · energia (barras) e passagens (cores)</span><span>fim</span></div></div>')


def timeline_html(items: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any]],
                  *, summary: bool = True) -> str:
    """Linha do tempo HORIZONTAL do set (rola para o lado).

    items: [{pos, title, artist, bpm, camelot, energy (0..1), thumb (url), duration ("3:12"),
             playing (opcional)}]
    links: len(items) - 1 itens {label: "suave"|"boa"|"atenção"|"difícil", tone (opcional), short}.
    Campos faltando não quebram: viram "–" ou somem. `summary=False` tira o resumo e o mapa.
    """
    items = [it for it in (items or []) if isinstance(it, Mapping)]
    links = list(links or [])
    if not items:
        return _root('<div class="djsk-empty">Nenhuma faixa no set ainda. Aprove músicas na triagem '
                     'para montar a linha do tempo.</div>')
    n = len(items)
    width = 3 if n > 99 else 2
    track: list[str] = []
    for i, it in enumerate(items):
        if i > 0:
            link = links[i - 1] if i - 1 < len(links) and isinstance(links[i - 1], Mapping) else None
            track.append(_connector(link, i, i + 1, width))
        track.append(_card(it, i, width))
    head = ""
    if summary:
        safe_links = [lk if isinstance(lk, Mapping) else None for lk in links[: n - 1]]
        head = _summary(items, safe_links)
        if n >= 2:
            head += _map(items, safe_links, width)
    return _root(f'<div class="djsk-tl">{head}<div class="djsk-tl-track" role="list" '
                 f'aria-label="Ordem do set">{"".join(track)}</div></div>')


# ---------------------------------------------------------------------------
# Funções que desenham na página
# ---------------------------------------------------------------------------
def render_html(markup: str) -> None:
    """Desenha um pedaço de HTML do tema (ignora texto vazio)."""
    if not markup:
        return
    import streamlit as st

    st.html(markup)


def kpi_grid(cards: Sequence[str], columns: int = 4) -> None:
    """Desenha a grade de cartões de número."""
    if not cards:
        return
    render_html(kpi_grid_html(cards, columns))


def hero(title: Any, subtitle: Any, spent_brl: Any, cap_brl: Any, credit_left_usd: Any,
         credit_total_usd: Any, insufficient: bool = False, *, kicker: Any = "Painel de curadoria",
         tags: Sequence[Any] | None = None) -> None:
    """Desenha o cabeçalho com o medidor de orçamento."""
    render_html(hero_html(title, subtitle, spent_brl, cap_brl, credit_left_usd, credit_total_usd,
                          insufficient, kicker=kicker, tags=tags))


def stepper(steps: Sequence[Sequence[Any]]) -> None:
    """Desenha a barra de etapas que fica presa no topo ao rolar.

    Chame no nível principal da página (fora de colunas/containers) para o "grudar no topo" funcionar.
    """
    if not steps:
        return
    render_html(stepper_html(steps))


def section_header(number: int, title: str, subtitle: str | None = None, anchor: str | None = None) -> None:
    """Desenha o título de seção (com âncora para a barra de etapas)."""
    render_html(section_header_html(number, title, subtitle, anchor))


def timeline(items: Sequence[Mapping[str, Any]], links: Sequence[Mapping[str, Any]], *,
             summary: bool = True) -> None:
    """Desenha a linha do tempo do set."""
    render_html(timeline_html(items, links, summary=summary))


def legend() -> None:
    """Desenha a legenda das cores das passagens."""
    render_html(legend_html())
