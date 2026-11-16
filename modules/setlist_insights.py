"""Leitura do set para quem não é técnico: estilos de ordenação, motivos de cada passagem,
nota do set (0 a 100) e resumo. Módulo PURO (sem streamlit), só usa models/config/order.

Tons (`tone`) usados nos motivos, do melhor para o pior:
  "ok"      perfeito          "good"  bom
  "warn"    atenção           "bad"   difícil
  "neutral" sem informação (ex.: tom não identificado)
`LABEL_TONES` liga os rótulos de transição ("suave", "boa", "atenção", "difícil") a esses tons.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from typing import Any

import config
from modules import order
from modules.models import Analysis, Transition

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Estilos de ordenação (presets de pesos)
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    "equilibrado": {
        "label": "Equilibrado",
        "icon": "⚖️",
        "description": "Um pouco de tudo: velocidade, energia, tom e voz na medida certa.",
        "weights": dict(config.DEFAULT_WEIGHTS),
        "energy_mode": "suave",
    },
    "bpm": {
        "label": "Velocidade travada",
        "icon": "🎯",
        "description": "Junta músicas de velocidade (BPM) parecida, para quase não mexer no pitch.",
        "weights": {"bpm": 2.0, "energia": 0.5, "tom": 0.1, "voz": 0.5},
        "energy_mode": "suave",
    },
    "subindo": {
        "label": "Energia subindo",
        "icon": "📈",
        "description": "Começa mais calmo e vai esquentando até o fim do set.",
        "weights": {"bpm": 0.8, "energia": 2.0, "tom": 0.1, "voz": 0.5},
        "energy_mode": "subindo",
    },
    "harmonico": {
        "label": "Tons que combinam",
        "icon": "🎹",
        "description": "Prefere emendar músicas cujos tons combinam, para a melodia não brigar.",
        "weights": {"bpm": 1.0, "energia": 0.6, "tom": 1.5, "voz": 0.5},
        "energy_mode": "suave",
    },
    "voz": {
        "label": "Sem atropelar MC",
        "icon": "🎤",
        "description": "Evita passagens com dois MCs cantando ao mesmo tempo (usa o Estúdio de Vocal).",
        "weights": {"bpm": 1.0, "energia": 0.6, "tom": 0.2, "voz": 2.0},
        "energy_mode": "suave",
    },
}
DEFAULT_PRESET = "equilibrado"

WEIGHT_HELP: dict[str, str] = {
    "bpm": "BPM: evita juntar músicas de velocidades diferentes.",
    "energia": "Energia: evita quedas ou saltos bruscos de pegada de uma música para a outra.",
    "tom": "Tom: prefere músicas cujos tons combinam, para a melodia não brigar.",
    "voz": "Voz: evita dois MCs cantando ao mesmo tempo na passagem (precisa do Estúdio de Vocal).",
}

TONES: dict[str, str] = {
    "ok": "perfeito", "good": "bom", "warn": "atenção", "bad": "difícil", "neutral": "sem informação",
}
LABEL_TONES: dict[str, str] = {"suave": "ok", "boa": "good", "atenção": "warn", "difícil": "bad"}
_TONE_RANK = {"ok": 0, "good": 1, "neutral": 2, "warn": 3, "bad": 4}

_HEADLINE_PREFIX = {
    "suave": "Passagem suave",
    "boa": "Boa passagem",
    "atenção": "Passagem que pede atenção",
    "difícil": "Passagem difícil",
}

# Faixas de ΔBPM e de energia usadas nos motivos.
_BPM_OK, _BPM_GOOD, _BPM_WARN = 1.0, 4.0, 8.0
_ENERGY_SAME = 0.10
_ENERGY_BIG = 0.30


def preset_weights(key: str) -> dict[str, float]:
    """Cópia dos pesos de um estilo. "equilibrado" (ou chave desconhecida) lê
    config.DEFAULT_WEIGHTS na hora da chamada."""
    if key not in PRESETS or key == DEFAULT_PRESET:
        return dict(config.DEFAULT_WEIGHTS)
    return dict(PRESETS[key]["weights"])


def preset_energy_mode(key: str) -> str:
    return str(PRESETS.get(key, PRESETS[DEFAULT_PRESET])["energy_mode"])


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------
def _num(x: float) -> str:
    """130.0 -> "130"; 6.5 -> "6,5"."""
    if abs(x - round(x)) < 0.05:
        return str(int(round(x)))
    return f"{x:.1f}".replace(".", ",")


def _finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# ---------------------------------------------------------------------------
# Motivos de uma passagem
# ---------------------------------------------------------------------------
def _reason(kind: str, icon: str, text: str, tone: str, short: str) -> dict[str, str]:
    return {"kind": kind, "icon": icon, "text": text, "tone": tone, "short": short}


def _bpm_reason(tr: Transition, a: Analysis | None, b: Analysis | None) -> dict[str, str] | None:
    diff = _finite_or_none(tr.bpm_diff)
    bpm_a = _finite_or_none(a.effective_bpm) if a is not None else None
    bpm_b = _finite_or_none(b.effective_bpm) if b is not None else None
    if diff is None:
        if bpm_a is None or bpm_b is None:
            return None
        diff = abs(bpm_a - bpm_b)
    arrow = f" ({_num(bpm_a)} → {_num(bpm_b)})" if bpm_a is not None and bpm_b is not None else ""
    if diff <= _BPM_OK:
        same = f" ({_num(bpm_a)} BPM)" if bpm_a is not None else ""
        return _reason("bpm", "🎯", f"Mesma velocidade{same}", "ok", "mesma velocidade")
    d = _num(diff)
    if diff <= _BPM_GOOD:
        return _reason("bpm", "🎚️", f"Velocidade muda {d} BPM{arrow}: ajuste leve no pitch",
                       "good", "velocidade quase igual")
    if diff <= _BPM_WARN:
        return _reason("bpm", "🎚️", f"Velocidade muda {d} BPM{arrow}: ajuste o pitch",
                       "warn", f"velocidade muda {d} BPM")
    return _reason("bpm", "⚠️", f"Velocidade muda {d} BPM{arrow}: troca brusca de ritmo",
                   "bad", f"velocidade muda {d} BPM")


def _energy_reason(tr: Transition, a: Analysis | None, b: Analysis | None) -> dict[str, str] | None:
    delta = None
    if a is not None and b is not None:
        out_a, in_b = _finite_or_none(a.energy_out), _finite_or_none(b.energy_in)
        if out_a is not None and in_b is not None:
            delta = in_b - out_a
    if delta is None:
        diff = _finite_or_none(tr.energy_diff)
        if diff is None:
            return None
        if diff <= _ENERGY_SAME:
            return _reason("energia", "➡️", "Energia parecida", "ok", "energia parecida")
        tone = "bad" if diff > _ENERGY_BIG else "warn"
        return _reason("energia", "⚡", "Energia muda bastante", tone, "energia muda bastante")
    if abs(delta) <= _ENERGY_SAME:
        return _reason("energia", "➡️", "Energia parecida", "ok", "energia parecida")
    if delta > 0:
        if delta > _ENERGY_BIG:
            return _reason("energia", "📈", "Energia sobe bastante: entrada forte", "warn",
                           "energia sobe bastante")
        return _reason("energia", "📈", "Energia sobe", "good", "energia sobe")
    if -delta > _ENERGY_BIG:
        return _reason("energia", "📉", "Energia cai bastante: a pista pode esfriar", "bad",
                       "energia cai bastante")
    return _reason("energia", "📉", "Energia cai um pouco", "warn", "energia cai um pouco")


def _key_reason(tr: Transition, a: Analysis | None, b: Analysis | None) -> dict[str, str]:
    cam_a = a.camelot if a is not None else None
    cam_b = b.camelot if b is not None else None
    dist = tr.camelot_dist
    if dist is None and cam_a and cam_b:
        dist = order.camelot_distance(cam_a, cam_b)
    if dist is None:
        return _reason("tom", "❔", "Tom não identificado", "neutral", "")
    pair = f" ({cam_a} → {cam_b})" if cam_a and cam_b else ""
    if dist == 0:
        same = f" (mesmo tom: {cam_a})" if cam_a else " (mesmo tom)"
        return _reason("tom", "🎹", f"Tons combinam{same}", "ok", "mesmo tom")
    if dist == 1:
        return _reason("tom", "🎹", f"Tons combinam{pair}", "ok", "tons combinam")
    if dist == 2:
        return _reason("tom", "🎹", f"Tons próximos{pair}", "good", "tons próximos")
    if dist == 3:
        return _reason("tom", "🎹", f"Tons um pouco diferentes{pair}", "warn", "tons um pouco diferentes")
    return _reason("tom", "🎹", f"Tons distantes{pair}: a melodia pode brigar", "bad", "tons distantes")


def _voice_reason(tr: Transition) -> dict[str, str] | None:
    voice = tr.voice
    if voice == "limpa":
        return _reason("voz", "🤫", "Passagem sem voz", "ok", "sem voz na passagem")
    if voice == "voz de um lado":
        return _reason("voz", "🎤", "Só um MC cantando", "good", "só um MC cantando")
    if voice == "choque de voz":
        return _reason("voz", "🗣️", "MCs se atropelam", "bad", "MCs se atropelam")
    return None   # "sem stems" (ou desconhecido): não há o que dizer


def transition_reasons(tr: Transition, a: Analysis | None, b: Analysis | None) -> list[dict[str, str]]:
    """Motivos da passagem A→B em português simples, na ordem velocidade, energia, tom, voz.

    Cada item: {"kind": "bpm"|"energia"|"tom"|"voz", "icon": emoji, "text": frase,
    "tone": "ok"|"good"|"warn"|"bad"|"neutral", "short": versão curta (minúscula, pode ser "")}.
    `a`/`b` podem ser None (usa só os números da transição).
    """
    out = []
    for r in (_bpm_reason(tr, a, b), _energy_reason(tr, a, b), _key_reason(tr, a, b), _voice_reason(tr)):
        if r is not None:
            out.append(r)
    return out


def _label_of(tr: Transition, reasons: list[dict[str, str]]) -> str:
    if tr.label in _HEADLINE_PREFIX:
        return tr.label
    worst = max((_TONE_RANK.get(r["tone"], 2) for r in reasons), default=0)
    return {0: "suave", 1: "boa", 2: "boa", 3: "atenção"}.get(worst, "difícil")


def transition_headline(tr: Transition, a: Analysis | None, b: Analysis | None) -> str:
    """Uma frase curta que resume a passagem.

    Ex.: "Passagem suave: mesma velocidade e energia parecida" ou
    "Passagem difícil: MCs se atropelam e velocidade muda 12 BPM".
    """
    reasons = [r for r in transition_reasons(tr, a, b) if r.get("short")]
    label = _label_of(tr, reasons)
    prefix = _HEADLINE_PREFIX[label]
    if label in ("suave", "boa"):
        picked = [r for r in reasons if r["tone"] in ("ok", "good")]
        picked.sort(key=lambda r: _TONE_RANK[r["tone"]])        # estável: mantém a ordem dos tipos
    else:
        picked = [r for r in reasons if r["tone"] in ("bad", "warn")]
        picked.sort(key=lambda r: -_TONE_RANK[r["tone"]])
    picked = picked[:2]
    if not picked:
        return prefix
    return f"{prefix}: " + " e ".join(r["short"] for r in picked)


# ---------------------------------------------------------------------------
# Nota do set (0 a 100) e comparação de ordens
# ---------------------------------------------------------------------------
def _used_weights(tr: Transition, w: dict[str, float]) -> float:
    used = dict(w)
    if tr.camelot_dist is None:
        used["tom"] = 0.0
    if tr.voice == order.NO_STEMS_LABEL or tr.voice not in config.VOICE_COST:
        used["voz"] = 0.0
    return float(sum(used.values()))


def _zero_score_at() -> float:
    """Custo normalizado a partir do qual a passagem vale 0 (limite de "difícil" = 0.6)."""
    labels = config.TRANSITION_LABELS
    try:
        v = float(labels[-2][0])
    except (IndexError, TypeError, ValueError):
        v = 0.6
    return v if math.isfinite(v) and v > 0 else 0.6


def transition_score(tr: Transition, weights: dict[str, float] | None = None) -> float:
    """Nota 0..100 de uma passagem (100 = perfeita; custo normalizado ≥ 0.6 = 0)."""
    w = order._clean_weights(weights)
    total = _used_weights(tr, w)
    if total <= 0:
        return 100.0
    try:
        cost = float(tr.cost)
    except (TypeError, ValueError):
        cost = math.nan
    if not math.isfinite(cost):        # como em order.classify: custo inválido = "difícil"
        return 0.0
    normalized = max(cost, 0.0) / total
    return 100.0 * (1.0 - min(normalized / _zero_score_at(), 1.0))


def set_score(transitions: list[Transition], weights: dict[str, float] | None = None) -> int:
    """Nota do set de 0 a 100: média das notas das passagens. Lista vazia -> 100.

    weights=None usa config.DEFAULT_WEIGHTS (como a ordenação).
    """
    if not transitions:
        return 100
    scores = [transition_score(tr, weights) for tr in transitions]
    return int(round(sum(scores) / len(scores)))


def score_verdict(score: float) -> tuple[str, str]:
    """(frase curta, tom) para mostrar ao lado da nota.

    Referência: uma passagem "suave" vale 75 ou mais; uma "boa", de 42 a 75.
    """
    if score >= 75:
        return "Set redondo", "ok"
    if score >= 55:
        return "Set bom", "good"
    if score >= 35:
        return "Algumas passagens pedem atenção", "warn"
    return "Set com passagens difíceis", "bad"


def compare_orders(baseline_ids: list[str], optimized_ids: list[str], analyses: dict[str, Analysis],
                   weights: dict[str, float] | None, voice_lookup: order.VoiceLookup | None = None,
                   energy_mode: str = "suave") -> dict[str, float | int]:
    """Compara uma ordem de referência (ex.: a ordem em que o DJ aprovou) com a otimizada.

    improvement_pct = redução % da soma das notas de dificuldade (custo) em relação à
    referência, com 1 casa; 0 se a referência já custa 0. Negativo = a otimizada ficou pior.
    """
    base = order.evaluate_order(list(baseline_ids), analyses, weights, voice_lookup,
                                energy_mode=energy_mode)
    opt = order.evaluate_order(list(optimized_ids), analyses, weights, voice_lookup,
                               energy_mode=energy_mode)
    b_cost, o_cost = float(base.total_cost), float(opt.total_cost)
    if b_cost > 1e-12 and math.isfinite(b_cost) and math.isfinite(o_cost):
        improvement = round(100.0 * (b_cost - o_cost) / b_cost, 1)
    else:
        improvement = 0.0
    return {
        "baseline_score": set_score(base.transitions, weights),
        "optimized_score": set_score(opt.transitions, weights),
        "improvement_pct": improvement,
        "baseline_cost": b_cost,
        "optimized_cost": o_cost,
    }


# ---------------------------------------------------------------------------
# Resumo do set
# ---------------------------------------------------------------------------
def set_summary(order_ids: list[str], analyses: dict[str, Analysis], top_keys: int = 5) -> dict[str, Any]:
    """Resumo do set na ordem dada. Ids sem análise são ignorados.

    {n, duration_s, bpm_min, bpm_max, bpm_avg, keys: [(camelot, quantidade)] (mais comuns
    primeiro, até `top_keys`), energy_start, energy_end (energia média da 1ª e da última
    faixa, 0..1), peak_pos (posição 1-based da faixa de maior energia)}.
    Set vazio: n=0, duration_s=0.0, keys=[] e o resto None.
    """
    items = [analyses[v] for v in order_ids if v in analyses]
    if not items:
        return {"n": 0, "duration_s": 0.0, "bpm_min": None, "bpm_max": None, "bpm_avg": None,
                "keys": [], "energy_start": None, "energy_end": None, "peak_pos": None}
    durations = [_finite_or_none(an.duration_s) or 0.0 for an in items]
    bpms = [b for b in (_finite_or_none(an.effective_bpm) for an in items) if b is not None and b > 0]
    energies = [_finite_or_none(an.energy_mean) for an in items]
    counts = Counter(an.camelot for an in items if an.camelot)
    keys = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max(int(top_keys), 0)]
    valid_e = [(e, i) for i, e in enumerate(energies) if e is not None]
    peak_pos = (max(valid_e, key=lambda t: (t[0], -t[1]))[1] + 1) if valid_e else None
    return {
        "n": len(items),
        "duration_s": float(sum(max(d, 0.0) for d in durations)),
        "bpm_min": min(bpms) if bpms else None,
        "bpm_max": max(bpms) if bpms else None,
        "bpm_avg": (sum(bpms) / len(bpms)) if bpms else None,
        "keys": keys,
        "energy_start": energies[0],
        "energy_end": energies[-1],
        "peak_pos": peak_pos,
    }


def label_counts(transitions: list[Transition]) -> dict[str, int]:
    """Quantas passagens de cada rótulo. Os 4 rótulos sempre aparecem (na ordem do config)."""
    out = {label: 0 for _, label in config.TRANSITION_LABELS}
    for tr in transitions:
        if tr.label:
            out[tr.label] = out.get(tr.label, 0) + 1
    return out
