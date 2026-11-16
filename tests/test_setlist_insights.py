"""Testes de modules/setlist_insights.py (puro, sem rede e sem streamlit)."""
from __future__ import annotations

import math

import pytest

import config
from modules import order
from modules import setlist_insights as si
from modules.models import Analysis, Transition

W = {"bpm": 1.0, "energia": 0.8, "tom": 0.2, "voz": 0.5}


def make(vid: str, bpm: float = 130.0, e_in: float = 0.5, e_out: float = 0.5, e_mean: float = 0.5,
         camelot: str | None = None, duration_s: float = 180.0, bpm_manual: float | None = None) -> Analysis:
    return Analysis(video_id=vid, duration_s=duration_s, bpm=bpm, bpm_raw=bpm, bpm_manual=bpm_manual,
                    energy_in=e_in, energy_out=e_out, energy_mean=e_mean, camelot=camelot)


def tr_for(a: Analysis, b: Analysis, voice: str | None = None, weights=None) -> Transition:
    lookup = (lambda x, y: (voice, 0.5)) if voice else None
    return order.transition(a, b, weights or W, lookup)


def by_kind(reasons):
    return {r["kind"]: r for r in reasons}


# ---------------------------------------------------------------------------
# Presets e textos de ajuda
# ---------------------------------------------------------------------------

def test_presets_shape_and_order():
    assert list(si.PRESETS) == ["equilibrado", "bpm", "subindo", "harmonico", "voz"]
    for key, p in si.PRESETS.items():
        assert {"label", "icon", "description", "weights", "energy_mode"} <= set(p), key
        assert set(p["weights"]) == {"bpm", "energia", "tom", "voz"}
        assert p["energy_mode"] in order.ENERGY_MODES
        assert p["description"].strip() and p["label"].strip() and p["icon"].strip()
    assert si.PRESETS["equilibrado"]["weights"] == config.DEFAULT_WEIGHTS
    assert si.PRESETS["equilibrado"]["energy_mode"] == "suave"
    assert si.PRESETS["bpm"]["weights"] == {"bpm": 2.0, "energia": 0.5, "tom": 0.1, "voz": 0.5}
    assert si.PRESETS["subindo"]["weights"] == {"bpm": 0.8, "energia": 2.0, "tom": 0.1, "voz": 0.5}
    assert si.PRESETS["subindo"]["energy_mode"] == "subindo"
    assert si.PRESETS["harmonico"]["weights"] == {"bpm": 1.0, "energia": 0.6, "tom": 1.5, "voz": 0.5}
    assert si.PRESETS["voz"]["weights"] == {"bpm": 1.0, "energia": 0.6, "tom": 0.2, "voz": 2.0}
    assert si.PRESETS["bpm"]["label"] == "Velocidade travada"
    assert si.PRESETS["voz"]["label"] == "Sem atropelar MC"
    for p in si.PRESETS.values():
        assert all(0 <= w <= config.WEIGHT_SLIDER_MAX for w in p["weights"].values())


def test_preset_helpers_return_copies_and_read_config(monkeypatch):
    w = si.preset_weights("bpm")
    w["bpm"] = 99
    assert si.PRESETS["bpm"]["weights"]["bpm"] == 2.0
    monkeypatch.setattr(config, "DEFAULT_WEIGHTS", {"bpm": 1.5, "energia": 0, "tom": 0, "voz": 0})
    assert si.preset_weights("equilibrado")["bpm"] == 1.5
    assert si.preset_weights("nao_existe")["bpm"] == 1.5
    assert si.preset_energy_mode("subindo") == "subindo"
    assert si.preset_energy_mode("nao_existe") == "suave"


def test_weight_help_is_plain_language():
    assert set(si.WEIGHT_HELP) == {"bpm", "energia", "tom", "voz"}
    text = " ".join(si.WEIGHT_HELP.values()).lower()
    for jargon in ("rms", "2-opt", "heurística", "custo", "camelot"):
        assert jargon not in text


# ---------------------------------------------------------------------------
# Motivos de cada passagem
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bpm_b, tone, needle", [
    (130.0, "ok", "Mesma velocidade (130 BPM)"),
    (130.8, "ok", "Mesma velocidade"),
    (133.0, "good", "Velocidade muda 3 BPM"),
    (137.0, "warn", "Velocidade muda 7 BPM"),
    (138.0, "warn", "ajuste o pitch"),
    (142.0, "bad", "Velocidade muda 12 BPM"),
])
def test_bpm_reason(bpm_b, tone, needle):
    a, b = make("a", bpm=130.0), make("b", bpm=bpm_b)
    r = by_kind(si.transition_reasons(tr_for(a, b), a, b))["bpm"]
    assert r["tone"] == tone
    assert needle in r["text"]
    assert r["icon"]


def test_bpm_reason_uses_manual_bpm_and_decimal_comma():
    a, b = make("a", bpm=130.0, bpm_manual=128.0), make("b", bpm=134.5)
    r = by_kind(si.transition_reasons(tr_for(a, b), a, b))["bpm"]
    assert "6,5 BPM" in r["text"] and "128 → 134,5" in r["text"]
    assert r["tone"] == "warn"


@pytest.mark.parametrize("e_out, e_in, tone, needle", [
    (0.5, 0.55, "ok", "Energia parecida"),
    (0.4, 0.6, "good", "Energia sobe"),
    (0.2, 0.8, "warn", "Energia sobe bastante"),
    (0.6, 0.4, "warn", "Energia cai um pouco"),
    (0.9, 0.2, "bad", "Energia cai bastante"),
])
def test_energy_reason(e_out, e_in, tone, needle):
    a, b = make("a", e_out=e_out), make("b", e_in=e_in)
    r = by_kind(si.transition_reasons(tr_for(a, b), a, b))["energia"]
    assert r["tone"] == tone
    assert r["text"].startswith(needle)


@pytest.mark.parametrize("cam_a, cam_b, tone, needle", [
    ("8A", "8A", "ok", "Tons combinam (mesmo tom: 8A)"),
    ("8A", "9A", "ok", "Tons combinam (8A → 9A)"),
    ("8A", "8B", "ok", "Tons combinam (8A → 8B)"),
    ("8A", "10A", "good", "Tons próximos"),
    ("8A", "11A", "warn", "Tons um pouco diferentes"),
    ("8A", "2B", "bad", "Tons distantes (8A → 2B)"),
    ("8A", "12A", "bad", "Tons distantes"),
])
def test_key_reason(cam_a, cam_b, tone, needle):
    a, b = make("a", camelot=cam_a), make("b", camelot=cam_b)
    r = by_kind(si.transition_reasons(tr_for(a, b), a, b))["tom"]
    assert r["tone"] == tone
    assert r["text"].startswith(needle)


def test_unknown_key_reason_is_neutral():
    a, b = make("a", camelot=None), make("b", camelot="8A")
    r = by_kind(si.transition_reasons(tr_for(a, b), a, b))["tom"]
    assert r["text"] == "Tom não identificado"
    assert r["tone"] == "neutral"


@pytest.mark.parametrize("voice, tone, text", [
    ("limpa", "ok", "Passagem sem voz"),
    ("voz de um lado", "good", "Só um MC cantando"),
    ("choque de voz", "bad", "MCs se atropelam"),
])
def test_voice_reason(voice, tone, text):
    a, b = make("a"), make("b")
    r = by_kind(si.transition_reasons(tr_for(a, b, voice=voice), a, b))["voz"]
    assert (r["text"], r["tone"]) == (text, tone)


def test_no_stems_omits_voice_and_order_is_fixed():
    a, b = make("a", camelot="8A"), make("b", camelot="9A")
    reasons = si.transition_reasons(tr_for(a, b), a, b)
    assert [r["kind"] for r in reasons] == ["bpm", "energia", "tom"]
    reasons = si.transition_reasons(tr_for(a, b, voice="limpa"), a, b)
    assert [r["kind"] for r in reasons] == ["bpm", "energia", "tom", "voz"]
    for r in reasons:
        assert set(r) == {"kind", "icon", "text", "tone", "short"}
        assert r["tone"] in si.TONES


def test_reasons_without_analyses_use_transition_numbers():
    tr = Transition(from_id="a", to_id="b", cost=1.0, bpm_diff=10.0, energy_diff=0.5,
                    camelot_dist=5, voice="sem stems", label="difícil")
    reasons = by_kind(si.transition_reasons(tr, None, None))
    assert reasons["bpm"]["tone"] == "bad" and "10 BPM" in reasons["bpm"]["text"]
    assert reasons["energia"]["tone"] == "bad"
    assert reasons["tom"]["tone"] == "bad"
    assert "voz" not in reasons


def test_reasons_tolerate_nan():
    tr = Transition(from_id="a", to_id="b", cost=float("nan"), bpm_diff=float("nan"),
                    energy_diff=float("nan"), camelot_dist=None, label="difícil")
    reasons = si.transition_reasons(tr, None, None)
    assert [r["kind"] for r in reasons] == ["tom"]
    assert si.transition_headline(tr, None, None) == "Passagem difícil"


# ---------------------------------------------------------------------------
# Frase-resumo
# ---------------------------------------------------------------------------

def test_headline_smooth():
    a, b = make("a", bpm=130), make("b", bpm=130)
    tr = tr_for(a, b)
    assert tr.label == "suave"
    assert si.transition_headline(tr, a, b) == "Passagem suave: mesma velocidade e energia parecida"


def test_headline_hard_lists_worst_reasons():
    a, b = make("a", bpm=130, e_out=0.9, camelot="8A"), make("b", bpm=142, e_in=0.2, camelot="2B")
    tr = tr_for(a, b, voice="choque de voz")
    assert tr.label == "difícil"
    h = si.transition_headline(tr, a, b)
    assert h.startswith("Passagem difícil: ")
    assert h.count(" e ") == 1
    assert "velocidade muda 12 BPM" in h          # ruins primeiro, na ordem velocidade/energia/...
    assert "energia cai bastante" in h


def test_headline_attention_and_unknown_label():
    a, b = make("a", bpm=130), make("b", bpm=134.5)      # 4,5/8 = 0.56 -> "atenção"
    tr = tr_for(a, b, weights={"bpm": 1.0})
    assert tr.label == "atenção"
    assert si.transition_headline(tr, a, b) == "Passagem que pede atenção: velocidade muda 4,5 BPM"
    tr.label = ""
    assert si.transition_headline(tr, a, b).startswith("Passagem que pede atenção")


# ---------------------------------------------------------------------------
# Nota do set, comparação, resumo e contagem de rótulos
# ---------------------------------------------------------------------------

def test_set_score_bounds_and_mapping():
    assert si.set_score([], W) == 100
    a, b = make("a"), make("b")
    perfect = tr_for(a, b)
    assert perfect.cost == 0.0
    assert si.set_score([perfect, perfect], W) == 100
    terrible = tr_for(make("x", bpm=120, e_out=1.0), make("y", bpm=160, e_in=0.0), weights={"bpm": 1.0})
    assert si.set_score([terrible], {"bpm": 1.0}) == 0
    # custo normalizado 0.3 (metade de 0.6) vale 50; média com uma perfeita = 75
    half = Transition(from_id="a", to_id="b", cost=0.3, bpm_diff=2.4, energy_diff=0.0,
                      camelot_dist=None, voice="sem stems")
    assert si.set_score([half], {"bpm": 1.0}) == 50
    assert si.set_score([half, perfect], {"bpm": 1.0}) == 75


def test_set_score_uses_only_present_weights():
    # Sem tom e sem stems, os pesos de tom e voz não diluem a nota.
    tr = Transition(from_id="a", to_id="b", cost=0.3, bpm_diff=2.4, energy_diff=0.0,
                    camelot_dist=None, voice="sem stems")
    assert si.set_score([tr], {"bpm": 1.0, "tom": 5.0, "voz": 5.0}) == 50
    tr2 = Transition(from_id="a", to_id="b", cost=0.3, bpm_diff=2.4, energy_diff=0.0,
                     camelot_dist=2, voice="limpa")
    assert si.set_score([tr2], {"bpm": 0.5, "tom": 0.25, "voz": 0.25}) == 50
    zero = {"bpm": 0, "energia": 0, "tom": 0, "voz": 0}
    assert si.set_score([tr], zero) == 100
    bad = Transition(from_id="a", to_id="b", cost=float("nan"), bpm_diff=0, energy_diff=0,
                     camelot_dist=None)
    assert si.set_score([bad], {"bpm": 1.0}) == 0


def test_set_score_weights_none_reads_config(monkeypatch):
    tr = Transition(from_id="a", to_id="b", cost=0.6, bpm_diff=4.8, energy_diff=0.0,
                    camelot_dist=None, voice="sem stems")
    monkeypatch.setattr(config, "DEFAULT_WEIGHTS", {"bpm": 2.0})
    assert si.set_score([tr], None) == 50


def test_score_verdict():
    assert si.score_verdict(100)[1] == "ok"
    assert si.score_verdict(60)[1] == "good"
    assert si.score_verdict(40)[1] == "warn"
    assert si.score_verdict(0)[1] == "bad"


def test_compare_orders():
    an = {v.video_id: v for v in (make("a", bpm=124, e_mean=0.1), make("b", bpm=150, e_mean=0.2),
                                  make("c", bpm=126, e_mean=0.3), make("d", bpm=148, e_mean=0.4))}
    baseline = ["a", "b", "c", "d"]
    optimized = order.order_set(list(an.values()), {"bpm": 1.0}).order
    res = si.compare_orders(baseline, optimized, an, {"bpm": 1.0})
    assert set(res) == {"baseline_score", "optimized_score", "improvement_pct", "baseline_cost",
                        "optimized_cost"}
    assert res["baseline_cost"] == pytest.approx((26 + 24 + 22) / config.BPM_DIFF_NORM)
    assert res["optimized_cost"] == pytest.approx((2 + 22 + 2) / config.BPM_DIFF_NORM)
    assert res["improvement_pct"] == pytest.approx(round(100 * (72 - 26) / 72, 1))
    assert res["optimized_score"] > res["baseline_score"]
    assert isinstance(res["baseline_score"], int)
    same = si.compare_orders(optimized, optimized, an, {"bpm": 1.0})
    assert same["improvement_pct"] == 0.0


def test_compare_orders_zero_baseline_and_energy_mode():
    an = {v.video_id: v for v in (make("a", e_mean=0.9), make("b", e_mean=0.1))}
    res = si.compare_orders(["a", "b"], ["b", "a"], an, {"energia": 1.0})
    assert res["baseline_cost"] == 0.0 and res["improvement_pct"] == 0.0
    rising = si.compare_orders(["a", "b"], ["b", "a"], an, {"energia": 1.0}, energy_mode="subindo")
    assert rising["baseline_cost"] == pytest.approx(0.8)
    assert rising["optimized_cost"] == pytest.approx(0.0)
    assert rising["improvement_pct"] == pytest.approx(100.0)


def test_set_summary():
    an = {
        "a": make("a", bpm=126, e_mean=0.3, camelot="8A", duration_s=170),
        "b": make("b", bpm=130, bpm_manual=132, e_mean=0.9, camelot="9A", duration_s=200),
        "c": make("c", bpm=140, e_mean=0.6, camelot="8A", duration_s=150),
        "d": make("d", bpm=128, e_mean=0.5, camelot=None, duration_s=180),
    }
    s = si.set_summary(["a", "b", "zzz", "c", "d"], an)
    assert s["n"] == 4
    assert s["duration_s"] == pytest.approx(700.0)
    assert (s["bpm_min"], s["bpm_max"]) == (126, 140)
    assert s["bpm_avg"] == pytest.approx((126 + 132 + 140 + 128) / 4)
    assert s["keys"] == [("8A", 2), ("9A", 1)]
    assert s["energy_start"] == pytest.approx(0.3)
    assert s["energy_end"] == pytest.approx(0.5)
    assert s["peak_pos"] == 2


def test_set_summary_empty_and_ties():
    s = si.set_summary([], {})
    assert s["n"] == 0 and s["duration_s"] == 0.0 and s["keys"] == []
    assert s["bpm_min"] is None and s["peak_pos"] is None and s["energy_start"] is None
    an = {"a": make("a", e_mean=0.7), "b": make("b", e_mean=0.7)}
    assert si.set_summary(["b", "a"], an)["peak_pos"] == 1      # empate: o primeiro pico
    an["c"] = make("c", e_mean=float("nan"))
    assert si.set_summary(["c", "a"], an)["peak_pos"] == 2


def test_label_counts():
    def t(label):
        return Transition(from_id="a", to_id="b", cost=0, bpm_diff=0, energy_diff=0,
                          camelot_dist=None, label=label)

    assert si.label_counts([]) == {"suave": 0, "boa": 0, "atenção": 0, "difícil": 0}
    counts = si.label_counts([t("suave"), t("suave"), t("difícil"), t("")])
    assert list(counts)[:4] == ["suave", "boa", "atenção", "difícil"]
    assert counts == {"suave": 2, "boa": 0, "atenção": 0, "difícil": 1}


def test_label_tones_cover_all_labels():
    for _, label in config.TRANSITION_LABELS:
        assert si.LABEL_TONES[label] in si.TONES
    assert not math.isnan(si.transition_score(Transition("a", "b", 0.0, 0.0, 0.0, None), W))
