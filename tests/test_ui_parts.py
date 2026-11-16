"""Testes das peças de interface (modules/ui_parts.py) e dos gráficos (modules/ui_charts.py)."""
from __future__ import annotations

import pytest

from modules import theme, ui_charts, ui_parts


def test_kbd_legend_lists_shortcuts_and_escapes():
    out = ui_parts.kbd_legend_html([("A", "aprovar"), ("<X>", "re<provar>")], note="no <card>")
    assert out.startswith('<div class="djsk-root">') and "\n" not in out
    assert '<kbd class="djsk-kbd">A</kbd>aprovar' in out
    assert "&lt;X&gt;" in out and "re&lt;provar&gt;" in out and "no &lt;card&gt;" in out
    # as teclas reservadas pelo Streamlit (C e R) nunca são usadas
    assert not {"C", "R"} & set(ui_parts.SHORTCUTS.values())
    assert set(ui_parts.SHORTCUTS) == {"approve", "reject", "inicio", "meio", "final"}


@pytest.mark.parametrize("bpm, expected", [
    (130.0, ("zona 130 · mandelão", "accent")),
    (125.0, ("zona 130 · mandelão", "accent")),
    (150.4, ("zona 150 · acelerado", "warn")),
    (140.0, ("fora da zona do funk", "muted")),
    (None, None),
    ("abc", None),
    (0, None),
])
def test_bpm_zone(bpm, expected):
    assert ui_parts.bpm_zone(bpm) == expected


def test_badges_and_reason_tone():
    assert ui_parts.badges_html([]) == ""
    out = ui_parts.badges_html([theme.badge("x", "ok")], compact=True)
    assert "djsk-badges--compact" in out and "djsk-tone-ok" in out
    assert ui_parts.reason_tone("neutral") == "muted"
    assert ui_parts.reason_tone("warn") == "warn"
    assert ui_parts.pct(0.555) == 56 and ui_parts.pct(2) == 100 and ui_parts.pct(None) == 0


def test_ref_chips_escape_and_empty():
    assert "nenhuma ainda" in ui_parts.ref_chips_html(["", "  "])
    out = ui_parts.ref_chips_html(["MC <b>X</b> - Faixa"])
    assert "MC &lt;b&gt;X&lt;/b&gt; - Faixa" in out and "<b>X" not in out


def test_integrations_html():
    out = ui_parts.integrations_html([
        {"name": "Spotify", "purpose": "playlists", "ok": True, "detail": "conectado como <DJ>"},
        {"name": "Gemini", "purpose": "explicar", "ok": False},
    ])
    assert out.count('class="djsk-int ') == 2
    assert "ligado" in out and "desligado" in out and "conectado como &lt;DJ&gt;" in out


def test_passage_html_escapes_and_colors():
    reasons = [{"kind": "bpm", "icon": "🎯", "text": "Mesma velocidade (130 BPM)", "tone": "ok", "short": "x"},
               {"kind": "tom", "icon": "❔", "text": "Tom não identificado", "tone": "neutral", "short": ""}]
    out = ui_parts.passage_html(1, 2, 2, "MC <A>", "MC B & C", "Passagem suave: mesma velocidade", "suave",
                                reasons, explanation="Entra <limpo>")
    assert "01 → 02" in out and "djsk-tone-ok" in out and "djsk-tone-muted" in out
    assert "MC &lt;A&gt;" in out and "MC B &amp; C" in out and "Entra &lt;limpo&gt;" in out
    assert "\n" not in out


def test_headline_short_and_compare_text():
    assert ui_parts.headline_short("Passagem suave: mesma velocidade") == "mesma velocidade"
    assert ui_parts.headline_short("Passagem suave") == ""
    assert ui_parts.compare_text(None) == ("", "muted")
    text, tone = ui_parts.compare_text({"improvement_pct": 37.4})
    assert text.startswith("37% mais fácil de mixar") and tone == "ok"
    text, tone = ui_parts.compare_text({"improvement_pct": -12.0})
    assert text.startswith("12% mais difícil") and tone == "warn"
    assert ui_parts.compare_text({"improvement_pct": 0.0})[1] == "info"


def test_verdict_html():
    out = ui_parts.verdict_html("Set <redondo>", "ok", ["linha 1", ""], [theme.badge("a")])
    assert "Set &lt;redondo&gt;" in out and "djsk-tone-ok" in out and out.count("djsk-verdict-sub") == 1


def test_arc_shapes():
    assert ui_parts.arc_shape("livre", 5) is None
    assert ui_parts.arc_shape("pico", 1) is None
    assert ui_parts.arc_shape("desconhecido", 5) is None
    peak = ui_parts.arc_shape("pico", 5)
    assert peak[0] == pytest.approx(0.0) and peak[-1] == pytest.approx(0.0, abs=1e-9)
    assert max(peak) == peak[2] == pytest.approx(1.0)
    rise = ui_parts.arc_shape("subida", 3)
    assert rise == pytest.approx([0.0, 0.5, 1.0])
    waves = ui_parts.arc_shape("ondas", 7)
    assert all(0.0 <= v <= 1.0 for v in waves)


def test_arc_target_and_adherence():
    energies = [0.4, 0.5, 0.6, 0.7]
    assert ui_parts.arc_adherence(energies, "subida") == 100
    assert ui_parts.arc_adherence(list(reversed(energies)), "subida") < 50
    assert ui_parts.arc_adherence(energies, "livre") is None
    assert ui_parts.arc_adherence([0.5, None], "subida") is None
    target = ui_parts.arc_target(energies, "subida")
    assert target == pytest.approx(energies)
    flat = ui_parts.arc_target([0.5, 0.5, 0.5], "pico")
    assert min(flat) == pytest.approx(0.4) and max(flat) == pytest.approx(0.6)


def test_spotify_helpers():
    out = ui_parts.spotify_result_html({"title": "A <b>", "artists": ["MC 1", "MC 2"], "album": "Disco",
                                        "duration_s": 150})
    assert "A &lt;b&gt;" in out and "MC 1, MC 2" in out and "Disco · 2:30" in out
    assert ui_parts.spotify_embed_url("playlist", "37i9dQZF1DX") .startswith(
        "https://open.spotify.com/embed/playlist/37i9dQZF1DX")
    assert ui_parts.spotify_embed_url("playlist", "x/../evil") is None
    assert ui_parts.spotify_embed_url("artist", "abc") is None
    assert ui_parts.spotify_embed_url("track", None) is None


def test_number_formats():
    assert ui_parts.fmt_minutes(12.4) == "12 min"
    assert ui_parts.fmt_minutes(75) == "1h15"
    assert ui_parts.fmt_minutes(None) == "–"
    assert ui_parts.fmt_range_minutes(10.2, 17.8) == "10–18 min"
    assert ui_parts.fmt_range_minutes(5, 5.2) == "5 min"
    assert ui_parts.set_duration_text(2890) == "48 min"
    assert ui_parts.set_duration_text(0) == "–"


def test_charts_build_valid_specs():
    rows = [{"pos": i + 1, "faixa": f"F{i}", "bpm": 128.0 + i, "tom": "8A", "energia": 40.0 + 10 * i,
             "passagem": ["início", "suave", "boa", "difícil"][i]} for i in range(4)]
    charts = [
        ui_charts.energy_map_chart(rows, [50, 60, 70, 80]),
        ui_charts.energy_map_chart(rows[:2]),
        ui_charts.manual_vs_panel_chart(25.0, 60.0, 90.0),
        ui_charts.manual_vs_panel_chart(None, 60.0, 90.0),
        ui_charts.funnel_chart([("Sugeridas", 40), ("Baixadas", 38), ("Decididas", 20), ("Aprovadas", 12),
                                ("Exportadas", 0)]),
        ui_charts.trend_chart([(1, 40.0), (2, 31.5)], 60.0, 90.0),
    ]
    for ch in charts:
        spec = ch.to_dict()
        assert spec.get("layer"), spec
    # BPM com eixo próprio (independente) no mapa de energia
    assert charts[0].to_dict()["resolve"]["scale"]["y"] == "independent"
