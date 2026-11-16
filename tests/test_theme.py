"""Testes do tema visual (modules/theme.py): HTML bem formado, texto escapado, casos-limite."""
from __future__ import annotations

import re
from html.parser import HTMLParser

import pytest

from modules import theme

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"}
EVIL = '<script>alert("x")</script>'


class _Checker(HTMLParser):
    """Confere se toda tag aberta foi fechada na ordem certa."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag not in _VOID:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        if tag in _VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"fechou </{tag}> com pilha {self.stack[-3:]}")
            return
        self.stack.pop()


def parse(markup: str) -> _Checker:
    assert isinstance(markup, str) and markup
    p = _Checker()
    p.feed(markup)
    p.close()
    assert not p.errors, p.errors
    assert not p.stack, f"tags sem fechar: {p.stack}"
    return p


def assert_safe(markup: str) -> None:
    assert "<script" not in markup.lower()
    assert "&lt;script&gt;" in markup


def items(n: int, **extra) -> list[dict]:
    return [{"pos": i + 1, "title": f"Faixa {i + 1}", "artist": f"MC {i + 1}", "bpm": 128 + (i % 5) * 0.5,
             "camelot": f"{(i % 12) + 1}A", "energy": (i % 10) / 10, "thumb": f"https://i.ytimg.com/vi/{i}/hq.jpg",
             "duration": "3:05", **extra} for i in range(n)]


def links(n: int) -> list[dict]:
    labels = ["suave", "boa", "atenção", "difícil"]
    return [{"label": labels[i % 4], "tone": theme.LABEL_TONE[labels[i % 4]], "short": "BPM igual · energia sobe"}
            for i in range(max(0, n - 1))]


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
def test_palette_and_tones():
    assert theme.PALETTE["accent"] == "#FF3D7F"
    assert theme.PALETTE["accent_2"] == "#7C5CFF"
    assert set(theme.TONES) == {"ok", "good", "warn", "bad", "info", "muted", "accent"}
    for color in [*theme.PALETTE.values(), *theme.TONES.values()]:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", color)
    assert theme.LABEL_TONE == {"suave": "ok", "boa": "good", "atenção": "warn", "difícil": "bad"}


@pytest.mark.parametrize("label,tone", [("suave", "ok"), ("Boa", "good"), ("ATENÇÃO", "warn"),
                                        ("atencao", "warn"), ("dificil", "bad"), ("??", "muted"), (None, "muted")])
def test_tone_for_label(label, tone):
    assert theme.tone_for_label(label) == tone


def test_tone_color_unknown():
    assert theme.tone_color("bad") == theme.PALETTE["bad"]
    assert theme.tone_color("nao-existe") == theme.PALETTE["muted"]


# ---------------------------------------------------------------------------
# Peças pequenas
# ---------------------------------------------------------------------------
def test_esc():
    assert theme.esc(EVIL) == "&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;"
    assert theme.esc("it's") == "it&#x27;s"
    assert theme.esc(None) == ""
    assert theme.esc(12) == "12"


def test_badge_escapes_and_tones():
    out = theme.badge(EVIL, "warn", icon='"><b>')
    parse(out)
    assert_safe(out)
    assert "djsk-tone-warn" in out
    assert '"><b>' not in out
    # tom desconhecido não quebra: cai no neutro
    out2 = theme.badge("x", "nao-existe")
    parse(out2)
    assert "djsk-tone-muted" in out2
    assert "djsk-playing" in theme.badge("tocando", "accent", pulse=True)


def test_kpi_card_and_grid():
    card = theme.kpi_card(EVIL, '1.234 "faixas"', sub=EVIL, tone="ok", icon="🎧")
    parse(card)
    assert_safe(card)
    assert "&quot;faixas&quot;" in card
    assert "djsk-tone-ok" in card
    assert "djsk-tone-default" in theme.kpi_card("a", 1)
    assert "djsk-tone-default" in theme.kpi_card("a", 1, tone="roxo-neon")
    assert "djsk-kpi-sub" not in theme.kpi_card("a", 1)
    grid = theme.kpi_grid_html([card, theme.kpi_card("b", 2)], columns=3)
    parse(grid)
    assert "--djsk-cols:3" in grid
    assert "--djsk-cols:8" in theme.kpi_grid_html([card], columns=99)
    assert "--djsk-cols:1" in theme.kpi_grid_html([card], columns=0)
    assert "--djsk-cols:4" in theme.kpi_grid_html([card], columns="x")  # type: ignore[arg-type]


@pytest.mark.parametrize("value,pct", [(0.5, "50"), (-1, "0"), (3, "100"), (None, "0"), (float("nan"), "0"),
                                       ("abc", "0")])
def test_energy_bar_clamps(value, pct):
    out = theme.energy_bar(value)
    parse(out)
    assert f'aria-valuenow="{pct}"' in out


def test_energy_bar_unknown_tone():
    assert "djsk-tone-accent" in theme.energy_bar(0.3, tone="??")
    assert "djsk-tone-bad" in theme.energy_bar(0.3, tone="bad")


@pytest.mark.parametrize("value,shown", [(87, 87), (150, 100), (-20, 0), (None, 0), ("x", 0), (99.6, 100),
                                         (float("inf"), 0)])
def test_score_ring_clamps(value, shown):
    out = theme.score_ring_svg(value, "Fluidez")
    parse(out)
    assert "<svg" not in out                 # o st.html do Streamlit remove SVG: o anel é CSS
    assert f"conic-gradient(" in out and f" 0 {shown}%," in out
    assert f">{shown}</span>" in out
    assert f"nota {shown} de 100" in out


def test_score_ring_tone_and_escape():
    assert "djsk-tone-ok" in theme.score_ring_svg(90, "a")
    assert "djsk-tone-good" in theme.score_ring_svg(65, "a")
    assert "djsk-tone-warn" in theme.score_ring_svg(45, "a")
    assert "djsk-tone-bad" in theme.score_ring_svg(10, "a")
    assert "djsk-tone-info" in theme.score_ring_svg(10, "a", tone="info")
    # tom desconhecido: usa o automático
    assert "djsk-tone-ok" in theme.score_ring_svg(95, "a", tone="??")
    out = theme.score_ring_svg(50, EVIL)
    parse(out)
    assert_safe(out)
    # cor do arco = cor do tom; nota zero sem brilho
    assert theme.PALETTE["ok"] in theme.score_ring_svg(90, "a")
    assert "box-shadow" not in theme.score_ring_svg(0, "a")
    assert "width:120px" in theme.score_ring_svg(50, "a", size=120)
    assert "width:88px" in theme.score_ring_svg(50, "a", size="x")  # type: ignore[arg-type]
    rings = theme.rings_html([theme.score_ring_svg(10, "a"), theme.score_ring_svg(90, "b")])
    parse(rings)
    assert rings.count("djsk-ring ") == 2


def test_chip_order():
    assert theme.chip("BPM", "128") == '<span class="djsk-chip"><b>128</b> BPM</span>'
    assert theme.chip("tom", "8A", key=True, strong_first=False) == \
        '<span class="djsk-chip djsk-chip--key">tom <b>8A</b></span>'
    assert theme.chip(EVIL, EVIL).count("&lt;script&gt;") == 2


def test_legend():
    out = theme.legend_html()
    parse(out)
    for label, tone in theme.LABEL_TONE.items():
        assert f"djsk-tone-{tone}" in out
        assert label.capitalize() in out


# ---------------------------------------------------------------------------
# Blocos de página
# ---------------------------------------------------------------------------
def test_hero_normal():
    out = theme.hero_html("DJ SK · Curadoria", "Funk do submundo", 0.42, 10.0, 4.8, 5.0)
    parse(out)
    assert "R$ 0,42" in out and "R$ 10,00" in out
    assert "US$ 4,80" in out and "US$ 5,00" in out
    assert "Sobram R$ 9,58" in out
    assert "width:4.2%" in out
    assert "djsk-budget-alert" not in out
    assert "\n" not in out


def test_hero_states():
    warn = theme.hero_html("t", "s", 8.5, 10, 1, 5)
    assert "djsk-budget djsk-tone-warn" in warn
    over = theme.hero_html("t", "s", 12, 10, 0, 5)
    assert "djsk-budget djsk-tone-bad" in over and "width:100.0%" in over
    blocked = theme.hero_html("t", "s", 1, 10, 0, 5, insufficient=True)
    assert "djsk-budget djsk-tone-bad" in blocked and "djsk-budget-alert" in blocked
    unknown = theme.hero_html("t", None, None, 10, None, None)
    parse(unknown)
    assert "indisponível" in unknown and "djsk-tone-muted" in unknown


def test_hero_escapes_and_tags():
    out = theme.hero_html(EVIL, EVIL, 1, 10, 1, 5, kicker=EVIL, tags=[EVIL, ("Spotify", "ok", "●")])
    parse(out)
    assert_safe(out)
    assert "djsk-tone-ok" in out


def test_stepper():
    steps = [("Referências", "done", "#referencias"), ("Triagem", "active", "triagem"),
             ("Setlist", "todo", "#setlist"), (EVIL, "???", None)]
    out = theme.stepper_html(steps)
    parse(out)
    assert_safe(out)
    assert 'href="#referencias"' in out and 'href="#triagem"' in out
    assert "djsk-step--done" in out and "djsk-step--active" in out
    assert out.count("djsk-step--todo") == 2      # status desconhecido vira "todo"
    assert "✓" in out
    assert 'aria-current="step"' in out
    assert out.count("djsk-step-sep") == 3
    assert "djsk-step-sep is-done" in out
    # âncora com caracteres estranhos não injeta nada
    bad = theme.stepper_html([("x", "todo", '#"><img src=x onerror=alert(1)>')])
    p = parse(bad)
    assert '"><img' not in bad
    assert not [t for t, _ in p.tags if t == "img"]
    anchors = [a for t, a in p.tags if t == "a"]
    assert len(anchors) == 1 and set(anchors[0]) == {"class", "href"}
    assert re.fullmatch(r"#[a-z0-9_-]+", anchors[0]["href"] or "")


def test_section_header():
    out = theme.section_header_html(2, "Triagem", "Ouça e aprove", anchor="#Triagem")
    parse(out)
    assert 'id="triagem"' in out
    assert "Ouça e aprove" in out
    no_anchor = theme.section_header_html(1, EVIL)
    parse(no_anchor)
    assert_safe(no_anchor)
    assert "djsk-anchor" not in no_anchor
    assert "djsk-section-sub" not in no_anchor


# ---------------------------------------------------------------------------
# Linha do tempo
# ---------------------------------------------------------------------------
def test_timeline_empty():
    out = theme.timeline_html([], [])
    parse(out)
    assert "djsk-empty" in out
    assert "djsk-tl-card" not in out


def test_timeline_one_item():
    out = theme.timeline_html(items(1), [])
    parse(out)
    assert out.count('class="djsk-tl-card') == 1
    assert "djsk-tl-link" not in out
    assert "djsk-tl-map" not in out       # mapa só faz sentido com 2+ faixas
    assert "1 faixa<" in out


def test_timeline_thirty_items():
    out = theme.timeline_html(items(30), links(30))
    parse(out)
    assert out.count('class="djsk-tl-card') == 30
    assert out.count('class="djsk-tl-link ') == 29
    assert len(re.findall(r'class="djsk-tl-bar[" ]', out)) == 30
    assert ">01<" in out and ">30<" in out
    assert "overflow" not in out           # a rolagem vem do CSS global
    assert "≈ 1h33 de set" in out          # 30 × 3:05 = 92,5 min
    assert "128–130 BPM" in out
    for tone in ("ok", "good", "warn", "bad"):
        assert f"djsk-tl-link djsk-tone-{tone}" in out
    assert "8 suaves" in out and "7 difíceis" in out
    assert "\n" not in out


def test_timeline_css_has_horizontal_scroll():
    css = theme.css()
    block = css[css.index(".djsk-tl-track{"):]
    assert "overflow-x:auto" in block[: block.index("}")]


def test_timeline_escapes_and_bad_urls():
    data = items(2)
    data[0].update(title=EVIL, artist='"><b>x', thumb="javascript:alert(1)")
    data[1].update(thumb='https://x.com/a.jpg" onerror="alert(1)')
    out = theme.timeline_html(data, [{"label": EVIL, "short": EVIL}])
    parse(out)
    assert_safe(out)
    assert "javascript:" not in out
    p = parse(out)
    imgs = [a for t, a in p.tags if t == "img"]
    assert len(imgs) == 1
    assert "onerror" not in imgs[0]                       # aspas escapadas: não vira atributo
    assert imgs[0]["src"].startswith("https://x.com/")


def test_timeline_missing_fields_and_mismatched_links():
    data = [{}, {"title": "B", "bpm": None, "energy": "alto", "duration": None}, {"title": "C", "pos": "x"}]
    out = theme.timeline_html(data, [{"label": "boa"}])   # faltando 1 link
    parse(out)
    assert out.count('class="djsk-tl-card') == 3
    assert out.count('class="djsk-tl-link ') == 2
    assert "Sem título" in out
    assert "<b>–</b> BPM" in out
    assert "djsk-chip--key" not in out                   # sem tom: sem chip de tom
    assert "djsk-tl-link djsk-tone-muted" in out          # link ausente: neutro
    # links a mais e itens inválidos são ignorados
    out2 = theme.timeline_html([{"title": "A"}, "lixo", {"title": "B"}], links(5))  # type: ignore[list-item]
    parse(out2)
    assert out2.count('class="djsk-tl-card') == 2


def test_timeline_tone_from_label_when_missing_or_unknown():
    out = theme.timeline_html(items(3), [{"label": "difícil"}, {"label": "suave", "tone": "roxo"}])
    assert "djsk-tl-link djsk-tone-bad" in out
    assert "djsk-tl-link djsk-tone-ok" in out


def test_timeline_playing_and_numeric_duration():
    data = items(2)
    data[1]["playing"] = True
    data[0]["duration"] = 185.0
    out = theme.timeline_html(data, links(2))
    parse(out)
    assert "djsk-tl-card djsk-playing" in out
    assert "tocando" in out
    assert ">3:05<" in out


def test_timeline_three_digit_positions():
    out = theme.timeline_html(items(100), links(100), summary=False)
    parse(out)
    assert ">001<" in out and ">100<" in out
    assert "djsk-tl-summary" not in out


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def test_css_contents():
    css = theme.css()
    assert "<style" not in css and "</" not in css
    assert css.lstrip().startswith("@import")             # @import precisa vir antes de tudo
    assert "Space+Grotesk" in css and "Inter" in css
    assert "#FF3D7F" in css and "#7C5CFF" in css
    assert "@keyframes djsk-pulse" in css and "@keyframes djsk-fade-up" in css
    assert "prefers-reduced-motion" in css
    assert css.count("{") == css.count("}")
    # toda classe própria usa o prefixo
    for cls in re.findall(r"\.([a-zA-Z][\w-]*)\{", css):
        assert cls.startswith(("djsk-", "stApp", "block-container", "element-container", "is-done")), cls
