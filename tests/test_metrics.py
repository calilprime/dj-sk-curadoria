"""Testes da medição de tempo (seção 6)."""
from __future__ import annotations

import json
import math

import pytest

import config
from modules.metrics import Comparison, SessionMetrics, compare_to_baseline

T0 = 1_760_000_000.0  # um instante qualquer de 2025 (evita timestamps inválidos no Windows)


class FakeClock:
    def __init__(self, t: float = T0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture(autouse=True)
def baseline(monkeypatch):
    # Fixa a linha de base do contrato, independentemente de ajustes futuros no config.
    monkeypatch.setattr(config, "BASELINE_SECONDS_PER_TRACK", (60.0, 90.0))
    monkeypatch.setattr(config, "TYPICAL_SET_SIZE", 100)


def test_compare_15s_per_track_is_4x_to_6x():
    c = compare_to_baseline(15.0)
    assert isinstance(c, Comparison)
    assert c.seconds_per_track == 15.0
    assert (c.baseline_min_s, c.baseline_max_s) == (60.0, 90.0)
    assert c.speedup_min == pytest.approx(4.0)
    assert c.speedup_max == pytest.approx(6.0)
    assert c.saved_per_track_min_s == pytest.approx(45.0)
    assert c.saved_per_track_max_s == pytest.approx(75.0)
    assert c.set_size == 100
    assert c.set_minutes_panel == pytest.approx(25.0)          # 100 × 15 s
    assert c.set_minutes_baseline_min == pytest.approx(100.0)  # 100 × 60 s
    assert c.set_minutes_baseline_max == pytest.approx(150.0)  # 100 × 90 s
    assert c.to_dict()["speedup_max"] == pytest.approx(6.0)


def test_compare_slower_than_baseline_gives_negative_savings():
    c = compare_to_baseline(120.0, set_size=10)
    assert c.speedup_min == pytest.approx(0.5)
    assert c.saved_per_track_min_s == pytest.approx(-60.0)
    assert c.set_size == 10
    assert c.set_minutes_panel == pytest.approx(20.0)


@pytest.mark.parametrize("spt", [0, 0.0, -5.0, float("nan"), float("inf"), None])
def test_compare_without_measurement_does_not_divide_by_zero(spt):
    c = compare_to_baseline(spt)
    assert c.seconds_per_track == 0.0
    assert c.speedup_min == c.speedup_max == 0.0
    assert c.saved_per_track_min_s == c.saved_per_track_max_s == 0.0
    assert c.set_minutes_panel == 0.0
    # A linha de base continua visível ao lado.
    assert c.set_minutes_baseline_min == pytest.approx(100.0)
    assert all(math.isfinite(v) for v in c.to_dict().values() if isinstance(v, float))


def test_compare_reads_config_at_call_time(monkeypatch):
    monkeypatch.setattr(config, "TYPICAL_SET_SIZE", 40)
    monkeypatch.setattr(config, "BASELINE_SECONDS_PER_TRACK", (30.0, 45.0))
    c = compare_to_baseline(15.0)
    assert c.set_size == 40
    assert (c.speedup_min, c.speedup_max) == (pytest.approx(2.0), pytest.approx(3.0))
    assert compare_to_baseline(15.0, set_size=0).set_minutes_panel == 0.0


def test_session_20_approved_in_300s(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    assert m.path == tmp_dirs["DATA_DIR"] / "metricas.json"
    assert not m.started and m.elapsed_s() == 0.0
    m.start()
    assert m.started
    clock.advance(120)
    assert m.elapsed_s() == pytest.approx(120.0)
    clock.advance(180)

    rec = m.finish(n_approved=20, n_reviewed=35)
    assert {"inicio", "fim", "duracao_s", "aprovadas", "revisadas", "s_por_faixa", "comparacao"} <= set(rec)
    assert rec["duracao_s"] == pytest.approx(300.0)
    assert rec["aprovadas"] == 20 and rec["revisadas"] == 35
    assert rec["s_por_faixa"] == pytest.approx(15.0)
    comp = rec["comparacao"]
    assert 4.0 <= comp["speedup_min"] <= comp["speedup_max"] <= 6.0
    assert comp["speedup_min"] == pytest.approx(4.0) and comp["speedup_max"] == pytest.approx(6.0)
    assert comp["set_size"] == 100 and comp["set_minutes_panel"] == pytest.approx(25.0)
    assert rec["inicio"] < rec["fim"]

    # O cronômetro para na exportação.
    clock.advance(1000)
    assert m.elapsed_s() == pytest.approx(300.0)

    saved = json.loads(m.path.read_text(encoding="utf-8"))
    assert isinstance(saved, list) and len(saved) == 1
    assert saved[0]["s_por_faixa"] == pytest.approx(15.0)
    assert m.history() == saved


def test_finish_twice_updates_same_record(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(300)
    m.finish(20, 30)
    clock.advance(100)          # reexportou depois de ajustar a ordem
    rec = m.finish(20, 30)
    hist = m.history()
    assert len(hist) == 1
    assert hist[0]["duracao_s"] == pytest.approx(400.0)
    assert rec["s_por_faixa"] == pytest.approx(20.0)

    # Buscar sugestões de novo abre uma sessão nova no histórico.
    m.start()
    clock.advance(60)
    m.finish(6, 10)
    hist = m.history()
    assert len(hist) == 2
    assert hist[0]["id"] != hist[1]["id"]
    assert hist[1]["s_por_faixa"] == pytest.approx(10.0)


def test_history_is_shared_between_instances(tmp_dirs, clock):
    a = SessionMetrics(clock=clock)
    a.start()
    clock.advance(50)
    a.finish(5, 5)
    b = SessionMetrics(clock=clock)
    b.start()
    clock.advance(30)
    b.finish(3, 4)
    assert len(SessionMetrics().history()) == 2


def test_finish_with_zero_approved_does_not_divide(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(200)
    rec = m.finish(0, 12)
    assert rec["s_por_faixa"] == 0.0
    assert rec["comparacao"]["speedup_min"] == 0.0
    assert rec["duracao_s"] == pytest.approx(200.0)
    assert len(m.history()) == 1


def test_finish_without_start_is_not_recorded(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    rec = m.finish(10, 10)
    assert rec["duracao_s"] == 0.0 and rec["s_por_faixa"] == 0.0
    assert not m.path.exists()
    assert m.history() == []


def test_start_again_before_finish_restarts_timer(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(500)
    m.start()                   # nova busca de sugestões antes de exportar
    clock.advance(100)
    assert m.elapsed_s() == pytest.approx(100.0)
    assert m.finish(10, 10)["duracao_s"] == pytest.approx(100.0)


def test_custom_path_and_default_path_follow_config(tmp_path, tmp_dirs, monkeypatch, clock):
    custom = tmp_path / "outro" / "hist.json"
    m = SessionMetrics(path=custom, clock=clock)
    m.start()
    clock.advance(10)
    m.finish(1, 1)
    assert custom.exists()
    assert not (tmp_dirs["DATA_DIR"] / "metricas.json").exists()

    default = SessionMetrics(clock=clock)
    other = tmp_path / "data2"
    monkeypatch.setattr(config, "DATA_DIR", other)
    assert default.path == other / "metricas.json"


def test_corrupted_history_is_quarantined(tmp_dirs, clock):
    path = tmp_dirs["DATA_DIR"] / "metricas.json"
    path.write_text("{isto não é json", encoding="utf-8")
    m = SessionMetrics(clock=clock)
    assert m.history() == []
    m.start()
    clock.advance(30)
    m.finish(3, 3)
    assert len(m.history()) == 1
    backup = path.with_name("metricas.json.corrompido")
    assert backup.read_text(encoding="utf-8") == "{isto não é json"


def test_history_ignores_non_dict_entries(tmp_dirs):
    path = tmp_dirs["DATA_DIR"] / "metricas.json"
    path.write_text(json.dumps([{"id": "a"}, 3, "x", None]), encoding="utf-8")
    assert SessionMetrics().history() == [{"id": "a"}]


def test_to_dict_and_restore_keep_the_timer(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(40)
    snap = json.loads(json.dumps(m.to_dict()))   # passa pelo JSON da sessão salva

    clock.advance(20)
    m2 = SessionMetrics.from_dict(snap, clock=clock)
    assert m2.started and not m2.finished
    assert m2.elapsed_s() == pytest.approx(60.0)
    m2.finish(6, 6)
    m.finish(6, 6)              # mesma sessão: atualiza o registro em vez de duplicar
    assert len(m2.history()) == 1


@pytest.mark.parametrize("bad", [None, [], "x", {"start": "abc"}, {"start": None, "end": 5}])
def test_restore_ignores_invalid_data(tmp_dirs, clock, bad):
    m = SessionMetrics.from_dict(bad, clock=clock)
    assert not m.started
    assert m.elapsed_s() == 0.0


def test_history_survives_read_error(tmp_dirs, clock, monkeypatch, caplog):
    """Leitura que falha (arquivo travado pelo antivírus, por exemplo) não pode apagar o histórico."""
    import builtins

    from modules import metrics as metrics_mod

    m = SessionMetrics(clock=clock)
    for n in (5, 6):
        m.start()
        clock.advance(60)
        m.finish(n, n)
    before = m.path.read_bytes()

    real_open = builtins.open
    locked = {"on": True}

    def locked_open(file, *a, **kw):
        if locked["on"] and str(file) == str(m.path):
            raise PermissionError(13, "arquivo em uso")
        return real_open(file, *a, **kw)

    monkeypatch.setattr(metrics_mod, "open", locked_open, raising=False)
    m.start()
    clock.advance(30)
    rec = m.finish(3, 3)                      # não derruba o app
    assert rec["s_por_faixa"] == pytest.approx(10.0)
    locked["on"] = False
    assert m.path.read_bytes() == before      # as duas sessões antigas continuam lá
    assert len(m.history()) == 2
    assert "Não foi possível" in caplog.text


def test_history_read_waits_for_writer(tmp_dirs, clock):
    """No Windows, ler o arquivo durante a troca atômica faz a gravação falhar: a leitura espera o lock."""
    import threading

    from modules import metrics as metrics_mod

    got: list = []
    with metrics_mod._lock:
        th = threading.Thread(target=lambda: got.append(SessionMetrics(clock=clock).history()))
        th.start()
        th.join(0.2)
        assert th.is_alive()                  # esperando o lock
    th.join(5)
    assert got == [[]]


def test_weird_clock_values_do_not_crash(tmp_dirs):
    clock = FakeClock(-100.0)   # relógio fora da faixa aceita pelo Windows
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(10)
    rec = m.finish(1, 1)
    assert rec["duracao_s"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Decisões da triagem e números ao vivo
# ---------------------------------------------------------------------------
def _triage(m: SessionMetrics, clock: FakeClock) -> None:
    """Início em T0; decisões a +5, +10, +15 e +60 s (intervalos 5, 5, 5, 45)."""
    m.start()
    for gap, vid, dec in ((5, "v1", "aprovada"), (5, "v2", "reprovada"),
                          (5, "v3", "aprovada"), (45, "v4", "aprovada")):
        clock.advance(gap)
        m.record_decision(vid, dec)


def test_record_decision_keeps_time_and_replaces_previous(clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(10)
    m.record_decision("a", "aprovada")
    clock.advance(5)
    m.record_decision("b", "reprovada")
    assert m.decisions == {"a": ("aprovada", T0 + 10), "b": ("reprovada", T0 + 15)}

    clock.advance(5)
    m.record_decision("a", "reprovada")          # mudou de ideia: substitui
    assert m.decisions["a"] == ("reprovada", T0 + 20)
    assert list(m.decisions) == ["b", "a"]        # ordem segue a hora da decisão
    assert len(m.decisions) == 2

    m.decisions["zzz"] = ("aprovada", 0.0)        # é uma cópia
    assert "zzz" not in m.decisions

    assert m.undo_decision("a") is True
    assert m.undo_decision("a") is False          # nada para desfazer
    assert m.undo_decision("nunca-decidida") is False
    assert m.decisions == {"b": ("reprovada", T0 + 15)}


@pytest.mark.parametrize("vid,dec", [("", "aprovada"), (None, "aprovada"), ("a", "talvez"), ("a", None)])
def test_record_decision_rejects_invalid_input(clock, vid, dec):
    m = SessionMetrics(clock=clock)
    with pytest.raises(ValueError):
        m.record_decision(vid, dec)
    assert m.decisions == {}


def test_record_decision_accepts_case_and_spaces(clock):
    m = SessionMetrics(clock=clock)
    m.record_decision(" a ", " Aprovada ")
    assert m.decisions == {"a": ("aprovada", T0)}


def test_start_opens_session_without_old_decisions(clock):
    m = SessionMetrics(clock=clock)
    _triage(m, clock)
    assert len(m.decisions) == 4
    m.start()
    assert m.decisions == {}


def test_live_stats_full_session(clock):
    m = SessionMetrics(clock=clock)
    _triage(m, clock)
    s = m.live_stats(n_suggested=20, n_downloaded=12, spent_brl=0.6)

    assert s["started"] is True and s["finished"] is False
    assert s["elapsed_s"] == pytest.approx(60.0)
    assert (s["decisions"], s["approved"], s["rejected"]) == (4, 3, 1)
    assert s["approval_rate"] == pytest.approx(0.75)
    assert s["avg_decision_s"] == pytest.approx(15.0)       # (5+5+5+45)/4
    assert s["median_decision_s"] == pytest.approx(5.0)
    assert s["s_per_approved"] == pytest.approx(20.0)       # 60 s / 3 aprovadas
    assert (s["baseline_lo_s"], s["baseline_hi_s"]) == (60.0, 90.0)
    assert s["speedup_lo"] == pytest.approx(3.0)
    assert s["speedup_hi"] == pytest.approx(4.5)
    assert s["saved_min_lo"] == pytest.approx(2.0)          # (3×60 − 60) / 60
    assert s["saved_min_hi"] == pytest.approx(3.5)          # (3×90 − 60) / 60
    assert s["saved_min_lo_raw"] == pytest.approx(2.0)
    proj = s["set_projection"]
    assert proj["size"] == 100
    assert proj["panel_min"] == pytest.approx(20.0 * 100 / 60)
    assert (proj["manual_lo_min"], proj["manual_hi_min"]) == (pytest.approx(100.0), pytest.approx(150.0))
    assert s["cost_brl"] == pytest.approx(0.6)
    assert s["cost_per_approved_brl"] == pytest.approx(0.2)
    assert s["funnel"] == [("Sugeridas", 20), ("Baixadas", 12), ("Decididas", 4),
                           ("Aprovadas", 3), ("Exportadas", 0)]

    s2 = m.live_stats(20, 12, n_exported=3, set_size=40)
    assert s2["funnel"][-1] == ("Exportadas", 3)
    assert s2["set_projection"]["size"] == 40
    assert s2["set_projection"]["panel_min"] == pytest.approx(20.0 * 40 / 60)
    assert s2["cost_brl"] == 0.0 and s2["cost_per_approved_brl"] == 0.0


def test_live_stats_follows_changed_decision(clock):
    m = SessionMetrics(clock=clock)
    _triage(m, clock)
    clock.advance(20)
    m.record_decision("v2", "aprovada")           # a reprovada virou aprovada
    s = m.live_stats(20, 12)
    assert (s["decisions"], s["approved"], s["rejected"]) == (4, 4, 0)
    assert s["approval_rate"] == pytest.approx(1.0)
    # horas agora: +5 (v1), +15 (v3), +60 (v4), +80 (v2) → intervalos 5, 10, 45, 20
    assert s["avg_decision_s"] == pytest.approx(20.0)
    assert s["median_decision_s"] == pytest.approx(15.0)
    m.undo_decision("v4")
    assert m.live_stats(20, 12)["approved"] == 3


def test_live_stats_is_safe_with_nothing_measured(clock):
    m = SessionMetrics(clock=clock)
    s = m.live_stats(0, 0)
    assert s["started"] is False and s["elapsed_s"] == 0.0
    assert (s["decisions"], s["approved"], s["rejected"]) == (0, 0, 0)
    for key in ("approval_rate", "avg_decision_s", "median_decision_s", "s_per_approved",
                "speedup_lo", "speedup_hi", "cost_per_approved_brl"):
        assert s[key] is None, key
    assert s["saved_min_lo"] == s["saved_min_hi"] == 0.0
    assert s["set_projection"]["panel_min"] is None
    assert s["set_projection"]["manual_lo_min"] == pytest.approx(100.0)
    assert [n for _, n in s["funnel"]] == [0, 0, 0, 0, 0]

    # Começou mas ainda não decidiu nada.
    m.start()
    clock.advance(30)
    s = m.live_stats(20, 0, spent_brl=1.0)
    assert s["elapsed_s"] == pytest.approx(30.0)
    assert s["approval_rate"] is None and s["s_per_approved"] is None
    assert s["saved_min_lo"] == 0.0
    assert s["saved_min_lo_raw"] == pytest.approx(-0.5)      # 30 s gastos sem aprovar nada
    assert s["cost_per_approved_brl"] is None


def test_live_stats_only_rejections(clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(12)
    m.record_decision("a", "reprovada")
    s = m.live_stats(5, 5)
    assert s["approval_rate"] == 0.0
    assert s["avg_decision_s"] == pytest.approx(12.0)
    assert s["s_per_approved"] is None and s["speedup_lo"] is None


def test_live_stats_slower_than_manual_shows_zero_saved(clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(300)
    m.record_decision("a", "aprovada")                       # 300 s para 1 faixa
    s = m.live_stats(1, 1)
    assert s["speedup_lo"] == pytest.approx(0.2)
    assert s["saved_min_lo"] == 0.0 and s["saved_min_hi"] == 0.0
    assert s["saved_min_lo_raw"] == pytest.approx(-4.0)      # (60 − 300) / 60
    assert s["saved_min_hi_raw"] == pytest.approx(-3.5)


def test_live_stats_without_start_uses_only_gaps_between_decisions(clock):
    m = SessionMetrics(clock=clock)
    m.record_decision("a", "aprovada")
    s = m.live_stats(1, 1)
    assert s["avg_decision_s"] is None                       # sem início não há 1º intervalo
    clock.advance(8)
    m.record_decision("b", "aprovada")
    s = m.live_stats(2, 2)
    assert s["avg_decision_s"] == pytest.approx(8.0)
    assert s["elapsed_s"] == 0.0 and s["s_per_approved"] is None


@pytest.mark.parametrize("bad", [None, "x", float("nan"), float("inf"), -3, -2.5])
def test_live_stats_ignores_bad_numbers(clock, bad):
    m = SessionMetrics(clock=clock)
    _triage(m, clock)
    s = m.live_stats(bad, bad, n_exported=bad, spent_brl=bad, set_size=-1 if bad == -3 else None)
    assert s["cost_brl"] == 0.0
    assert all(isinstance(n, int) and n >= 0 for _, n in s["funnel"])
    assert s["set_projection"]["size"] >= 0
    for v in (s["avg_decision_s"], s["s_per_approved"], s["saved_min_lo"], s["cost_brl"]):
        assert math.isfinite(v)


def test_live_stats_freezes_at_export(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    _triage(m, clock)
    m.finish(3, 4)
    clock.advance(500)
    s = m.live_stats(20, 12, n_exported=3)
    assert s["finished"] is True
    assert s["elapsed_s"] == pytest.approx(60.0)
    assert s["s_per_approved"] == pytest.approx(20.0)


def test_finish_records_decision_numbers(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    _triage(m, clock)
    rec = m.finish(n_approved=3, n_reviewed=4)
    # As chaves antigas continuam lá.
    assert {"id", "inicio", "fim", "duracao_s", "aprovadas", "revisadas", "s_por_faixa",
            "comparacao"} <= set(rec)
    assert rec["decisoes"] == 4 and rec["reprovadas"] == 1
    assert rec["taxa_aprovacao"] == pytest.approx(0.75)
    assert rec["s_por_decisao"] == pytest.approx(15.0)
    saved = m.history()[0]
    assert saved["decisoes"] == 4 and saved["taxa_aprovacao"] == pytest.approx(0.75)


def test_finish_without_recorded_decisions_uses_given_counts(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(100)
    rec = m.finish(n_approved=6, n_reviewed=10)
    assert rec["decisoes"] == 10 and rec["reprovadas"] == 4
    assert rec["taxa_aprovacao"] == pytest.approx(0.6)
    assert rec["s_por_decisao"] == pytest.approx(10.0)       # 100 s / 10 decisões


def test_finish_with_no_decisions_at_all(tmp_dirs, clock):
    m = SessionMetrics(clock=clock)
    m.start()
    clock.advance(40)
    rec = m.finish(0, 0)
    assert rec["decisoes"] == 0 and rec["reprovadas"] == 0
    assert rec["taxa_aprovacao"] is None and rec["s_por_decisao"] is None
    rec2 = SessionMetrics(clock=clock).finish(0, 0)          # sem start
    assert rec2["taxa_aprovacao"] is None and rec2["s_por_decisao"] is None
    json.dumps(rec)                                          # continua gravável


def test_to_dict_and_restore_keep_decisions(clock):
    m = SessionMetrics(clock=clock)
    _triage(m, clock)
    snap = json.loads(json.dumps(m.to_dict()))
    assert snap["decisions"]["v2"] == ["reprovada", T0 + 10]

    m2 = SessionMetrics.from_dict(snap, clock=clock)
    assert m2.decisions == m.decisions
    assert list(m2.decisions) == list(m.decisions)
    assert m2.live_stats(20, 12)["avg_decision_s"] == pytest.approx(15.0)


def test_restore_old_snapshot_without_decisions(clock):
    old = {"session_id": "abc", "start": T0, "end": None}   # formato antes das decisões
    m = SessionMetrics.from_dict(old, clock=clock)
    assert m.started and m.decisions == {}
    clock.advance(10)
    assert m.live_stats(1, 1)["elapsed_s"] == pytest.approx(10.0)


def test_restore_drops_invalid_decisions(clock):
    snap = {"session_id": "abc", "start": T0, "end": None, "decisions": {
        "ok": ["aprovada", T0 + 3],
        "dict": {"decision": "reprovada", "ts": T0 + 1},
        "ruim1": ["talvez", T0],
        "ruim2": ["aprovada", "abc"],
        "ruim3": "aprovada",
        "ruim4": ["aprovada"],
        "ruim5": ["aprovada", float("nan")],
        "": ["aprovada", T0],
    }}
    m = SessionMetrics.from_dict(snap, clock=clock)
    assert m.decisions == {"dict": ("reprovada", T0 + 1), "ok": ("aprovada", T0 + 3)}
    assert list(m.decisions) == ["dict", "ok"]               # reordenado pela hora

    m.restore({"start": T0, "decisions": "lixo"})
    assert m.decisions == {}
    m.restore({"start": None, "decisions": {"a": ["aprovada", T0]}})
    assert m.decisions == {}                                 # sem sessão, sem decisões


# ---------------------------------------------------------------------------
# Histórico: apagar e resumir
# ---------------------------------------------------------------------------
def _three_sessions(clock: FakeClock) -> SessionMetrics:
    m = SessionMetrics(clock=clock)
    for duration, approved in ((300, 20), (120, 0), (100, 10)):
        m.start()
        clock.advance(duration)
        m.finish(approved, approved + 2)
    return m


def test_clear_history_returns_number_of_sessions(tmp_dirs, clock):
    from modules.metrics import clear_history, load_history

    m = _three_sessions(clock)
    assert len(load_history()) == 3
    assert clear_history() == 3
    assert not m.path.exists()
    assert m.history() == [] and load_history() == []
    assert clear_history() == 0                              # já vazio: sem erro

    # Depois de zerar, a próxima exportação começa um histórico novo.
    m.start()
    clock.advance(10)
    m.finish(1, 1)
    assert len(m.history()) == 1


def test_clear_history_custom_path_and_corrupted_file(tmp_path, tmp_dirs, clock):
    from modules.metrics import clear_history

    custom = tmp_path / "x" / "hist.json"
    m = SessionMetrics(path=custom, clock=clock)
    m.start()
    clock.advance(5)
    m.finish(1, 1)
    default_file = tmp_dirs["DATA_DIR"] / "metricas.json"
    default_file.write_text("[]", encoding="utf-8")
    assert clear_history(custom) == 1
    assert not custom.exists() and default_file.exists()     # só apaga o que foi pedido

    default_file.write_text("{corrompido", encoding="utf-8")
    assert clear_history() == 0
    assert not default_file.exists()


def test_clear_history_on_locked_file_raises_oserror(tmp_dirs, clock, monkeypatch):
    from pathlib import Path

    from modules.metrics import clear_history

    m = _three_sessions(clock)

    def locked(self, missing_ok=False):
        raise PermissionError(13, "arquivo em uso")

    with monkeypatch.context() as mp:            # só o unlink; as pastas do tmp_dirs continuam
        mp.setattr(Path, "unlink", locked)
        with pytest.raises(OSError):
            clear_history()
    assert len(m.history()) == 3


def test_history_stats(tmp_dirs, clock):
    from modules.metrics import history_stats

    m = _three_sessions(clock)
    stats = history_stats(m.history())
    assert stats["sessions"] == 3
    assert stats["total_approved"] == 30
    assert stats["total_decisions"] == 22 + 2 + 12           # decisoes = revisadas (sem record_decision)
    assert stats["total_minutes"] == pytest.approx(520 / 60)
    # (20×60 − 300) + (0 − 120) + (10×60 − 100) = 900 − 120 + 500 = 1280 s
    assert stats["total_saved_min_lo"] == pytest.approx(1280 / 60)
    # (20×90 − 300) + (0 − 120) + (10×90 − 100) = 1500 − 120 + 800 = 2180 s
    assert stats["total_saved_min_hi"] == pytest.approx(2180 / 60)
    assert stats["best_s_per_track"] == pytest.approx(10.0)
    assert stats["avg_s_per_track"] == pytest.approx(520 / 30)
    assert stats["trend"] == [(1, pytest.approx(15.0)), (3, pytest.approx(10.0))]


def test_history_stats_is_robust():
    from modules.metrics import history_stats

    empty = history_stats([])
    assert empty == {"sessions": 0, "total_approved": 0, "total_decisions": 0, "total_minutes": 0.0,
                     "total_saved_min_lo": 0.0, "total_saved_min_hi": 0.0,
                     "best_s_per_track": None, "avg_s_per_track": None, "trend": []}
    assert history_stats(None)["sessions"] == 0

    messy = [
        {"aprovadas": "abc", "duracao_s": None},
        "não é sessão",
        {"aprovadas": 4, "duracao_s": 40},                   # sem s_por_faixa: calcula
        {"aprovadas": 2, "duracao_s": float("nan"), "s_por_faixa": None},
        {"aprovadas": 1, "duracao_s": 1000, "s_por_faixa": 1000},  # bem mais lento que o manual
    ]
    s = history_stats(messy)
    assert s["sessions"] == 4
    assert s["total_approved"] == 7
    assert s["trend"] == [(2, pytest.approx(10.0)), (4, pytest.approx(1000.0))]
    assert s["best_s_per_track"] == pytest.approx(10.0)
    # 4×60 − 40 + 2×60 − 0 + 60 − 1000 = 200 + 120 − 940 < 0 → nunca negativo
    assert s["total_saved_min_lo"] == 0.0
