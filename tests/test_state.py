"""Testes da sessão salva em disco."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import config
from modules import state
from modules.models import Track


def make_snapshot() -> dict:
    refs = [Track("ref1", "Baile do Ó", ["MC Ção"], 190.0)]
    sugs = [Track(f"v{i}", f"Faixa {i}", ["MC Teste"], 150.0 + i, score=3 - i % 3) for i in range(5)]
    return {
        "references": [t.to_dict() for t in refs],
        "suggestions": [t.to_dict() for t in sugs],
        "page": 1,
        "decisions": {"v0": "aprovada", "v1": "reprovada", "v2": "aprovada"},
        "order": ["v2", "v0"],
        "weights": {"bpm": 1.5, "energia": 0.8, "tom": 0.0, "voz": 0.5},
        "metrics": {"session_id": "abc123", "start": 1_760_000_000.0, "end": None},
    }


def test_session_path_follows_config_at_call_time(tmp_dirs, tmp_path, monkeypatch):
    assert state.session_path() == tmp_dirs["DATA_DIR"] / "sessao_atual.json"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "outra")
    assert state.session_path() == tmp_path / "outra" / "sessao_atual.json"
    assert state.session_path(tmp_path / "x") == tmp_path / "x" / "sessao_atual.json"


def test_round_trip(tmp_dirs):
    snap = make_snapshot()
    original = json.loads(json.dumps(snap))
    state.save_session(snap)
    assert snap == original                     # não altera o dicionário recebido
    assert "saved_at" not in snap

    loaded = state.load_session()
    assert loaded is not None
    saved_at = loaded.pop("saved_at")
    assert datetime.fromisoformat(saved_at)     # ISO válido
    assert loaded == original
    assert [Track.from_dict(d) for d in loaded["suggestions"]][2].title == "Faixa 2"
    assert Track.from_dict(loaded["references"][0]).artist_str == "MC Ção"


def test_file_is_utf8_json_with_accents(tmp_dirs):
    state.save_session(make_snapshot())
    raw = state.session_path().read_bytes()
    assert "Baile do Ó".encode("utf-8") in raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    data = json.loads(raw.decode("utf-8"))
    assert data["decisions"]["v0"] == "aprovada"


def test_save_overwrites_previous(tmp_dirs):
    state.save_session(make_snapshot())
    snap = make_snapshot()
    snap["page"] = 3
    snap["decisions"]["v4"] = "aprovada"
    state.save_session(snap)
    loaded = state.load_session()
    assert loaded["page"] == 3 and loaded["decisions"]["v4"] == "aprovada"
    # Sem sobras de arquivo temporário da gravação atômica.
    assert [p.name for p in tmp_dirs["DATA_DIR"].iterdir()] == ["sessao_atual.json"]


def test_explicit_data_dir(tmp_dirs, tmp_path):
    other = tmp_path / "pasta_nova"
    state.save_session(make_snapshot(), data_dir=other)
    assert (other / "sessao_atual.json").exists()
    assert state.load_session() is None
    assert state.load_session(other)["page"] == 1
    state.clear_session(other)
    assert state.load_session(other) is None


def test_missing_file_returns_none(tmp_dirs):
    assert state.load_session() is None


@pytest.mark.parametrize("content", ["", "{quebrado", "[1, 2, 3]", "null", '"texto"', "\x00\x01"])
def test_corrupted_file_returns_none_and_is_kept_aside(tmp_dirs, content):
    path = state.session_path()
    path.write_text(content, encoding="utf-8")
    assert state.load_session() is None
    assert not path.exists()
    assert path.with_name("sessao_atual.json.corrompido").read_text(encoding="utf-8") == content
    # Depois disso dá para salvar normalmente.
    state.save_session(make_snapshot())
    assert state.load_session()["page"] == 1


def test_invalid_utf8_returns_none(tmp_dirs):
    state.session_path().write_bytes(b'{"page": "\xff\xfe"}')
    assert state.load_session() is None


def test_partial_or_invalid_fields_are_normalized(tmp_dirs):
    path = state.session_path()
    path.write_text(json.dumps({
        "references": [{"video_id": "r1", "title": "Ok"}, {"title": "sem id"}, "lixo"],
        "suggestions": "não é lista",
        "page": -2,
        "decisions": {"a": "aprovada", "b": "talvez", "c": None},
        "order": ["a", 3, None, ""],
        "weights": {"bpm": 2, "tom": "alto", "voz": True},
        "metrics": [],
        "extra_do_app": {"x": 1},
    }), encoding="utf-8")
    s = state.load_session()
    assert s["references"] == [{"video_id": "r1", "title": "Ok"}]
    assert s["suggestions"] == []
    assert s["page"] == 0
    assert s["decisions"] == {"a": "aprovada"}
    assert s["order"] == ["a"]
    assert s["weights"]["bpm"] == 2.0
    assert s["weights"]["tom"] == config.DEFAULT_WEIGHTS["tom"]   # inválido volta ao padrão
    assert s["weights"]["voz"] == config.DEFAULT_WEIGHTS["voz"]
    assert s["metrics"] == {}
    assert s["saved_at"] is None
    assert s["extra_do_app"] == {"x": 1}                          # chaves extras passam intactas


def test_empty_snapshot_gets_defaults(tmp_dirs):
    state.save_session({})
    s = state.load_session()
    assert s["references"] == [] and s["suggestions"] == [] and s["order"] == []
    assert s["page"] == 0 and s["decisions"] == {} and s["metrics"] == {}
    assert s["weights"] == config.DEFAULT_WEIGHTS
    assert isinstance(s["saved_at"], str)


def test_non_json_values_are_converted(tmp_dirs):
    snap = make_snapshot()
    snap["references"] = [Track("r9", "Direto da dataclass", ["MC"])]   # sem to_dict()
    snap["extra"] = {"pasta": Path("C:/x/y"), "ids": {"b"}, "tupla": (1, 2),
                     "bpm": np.float64(130.5), "n": np.int64(3), "nan": float("nan")}
    state.save_session(snap)
    s = state.load_session()
    assert s["references"][0]["video_id"] == "r9"
    assert s["extra"]["pasta"] == str(Path("C:/x/y"))
    assert s["extra"]["ids"] == ["b"] and s["extra"]["tupla"] == [1, 2]
    assert s["extra"]["bpm"] == 130.5 and s["extra"]["n"] == 3
    assert s["extra"]["nan"] is None


def test_numpy_arrays_become_lists(tmp_dirs):
    """Arrays numpy (curva de energia, waveform) não podem virar texto truncado com "..."."""
    snap = make_snapshot()
    curve = np.linspace(0.0, 1.0, 2000)
    snap["extra"] = {"curva": curve, "vazio": np.array([]), "um": np.array([0.5], dtype=np.float32),
                     "matriz": np.array([[1, 2], [3, 4]])}
    state.save_session(snap)
    s = state.load_session()
    assert s["extra"]["curva"] == pytest.approx(curve.tolist())
    assert s["extra"]["vazio"] == []
    assert s["extra"]["um"] == [0.5]
    assert s["extra"]["matriz"] == [[1, 2], [3, 4]]


def test_weights_are_clamped_to_slider_range(tmp_dirs, monkeypatch):
    """Peso fora da faixa do slider derrubaria o app em toda recarga (st.slider recusa o valor)."""
    monkeypatch.setattr(config, "WEIGHT_SLIDER_MAX", 2.0)
    snap = make_snapshot()
    snap["weights"] = {"bpm": 5.0, "energia": -1.0, "tom": 0.3, "voz": 2.0}
    state.save_session(snap)
    w = state.load_session()["weights"]
    assert w == {"bpm": 2.0, "energia": 0.0, "tom": 0.3, "voz": 2.0}


def test_save_error_does_not_raise(tmp_dirs, monkeypatch, caplog):
    def boom(path, obj):
        raise PermissionError("disco bloqueado")

    monkeypatch.setattr(state, "atomic_write_json", boom)
    state.save_session(make_snapshot())          # não derruba o app
    assert "Não foi possível salvar a sessão" in caplog.text
    assert state.load_session() is None


def test_clear_session(tmp_dirs):
    state.clear_session()                        # sem arquivo: não dá erro
    state.save_session(make_snapshot())
    other = tmp_dirs["DATA_DIR"] / "orcamento.json"
    other.write_text("{}", encoding="utf-8")
    state.clear_session()
    assert state.load_session() is None
    assert other.exists()                        # só apaga a sessão
