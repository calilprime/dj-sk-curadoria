"""Testes do "Zerar painel" (modules/maintenance.py)."""
from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

import config
from modules import maintenance
from modules.maintenance import CATEGORIES, DEFAULT_RESET, ResetReport, reset, usage


@pytest.fixture
def base(tmp_dirs, tmp_path, monkeypatch):
    """Projeto falso em tmp_path: pastas do config + arquivos que nunca podem ser apagados."""
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    setlists = tmp_path / "Setlists"
    setlists.mkdir()
    monkeypatch.setattr(config, "SETLISTS_DIR", setlists)
    return tmp_path


def _write(path: Path, size: int = 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    return path


def _populate(root: Path) -> dict[str, list[Path]]:
    """Cria arquivos de todas as categorias e os protegidos. Devolve {categoria: [arquivos]}."""
    cache, stems = Path(config.CACHE_DIR), Path(config.STEMS_DIR)
    data, setlists, legacy = Path(config.DATA_DIR), Path(config.SETLISTS_DIR), Path(config.EXPORT_DIR)
    files = {
        "cache": [_write(cache / "abc123.mp3", 100), _write(cache / "abc123.json", 20),
                  _write(cache / "previas" / "abc123_intro.mp3", 30),
                  _write(cache / "abc123.mp3.part", 5)],
        "sessao": [_write(data / "sessao_atual.json", 40),
                   _write(data / "sessao_atual.json.corrompido", 4)],
        "metricas": [_write(data / "metricas.json", 50),
                     _write(data / "metricas.json.corrompido", 5),
                     _write(data / ".metricas.json.k2j3.tmp", 1)],
        "stems": [_write(stems / "Artista - Música" / "vocals.mp3", 60),
                  _write(stems / "Artista - Música" / "meta.json", 6), _write(stems / "_nomes.json", 3)],
        "setlists": [_write(setlists / "Baile" / "01 - Faixa.mp3", 70),
                     _write(setlists / "Baile" / "Baile.m3u8", 7),
                     _write(legacy / "Setlist_Atual.csv", 8)],
        "orcamento": [_write(data / "orcamento.json", 80),
                      _write(data / "orcamento.json.20261022-101010.corrompido", 8)],
    }
    files["_protegidos"] = [
        _write(root / ".env", 1), _write(root / "cookies.txt", 1), _write(root / ".cache", 1),
        _write(data / ".spotify_token", 1), _write(root / ".venv" / "Scripts" / "python.exe", 1),
        _write(root / "app.py", 1), _write(root / "config.py", 1), _write(root / "modules" / "metrics.py", 1),
        _write(data / "validacao_librosa.csv", 1),   # outro arquivo de data/ que não é de nenhuma categoria
    ]
    return files


def _size(paths: list[Path]) -> int:
    return sum(p.stat().st_size for p in paths)


# ---------------------------------------------------------------------------
def test_constants_cover_all_categories():
    assert set(CATEGORIES) == {"cache", "sessao", "metricas", "stems", "setlists", "orcamento"}
    assert DEFAULT_RESET == {"cache", "sessao", "metricas"}
    assert set(maintenance.LABELS) == set(CATEGORIES) == set(maintenance.DESCRIPTIONS)
    assert all(maintenance.LABELS[c] and maintenance.DESCRIPTIONS[c] for c in CATEGORIES)
    assert "dinheiro gasto na Replicate foi real" in maintenance.DESCRIPTIONS["orcamento"]


def test_usage_counts_each_category(base):
    files = _populate(base)
    u = usage()
    assert list(u) == list(CATEGORIES)
    for cat in CATEGORIES:
        assert u[cat]["files"] == len(files[cat]), cat
        assert u[cat]["bytes"] == _size(files[cat]), cat
        assert u[cat]["safe"] is True
    assert u["cache"]["path"] == str(config.CACHE_DIR)
    assert u["sessao"]["path"] == str(Path(config.DATA_DIR) / "sessao_atual.json")
    assert u["metricas"]["path"] == str(Path(config.DATA_DIR) / "metricas.json")
    assert u["orcamento"]["path"] == str(Path(config.DATA_DIR) / "orcamento.json")
    assert u["setlists"]["paths"] == [str(config.SETLISTS_DIR), str(config.EXPORT_DIR)]


def test_usage_with_missing_folders(base):
    import shutil

    shutil.rmtree(config.CACHE_DIR)
    u = usage()
    assert u["cache"] == {"files": 0, "bytes": 0, "path": str(config.CACHE_DIR),
                          "paths": [str(config.CACHE_DIR)], "safe": True}


# Bytes que _populate grava em cada categoria.
_BYTES = {"cache": 155, "sessao": 44, "metricas": 56, "stems": 69, "setlists": 85, "orcamento": 88}


def test_default_reset_clears_cache_session_and_metrics_only(base):
    files = _populate(base)
    rep = reset(DEFAULT_RESET)
    assert isinstance(rep, ResetReport)
    assert rep.ok and rep.failures == []

    for cat in ("cache", "sessao", "metricas"):
        assert not any(p.exists() for p in files[cat]), cat
        assert rep.per_category[cat] == {"files": len(files[cat]), "bytes": _BYTES[cat]}
    for cat in ("stems", "setlists", "orcamento", "_protegidos"):
        assert all(p.exists() for p in files[cat]), cat

    assert rep.removed_files == sum(len(files[c]) for c in DEFAULT_RESET)
    assert rep.freed_bytes == sum(_BYTES[c] for c in DEFAULT_RESET)
    # A pasta do cache foi recriada vazia (sem a subpasta de prévias).
    assert Path(config.CACHE_DIR).is_dir() and list(Path(config.CACHE_DIR).iterdir()) == []
    assert set(rep.per_category) == set(DEFAULT_RESET)


@pytest.mark.parametrize("cat", CATEGORIES)
def test_each_category_removes_only_its_files(base, cat):
    files = _populate(base)
    assert _size(files[cat]) == _BYTES[cat]
    rep = reset({cat})
    assert rep.ok, rep.failures
    assert rep.removed_files == len(files[cat])
    assert rep.freed_bytes == _BYTES[cat]
    assert rep.per_category == {cat: {"files": len(files[cat]), "bytes": _BYTES[cat]}}
    assert not any(p.exists() for p in files[cat])
    for other, paths in files.items():
        if other != cat:
            assert all(p.exists() for p in paths), other
    assert usage()[cat]["files"] == 0


def test_reset_everything_keeps_secrets_and_code(base):
    files = _populate(base)
    rep = reset(set(CATEGORIES))
    assert rep.ok, rep.failures
    assert rep.removed_files == sum(len(v) for k, v in files.items() if k != "_protegidos")
    assert all(p.exists() for p in files["_protegidos"])
    for d in (config.CACHE_DIR, config.STEMS_DIR, config.SETLISTS_DIR, config.EXPORT_DIR, config.DATA_DIR):
        assert Path(d).is_dir()
    assert all(u["files"] == 0 for u in usage().values())
    assert sorted(p.name for p in Path(config.DATA_DIR).iterdir()) == [".spotify_token",
                                                                      "validacao_librosa.csv"]


def test_reset_recreates_missing_folders(base):
    import shutil

    shutil.rmtree(config.STEMS_DIR)
    rep = reset({"stems", "setlists"})
    assert rep.ok and rep.removed_files == 0
    assert Path(config.STEMS_DIR).is_dir() and Path(config.SETLISTS_DIR).is_dir()


def test_unknown_category_raises_and_deletes_nothing(base):
    files = _populate(base)
    with pytest.raises(ValueError):
        reset({"cache", "tudo"})
    assert all(p.exists() for paths in files.values() for p in paths)


def test_empty_selection_and_single_string(base):
    files = _populate(base)
    rep = reset(set())
    assert rep.removed_files == 0 and rep.per_category == {}
    rep = reset("sessao")                     # uma categoria só, como texto
    assert rep.per_category["sessao"]["files"] == len(files["sessao"])


def test_folder_outside_the_project_is_never_touched(base, tmp_path_factory, monkeypatch):
    outside = tmp_path_factory.mktemp("fora_do_projeto")
    victim = _write(outside / "importante.mp3", 10)
    monkeypatch.setattr(config, "CACHE_DIR", outside)
    u = usage()
    assert u["cache"]["safe"] is False
    rep = reset({"cache"})
    assert victim.exists()
    assert rep.removed_files == 0 and not rep.ok
    assert "fora do projeto" in rep.failures[0]


def test_data_dir_outside_the_project_is_refused(base, tmp_path_factory, monkeypatch):
    outside = tmp_path_factory.mktemp("data_fora")
    victim = _write(outside / "metricas.json", 10)
    monkeypatch.setattr(config, "DATA_DIR", outside)
    rep = reset({"metricas"})
    assert victim.exists() and rep.failures


def test_project_folder_itself_is_refused(base, monkeypatch):
    files = _populate(base)
    monkeypatch.setattr(config, "CACHE_DIR", base)        # configuração errada: cache = projeto
    rep = reset({"cache"})
    assert rep.removed_files == 0 and rep.failures
    assert all(p.exists() for paths in files.values() for p in paths)


def test_folder_inside_venv_is_refused(base, monkeypatch):
    venv_lib = base / ".venv" / "Lib"
    keep = _write(venv_lib / "site.py", 1)
    other = _write(venv_lib / "dados.bin", 1)
    monkeypatch.setattr(config, "STEMS_DIR", venv_lib)
    rep = reset({"stems"})
    assert keep.exists() and other.exists() and rep.failures


def test_protected_names_inside_a_category_folder_survive(base):
    cache = Path(config.CACHE_DIR)
    env = _write(cache / ".env", 1)
    script = _write(cache / "ferramenta.py", 1)
    token = _write(cache / ".spotify_token", 1)
    song = _write(cache / "x.mp3", 1)
    rep = reset({"cache"})
    assert not song.exists()
    assert env.exists() and script.exists() and token.exists()
    assert rep.ok and rep.removed_files == 1


def test_symlink_inside_cache_removes_only_the_link(base, tmp_path_factory):
    outside = tmp_path_factory.mktemp("alvo_link")
    target_file = _write(outside / "musica.mp3", 10)
    target_dir = outside / "pasta"
    inner = _write(target_dir / "dentro.mp3", 10)
    cache = Path(config.CACHE_DIR)
    try:
        os.symlink(target_file, cache / "link.mp3")
        os.symlink(target_dir, cache / "link_pasta", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Sem permissão para criar links simbólicos nesta máquina")
    rep = reset({"cache"})
    assert not os.path.lexists(cache / "link.mp3")
    assert not os.path.lexists(cache / "link_pasta")
    assert target_file.exists() and inner.exists()
    assert rep.ok, rep.failures


@pytest.mark.skipif(sys.platform != "win32", reason="junção (atalho de pasta) só existe no Windows")
def test_junction_inside_cache_removes_only_the_link(base, tmp_path_factory):
    import subprocess

    outside = tmp_path_factory.mktemp("alvo_juncao")
    inner = _write(outside / "dentro.mp3", 10)
    junction = Path(config.CACHE_DIR) / "juncao"
    r = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        pytest.skip(f"mklink /J falhou: {r.stderr or r.stdout}")
    rep = reset({"cache"})
    assert not os.path.lexists(junction)
    assert inner.exists()
    assert rep.ok, rep.failures


def test_read_only_file_is_removed(base):
    f = _write(Path(config.CACHE_DIR) / "somente_leitura.mp3", 3)
    os.chmod(f, stat.S_IREAD)
    rep = reset({"cache"})
    assert not f.exists() and rep.ok


def test_locked_file_becomes_failure_not_exception(base, monkeypatch):
    files = _populate(base)
    locked = files["cache"][0]
    real_unlink = maintenance._unlink

    def fake_unlink(path):
        if Path(path) == locked:
            raise PermissionError(32, "O arquivo já está sendo usado por outro processo")
        real_unlink(path)

    monkeypatch.setattr(maintenance, "_unlink", fake_unlink)
    rep = reset({"cache", "metricas"})
    assert locked.exists()
    assert not rep.ok and len(rep.failures) == 1
    assert locked.name in rep.failures[0]
    assert all(not p.exists() for p in files["cache"][1:])
    assert all(not p.exists() for p in files["metricas"])
    assert rep.per_category["cache"]["files"] == len(files["cache"]) - 1
    assert "não puderam ser apagados" in rep.summary()


@pytest.mark.skipif(sys.platform != "win32", reason="no Windows um arquivo aberto não pode ser apagado")
def test_really_open_file_on_windows(base):
    f = _write(Path(config.CACHE_DIR) / "tocando.mp3", 5)
    other = _write(Path(config.CACHE_DIR) / "sub" / "outra.mp3", 5)
    with open(f, "rb"):
        rep = reset({"cache"})
        assert f.exists()
    assert not other.exists()
    assert len(rep.failures) == 1 and "tocando.mp3" in rep.failures[0]
    assert Path(config.CACHE_DIR).is_dir()


def test_reset_works_with_the_real_modules(base):
    """Depois de zerar, métricas, sessão e orçamento voltam vazios."""
    from modules import budget, state
    from modules.metrics import SessionMetrics

    m = SessionMetrics()
    m.start()
    m.finish(1, 1)
    state.save_session({"page": 2, "decisions": {"a": "aprovada"}})
    Path(config.DATA_DIR, "orcamento.json").write_text(json.dumps({"entries": []}), encoding="utf-8")
    assert m.history() and state.load_session() is not None

    rep = reset({"metricas", "sessao", "orcamento"})
    assert rep.ok, rep.failures
    assert m.history() == []
    assert state.load_session() is None
    assert not Path(config.DATA_DIR, "orcamento.json").exists()
    budget.Budget()                           # abre sem erro depois de zerado


def test_format_bytes_and_summary():
    fb = maintenance.format_bytes
    assert fb(0) == "0 B" and fb(512) == "512 B" and fb(None) == "0 B" and fb(-5) == "0 B"
    assert fb(1536) == "1,5 KB"
    assert fb(45.3 * 1024 * 1024) == "45,3 MB"
    assert fb(3 * 1024 ** 4) == "3072,0 GB"
    rep = ResetReport(removed_files=3, freed_bytes=2048)
    assert rep.summary() == "3 arquivo(s) apagado(s), 2,0 KB liberados."
    assert rep.ok
