"""Testes da exportação do set (seção 4.9)."""
from __future__ import annotations

import csv
import shutil
import sys
import threading
import time
import unicodedata
from pathlib import Path

import pytest
from mutagen.id3 import ID3

import config
from modules import export
from modules.export import CSV_COLUMNS, build_filename, export_setlist, short_key
from modules.models import Analysis, Track, Transition
from tests.conftest import synth_track, write_audio

FORBIDDEN = set('<>:"/\\|?*')


@pytest.fixture(scope="module")
def audio_src(tmp_path_factory) -> dict[str, Path]:
    """Um mp3 e um wav sintéticos curtos, gerados uma vez só para o módulo."""
    d = tmp_path_factory.mktemp("audio_src")
    y = synth_track(bpm=130, duration_s=2.0)
    return {"mp3": write_audio(d / "x.mp3", y), "wav": write_audio(d / "x.wav", y)}


@pytest.fixture(autouse=True)
def export_defaults(monkeypatch):
    # Fixa as opções do contrato, independentemente de ajustes futuros no config.
    monkeypatch.setattr(config, "M3U_ABSOLUTE_PATHS", True)
    monkeypatch.setattr(config, "CSV_DELIMITER", ";")
    monkeypatch.setattr(config, "CSV_ENCODING", "utf-8-sig")
    monkeypatch.setattr(config, "WRITE_ID3_TAGS", True)
    monkeypatch.setattr(config, "PLAYLIST_NAME", "Setlist_Atual")


def cache_mp3(tmp_dirs, audio_src, video_id: str) -> Path:
    dst = tmp_dirs["CACHE_DIR"] / f"{video_id}.mp3"
    shutil.copy(audio_src["mp3"], dst)
    return dst


def analysis(video_id: str, bpm: float = 130.04, **kw) -> Analysis:
    params = dict(video_id=video_id, duration_s=185.6, bpm=bpm, bpm_raw=bpm, energy_mean=0.5,
                  energy_in=0.333, energy_out=0.756, key="A minor", camelot="8A", key_confidence=0.8)
    params.update(kw)
    return Analysis(**params)


def m3u_lines(path: Path) -> list[str]:
    """Linhas do .m3u8 sem a tradução de quebras de linha do modo texto."""
    return path.read_bytes().decode("utf-8").split("\r\n")


def read_csv(path: Path) -> list[list[str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f, delimiter=";"))


def build_set(tmp_dirs, audio_src, tmp_path):
    """4 faixas exportáveis (3 do cache + 1 wav local) e 1 sem áudio no meio."""
    for vid in ("a", "b", "d"):
        cache_mp3(tmp_dirs, audio_src, vid)
    local_dir = tmp_path / "minhas_musicas"
    local_dir.mkdir()
    local_wav = local_dir / "Minha Faixa.WAV"
    shutil.copy(audio_src["wav"], local_wav)
    tracks = [
        Track("a", "Vai Descendo", ["MC Alfa"], 180.0),
        Track("b", "Tá Tranquilo", ["MC Beta", "DJ Gama"], 200.0),
        Track("c", "Sem Áudio", ["MC Fantasma"], 150.0),
        Track("local_1", "Faixa Local", ["DJ SK"], None, source="local", local_path=str(local_wav)),
        Track("d", "Final", ["MC Delta"], None),
    ]
    analyses = {
        "a": analysis("a", bpm=130.04, bpm_manual=128.0),
        "b": analysis("b", bpm=131.0, key="C# major", camelot="3B", energy_mean=0.9),
        "c": analysis("c"),
        "local_1": analysis("local_1", bpm=132.0, key=None, camelot=None, duration_s=0.0),
    }
    transitions = [
        Transition("a", "b", cost=0.12345, bpm_diff=1.0, energy_diff=0.1, camelot_dist=1,
                   voice="limpa", label="suave", explanation="Entra no\nrefrão."),
        Transition("b", "c", cost=0.5, bpm_diff=2.0, energy_diff=0.2, camelot_dist=2, label="atenção"),
        Transition("c", "local_1", cost=0.7, bpm_diff=3.0, energy_diff=0.3, camelot_dist=None, label="difícil"),
        Transition("local_1", "d", cost=0.2, bpm_diff=1.0, energy_diff=0.1, camelot_dist=None,
                   voice="choque de voz", label="boa"),
    ]
    return tracks, analyses, transitions, local_wav


def test_build_filename_numbering_and_extension():
    t = Track("x", "Título", ["MC Ção"])
    assert build_filename(1, t, 20) == "01 - MC Ção - Título.mp3"
    assert build_filename(99, t, 99) == "99 - MC Ção - Título.mp3"
    assert build_filename(5, t, 100) == "005 - MC Ção - Título.mp3"
    assert build_filename(123, t, 150, ext=".WAV") == "123 - MC Ção - Título.wav"
    assert build_filename(2, t, 3, ext="flac") == "02 - MC Ção - Título.flac"
    assert build_filename(1, Track("y", "Só Título"), 1) == "01 - Desconhecido - Só Título.mp3"


def test_build_filename_forbidden_chars_and_accents():
    nfd_title = unicodedata.normalize("NFD", 'Vai: "Tá" / Ó *Bruxa*? <Já> | Coração.')
    t = Track("x", nfd_title, ["MC Kevinho\\Ação"])
    name = build_filename(7, t, 10)
    assert not (FORBIDDEN & set(name))
    assert name == unicodedata.normalize("NFC", name)
    assert "Tá" in name and "Ó" in name and "Coração" in name and "Ação" in name
    assert name.startswith("07 - MC Kevinho Ação - Vai")
    assert name.endswith("Coração.mp3")          # ponto final do título não vira "..mp3"


def test_build_filename_long_names_are_truncated():
    t = Track("x", "Muito " * 80, ["MC Longo"])
    name = build_filename(1, t, 1)
    assert len(name) <= 150 + len(".mp3") and name.endswith(".mp3")
    assert len(build_filename(1, t, 1, max_len=50)) <= 54


@pytest.mark.parametrize("key,expected", [
    ("A minor", "Am"), ("C# major", "C#"), ("Bb minor", "Bbm"), ("f# minor", "F#m"),
    ("E", "E"), ("Db maior", "Db"), (None, None), ("", None), ("H minor", None), ("C lydian", None),
])
def test_short_key(key, expected):
    assert short_key(key) == expected


def test_full_export(tmp_dirs, audio_src, tmp_path):
    tracks, analyses, transitions, local_wav = build_set(tmp_dirs, audio_src, tmp_path)
    out = tmp_dirs["EXPORT_DIR"]
    res = export_setlist(tracks, analyses, transitions)

    assert res.folder == out
    assert res.m3u8_path == out / "Setlist_Atual.m3u8"
    assert res.csv_path == out / "Setlist_Atual.csv"
    assert res.skipped == ["MC Fantasma - Sem Áudio"]
    names = [p.name for p in res.files]
    assert names == [
        "01 - MC Alfa - Vai Descendo.mp3",
        "02 - MC Beta, DJ Gama - Tá Tranquilo.mp3",
        "03 - DJ SK - Faixa Local.wav",              # local não-mp3 mantém a extensão
        "04 - MC Delta - Final.mp3",
    ]
    assert all(p.parent == out and p.is_file() for p in res.files)
    # Cópias, não movimentos: o cache e o arquivo local continuam lá, intactos.
    assert (tmp_dirs["CACHE_DIR"] / "a.mp3").read_bytes() == audio_src["mp3"].read_bytes()
    assert local_wav.read_bytes() == audio_src["wav"].read_bytes()
    assert (out / names[2]).read_bytes() == audio_src["wav"].read_bytes()


def test_m3u8_exact_content(tmp_dirs, audio_src, tmp_path):
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    res = export_setlist(tracks, analyses, transitions)
    raw = res.m3u8_path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")                 # UTF-8 sem BOM
    assert raw.count(b"\n") == raw.count(b"\r\n")              # só CRLF
    files = [str(p.resolve()) for p in res.files]
    # Duração: análise > Track.duration_s > leitura do arquivo (wav e mp3 de ~2 s).
    expected = (
        "#EXTM3U\r\n"
        "#EXTINF:186,MC Alfa - Vai Descendo\r\n" f"{files[0]}\r\n"
        "#EXTINF:186,MC Beta, DJ Gama - Tá Tranquilo\r\n" f"{files[1]}\r\n"
        "#EXTINF:2,DJ SK - Faixa Local\r\n" f"{files[2]}\r\n"
        "#EXTINF:2,MC Delta - Final\r\n" f"{files[3]}\r\n"
    )
    assert raw.decode("utf-8") == expected
    assert all(Path(f).is_absolute() and Path(f).exists() for f in files)


def test_m3u8_relative_paths(tmp_dirs, audio_src, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "M3U_ABSOLUTE_PATHS", False)
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    res = export_setlist(tracks, analyses, transitions)
    lines = m3u_lines(res.m3u8_path)
    assert lines[2] == "01 - MC Alfa - Vai Descendo.mp3"
    assert lines[6] == "03 - DJ SK - Faixa Local.wav"


def test_csv_columns_and_decimal_comma(tmp_dirs, audio_src, tmp_path):
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    res = export_setlist(tracks, analyses, transitions)
    raw = res.csv_path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")                     # BOM para o Excel
    rows = read_csv(res.csv_path)
    assert rows[0] == CSV_COLUMNS == [
        "ordem", "arquivo", "artista", "titulo", "video_id", "duracao", "bpm", "bpm_manual",
        "energia_media", "energia_entrada", "energia_saida", "tom", "camelot",
        "transicao_para_proxima", "custo_transicao", "janela_voz", "explicacao",
    ]
    assert len(rows) == 5
    r = [dict(zip(CSV_COLUMNS, row)) for row in rows[1:]]

    assert r[0]["ordem"] == "1"
    assert r[0]["arquivo"] == "01 - MC Alfa - Vai Descendo.mp3"
    assert r[0]["artista"] == "MC Alfa" and r[0]["titulo"] == "Vai Descendo" and r[0]["video_id"] == "a"
    assert r[0]["duracao"] == "3:06"
    assert r[0]["bpm"] == "130,0" and r[0]["bpm_manual"] == "128,0"
    assert r[0]["energia_media"] == "0,50"
    assert r[0]["energia_entrada"] == "0,33" and r[0]["energia_saida"] == "0,76"
    assert r[0]["tom"] == "A minor" and r[0]["camelot"] == "8A"
    assert r[0]["transicao_para_proxima"] == "suave"
    assert r[0]["custo_transicao"] == "0,12"
    assert r[0]["janela_voz"] == "limpa"
    assert r[0]["explicacao"] == "Entra no refrão."            # sem quebra de linha no CSV

    assert r[1]["artista"] == "MC Beta, DJ Gama" and r[1]["titulo"] == "Tá Tranquilo"
    assert r[1]["bpm"] == "131,0" and r[1]["bpm_manual"] == ""
    assert r[1]["energia_media"] == "0,90" and r[1]["tom"] == "C# major"
    # A faixa seguinte (sem áudio) ficou de fora: a transição b->c não vale para b->local_1.
    assert r[1]["transicao_para_proxima"] == "" and r[1]["custo_transicao"] == ""

    assert r[2]["arquivo"] == "03 - DJ SK - Faixa Local.wav"
    assert r[2]["bpm"] == "132,0" and r[2]["tom"] == "" and r[2]["camelot"] == ""
    assert r[2]["duracao"] == "0:02"
    assert r[2]["transicao_para_proxima"] == "boa" and r[2]["custo_transicao"] == "0,20"
    assert r[2]["janela_voz"] == "choque de voz"

    # Faixa sem análise e última do set: colunas vazias, sem "None".
    assert r[3]["ordem"] == "4" and r[3]["bpm"] == "" and r[3]["energia_media"] == ""
    assert r[3]["transicao_para_proxima"] == "" and r[3]["explicacao"] == ""
    assert not any("None" in v or "." in v for row in r for k, v in row.items()
                   if k in ("bpm", "bpm_manual", "energia_media", "energia_entrada",
                            "energia_saida", "custo_transicao"))


def test_id3_tags_only_on_copies(tmp_dirs, audio_src, tmp_path):
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    res = export_setlist(tracks, analyses, transitions)

    t1 = ID3(res.files[0])
    assert str(t1["TIT2"]) == "Vai Descendo"
    assert str(t1["TPE1"]) == "MC Alfa"
    assert str(t1["TBPM"]) == "128"                  # bpm_manual tem prioridade, inteiro
    assert str(t1["TKEY"]) == "Am"
    assert str(t1["TRCK"]) == "1/4"
    assert t1.version[:2] == (2, 3)

    t2 = ID3(res.files[1])
    assert str(t2["TIT2"]) == "Tá Tranquilo"
    assert str(t2["TPE1"]) == "MC Beta, DJ Gama"
    assert str(t2["TBPM"]) == "131" and str(t2["TKEY"]) == "C#" and str(t2["TRCK"]) == "2/4"

    t4 = ID3(res.files[3])                           # sem análise: sem BPM nem tom
    assert str(t4["TIT2"]) == "Final" and str(t4["TRCK"]) == "4/4"
    assert "TBPM" not in t4 and "TKEY" not in t4

    try:
        src_tags = ID3(tmp_dirs["CACHE_DIR"] / "a.mp3")
    except Exception:  # noqa: BLE001 - sem cabeçalho ID3 também prova que o cache ficou intacto
        src_tags = {}
    assert "TIT2" not in src_tags


def test_tag_failure_does_not_abort(tmp_dirs, audio_src, tmp_path, monkeypatch, caplog):
    def boom(self, *a, **kw):
        raise OSError("arquivo travado")

    monkeypatch.setattr(ID3, "save", boom)
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    res = export_setlist(tracks, analyses, transitions)
    assert len(res.files) == 4 and all(p.exists() for p in res.files)
    assert res.m3u8_path.exists() and res.csv_path.exists()
    assert "Não foi possível gravar as tags" in caplog.text


def test_tags_disabled(tmp_dirs, audio_src, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WRITE_ID3_TAGS", False)
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    res = export_setlist(tracks, analyses, transitions)
    assert res.files[0].read_bytes() == audio_src["mp3"].read_bytes()


def test_cleanup_only_removes_previous_export(tmp_dirs, audio_src, tmp_path):
    out = tmp_dirs["EXPORT_DIR"]
    old_export = ["01 - Velho - Antigo.mp3", "07 - X - Y.flac", "100 - Muitas - Faixas.wav",
                  "Setlist_Atual.m3u8", "Setlist_Atual (pen drive).m3u8", "Setlist_Atual.csv"]
    user_files = ["notas.txt", "1 - um digito.mp3", "0001 - quatro digitos.mp3",
                  "01 - sem extensao de audio.txt", "01-sem espacos.mp3", "Outra.m3u8",
                  "Setlist_Atual.m3u8.bak", "capa.jpg"]
    for n in old_export + user_files:
        (out / n).write_bytes(b"conteudo")
    (out / "01 - pasta do usuario.mp3").mkdir()           # pasta com nome parecido não é apagada
    (out / "01 - pasta do usuario.mp3" / "dentro.txt").write_text("x", encoding="utf-8")

    cache_mp3(tmp_dirs, audio_src, "a")
    res = export_setlist([Track("a", "Nova", ["MC Novo"])], {})

    left = sorted(p.name for p in out.iterdir())
    expected = sorted(user_files + ["01 - pasta do usuario.mp3", "01 - MC Novo - Nova.mp3",
                                    "Setlist_Atual.m3u8", "Setlist_Atual (pen drive).m3u8",
                                    "Setlist_Atual.csv"])
    assert left == expected
    assert (out / "notas.txt").read_bytes() == b"conteudo"
    assert (out / "01 - pasta do usuario.mp3" / "dentro.txt").exists()
    assert res.files == [out / "01 - MC Novo - Nova.mp3"]


def test_reexport_with_fewer_tracks_removes_stale_files(tmp_dirs, audio_src, tmp_path):
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    export_setlist(tracks, analyses, transitions)
    res = export_setlist(tracks[:1], analyses, transitions[:0])
    out = tmp_dirs["EXPORT_DIR"]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["01 - MC Alfa - Vai Descendo.mp3", "Setlist_Atual.m3u8", "Setlist_Atual (pen drive).m3u8",
         "Setlist_Atual.csv"])
    assert len(read_csv(res.csv_path)) == 2


def test_local_file_inside_export_folder_survives_cleanup(tmp_dirs, audio_src):
    """Uma faixa local que É um arquivo da exportação anterior não pode ser apagada antes da cópia."""
    out = tmp_dirs["EXPORT_DIR"]
    inside = out / "02 - Antiga - Local.wav"
    shutil.copy(audio_src["wav"], inside)
    cache_mp3(tmp_dirs, audio_src, "a")
    tracks = [Track("local_x", "Local", ["Antiga"], 2.0, source="local", local_path=str(inside)),
              Track("a", "Outra", ["MC A"], 2.0)]
    res = export_setlist(tracks, {})
    assert [p.name for p in res.files] == ["01 - Antiga - Local.wav", "02 - MC A - Outra.mp3"]
    assert res.files[0].read_bytes() == audio_src["wav"].read_bytes()
    assert not inside.exists()
    assert not [p for p in out.iterdir() if p.is_dir()]     # pasta temporária removida


def test_duplicate_artist_title_do_not_overwrite(tmp_dirs, audio_src):
    cache_mp3(tmp_dirs, audio_src, "v1")
    (tmp_dirs["CACHE_DIR"] / "v2.mp3").write_bytes(b"outro audio")
    tracks = [Track("v1", "Mesma", ["MC Igual"], 2.0), Track("v2", "Mesma", ["MC Igual"], 2.0),
              Track("v1", "Mesma", ["MC Igual"], 2.0)]       # mesma faixa repetida no set
    res = export_setlist(tracks, {})
    assert [p.name for p in res.files] == ["01 - MC Igual - Mesma.mp3", "02 - MC Igual - Mesma.mp3",
                                           "03 - MC Igual - Mesma.mp3"]
    assert len({p.resolve() for p in res.files}) == 3
    assert res.files[1].read_bytes().endswith(b"outro audio")


def test_more_than_99_tracks_use_three_digits(tmp_dirs, monkeypatch):
    monkeypatch.setattr(config, "WRITE_ID3_TAGS", False)
    tracks = []
    for i in range(101):
        (tmp_dirs["CACHE_DIR"] / f"v{i}.mp3").write_bytes(b"x")
        tracks.append(Track(f"v{i}", f"Faixa {i}", ["MC"], 120.0))
    res = export_setlist(tracks, {})
    names = [p.name for p in res.files]
    assert len(names) == 101 and len(set(names)) == 101
    assert names[0] == "001 - MC - Faixa 0.mp3"
    assert names[98] == "099 - MC - Faixa 98.mp3"
    assert names[100] == "101 - MC - Faixa 100.mp3"
    lines = m3u_lines(res.m3u8_path)
    assert lines[1] == "#EXTINF:120,MC - Faixa 0"
    assert read_csv(res.csv_path)[-1][:2] == ["101", "101 - MC - Faixa 100.mp3"]

    # Exportar de novo com 99 faixas volta a 2 dígitos e apaga os de 3 dígitos.
    res2 = export_setlist(tracks[:99], {})
    assert res2.files[0].name == "01 - MC - Faixa 0.mp3"
    assert not list(tmp_dirs["EXPORT_DIR"].glob("1?? - *"))
    assert not list(tmp_dirs["EXPORT_DIR"].glob("0?? - *"))


def test_skipped_numbering_is_continuous(tmp_dirs, audio_src, tmp_path):
    cache_mp3(tmp_dirs, audio_src, "a")
    cache_mp3(tmp_dirs, audio_src, "c")
    (tmp_dirs["CACHE_DIR"] / "vazio.mp3").write_bytes(b"")      # arquivo vazio não conta como áudio
    tracks = [Track("a", "Um", ["MC"]), Track("b", "Dois", ["MC"]),
              Track("vazio", "Três", ["MC"]),
              Track("l", "Quatro", ["MC"], local_path=str(tmp_path / "nao_existe.mp3")),
              Track("c", "Cinco", ["MC"])]
    res = export_setlist(tracks, {})
    assert [p.name for p in res.files] == ["01 - MC - Um.mp3", "02 - MC - Cinco.mp3"]
    assert res.skipped == ["MC - Dois", "MC - Três", "MC - Quatro"]
    assert str(ID3(res.files[1])["TRCK"]) == "2/2"


def test_local_path_missing_falls_back_to_cache(tmp_dirs, audio_src, tmp_path):
    cache_mp3(tmp_dirs, audio_src, "yt1")
    t = Track("yt1", "Do Cache", ["MC"], 2.0, local_path=str(tmp_path / "sumiu.wav"))
    res = export_setlist([t], {})
    assert [p.name for p in res.files] == ["01 - MC - Do Cache.mp3"]


def test_forbidden_chars_and_accents_in_export(tmp_dirs, audio_src):
    cache_mp3(tmp_dirs, audio_src, "z")
    title = unicodedata.normalize("NFD", 'Ó "Bruxa": Tá/Lá? *Coração*')
    t = Track("z", title, ["MC Ção|Zé"], 2.0)
    res = export_setlist([t], {"z": analysis("z", duration_s=2.0)})
    name = res.files[0].name
    assert not (FORBIDDEN & set(name))
    assert unicodedata.normalize("NFC", "Ó") in name and "Coração" in name and "Zé" in name
    assert res.files[0].exists()
    m3u = m3u_lines(res.m3u8_path)
    assert m3u[1] == f"#EXTINF:2,MC Ção|Zé - {title}"           # nome original na playlist
    assert Path(m3u[2]).name == name
    row = read_csv(res.csv_path)[1]
    assert row[1] == name and row[2] == "MC Ção|Zé" and row[3] == title
    assert str(ID3(res.files[0])["TIT2"]) == title


def test_unknown_duration_writes_minus_one(tmp_dirs):
    (tmp_dirs["CACHE_DIR"] / "q.mp3").write_bytes(b"isto nao e um mp3")
    res = export_setlist([Track("q", "Sem Duração", ["MC"])], {})
    lines = m3u_lines(res.m3u8_path)
    assert lines[1] == "#EXTINF:-1,MC - Sem Duração"
    assert read_csv(res.csv_path)[1][5] == ""


def test_empty_set_and_explicit_dirs(tmp_dirs, audio_src, tmp_path):
    out = tmp_path / "nova" / "pasta"                           # criada se não existir
    cache = tmp_path / "meu_cache"
    cache.mkdir()
    shutil.copy(audio_src["mp3"], cache / "k.mp3")
    res = export_setlist([], {}, out_dir=out, playlist_name="Teste")
    assert res.files == [] and res.skipped == []
    assert (out / "Teste.m3u8").read_bytes() == b"#EXTM3U\r\n"
    assert read_csv(out / "Teste.csv") == [CSV_COLUMNS]

    res = export_setlist([Track("k", "Faixa", ["MC"])], {}, out_dir=out, playlist_name="Teste",
                         cache_dir=cache)
    assert [p.name for p in res.files] == ["01 - MC - Faixa.mp3"]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["01 - MC - Faixa.mp3", "Teste.csv", "Teste.m3u8", "Teste (pen drive).m3u8"])
    assert not any(tmp_dirs["EXPORT_DIR"].iterdir())           # pasta padrão não foi tocada


def test_defaults_read_config_at_call_time(tmp_dirs, audio_src, tmp_path, monkeypatch):
    other_out = tmp_path / "out2"
    other_cache = tmp_path / "cache2"
    other_cache.mkdir()
    shutil.copy(audio_src["mp3"], other_cache / "m.mp3")
    monkeypatch.setattr(config, "EXPORT_DIR", other_out)
    monkeypatch.setattr(config, "CACHE_DIR", other_cache)
    monkeypatch.setattr(config, "PLAYLIST_NAME", "Baile")
    res = export_setlist([Track("m", "Faixa", ["MC"])], {})
    assert res.m3u8_path == other_out / "Baile.m3u8" and res.csv_path == other_out / "Baile.csv"
    assert res.files == [other_out / "01 - MC - Faixa.mp3"]


def test_transitions_matched_by_pair_when_list_is_misaligned(tmp_dirs, audio_src):
    for vid in ("a", "b", "c"):
        cache_mp3(tmp_dirs, audio_src, vid)
    tracks = [Track(v, v.upper(), ["MC"], 2.0) for v in ("a", "b", "c")]
    trs = [Transition("b", "c", 0.3, 1.0, 0.1, None, label="boa"),
           Transition("a", "b", 0.1, 1.0, 0.1, None, label="suave")]   # fora de ordem
    res = export_setlist(tracks, {}, trs)
    rows = read_csv(res.csv_path)[1:]
    assert [r[13] for r in rows] == ["suave", "boa", ""]
    assert [r[14] for r in rows] == ["0,10", "0,30", ""]


def test_module_has_no_print():
    src = Path(export.__file__).read_text(encoding="utf-8")
    assert "print(" not in src


# ---------------------------------------------------------------------------
# Revisão: arquivos abertos, desfazer em caso de erro, faixas locais, tags, CSV
# ---------------------------------------------------------------------------
def _snapshot_folder(folder: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}


@pytest.mark.skipif(sys.platform != "win32", reason="trava de arquivo aberto é do Windows")
def test_reexport_with_csv_open_in_excel_keeps_previous_export(tmp_dirs, audio_src, tmp_path):
    """O DJ abre o CSV no Excel (arquivo travado) e exporta de novo: nada pode ser apagado."""
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    res = export_setlist(tracks, analyses, transitions)
    out = tmp_dirs["EXPORT_DIR"]
    before = _snapshot_folder(out)
    with open(res.csv_path, "rb"):                 # no Windows, arquivo aberto não pode ser apagado
        with pytest.raises(OSError, match="aberto"):
            export_setlist(tracks[:1], analyses, [])
    assert _snapshot_folder(out) == before
    assert not [p for p in out.iterdir() if p.is_dir()]


def test_failed_export_rolls_back_to_previous_export(tmp_dirs, audio_src, tmp_path, monkeypatch):
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    export_setlist(tracks, analyses, transitions)
    out = tmp_dirs["EXPORT_DIR"]
    before = _snapshot_folder(out)

    real_copy = shutil.copy2
    calls = {"n": 0}

    def flaky_copy(src, dst, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disco cheio")
        return real_copy(src, dst, *a, **kw)

    monkeypatch.setattr(export.shutil, "copy2", flaky_copy)
    with pytest.raises(OSError, match="disco cheio"):
        export_setlist(list(reversed(tracks)), analyses, [])
    assert _snapshot_folder(out) == before          # exportação anterior intacta
    assert not [p for p in out.iterdir() if p.is_dir()]


def test_failed_export_puts_local_files_back(tmp_dirs, audio_src, monkeypatch):
    """Faixa local que mora na pasta de exportação: se a exportação falhar, o arquivo volta."""
    out = tmp_dirs["EXPORT_DIR"]
    inside = out / "02 - Antiga - Local.wav"
    inside.write_bytes(b"audio local do DJ")
    t = Track("local_x", "Local", ["Antiga"], 2.0, source="local", local_path=str(inside))

    def boom(*a, **kw):
        raise OSError("falha na cópia")

    monkeypatch.setattr(export.shutil, "copy2", boom)
    with pytest.raises(OSError):
        export_setlist([t], {})
    assert inside.read_bytes() == b"audio local do DJ"
    assert not [p for p in out.iterdir() if p.is_dir()]
    assert t.local_path == str(inside)


def test_reorder_and_reexport_local_files_from_export_folder(tmp_dirs):
    """O DJ carrega a própria Setlist_Atual como pasta local, reordena e exporta duas vezes.

    Os nomes mudam de número a cada exportação; a segunda não pode perder faixas.
    """
    out = tmp_dirs["EXPORT_DIR"]
    a, b = out / "01 - MC A - Um.wav", out / "02 - MC B - Dois.wav"
    a.write_bytes(b"audio-A")
    b.write_bytes(b"audio-B")
    ta = Track("local_a", "Um", ["MC A"], 2.0, source="local", local_path=str(a))
    tb = Track("local_b", "Dois", ["MC B"], 2.0, source="local", local_path=str(b))

    r1 = export_setlist([tb, ta], {})
    assert [p.name for p in r1.files] == ["01 - MC B - Dois.wav", "02 - MC A - Um.wav"]
    assert [p.read_bytes() for p in r1.files] == [b"audio-B", b"audio-A"]
    # As faixas passam a apontar para as cópias novas (os nomes antigos não existem mais).
    assert Path(ta.local_path).read_bytes() == b"audio-A"
    assert Path(tb.local_path).read_bytes() == b"audio-B"

    r2 = export_setlist([ta, tb], {})
    assert r2.skipped == []
    assert [p.name for p in r2.files] == ["01 - MC A - Um.wav", "02 - MC B - Dois.wav"]
    assert [p.read_bytes() for p in r2.files] == [b"audio-A", b"audio-B"]


def test_id3_tags_are_valid_v23(tmp_dirs, audio_src):
    """Tag ID3v2.4 de origem (ex.: ano em TDRC) precisa ser convertida antes de salvar como v2.3."""
    from mutagen.id3 import TDRC, TSOP

    src = cache_mp3(tmp_dirs, audio_src, "v24")
    tags = ID3(src)
    tags.add(TDRC(encoding=3, text="2024"))
    tags.add(TSOP(encoding=3, text="Ordenação"))      # quadro que só existe na v2.4
    tags.save(src, v2_version=4)

    res = export_setlist([Track("v24", "Faixa", ["MC"], 2.0)], {"v24": analysis("v24", duration_s=2.0)})
    raw = ID3(res.files[0], translate=False)
    assert raw.version[:2] == (2, 3)
    assert "TDRC" not in raw and "TSOP" not in raw
    assert str(raw["TYER"]) == "2024"
    assert str(raw["TIT2"]) == "Faixa" and str(raw["TBPM"]) == "130"


def test_csv_text_cells_cannot_become_formulas(tmp_dirs, audio_src):
    """Títulos vêm do YouTube (qualquer um escreve): no Excel, "=..." ou "-..." viram fórmula."""
    cache_mp3(tmp_dirs, audio_src, "-dQw4w9WgXc")
    t = Track("-dQw4w9WgXc", '=HYPERLINK("http://exemplo.invalid","clique")', ["+MC Mais", "@MC"], 2.0)
    tr = Transition("-dQw4w9WgXc", "x", 0.1, 1.0, 0.1, None, label="suave", explanation="-entra seco")
    cache_mp3(tmp_dirs, audio_src, "x")
    t2 = Track("x", "Normal", ["MC Normal"], 2.0)
    res = export_setlist([t, t2], {}, [tr])
    rows = read_csv(res.csv_path)[1:]
    r = dict(zip(CSV_COLUMNS, rows[0]))
    assert r["titulo"] == "'" + t.title
    assert r["artista"] == "'+MC Mais, @MC"
    assert r["video_id"] == "'-dQw4w9WgXc"
    assert r["explicacao"] == "'-entra seco"
    # Valores comuns não mudam.
    r2 = dict(zip(CSV_COLUMNS, rows[1]))
    assert r2["titulo"] == "Normal" and r2["artista"] == "MC Normal" and r2["video_id"] == "x"
    # A playlist e as tags guardam o texto original.
    assert m3u_lines(res.m3u8_path)[1] == f"#EXTINF:2,+MC Mais, @MC - {t.title}"
    assert str(ID3(res.files[0])["TIT2"]) == t.title


def test_playlist_name_is_sanitized(tmp_dirs, audio_src):
    """Nome digitado pelo DJ com ":" ou "/" não pode virar fluxo NTFS nem subpasta."""
    cache_mp3(tmp_dirs, audio_src, "a")
    out = tmp_dirs["EXPORT_DIR"]
    res = export_setlist([Track("a", "Faixa", ["MC"], 2.0)], {}, playlist_name="Baile: Sexta/Sábado")
    assert res.m3u8_path.parent == out and res.csv_path.parent == out
    assert not (FORBIDDEN & set(res.m3u8_path.name)) and res.m3u8_path.suffix == ".m3u8"
    assert res.m3u8_path.read_bytes().startswith(b"#EXTM3U")
    assert not (FORBIDDEN & set(res.portable_m3u8_path.name))
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["01 - MC - Faixa.mp3", res.m3u8_path.name, res.portable_m3u8_path.name, res.csv_path.name])
    # Exportar de novo com o mesmo nome limpa as playlists anteriores (absoluta + pen drive).
    export_setlist([Track("a", "Faixa", ["MC"], 2.0)], {}, playlist_name="Baile: Sexta/Sábado")
    assert len(list(out.glob("*.m3u8"))) == 2


def test_concurrent_exports_are_serialized(tmp_dirs, audio_src, monkeypatch):
    """Duas abas do app exportando ao mesmo tempo não podem misturar as pastas."""
    monkeypatch.setattr(config, "WRITE_ID3_TAGS", False)
    for v in ("a", "b"):
        cache_mp3(tmp_dirs, audio_src, v)
    active = {"now": 0, "max": 0}
    guard = threading.Lock()
    real_copy = shutil.copy2

    def slow_copy(src, dst, *a, **kw):
        with guard:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        try:
            time.sleep(0.15)
            return real_copy(src, dst, *a, **kw)
        finally:
            with guard:
                active["now"] -= 1

    monkeypatch.setattr(export.shutil, "copy2", slow_copy)
    errors: list[BaseException] = []

    def run(tracks):
        try:
            export_setlist(tracks, {})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = [threading.Thread(target=run, args=([Track("a", "Um", ["MC"], 2.0)],)),
          threading.Thread(target=run, args=([Track("b", "Dois", ["MC"], 2.0)],))]
    for t in th:
        t.start()
    for t in th:
        t.join(10)
    assert not errors
    assert active["max"] == 1
    names = sorted(p.name for p in tmp_dirs["EXPORT_DIR"].iterdir())
    assert len([n for n in names if n.endswith(".mp3")]) == 1     # só a última exportação


def test_cleanup_follows_configured_audio_extensions(tmp_dirs, audio_src, tmp_path, monkeypatch):
    """Extensão nova em LOCAL_AUDIO_EXTENSIONS: a cópia exportada também precisa ser limpa depois."""
    monkeypatch.setattr(config, "LOCAL_AUDIO_EXTENSIONS", (*config.LOCAL_AUDIO_EXTENSIONS, ".opus"))
    local = tmp_path / "x.opus"
    local.write_bytes(b"opus")
    export_setlist([Track("local_o", "Opus", ["MC"], 2.0, source="local", local_path=str(local))], {})
    assert (tmp_dirs["EXPORT_DIR"] / "01 - MC - Opus.opus").exists()
    cache_mp3(tmp_dirs, audio_src, "a")
    export_setlist([Track("a", "Outra", ["MC"], 2.0)], {})
    assert sorted(p.name for p in tmp_dirs["EXPORT_DIR"].iterdir()) == sorted([
        "01 - MC - Outra.mp3", "Setlist_Atual.csv", "Setlist_Atual.m3u8", "Setlist_Atual (pen drive).m3u8"])


def test_portable_playlist_uses_relative_paths(tmp_dirs, audio_src, tmp_path):
    """A playlist "(pen drive)" funciona depois de copiar a pasta para outro computador."""
    tracks, analyses, transitions, _ = build_set(tmp_dirs, audio_src, tmp_path)
    res = export_setlist(tracks, analyses, transitions)
    absolute = m3u_lines(res.m3u8_path)
    portable = m3u_lines(res.portable_m3u8_path)
    assert portable[0] == "#EXTM3U" and len(portable) == len(absolute)
    assert portable[1::2] == absolute[1::2]                     # mesmas linhas #EXTINF
    assert portable[2::2] == [p.name for p in res.files]        # só o nome do arquivo
    assert all(Path(a).is_absolute() for a in absolute[2::2])


def test_no_portable_playlist_when_paths_are_already_relative(tmp_dirs, audio_src, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "M3U_ABSOLUTE_PATHS", False)
    cache_mp3(tmp_dirs, audio_src, "a")
    res = export_setlist([Track("a", "Faixa", ["MC"], 2.0)], {})
    assert res.portable_m3u8_path is None
    assert len(list(tmp_dirs["EXPORT_DIR"].glob("*.m3u8"))) == 1


def test_real_mp3_export(tmp_dirs, real_audio_dir):
    """Com mp3 reais (DJSK_AUDIO_FIXTURES): cópia, duração lida do arquivo e tags gravadas."""
    srcs = sorted(real_audio_dir.glob("*.mp3"))[:2]
    if not srcs:
        pytest.skip("nenhum mp3 em DJSK_AUDIO_FIXTURES")
    tracks = []
    for src in srcs:
        shutil.copy(src, tmp_dirs["CACHE_DIR"] / src.name)
        tracks.append(Track(src.stem, f"Faixa {src.stem}", ["MC Real"]))
    res = export_setlist(tracks, {tracks[0].video_id: analysis(tracks[0].video_id, duration_s=0.0)})
    assert len(res.files) == len(srcs) and not res.skipped
    lines = m3u_lines(res.m3u8_path)
    for i in range(len(srcs)):
        seconds = int(lines[1 + 2 * i].split(":", 1)[1].split(",", 1)[0])
        assert 60 <= seconds <= 480
    tags = ID3(res.files[0])
    assert str(tags["TBPM"]) == "130" and str(tags["TKEY"]) == "Am"
    assert str(tags["TRCK"]) == f"1/{len(srcs)}"
