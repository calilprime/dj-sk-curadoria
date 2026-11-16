"""Testes das janelas de mixagem (4.8). Só usam models/config/util e o módulo de ordenação."""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
import pytest

import config
from conftest import SR, synth_vocals, write_audio
from modules import mixwindows as mw
from modules import util
from modules.models import Analysis, VoiceActivity

TOL_S = 0.3


def assert_intervals_close(got, expected, tol=TOL_S):
    assert len(got) == len(expected), f"{got} != {expected}"
    for (ga, gb), (ea, eb) in zip(got, expected):
        assert abs(ga - ea) <= tol and abs(gb - eb) <= tol, f"{got} != {expected}"


def va(intervals, duration=60.0, vid="x") -> VoiceActivity:
    return VoiceActivity(video_id=vid, duration_s=duration, intervals=list(intervals))


def make_stem(stems_dir, vid, duration, intervals, ext=".mp3"):
    return write_audio(stems_dir / vid / f"vocals{ext}", synth_vocals(duration, intervals))


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ext", [".wav", ".mp3"])
def test_detect_known_intervals(tmp_path, ext):
    expected = [(2.0, 5.0), (8.3, 12.7), (20.05, 21.0), (27.0, 30.0)]
    path = write_audio(tmp_path / f"vocals{ext}", synth_vocals(30.0, expected))
    res = mw.detect_voice(path, "abc")
    assert res.video_id == "abc"
    assert res.duration_s == pytest.approx(30.0, abs=0.1)
    assert_intervals_close(res.intervals, expected)


def test_gaps_are_merged_and_short_segments_dropped(tmp_path):
    y = synth_vocals(20.0, [(1.0, 3.0), (3.3, 5.0),     # lacuna 0.3 s < 0.6 -> une
                            (6.5, 8.0),                  # lacuna 1.5 s -> separa
                            (12.0, 12.2),                # 0.2 s < 0.5 -> descarta
                            (15.0, 16.0)])
    res = mw.detect_voice(write_audio(tmp_path / "v.wav", y), "x")
    assert_intervals_close(res.intervals, [(1.0, 5.0), (6.5, 8.0), (15.0, 16.0)], tol=0.15)


def test_silence_and_stereo(tmp_path):
    silent = mw.detect_voice(write_audio(tmp_path / "s.wav", np.zeros(SR * 5, np.float32)), "s")
    assert silent.intervals == [] and silent.duration_s == pytest.approx(5.0)
    mono = synth_vocals(10.0, [(2.0, 6.0)])
    stereo = np.stack([mono, mono * 0.5], axis=1)
    res = mw.detect_voice(write_audio(tmp_path / "st.wav", stereo), "st")
    assert_intervals_close(res.intervals, [(2.0, 6.0)])


def test_threshold_read_from_config_at_call_time(tmp_path, monkeypatch):
    # Ruído baixo (~ -46 dBFS): não é voz no limiar padrão (-35), passa a ser em -55.
    y = (np.random.default_rng(0).standard_normal(SR * 6) * 0.005).astype(np.float32)
    y[: SR * 2] = 0.0
    path = write_audio(tmp_path / "quiet.wav", y)
    assert mw.detect_voice(path, "q").intervals == []
    monkeypatch.setattr(config, "VOICE_THRESHOLD_DB", -55.0)
    assert_intervals_close(mw.detect_voice(path, "q").intervals, [(2.0, 6.0)])


def test_empty_and_unreadable_files(tmp_path):
    empty = mw.detect_voice(write_audio(tmp_path / "e.wav", np.zeros(0, np.float32)), "e")
    assert empty.intervals == [] and empty.duration_s == 0.0
    bad = tmp_path / "vocals.mp3"
    bad.write_bytes(b"isto nao e audio")
    with pytest.raises(mw.VoiceDetectionError) as info:
        mw.detect_voice(bad, "b")
    # Mensagem em português, sem o texto em inglês da biblioteca nem o caminho completo
    # (o detalhe técnico fica na causa encadeada, para o log).
    msg = str(info.value)
    assert "vocals.mp3" in msg
    assert "Error" not in msg and str(tmp_path) not in msg
    assert info.value.__cause__ is not None


# ---------------------------------------------------------------------------
# Cache (voice.json)
# ---------------------------------------------------------------------------

def test_voice_path_reads_config_at_call_time(tmp_dirs, tmp_path):
    assert mw.voice_path("abc") == tmp_dirs["STEMS_DIR"] / "abc" / "voice.json"
    assert mw.voice_path("abc", tmp_path) == tmp_path / "abc" / "voice.json"


def test_load_or_detect_without_stem_is_none(tmp_dirs):
    assert mw.load_or_detect("nada") is None
    (tmp_dirs["STEMS_DIR"] / "so_drums").mkdir()
    write_audio(tmp_dirs["STEMS_DIR"] / "so_drums" / "drums.wav", np.zeros(SR, np.float32))
    assert mw.load_or_detect("so_drums") is None
    assert not mw.voice_path("so_drums").exists()


def test_load_or_detect_writes_and_reuses_cache(tmp_dirs, monkeypatch):
    make_stem(tmp_dirs["STEMS_DIR"], "v1", 20.0, [(3.0, 7.0)])
    first = mw.load_or_detect("v1")
    assert_intervals_close(first.intervals, [(3.0, 7.0)])
    cache = mw.voice_path("v1")
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert VoiceActivity.from_dict(data) == first
    assert data["version"] == mw.VOICE_CACHE_VERSION

    def boom(*a, **k):
        raise AssertionError("não devia detectar de novo")

    monkeypatch.setattr(mw, "detect_voice", boom)
    assert mw.load_or_detect("v1") == first


def test_cache_is_refreshed_when_stem_or_params_change(tmp_dirs, monkeypatch):
    stems = tmp_dirs["STEMS_DIR"]
    path = make_stem(stems, "v2", 20.0, [(3.0, 7.0)], ext=".wav")
    assert_intervals_close(mw.load_or_detect("v2").intervals, [(3.0, 7.0)])

    write_audio(path, synth_vocals(20.0, [(10.0, 15.0)]))       # stem novo
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert_intervals_close(mw.load_or_detect("v2").intervals, [(10.0, 15.0)])

    calls = []
    real = mw.detect_voice
    monkeypatch.setattr(mw, "detect_voice", lambda p, v: calls.append(v) or real(p, v))
    monkeypatch.setattr(config, "VOICE_MIN_SEGMENT_S", 6.0)     # parâmetro mudou
    assert mw.load_or_detect("v2").intervals == []
    assert calls == ["v2"]


def test_cache_without_stem_is_still_used(tmp_dirs):
    cached = va([(1.0, 2.0)], duration=30.0, vid="v3")
    p = mw.voice_path("v3")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(cached.to_dict()), encoding="utf-8")
    assert mw.load_or_detect("v3") == cached


def test_corrupt_cache_is_redone_and_bad_stem_gives_none(tmp_dirs, caplog):
    stems = tmp_dirs["STEMS_DIR"]
    make_stem(stems, "v4", 15.0, [(2.0, 4.0)], ext=".wav")
    mw.voice_path("v4").write_text("{quebrado", encoding="utf-8")
    assert_intervals_close(mw.load_or_detect("v4").intervals, [(2.0, 4.0)])

    (stems / "v5").mkdir()
    (stems / "v5" / "vocals.mp3").write_bytes(b"lixo")
    with caplog.at_level(logging.WARNING, logger="modules.mixwindows"):
        assert mw.load_or_detect("v5") is None
    assert any("v5" in r.getMessage() for r in caplog.records)


def test_vocals_mp3_preferred_over_other_extensions(tmp_dirs):
    stems = tmp_dirs["STEMS_DIR"]
    make_stem(stems, "v6", 12.0, [(1.0, 3.0)], ext=".mp3")
    make_stem(stems, "v6", 12.0, [(6.0, 9.0)], ext=".wav")
    assert_intervals_close(mw.load_or_detect("v6").intervals, [(1.0, 3.0)])


def test_empty_preferred_stem_does_not_hide_valid_one(tmp_dirs):
    """Um vocals.mp3 vazio (gravação interrompida) não pode esconder um vocals.wav válido;
    o modules/stems.py também ignora stems de tamanho 0."""
    stems = tmp_dirs["STEMS_DIR"]
    (stems / "v7").mkdir()
    (stems / "v7" / "vocals.mp3").write_bytes(b"")
    make_stem(stems, "v7", 12.0, [(2.0, 5.0)], ext=".wav")
    res = mw.load_or_detect("v7")
    assert res is not None
    assert_intervals_close(res.intervals, [(2.0, 5.0)])


def test_only_empty_stem_is_no_stem(tmp_dirs):
    stems = tmp_dirs["STEMS_DIR"]
    (stems / "v8").mkdir()
    (stems / "v8" / "vocals.mp3").write_bytes(b"")
    assert mw.load_or_detect("v8") is None


def test_folder_name_matches_other_modules(tmp_dirs):
    """stems.py e analyze.py nomeiam a pasta/arquivo com util.safe_filename(video_id);
    o voice.json e o vocals.* precisam ser procurados na mesma pasta."""
    stems = tmp_dirs["STEMS_DIR"]
    vid = "a:b|c"
    folder = stems / util.safe_filename(vid)
    assert mw.voice_path(vid) == folder / "voice.json"
    make_stem(stems, util.safe_filename(vid), 12.0, [(2.0, 5.0)], ext=".wav")
    res = mw.load_or_detect(vid)
    assert res is not None and res.video_id == vid
    assert mw.voice_path(vid).is_file()
    # id malformado não escapa da pasta de stems
    assert mw.voice_path("../fora").parent.parent == stems


# ---------------------------------------------------------------------------
# voice_fraction / classify_pair
# ---------------------------------------------------------------------------

def test_voice_fraction():
    v = va([(0.0, 2.0), (5.0, 10.0), (9.0, 12.0)])           # sobrepostos são unidos
    assert mw.voice_fraction(v, 0.0, 4.0) == pytest.approx(0.5)
    assert mw.voice_fraction(v, 5.0, 12.0) == pytest.approx(1.0)
    assert mw.voice_fraction(v, 2.0, 5.0) == 0.0
    assert mw.voice_fraction(v, 1.0, 6.0) == pytest.approx(2.0 / 5.0)
    assert mw.voice_fraction(v, 4.0, 4.0) == 0.0
    assert mw.voice_fraction(v, 6.0, 3.0) == 0.0
    assert mw.voice_fraction(va([]), 0.0, 10.0) == 0.0


def test_classify_all_labels():
    n = config.TRANSITION_WINDOW_S
    dur = 120.0
    silent = va([], dur)
    a_end = va([(dur - n, dur)], dur)                     # voz na janela inteira de saída
    b_start = va([(0.0, n)], dur)                         # voz na janela inteira de entrada
    a_middle = va([(30.0, 60.0)], dur)                    # voz só fora das janelas

    assert mw.classify_pair(silent, silent) == ("limpa", 0.0)
    assert mw.classify_pair(a_middle, a_middle) == ("limpa", 0.0)
    assert mw.classify_pair(a_end, silent) == ("voz de um lado", 0.0)
    assert mw.classify_pair(silent, b_start) == ("voz de um lado", 0.0)
    label, clash = mw.classify_pair(a_end, b_start)
    assert label == "choque de voz" and clash == pytest.approx(1.0)
    assert mw.classify_pair(None, b_start) == ("sem stems", 0.0)
    assert mw.classify_pair(a_end, None) == ("sem stems", 0.0)
    assert mw.classify_pair(None, None) == ("sem stems", 0.0)
    assert set(config.VOICE_COST) == {"limpa", "voz de um lado", "choque de voz", "sem stems"}


def test_both_sides_with_voice_but_not_simultaneous_is_one_side():
    n, dur = 16.0, 100.0
    a = va([(dur - n, dur - n + 6.0)], dur)       # A canta nos 6 primeiros s da mixagem
    b = va([(10.0, 16.0)], dur)                   # B entra cantando só no fim
    assert mw.classify_pair(a, b, n) == ("voz de um lado", 0.0)


def test_clash_threshold_and_fraction():
    n, dur = 10.0, 100.0
    thr = config.VOICE_CLASH_MIN_OVERLAP
    a = va([(dur - n, dur)], dur)
    at_limit = va([(0.0, thr * n)], dur)                      # exatamente no limite
    below = va([(0.0, thr * n - 0.5)], dur)
    label, clash = mw.classify_pair(a, at_limit, n)
    assert label == "choque de voz" and clash == pytest.approx(thr)
    label, clash = mw.classify_pair(a, below, n)
    assert label == "voz de um lado" and clash == pytest.approx(thr - 0.05)


def test_alignment_is_directional():
    """A→B usa o fim de A com o começo de B; B→A é outra transição."""
    dur = 60.0
    a = va([(0.0, 10.0)], dur)              # voz só no começo
    b = va([(50.0, 60.0)], dur)             # voz só no fim
    assert mw.classify_pair(a, b) == ("limpa", 0.0)
    # B→A: voz de B em [50, 60] vira [6, 16] na mixagem; A canta em [0, 10] -> 4 s juntos.
    label, clash = mw.classify_pair(b, a)
    assert label == "choque de voz" and clash == pytest.approx(4.0 / 16.0)


def test_short_tracks_use_common_smaller_window():
    # Faixa B de 10 s: N = min(16, 30, 5) = 5 nos dois lados.
    a = va([(40.0, 52.0)], 60.0)            # voz até 8 s antes do fim de A: fora dos últimos 5 s
    b = va([(0.0, 5.0)], 10.0)
    assert mw.classify_pair(a, b) == ("voz de um lado", 0.0)
    a2 = va([(56.0, 60.0)], 60.0)           # últimos 4 s de A
    label, clash = mw.classify_pair(a2, b)
    assert label == "choque de voz" and clash == pytest.approx(4.0 / 5.0)
    # Janela explícita menor
    assert mw.classify_pair(a, va([(0.0, 5.0)], 60.0), window_s=4.0) == ("voz de um lado", 0.0)


def test_zero_length_stem_is_no_stems():
    assert mw.classify_pair(va([], 0.0), va([], 60.0)) == ("sem stems", 0.0)
    assert mw.classify_pair(va([], 60.0), va([], 60.0), window_s=0.0) == ("sem stems", 0.0)


# ---------------------------------------------------------------------------
# make_voice_lookup
# ---------------------------------------------------------------------------

def test_make_voice_lookup_and_memory_cache(tmp_dirs, monkeypatch):
    stems = tmp_dirs["STEMS_DIR"]
    make_stem(stems, "a", 40.0, [(30.0, 40.0)], ext=".wav")     # voz no fim
    make_stem(stems, "b", 40.0, [(0.0, 10.0)], ext=".wav")      # voz no começo
    make_stem(stems, "c", 40.0, [(18.0, 22.0)], ext=".wav")     # voz só no meio (fora das janelas)

    calls = []
    real = mw.load_or_detect
    monkeypatch.setattr(mw, "load_or_detect", lambda v, d=None: calls.append(v) or real(v, d))
    lookup = mw.make_voice_lookup()
    assert lookup("a", "b")[0] == "choque de voz"
    assert lookup("b", "a") == ("limpa", 0.0)
    assert lookup("a", "c") == ("voz de um lado", 0.0)
    assert lookup("c", "b") == ("voz de um lado", 0.0)
    assert lookup("c", "sem_stem") == ("sem stems", 0.0)
    assert lookup("sem_stem", "a") == ("sem stems", 0.0)
    assert sorted(calls) == ["a", "b", "c", "sem_stem"]          # cada faixa carregada uma vez
    lookup.cache_clear()
    lookup("a", "b")
    assert calls.count("a") == 2


def test_lookup_drives_order_set(tmp_dirs):
    from modules import order

    stems = tmp_dirs["STEMS_DIR"]
    make_stem(stems, "a", 40.0, [(30.0, 40.0)], ext=".wav")
    make_stem(stems, "b", 40.0, [(0.0, 10.0)], ext=".wav")
    make_stem(stems, "c", 40.0, [], ext=".wav")
    an = [Analysis(video_id=v, duration_s=40.0, bpm=130.0, bpm_raw=130.0, energy_mean=e)
          for v, e in (("a", 0.1), ("b", 0.5), ("c", 0.5))]
    w = {"bpm": 1.0, "energia": 0.8, "tom": 0.2, "voz": 0.5}
    res = order.order_set(an, w, voice_lookup=mw.make_voice_lookup())
    assert res.order == ["a", "c", "b"]          # evita a→b (choque); ordem natural seria a, b, c
    assert "choque de voz" not in [t.voice for t in res.transitions]


# ---------------------------------------------------------------------------
# Áudio real (opcional)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_detect_on_real_mp3_runs_fast(real_audio_dir):
    files = sorted(real_audio_dir.glob("*.mp3"))
    if not files:
        pytest.skip("sem mp3 na pasta de áudios reais")
    t0 = time.perf_counter()
    res = mw.detect_voice(files[0], files[0].stem)
    elapsed = time.perf_counter() - t0
    print(f"\n{files[0].name}: {res.duration_s:.1f} s, {len(res.intervals)} trechos, {elapsed:.2f} s")
    assert res.duration_s > 30
    assert all(0 <= a < b <= res.duration_s + 1e-6 for a, b in res.intervals)
    assert elapsed < 10
