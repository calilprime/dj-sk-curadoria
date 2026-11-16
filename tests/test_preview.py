"""Testes da prévia de passagem A→B (modules/preview.py), com áudio sintético."""
from __future__ import annotations

import time

import numpy as np
import pytest
import soundfile as sf

import config
from modules import preview
from tests.conftest import SR, synth_track, write_audio

PEAK = 10 ** (preview.PEAK_DBFS / 20)


@pytest.fixture
def tracks(tmp_path):
    a = write_audio(tmp_path / "src" / "a.wav", synth_track(bpm=130, duration_s=60, seed=1))
    b = write_audio(tmp_path / "src" / "b.wav", synth_track(bpm=130, duration_s=60, tone_hz=330, seed=2))
    return a, b


def _info(path):
    return sf.info(str(path))


def test_preview_path(tmp_path, tmp_dirs):
    p = preview.preview_path("abc", "x/y:z", tmp_path / "c")
    assert p.parent == tmp_path / "c" / "previews"
    assert p.suffix == ".wav" and p.name.startswith("abc__")
    assert "/" not in p.name and ":" not in p.name
    # None = config.CACHE_DIR lido na hora da chamada
    assert preview.preview_path("a", "b") == tmp_dirs["CACHE_DIR"] / "previews" / "a__b.wav"


def test_preview_duration_format_peak_and_fades(tracks, tmp_path):
    a, b = tracks
    out = preview.make_transition_preview(a, b, "a", "b", bpm_a=130, bpm_b=130, cache_dir=tmp_path / "c")
    assert out == preview.preview_path("a", "b", tmp_path / "c")
    info = _info(out)
    assert info.samplerate == preview.PREVIEW_SR
    assert info.channels == 1
    assert info.subtype == "PCM_16"
    assert info.duration == pytest.approx(8 + 12 + 8, abs=0.02)
    y, _ = sf.read(str(out))
    assert np.max(np.abs(y)) == pytest.approx(PEAK, abs=2e-3)
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3
    # sem arquivos temporários sobrando
    assert not list(out.parent.glob("*.tmp"))


def test_custom_windows(tracks, tmp_path):
    a, b = tracks
    out = preview.make_transition_preview(a, b, "a", "b", lead_s=4, overlap_s=6, tail_s=3,
                                          cache_dir=tmp_path / "c")
    assert _info(out).duration == pytest.approx(13, abs=0.02)
    out2 = preview.make_transition_preview(a, b, "a", "b2", lead_s=5, overlap_s=0, tail_s=5,
                                           cache_dir=tmp_path / "c")
    assert _info(out2).duration == pytest.approx(10, abs=0.02)


def test_cache_is_reused_and_force_rebuilds(tracks, tmp_path, monkeypatch):
    a, b = tracks
    calls = []
    real_build = preview._build

    def counting(*args, **kwargs):
        calls.append(args)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(preview, "_build", counting)
    c = tmp_path / "c"
    p1 = preview.make_transition_preview(a, b, "a", "b", bpm_a=130, bpm_b=128, cache_dir=c)
    mtime = p1.stat().st_mtime_ns
    p2 = preview.make_transition_preview(a, b, "a", "b", bpm_a=130, bpm_b=128, cache_dir=c)
    assert p2 == p1 and len(calls) == 1 and p1.stat().st_mtime_ns == mtime
    preview.make_transition_preview(a, b, "a", "b", bpm_a=130, bpm_b=128, cache_dir=c, force=True)
    assert len(calls) == 2
    # parâmetros diferentes (ex.: BPM corrigido pelo DJ) geram a prévia de novo
    preview.make_transition_preview(a, b, "a", "b", bpm_a=130, bpm_b=129, cache_dir=c)
    assert len(calls) == 3
    # prévia antiga sem o arquivo de parâmetros: reaproveita
    preview._meta_path(p1).unlink()
    preview.make_transition_preview(a, b, "a", "b", bpm_a=125, bpm_b=125, cache_dir=c)
    assert len(calls) == 3


def test_short_tracks_shrink_windows(tmp_path):
    a = write_audio(tmp_path / "a.wav", synth_track(bpm=130, duration_s=10))
    b = write_audio(tmp_path / "b.wav", synth_track(bpm=130, duration_s=6, seed=3))
    out = preview.make_transition_preview(a, b, "a", "b", cache_dir=tmp_path / "c")
    f = min(10 / 20, 6 / 20)                          # janelas encolhem na mesma proporção
    assert _info(out).duration == pytest.approx(28 * f, abs=0.02)
    tiny_a = write_audio(tmp_path / "ta.wav", synth_track(duration_s=0.5))
    tiny_b = write_audio(tmp_path / "tb.wav", synth_track(duration_s=0.4, seed=4))
    out2 = preview.make_transition_preview(tiny_a, tiny_b, "ta", "tb", bpm_a=130, bpm_b=125,
                                           cache_dir=tmp_path / "c")
    d = _info(out2).duration
    assert 0 < d <= 28
    y, _ = sf.read(str(out2))
    assert np.all(np.isfinite(y))


def test_silent_track_does_not_break(tmp_path):
    a = write_audio(tmp_path / "a.wav", np.zeros(SR * 30, dtype=np.float32))
    b = write_audio(tmp_path / "b.wav", synth_track(duration_s=30))
    out = preview.make_transition_preview(a, b, "a", "b", cache_dir=tmp_path / "c")
    assert _info(out).duration == pytest.approx(28, abs=0.02)


def test_tempo_rate_rule():
    assert preview.tempo_rate(130, 125) == pytest.approx(130 / 125)
    assert preview.tempo_rate(130, 120.4) == pytest.approx(130 / 120.4)       # 7,97%
    assert preview.tempo_rate(130, 120) is None                               # 8,3%
    assert preview.tempo_rate(130, 110) is None
    assert preview.tempo_rate(130, 130) is None
    assert preview.tempo_rate(None, 130) is None
    assert preview.tempo_rate(130, None) is None
    assert preview.tempo_rate(130, 0) is None
    assert preview.tempo_rate(float("nan"), 130) is None
    assert preview.tempo_rate("x", 130) is None


def test_tempo_match_applied_only_up_to_8_percent(tracks, tmp_path, monkeypatch):
    import librosa

    a, b = tracks
    rates = []
    real = librosa.effects.time_stretch

    def spy(y, *, rate, **kw):
        rates.append(rate)
        return real(y, rate=rate, **kw)

    monkeypatch.setattr(librosa.effects, "time_stretch", spy)
    c = tmp_path / "c"
    out = preview.make_transition_preview(a, b, "a", "b", bpm_a=130, bpm_b=125, cache_dir=c)
    assert rates == [pytest.approx(130 / 125)]
    assert _info(out).duration == pytest.approx(28, abs=0.02)

    preview.make_transition_preview(a, b, "a", "c", bpm_a=130, bpm_b=110, cache_dir=c)
    preview.make_transition_preview(a, b, "a", "d", bpm_a=130, bpm_b=None, cache_dir=c)
    preview.make_transition_preview(a, b, "a", "e", cache_dir=c)
    assert len(rates) == 1

    preview.make_transition_preview(a, b, "a", "f", bpm_a=125, bpm_b=133, cache_dir=c)   # 6%
    assert len(rates) == 2 and rates[-1] == pytest.approx(125 / 133)


def test_best_skip_finds_beat_offset():
    env_a = np.zeros(400)
    env_a[::20] = 1.0
    env_b = np.zeros(440)
    env_b[7::20] = 1.0
    assert preview.best_skip(env_a - env_a.mean(), env_b - env_b.mean(), 20) == 7
    assert preview.best_skip(np.zeros(0), env_b, 20) == 0
    assert preview.best_skip(env_a, env_b, 0) == 0
    assert preview.best_skip(np.zeros(400), env_b, 20) == 0


def test_beats_of_b_are_aligned_with_a(tmp_path):
    """B começa 0,15 s atrasada: a prévia pula esse trecho para as batidas caírem juntas."""
    a = write_audio(tmp_path / "a.wav", synth_track(bpm=130, duration_s=60, seed=1))
    y_b = synth_track(bpm=130, duration_s=60, tone_hz=330, seed=2)
    y_b = np.concatenate((np.zeros(int(0.15 * SR), dtype=np.float32), y_b))
    b = write_audio(tmp_path / "b.wav", y_b)
    c = tmp_path / "c"
    out = preview.make_transition_preview(a, b, "a", "b", bpm_a=130, bpm_b=130, cache_dir=c)
    meta = preview.read_json(preview._meta_path(out))
    assert meta["info"]["skip_s"] == pytest.approx(0.15, abs=0.03)
    out2 = preview.make_transition_preview(a, b, "a", "b_raw", bpm_a=130, bpm_b=130, cache_dir=c,
                                           align_beats=False)
    assert preview.read_json(preview._meta_path(out2))["info"]["skip_s"] == 0.0
    assert _info(out2).duration == pytest.approx(_info(out).duration, abs=0.01)


def test_errors(tracks, tmp_path):
    a, b = tracks
    with pytest.raises(FileNotFoundError):
        preview.make_transition_preview(tmp_path / "nao.mp3", b, "x", "b", cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        preview.make_transition_preview(a, tmp_path / "nao.mp3", "a", "x", cache_dir=tmp_path)
    with pytest.raises(ValueError):
        preview.make_transition_preview(a, b, "a", "b", lead_s=-1, cache_dir=tmp_path)
    with pytest.raises(ValueError):
        preview.make_transition_preview(a, b, "a", "b", lead_s=0, overlap_s=0, tail_s=0,
                                        cache_dir=tmp_path)
    broken = tmp_path / "quebrado.mp3"
    broken.write_bytes(b"isto nao e audio")
    with pytest.raises(preview.PreviewError):
        preview.make_transition_preview(broken, b, "q", "b", cache_dir=tmp_path)


def test_cache_dir_none_uses_config(tracks, tmp_dirs):
    a, b = tracks
    out = preview.make_transition_preview(a, b, "a", "b")
    assert out.parent == config.CACHE_DIR / "previews"
    assert out.is_file()


def test_warm_up_never_raises(monkeypatch):
    assert preview.warm_up() >= 0.0

    def boom(*a, **k):
        raise RuntimeError("falhou")

    monkeypatch.setattr(preview, "_time_stretch", boom)
    assert preview.warm_up() >= 0.0


@pytest.mark.slow
def test_real_mp3_speed(real_audio_dir, tmp_path):
    mp3s = sorted(real_audio_dir.glob("*.mp3"))
    if len(mp3s) < 2:
        pytest.skip("precisa de 2 mp3")
    # No painel o librosa já está aquecido (análise ou warm_up()); a 1ª chamada "fria" do
    # processo leva alguns segundos a mais por compilar código.
    preview.warm_up()
    t0 = time.perf_counter()
    out = preview.make_transition_preview(mp3s[0], mp3s[1], "ra", "rb", bpm_a=130, bpm_b=126,
                                          cache_dir=tmp_path)
    elapsed = time.perf_counter() - t0
    print(f"\nprévia real: {elapsed:.2f} s")
    assert _info(out).duration == pytest.approx(28, abs=0.05)
    assert elapsed < 4.0
