"""Fixtures compartilhadas: pastas temporárias e áudio sintético com BPM conhecido."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import config

SR = 22050


def synth_track(bpm: float = 130.0, duration_s: float = 60.0, sr: int = SR,
                intro_s: float = 0.0, outro_s: float = 0.0, tone_hz: float | None = 220.0,
                seed: int = 0) -> np.ndarray:
    """Batida de bumbo em `bpm` com chimbal nos contratempos.

    Durante a intro e a saída o volume cai para 15%, para testar intro/saída.
    `tone_hz` acrescenta um acorde menor (tônica, terça menor, quinta) para testar tom.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm
    kick_len = int(0.12 * sr)
    t = np.arange(kick_len) / sr
    kick = (np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 25)).astype(np.float32)
    hat_len = int(0.03 * sr)
    hat = (rng.standard_normal(hat_len) * np.exp(-np.arange(hat_len) / (0.005 * sr))).astype(np.float32) * 0.3
    k = 0
    while True:
        start = int(k * beat * sr)
        if start >= n:
            break
        end = min(start + kick_len, n)
        y[start:end] += kick[: end - start]
        h = int((k + 0.5) * beat * sr)
        if h < n:
            he = min(h + hat_len, n)
            y[h:he] += hat[: he - h]
        k += 1
    if tone_hz:
        tt = np.arange(n) / sr
        for ratio in (1.0, 2 ** (3 / 12), 2 ** (7 / 12)):
            y += 0.08 * np.sin(2 * np.pi * tone_hz * ratio * tt).astype(np.float32)
    env = np.ones(n, dtype=np.float32)
    if intro_s > 0:
        env[: int(intro_s * sr)] = 0.15
    if outro_s > 0:
        env[n - int(outro_s * sr):] = 0.15
    y *= env
    peak = float(np.max(np.abs(y))) or 1.0
    return (0.9 * y / peak).astype(np.float32)


def synth_vocals(duration_s: float, voice_intervals: list[tuple[float, float]], sr: int = SR) -> np.ndarray:
    """Stem de voz falso: ruído modulado (fala) só nos intervalos indicados, silêncio no resto."""
    rng = np.random.default_rng(1)
    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    for a, b in voice_intervals:
        s, e = int(a * sr), min(int(b * sr), n)
        seg = rng.standard_normal(e - s).astype(np.float32) * 0.3
        mod = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * np.arange(e - s) / sr)
        y[s:e] = seg * mod.astype(np.float32)
    return y


def write_audio(path: Path, y: np.ndarray, sr: int = SR) -> Path:
    """Grava WAV (soundfile) ou MP3 (via ffmpeg) conforme a extensão."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".wav":
        sf.write(path, y, sr)
        return path
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "x.wav"
        sf.write(wav, y, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), str(path)], check=True)
    return path


@pytest.fixture
def tmp_dirs(tmp_path, monkeypatch):
    """Redireciona as pastas do config para uma pasta temporária."""
    dirs = {
        "CACHE_DIR": tmp_path / "cache",
        "STEMS_DIR": tmp_path / "stems",
        "EXPORT_DIR": tmp_path / "Setlist_Atual",
        "DATA_DIR": tmp_path / "data",
    }
    for name, d in dirs.items():
        d.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(config, name, d)
    return dirs


@pytest.fixture
def real_audio_dir():
    """Pasta com mp3 reais (fora do repositório). Defina DJSK_AUDIO_FIXTURES para usar."""
    p = os.getenv("DJSK_AUDIO_FIXTURES")
    if not p or not Path(p).is_dir():
        pytest.skip("DJSK_AUDIO_FIXTURES não definido")
    return Path(p)
