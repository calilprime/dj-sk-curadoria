"""Prévia da passagem A→B ("Ouvir a passagem").

Monta um WAV curto com o fim da música A emendado no começo da música B, do jeito que o
DJ faria na pista:

  [ lead_s só A ][ overlap_s: A saindo e B entrando ][ tail_s só B ]

Passos: lê só os trechos necessários (mono, 44,1 kHz); se as velocidades (BPM) são
próximas (até 8%), acelera ou desacelera B para a velocidade de A; alinha as batidas de B
com as de A (opcional, padrão ligado); iguala o volume de B ao de A; faz a troca com
crossfade de potência constante; põe um fade-in curto no início e um fade-out no fim;
normaliza o pico em -1 dBFS e grava WAV 16-bit de forma atômica em
cache/previews/{A}__{B}.wav. Se a prévia já existe com os mesmos parâmetros, só devolve o
caminho (a não ser com force=True).
"""
from __future__ import annotations

import logging
import math
import os
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

import config
from modules.util import atomic_write_json, read_json, safe_filename

logger = logging.getLogger(__name__)

PREVIEW_SR = 44100
PREVIEW_DIRNAME = "previews"
PREVIEW_VERSION = 1                 # aumente para invalidar prévias antigas
MAX_TEMPO_MATCH = 0.08              # só ajusta a velocidade de B se a diferença for até 8%
_MIN_TEMPO_CHANGE = 1e-3            # abaixo disso não vale a pena mexer na velocidade
FADE_IN_S = 0.3
FADE_OUT_S = 1.5
PEAK_DBFS = -1.0
_GAIN_LIMITS = (0.25, 4.0)          # ajuste de volume de B limitado a ±12 dB
_ONSET_HOP = 512
_DEFAULT_BEAT_S = 0.5               # 1 batida a 120 BPM: busca de alinhamento sem BPM
_READ_MARGIN_S = 1.0                # folga na leitura do fim de A


class PreviewError(RuntimeError):
    """Falha ao montar a prévia (arquivo ilegível etc.). Mensagem pronta para o usuário."""


def preview_path(a_id: str, b_id: str, cache_dir: Path | None = None) -> Path:
    """`{cache}/previews/{A}__{B}.wav`. cache_dir=None lê config.CACHE_DIR na hora."""
    base = Path(cache_dir) if cache_dir is not None else Path(config.CACHE_DIR)
    name = f"{safe_filename(str(a_id))}__{safe_filename(str(b_id))}.wav"
    return base / PREVIEW_DIRNAME / name


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


# ---------------------------------------------------------------------------
# Leitura dos trechos
# ---------------------------------------------------------------------------
def _duration(path: Path) -> float:
    import soundfile as sf

    try:
        return float(sf.info(str(path)).duration)
    except Exception:
        pass
    import librosa

    try:
        return float(librosa.get_duration(path=str(path)))
    except Exception as exc:
        raise PreviewError(f"Não foi possível ler o áudio '{path.name}': {exc}") from exc


def _time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    import librosa

    with warnings.catch_warnings():
        # librosa 1.0 avisa sobre parâmetros que ele mesmo repassa ao phase_vocoder.
        warnings.simplefilter("ignore", FutureWarning)
        return np.asarray(librosa.effects.time_stretch(y, rate=rate), dtype=np.float32)


def _load_segment(path: Path, offset_s: float, duration_s: float) -> np.ndarray:
    """Trecho mono em PREVIEW_SR (float32). Nunca devolve None."""
    import librosa

    try:
        y, _ = librosa.load(str(path), sr=PREVIEW_SR, mono=True,
                            offset=max(float(offset_s), 0.0), duration=max(float(duration_s), 0.01))
    except Exception as exc:
        raise PreviewError(f"Não foi possível ler o áudio '{path.name}': {exc}") from exc
    y = np.asarray(y, dtype=np.float32)
    if not np.all(np.isfinite(y)):
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return y


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    """Corta ou completa com silêncio até n amostras."""
    if len(y) >= n:
        return y[:n]
    return np.concatenate((y, np.zeros(n - len(y), dtype=np.float32)))


# ---------------------------------------------------------------------------
# Velocidade, alinhamento e volume
# ---------------------------------------------------------------------------
def tempo_rate(bpm_a: float | None, bpm_b: float | None) -> float | None:
    """Fator para tocar B na velocidade de A (bpm_a / bpm_b), ou None se não deve ajustar
    (BPM ausente/ inválido, diferença acima de 8% ou praticamente igual)."""
    try:
        a, b = float(bpm_a), float(bpm_b)          # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(a) and math.isfinite(b)) or a <= 0 or b <= 0:
        return None
    rate = a / b
    if abs(a - b) / b > MAX_TEMPO_MATCH + 1e-9:
        return None
    if abs(rate - 1.0) < _MIN_TEMPO_CHANGE:
        return None
    return rate


def _onset_env(y: np.ndarray) -> np.ndarray:
    import librosa

    if len(y) < 2048 or not np.any(y):
        return np.zeros(0, dtype=np.float32)
    env = librosa.onset.onset_strength(y=y, sr=PREVIEW_SR, hop_length=_ONSET_HOP)
    env = np.asarray(env, dtype=np.float64)
    env = env - env.mean()
    return env


def best_skip(env_a: np.ndarray, env_b: np.ndarray, max_skip: int) -> int:
    """Quantos quadros pular no começo de B para as batidas caírem junto com as de A.

    Procura em [0, max_skip) o deslocamento que maximiza Σ env_a[t]·env_b[t + skip].
    Sem sinal útil devolve 0.
    """
    max_skip = int(max(max_skip, 0))
    if max_skip <= 1 or len(env_a) == 0 or len(env_b) == 0:
        return 0
    n = min(len(env_a), len(env_b) - max_skip)
    if n <= 4:
        return 0
    a = env_a[:n]
    if not np.any(a):
        return 0
    scores = np.array([float(np.dot(a, env_b[s:s + n])) for s in range(max_skip)])
    if not np.all(np.isfinite(scores)) or scores.max() <= 0:
        return 0
    return int(np.argmax(scores))


def _rms(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


def _match_gain(ref: np.ndarray, y: np.ndarray) -> float:
    r_ref, r_y = _rms(ref), _rms(y)
    if r_ref <= 1e-6 or r_y <= 1e-6:
        return 1.0
    lo, hi = _GAIN_LIMITS
    return float(min(max(r_ref / r_y, lo), hi))


# ---------------------------------------------------------------------------
# Montagem
# ---------------------------------------------------------------------------
def _scale_windows(dur_a: float, dur_b: float, lead_s: float, overlap_s: float,
                   tail_s: float) -> tuple[float, float, float]:
    """Encolhe as três janelas na mesma proporção quando alguma faixa é curta demais."""
    need_a, need_b = lead_s + overlap_s, overlap_s + tail_s
    factors = [1.0]
    if need_a > 0:
        factors.append(max(dur_a, 0.0) / need_a)
    if need_b > 0:
        factors.append(max(dur_b, 0.0) / need_b)
    f = max(min(factors), 0.0)
    return lead_s * f, overlap_s * f, tail_s * f


def _build(path_a: Path, path_b: Path, bpm_a: float | None, bpm_b: float | None,
           lead_s: float, overlap_s: float, tail_s: float, align_beats: bool) -> tuple[np.ndarray, dict]:
    sr = PREVIEW_SR
    dur_a, dur_b = _duration(path_a), _duration(path_b)
    lead_s, overlap_s, tail_s = _scale_windows(dur_a, dur_b, lead_s, overlap_s, tail_s)
    n_lead, n_ov, n_tail = (int(round(x * sr)) for x in (lead_s, overlap_s, tail_s))
    n_a, n_b = n_lead + n_ov, n_ov + n_tail
    if n_a + n_tail <= 0:
        raise PreviewError("As músicas são curtas demais para montar a prévia.")

    # A: os últimos (lead + overlap) segundos. Lê com 1 s de folga antes e fica com as
    # últimas amostras lidas: a duração de mp3 é estimada e pode passar um pouco do fim real.
    if n_a > 0:
        raw_a = _load_segment(path_a, dur_a - n_a / sr - _READ_MARGIN_S, n_a / sr + _READ_MARGIN_S + 0.05)
        seg_a = raw_a[-n_a:] if len(raw_a) >= n_a else \
            np.concatenate((np.zeros(n_a - len(raw_a), dtype=np.float32), raw_a))
    else:
        seg_a = np.zeros(0, dtype=np.float32)

    # B: os primeiros (overlap + tail) segundos, mais folga para velocidade e alinhamento.
    rate = tempo_rate(bpm_a, bpm_b)
    beat_s = min(max(60.0 / float(bpm_a), 0.25), 1.0) if _valid_bpm(bpm_a) else _DEFAULT_BEAT_S
    slack_s = beat_s if (align_beats and n_ov > 0) else 0.0
    out_need_s = n_b / sr + slack_s
    src_need_s = out_need_s * (rate or 1.0) + 0.05
    seg_b = _load_segment(path_b, 0.0, min(src_need_s, max(dur_b, 0.01) + 0.05))
    if rate is not None and len(seg_b) > 2048:
        seg_b = _time_stretch(seg_b, rate)

    skip = 0
    if align_beats and n_ov > 0 and slack_s > 0:
        env_a = _onset_env(seg_a[n_lead:n_lead + n_ov])
        env_b = _onset_env(seg_b[:n_ov + int(slack_s * sr)])
        max_frames = int(round(slack_s * sr / _ONSET_HOP))
        skip = best_skip(env_a, env_b, max_frames) * _ONSET_HOP
        if len(seg_b) - skip < n_b:          # sem folga suficiente: não alinha
            skip = 0
    seg_b = _fit(seg_b[skip:], n_b)

    # Volume de B igual ao de A.
    gain = _match_gain(seg_a, seg_b)
    seg_b = seg_b * np.float32(gain)

    # Crossfade de potência constante (cos/sin) na sobreposição.
    total = n_lead + n_ov + n_tail
    out = np.zeros(total, dtype=np.float32)
    if n_ov > 0:
        theta = np.linspace(0.0, np.pi / 2, n_ov, dtype=np.float64)
        fade_out_a, fade_in_b = np.cos(theta).astype(np.float32), np.sin(theta).astype(np.float32)
    else:
        fade_out_a = fade_in_b = np.zeros(0, dtype=np.float32)
    out[:n_lead] = seg_a[:n_lead]
    out[n_lead:n_lead + n_ov] = seg_a[n_lead:n_lead + n_ov] * fade_out_a + seg_b[:n_ov] * fade_in_b
    out[n_lead + n_ov:] = seg_b[n_ov:n_ov + n_tail]

    # Fades nas pontas (limitados a metade da prévia, para prévias curtinhas).
    n_in = min(int(FADE_IN_S * sr), total // 2)
    n_out = min(int(FADE_OUT_S * sr), total // 2)
    if n_in > 0:
        out[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)
    if n_out > 0:
        out[total - n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)

    peak = float(np.max(np.abs(out))) if total else 0.0
    if peak > 1e-9:
        out *= np.float32(10 ** (PEAK_DBFS / 20) / peak)
    info = {"lead_s": lead_s, "overlap_s": overlap_s, "tail_s": tail_s, "rate": rate,
            "skip_s": skip / sr, "gain_b": gain}
    return out, info


def _valid_bpm(bpm: float | None) -> bool:
    try:
        v = float(bpm)                               # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return math.isfinite(v) and v > 0


def _write_wav_atomic(path: Path, y: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    os.close(fd)
    try:
        sf.write(tmp, y, sr, format="WAV", subtype="PCM_16")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _source_sig(path: Path) -> list:
    try:
        st = path.stat()
        return [path.name, int(st.st_size), int(st.st_mtime)]
    except OSError:
        return [path.name, None, None]


def warm_up() -> float:
    """Prepara o librosa (a 1ª prévia do processo compila código e demora alguns segundos a
    mais). Pode rodar numa thread no início do painel. Nunca levanta erro; devolve os
    segundos gastos."""
    t0 = time.perf_counter()
    try:
        rng = np.random.default_rng(0)
        y = (rng.standard_normal(PREVIEW_SR * 2) * 0.1).astype(np.float32)
        _onset_env(y)
        _time_stretch(y, 1.02)
    except Exception:
        logger.warning("Falha ao preparar a prévia de passagem.", exc_info=True)
    elapsed = time.perf_counter() - t0
    logger.info("Prévia de passagem preparada em %.2f s.", elapsed)
    return elapsed


def make_transition_preview(path_a: str | Path, path_b: str | Path, a_id: str, b_id: str,
                            bpm_a: float | None = None, bpm_b: float | None = None,
                            lead_s: float = 8.0, overlap_s: float = 12.0, tail_s: float = 8.0,
                            cache_dir: Path | None = None, force: bool = False,
                            align_beats: bool = True) -> Path:
    """Gera (ou reaproveita) a prévia da passagem A→B e devolve o caminho do WAV.

    bpm_a/bpm_b: velocidades (use `Analysis.effective_bpm`); com as duas e diferença de até
    8%, B é tocada na velocidade de A. Faixas mais curtas que as janelas encolhem as janelas
    na mesma proporção. Levanta FileNotFoundError se algum arquivo não existe, ValueError se
    as janelas são inválidas e PreviewError se o áudio não pode ser lido.
    """
    path_a, path_b = Path(path_a), Path(path_b)
    for p in (path_a, path_b):
        if not p.is_file():
            raise FileNotFoundError(f"Arquivo de áudio não encontrado: {p}")
    windows = []
    for name, v in (("lead_s", lead_s), ("overlap_s", overlap_s), ("tail_s", tail_s)):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{name} inválido: {v!r}") from None
        if not math.isfinite(fv) or fv < 0:
            raise ValueError(f"{name} precisa ser um número >= 0 (recebi {v!r}).")
        windows.append(fv)
    lead_s, overlap_s, tail_s = windows
    if lead_s + overlap_s + tail_s <= 0:
        raise ValueError("A prévia precisa ter alguma duração.")

    out_path = preview_path(a_id, b_id, cache_dir)
    params = {
        "version": PREVIEW_VERSION, "lead_s": lead_s, "overlap_s": overlap_s, "tail_s": tail_s,
        "bpm_a": float(bpm_a) if _valid_bpm(bpm_a) else None,
        "bpm_b": float(bpm_b) if _valid_bpm(bpm_b) else None,
        "align_beats": bool(align_beats),
        "source_a": _source_sig(path_a), "source_b": _source_sig(path_b),
    }
    meta_path = _meta_path(out_path)
    if not force and out_path.is_file():
        meta = read_json(meta_path)
        # Sem o arquivo de parâmetros (prévia antiga): usa a prévia que já existe.
        if meta is None or meta.get("params") == params:
            return out_path

    t0 = time.perf_counter()
    y, info = _build(path_a, path_b, bpm_a, bpm_b, lead_s, overlap_s, tail_s, bool(align_beats))
    _write_wav_atomic(out_path, y, PREVIEW_SR)
    try:
        atomic_write_json(meta_path, {"params": params, "info": info})
    except OSError:
        logger.warning("Não foi possível gravar os parâmetros da prévia %s.", out_path.name,
                       exc_info=True)
    logger.info("Prévia %s -> %s: %.1f s de áudio em %.2f s (velocidade %s, alinhamento %.2f s)",
                a_id, b_id, len(y) / PREVIEW_SR, time.perf_counter() - t0,
                f"x{info['rate']:.3f}" if info["rate"] else "original", info["skip_s"])
    return out_path
