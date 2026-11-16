"""Análise de áudio com librosa, faixa inteira (seção 4.4).

CONTRATO (implementar mantendo as assinaturas):

- librosa 1.0.0 instalado (Python 3.14). `librosa.beat.beat_track` e
  `librosa.feature.rhythm.tempo` existem. Primeira chamada compila numba (~12 s).
- VALIDADO em 22/10/2026 com 5 faixas reais de mandelão: beat_track deu 129.2 em três,
  mas 80.7 e 107.7 em duas. Ou seja, só dobrar/metade NÃO basta: o estimador deve
  considerar candidatos (tempogram/razões 1/2, 2/3, 3/4, 1, 4/3, 3/2, 2) e escolher o
  mais forte DENTRO de [BPM_MIN, BPM_MAX]; `normalize_bpm` faz o ajuste final de dobro/metade.
- Energia: RMS por janela de ENERGY_WINDOW_S, normalizada 0..1 (divida pelo percentil 95
  e corte em 1). Intro = tempo até a energia passar de INTRO_THRESHOLD pela 1ª vez.
  Saída = do último ponto acima do limiar até o fim.
- Tom: chroma (cqt) médio + perfis de Krumhansl-Kessler (24 tons) → "A minor" e Camelot "8A".
- Resultado salvo em `cache/{videoId}.json` (Analysis.to_dict()).

Estimador de BPM (por que não basta o beat_track):
1. Envelope de onset (hop de 256 amostras, ~11,6 ms) sem a tendência lenta de volume.
2. Autocorrelação global do envelope (tempograma da faixa inteira, até 8 s de atraso).
3. "Força" de um BPM = média da autocorrelação em todos os múltiplos do período
   (filtro em pente). É contínua, então não fica presa à grade de atrasos do tempograma.
4. BPM bruto = pico mais forte entre 40 e 260 BPM (com prior log-normal em 120, como o librosa).
5. Candidatos = bruto × {1/2, 2/3, 3/4, 1, 4/3, 3/2, 2} e os picos do pente dentro da faixa;
   cada um é refinado no pico local (interpolação parabólica). Vence o de maior força
   DENTRO de [lo, hi]; `normalize_bpm` faz o ajuste final e o valor é arredondado a 0,1.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from pathlib import Path

import numpy as np

import config
from modules.models import Analysis
from modules.util import atomic_write_json, read_json, safe_filename

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tons e roda Camelot
# ---------------------------------------------------------------------------
# Roda Camelot: tom -> código. Ex.: "C major" -> "8B", "A minor" -> "8A".
# Grafia canônica (a mesma que `estimate_key` devolve); `camelot_of` aceita enarmônicos.
CAMELOT: dict[str, str] = {
    "B major": "1B", "F# major": "2B", "Db major": "3B", "Ab major": "4B",
    "Eb major": "5B", "Bb major": "6B", "F major": "7B", "C major": "8B",
    "G major": "9B", "D major": "10B", "A major": "11B", "E major": "12B",
    "G# minor": "1A", "Eb minor": "2A", "Bb minor": "3A", "F minor": "4A",
    "C minor": "5A", "G minor": "6A", "D minor": "7A", "A minor": "8A",
    "E minor": "9A", "B minor": "10A", "F# minor": "11A", "C# minor": "12A",
}

_MAJOR_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_MINOR_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B")
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_MAJOR_WORDS = {"", "major", "maj", "ma", "maior", "dur"}
_MINOR_WORDS = {"m", "minor", "min", "mi", "menor", "moll", "-"}
_KEY_RE = re.compile(r"^([A-Ga-g])\s*([#b]{0,2})\s*(.*)$")
_CODE_RE = re.compile(r"^(1[0-2]|0?[1-9])\s*([ABab])$")

# Perfis de Krumhansl-Kessler (tônica em C).
_KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# ---------------------------------------------------------------------------
# Parâmetros internos da análise (não precisam ir para o config.py)
# ---------------------------------------------------------------------------
_ONSET_HOP = 256                 # ~11,6 ms a 22050 Hz: resolução fina do envelope de onset
_AC_MAX_LAG_S = 8.0              # maior atraso da autocorrelação
_DETREND_S = 2.0                 # média móvel retirada do envelope (variações lentas de volume)
_RAW_RANGE = (40.0, 260.0)       # onde se procura o BPM bruto
_PRIOR_CENTER_BPM = 120.0        # prior log-normal do BPM bruto (mesmo centro do librosa)
_RATIOS = (0.5, 2 / 3, 0.75, 1.0, 4 / 3, 1.5, 2.0)
_PEAK_SEARCH = 0.03              # busca o pico local em ±3% de cada candidato
_RANGE_TOL = 0.003               # folga nas bordas da faixa (arredondamento do pico)
_MIN_BPM_AUDIO_S = 2.0           # abaixo disso não há batidas suficientes
_SILENCE_PEAK = 1e-4             # pico abaixo de -80 dBFS = silêncio
_LOUDNESS_FLOOR_DB = -120.0
_CHROMA_HOP = 2048
_CHROMA_FMIN_NOTE = "C2"         # ignora o sub-grave do bumbo (abaixo de ~65 Hz)
_CHROMA_OCTAVES = 6
_TUNING_EXCERPT_S = 60.0         # afinação estimada só num trecho central (custo)

_cache_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Auxiliares de sinal
# ---------------------------------------------------------------------------
def _to_mono(y) -> np.ndarray:
    """Aceita mono, (canais, amostras) ou (amostras, canais); troca NaN/inf por 0."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=0 if y.shape[0] <= y.shape[1] else 1)
    elif y.ndim > 2:
        y = y.reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(y, dtype=np.float32)


def _is_silent(y: np.ndarray) -> bool:
    return y.size == 0 or float(np.max(np.abs(y))) < _SILENCE_PEAK


def _bpm_range(lo: float | None, hi: float | None) -> tuple[float, float]:
    lo = float(config.BPM_MIN if lo is None else lo)
    hi = float(config.BPM_MAX if hi is None else hi)
    return (lo, hi) if lo <= hi else (hi, lo)


# ---------------------------------------------------------------------------
# BPM
# ---------------------------------------------------------------------------
def normalize_bpm(bpm: float, lo: float | None = None, hi: float | None = None) -> float:
    """Dobra/divide por 2 até cair em [lo, hi] (padrão BPM_MIN..BPM_MAX). Se impossível, o mais próximo.

    "Mais próximo" = entre as oitavas do valor (x/2, x, 2x...), a que fica mais perto da faixa.
    Valor não positivo ou inválido devolve o ponto médio da faixa.
    """
    lo, hi = _bpm_range(lo, hi)
    try:
        x = float(bpm)
    except (TypeError, ValueError):
        return (lo + hi) / 2
    if not math.isfinite(x) or x <= 0 or lo <= 0:
        return (lo + hi) / 2
    while x < lo:
        x *= 2
    while x > hi:
        x /= 2
    if x >= lo:
        return x
    up = x * 2                       # a faixa é mais estreita que uma oitava: x < lo e 2x > hi
    return x if (lo - x) <= (up - hi) else up


def _onset_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=_ONSET_HOP)


def _onset_autocorr(oenv: np.ndarray, fps: float) -> np.ndarray | None:
    """Autocorrelação normalizada (ac[0] = 1, sem viés de tamanho) do envelope sem tendência."""
    import librosa
    from scipy.ndimage import uniform_filter1d

    n = len(oenv)
    if n < 8:
        return None
    x = oenv.astype(np.float64)
    x = x - uniform_filter1d(x, size=max(3, int(_DETREND_S * fps)), mode="nearest")
    max_lag = min(int(_AC_MAX_LAG_S * fps), n // 2)
    if max_lag < 4:
        return None
    ac = librosa.autocorrelate(x, max_size=max_lag + 1)
    if not np.isfinite(ac[0]) or ac[0] <= 1e-12:
        return None
    ac = ac / ac[0]
    return ac * (n / (n - np.arange(len(ac))))


def _comb_strength(ac: np.ndarray, fps: float, bpms) -> np.ndarray:
    """Força de cada BPM: média da autocorrelação nos múltiplos do período (filtro em pente)."""
    bpms = np.atleast_1d(np.asarray(bpms, dtype=np.float64))
    max_lag = len(ac) - 1
    period = 60.0 * fps / bpms                       # período em quadros do envelope
    kmax = max(1, int(max_lag / period.min()))
    lags = period[:, None] * np.arange(1, kmax + 1)[None, :]
    valid = lags <= max_lag
    vals = np.interp(np.where(valid, lags, 0.0), np.arange(len(ac)), ac)
    count = valid.sum(axis=1)
    total = np.where(valid, vals, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), -1.0)


def _refine_peak(ac: np.ndarray, fps: float, bpm: float,
                 rel: float = _PEAK_SEARCH) -> tuple[float, float, bool]:
    """Pico local do pente em bpm·(1 ± rel), com interpolação parabólica.

    Devolve (bpm, força, é_pico); é_pico=False quando o máximo cai na borda da busca.
    """
    grid = bpm * np.linspace(1 - rel, 1 + rel, 241)
    s = _comb_strength(ac, fps, grid)
    i = int(np.argmax(s))
    if i == 0 or i == len(grid) - 1:
        return float(grid[i]), float(s[i]), False
    a, b, c = s[i - 1], s[i], s[i + 1]
    denom = a - 2 * b + c
    off = 0.5 * (a - c) / denom if denom < 0 else 0.0
    off = float(np.clip(off, -1.0, 1.0))
    step = grid[1] - grid[0]
    best = float(grid[i] + off * step)
    return best, float(_comb_strength(ac, fps, best)[0]), True


def _bpm_candidates(ac: np.ndarray, fps: float, lo: float,
                    hi: float) -> tuple[float, list[tuple[float, float, str]]]:
    """BPM bruto e candidatos [(força, bpm, origem)] dentro de [lo, hi]."""
    from scipy.signal import find_peaks

    grid = np.geomspace(_RAW_RANGE[0], _RAW_RANGE[1], 3000)
    strength = _comb_strength(ac, fps, grid)
    prior = np.exp(-0.5 * np.log2(grid / _PRIOR_CENTER_BPM) ** 2)
    raw, _, _ = _refine_peak(ac, fps, float(grid[int(np.argmax(strength * prior))]))

    lo_ok, hi_ok = lo * (1 - _RANGE_TOL), hi * (1 + _RANGE_TOL)
    found: list[tuple[float, float, str]] = []

    def consider(bpm: float, origin: str) -> None:
        b, s, is_peak = _refine_peak(ac, fps, bpm)
        if is_peak and lo_ok <= b <= hi_ok and all(abs(b - f[1]) > 0.05 for f in found):
            found.append((s, b, origin))

    for r in _RATIOS:
        c = raw * r
        if lo * (1 - _PEAK_SEARCH) <= c <= hi * (1 + _PEAK_SEARCH):
            consider(c, f"bruto×{r:.3g}")
    # Picos do pente dentro da faixa: cobre o caso em que nenhuma razão do bruto cai nela.
    inner = np.geomspace(lo_ok, hi_ok, 400)
    s_inner = _comb_strength(ac, fps, inner)
    peaks, _ = find_peaks(s_inner)
    for p in peaks:
        consider(float(inner[p]), "pico na faixa")
    return raw, found


def estimate_bpm(y, sr: int, lo: float | None = None, hi: float | None = None) -> tuple[float, float]:
    """Devolve (bpm_normalizado, bpm_bruto) escolhendo o candidato mais forte dentro da faixa.

    Ambos arredondados a 0,1. Silêncio ou áudio curto demais: (ponto médio da faixa, 0.0).
    """
    lo, hi = _bpm_range(lo, hi)
    mid = round((lo + hi) / 2, 1)
    y = _to_mono(y)
    if _is_silent(y) or y.size < _MIN_BPM_AUDIO_S * sr:
        return mid, 0.0
    fps = sr / _ONSET_HOP
    ac = _onset_autocorr(_onset_envelope(y, sr), fps)
    if ac is None:
        return mid, 0.0
    raw, candidates = _bpm_candidates(ac, fps, lo, hi)
    if candidates:
        strength, best, origin = max(candidates)
        log.debug("BPM: bruto %.2f, escolhido %.2f (%s, força %.3f) entre %d candidatos",
                  raw, best, origin, strength, len(candidates))
    else:
        best = raw
        log.debug("BPM: nenhum pico dentro de [%.0f, %.0f]; normalizando o bruto %.2f", lo, hi, raw)
    return round(normalize_bpm(best, lo, hi), 1), round(raw, 1)


# ---------------------------------------------------------------------------
# Tom
# ---------------------------------------------------------------------------
def _parse_key(name: str) -> tuple[int, str] | None:
    """"F#m" / "Gb minor" / "C# major" -> (classe de altura 0..11, "major" | "minor")."""
    s = name.strip().replace("♯", "#").replace("♭", "b")
    m = _KEY_RE.match(s)
    if not m:
        return None
    letter, acc, rest = m.groups()
    pc = (_LETTER_PC[letter.upper()] + acc.count("#") - acc.count("b")) % 12
    rest = rest.strip()
    if rest == "M":
        return pc, "major"
    word = rest.lower().strip(" .")
    if word in _MAJOR_WORDS:
        return pc, "major"
    if word in _MINOR_WORDS:
        return pc, "minor"
    return None


_CAMELOT_BY_PC: dict[tuple[int, str], str] = {_parse_key(k): v for k, v in CAMELOT.items()}  # type: ignore[misc]


def _key_name(pc: int, mode: str) -> str:
    names = _MAJOR_NAMES if mode == "major" else _MINOR_NAMES
    return f"{names[pc % 12]} {mode}"


def camelot_of(key_name: str | None) -> str | None:
    """"A minor" -> "8A"; aceita sustenido/bemol enarmônicos ("Db major" == "C# major").

    Também aceita a forma curta ("Am", "F#m", "Bb") e um código Camelot pronto ("8a" -> "8A").
    Nome inválido devolve None.
    """
    if not key_name or not isinstance(key_name, str):
        return None
    s = key_name.strip()
    code = _CODE_RE.match(s)
    if code:
        return f"{int(code.group(1))}{code.group(2).upper()}"
    parsed = _parse_key(s)
    return _CAMELOT_BY_PC.get(parsed) if parsed else None


def _mean_chroma(y: np.ndarray, sr: int) -> np.ndarray | None:
    """Chroma CQT médio, ponderado pela energia (quadros altos pesam mais que os baixos)."""
    import librosa

    fmin = float(librosa.note_to_hz(_CHROMA_FMIN_NOTE))
    n = y.size
    excerpt = int(_TUNING_EXCERPT_S * sr)
    start = max(0, (n - excerpt) // 2)
    tuning = float(librosa.estimate_tuning(y=y[start:start + excerpt], sr=sr, bins_per_octave=36))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_CHROMA_HOP, fmin=fmin,
                                        n_octaves=_CHROMA_OCTAVES, norm=None, tuning=tuning)
    prof = np.asarray(chroma, dtype=np.float64).mean(axis=1)
    if not np.all(np.isfinite(prof)) or prof.max() <= 1e-12:
        return None
    return prof


def _key_scores(prof: np.ndarray) -> np.ndarray:
    """Correlação de Pearson com os 24 perfis: índices 0..11 maior (tônica C..B), 12..23 menor."""
    p = prof - prof.mean()
    norm_p = np.linalg.norm(p)
    out = np.zeros(24)
    if norm_p <= 1e-12:
        return out
    for mode_i, base in enumerate((_KK_MAJOR, _KK_MINOR)):
        for pc in range(12):
            q = np.roll(base, pc)
            q = q - q.mean()
            out[mode_i * 12 + pc] = float(p @ q / (norm_p * np.linalg.norm(q)))
    return out


def _key_confidence(scores: np.ndarray) -> float:
    """Distância do melhor para o segundo, relativa ao espaço que sobra acima do segundo (0..1)."""
    order = np.argsort(scores)[::-1]
    r1, r2 = float(scores[order[0]]), float(scores[order[1]])
    if r1 <= 0:
        return 0.0
    return float(np.clip((r1 - r2) / max(1.0 - r2, 1e-9), 0.0, 1.0))


def estimate_key(y, sr: int) -> tuple[str | None, str | None, float]:
    """Devolve (nome do tom, código Camelot, confiança 0..1). (None, None, conf) abaixo de KEY_MIN_CONFIDENCE."""
    y = _to_mono(y)
    if _is_silent(y) or y.size < 0.5 * sr:
        return None, None, 0.0
    try:
        prof = _mean_chroma(y, sr)
    except Exception as exc:  # áudio estranho não pode derrubar a análise inteira
        log.warning("Falha ao calcular o chroma: %s", exc)
        return None, None, 0.0
    if prof is None:
        return None, None, 0.0
    scores = _key_scores(prof)
    conf = round(_key_confidence(scores), 3)
    best = int(np.argmax(scores))
    if conf < config.KEY_MIN_CONFIDENCE:
        return None, None, conf
    name = _key_name(best % 12, "major" if best < 12 else "minor")
    return name, CAMELOT[name], conf


# ---------------------------------------------------------------------------
# Energia, waveform
# ---------------------------------------------------------------------------
def _window_rms(y: np.ndarray, sr: int, window_s: float) -> np.ndarray:
    """RMS por janela de `window_s` (a última janela, parcial, usa só as suas amostras)."""
    win = max(1, int(round(window_s * sr)))
    n = y.size
    if n == 0:
        return np.zeros(0)
    nwin = int(math.ceil(n / win))
    pad = nwin * win - n
    sq = (np.pad(y.astype(np.float64), (0, pad)) ** 2).reshape(nwin, win).sum(axis=1)
    counts = np.full(nwin, float(win))
    counts[-1] = win - pad
    return np.sqrt(sq / counts)


def _normalize_energy(rms: np.ndarray) -> np.ndarray:
    if rms.size == 0:
        return rms
    ref = float(np.percentile(rms, 95))
    if ref <= 1e-9:
        ref = float(rms.max())
    if ref <= 1e-9:
        return np.zeros_like(rms)
    return np.clip(rms / ref, 0.0, 1.0)


def _intro_outro(energy: np.ndarray, duration_s: float, window_s: float,
                 threshold: float) -> tuple[float, float]:
    loud = np.flatnonzero(energy >= threshold)
    if loud.size == 0:
        return 0.0, 0.0
    intro = min(float(loud[0]) * window_s, duration_s)
    outro = max(0.0, duration_s - min(float(loud[-1] + 1) * window_s, duration_s))
    return intro, outro


def _edge_energy(energy: np.ndarray, duration_s: float, window_s: float) -> tuple[float, float]:
    """Energia média dos primeiros/últimos TRANSITION_WINDOW_S (no máximo metade da faixa)."""
    if energy.size == 0:
        return 0.0, 0.0
    span = min(float(config.TRANSITION_WINDOW_S), duration_s / 2)
    k = int(min(energy.size, max(1, round(span / window_s))))
    return float(energy[:k].mean()), float(energy[-k:].mean())


def _waveform(y: np.ndarray, points: int) -> list[float]:
    """Pico absoluto por bloco, normalizado 0..1."""
    points = max(1, int(points))
    a = np.abs(y)
    if a.size < points:
        a = np.pad(a, (0, points - a.size))
    edges = np.linspace(0, a.size, points + 1).astype(int)
    peaks = np.maximum.reduceat(a, edges[:-1])
    top = float(peaks.max())
    if top <= 1e-12:
        return [0.0] * points
    return [round(float(v), 4) for v in peaks / top]


def _analyze_signal(y, sr: int, video_id: str) -> Analysis:
    """Análise de um sinal já carregado (separado de `analyze_file` para os testes)."""
    y = _to_mono(y)
    duration = y.size / sr if sr else 0.0
    window_s = float(config.ENERGY_WINDOW_S)

    rms = _window_rms(y, sr, window_s)
    energy = _normalize_energy(rms)
    intro, outro = _intro_outro(energy, duration, window_s, float(config.INTRO_THRESHOLD))
    e_in, e_out = _edge_energy(energy, duration, window_s)
    mean_rms = float(rms.mean()) if rms.size else 0.0
    loudness = 20 * math.log10(mean_rms) if mean_rms > 0 else _LOUDNESS_FLOOR_DB

    bpm, bpm_raw = estimate_bpm(y, sr)
    key, camelot, conf = estimate_key(y, sr)
    return Analysis(
        video_id=video_id,
        duration_s=round(duration, 3),
        bpm=bpm,
        bpm_raw=bpm_raw,
        bpm_manual=None,
        energy=[round(float(v), 4) for v in energy],
        intro_s=round(intro, 2),
        outro_s=round(outro, 2),
        energy_in=round(e_in, 4),
        energy_out=round(e_out, 4),
        energy_mean=round(float(energy.mean()), 4) if energy.size else 0.0,
        loudness_db=round(max(loudness, _LOUDNESS_FLOOR_DB), 2),
        waveform=_waveform(y, config.WAVEFORM_POINTS),
        key=key,
        camelot=camelot,
        key_confidence=conf,
        version=config.ANALYSIS_VERSION,
    )


def analyze_file(path: str | Path, video_id: str) -> Analysis:
    """Analisa o arquivo inteiro (mono, ANALYSIS_SAMPLE_RATE). Não grava cache."""
    import librosa

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de áudio não encontrado: {path}")
    sr = int(config.ANALYSIS_SAMPLE_RATE)
    t0 = time.perf_counter()
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True)
    except Exception as exc:
        raise RuntimeError(f"Não foi possível ler o áudio '{path.name}': {exc}") from exc
    analysis = _analyze_signal(y, sr, video_id)
    log.info("Análise de %s: %.1f s de áudio em %.2f s (BPM %.1f, tom %s)", video_id,
             analysis.duration_s, time.perf_counter() - t0, analysis.bpm, analysis.key or "?")
    return analysis


# ---------------------------------------------------------------------------
# Cache JSON
# ---------------------------------------------------------------------------
def analysis_path(video_id: str, cache_dir: Path | None = None) -> Path:
    """`cache/{videoId}.json`."""
    base = Path(cache_dir) if cache_dir is not None else Path(config.CACHE_DIR)
    return base / f"{safe_filename(str(video_id))}.json"


def _clean_manual_bpm(value) -> float | None:
    """None/0 = sem correção; número positivo e finito = correção; o resto é inválido."""
    if value is None:
        return None
    try:
        bpm = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"BPM manual inválido: {value!r}") from None
    if not math.isfinite(bpm) or bpm < 0:
        raise ValueError(f"BPM manual inválido: {value!r}")
    return round(bpm, 2) if bpm > 0 else None


def _stored_manual_bpm(video_id: str, cache_dir: Path | None) -> float | None:
    """`bpm_manual` gravado no JSON, mesmo de versão antiga ou com outros campos estragados."""
    data = read_json(analysis_path(video_id, cache_dir))
    if not isinstance(data, dict):
        return None
    try:
        return _clean_manual_bpm(data.get("bpm_manual"))
    except ValueError:
        return None


def load_analysis(video_id: str, cache_dir: Path | None = None) -> Analysis | None:
    """Lê do cache; None se não existir, estiver corrompido ou com `version` != ANALYSIS_VERSION."""
    data = read_json(analysis_path(video_id, cache_dir))
    if not isinstance(data, dict):
        return None
    version = data.get("version")
    if isinstance(version, bool) or version != config.ANALYSIS_VERSION:
        return None
    if data.get("video_id") != video_id:
        return None
    try:
        analysis = Analysis.from_dict(data)
        if not (math.isfinite(float(analysis.bpm)) and math.isfinite(float(analysis.duration_s))):
            return None
        analysis.bpm_manual = _clean_manual_bpm(analysis.bpm_manual)
    except (TypeError, ValueError):
        return None
    return analysis


def save_analysis(analysis: Analysis, cache_dir: Path | None = None) -> None:
    with _cache_lock:
        atomic_write_json(analysis_path(analysis.video_id, cache_dir), analysis.to_dict())


def load_or_analyze(video_id: str, path: str | Path, force: bool = False,
                    cache_dir: Path | None = None) -> Analysis:
    """Usa o cache se válido; senão analisa e grava. Preserva `bpm_manual` já salvo."""
    if not force:
        cached = load_analysis(video_id, cache_dir)
        if cached is not None:
            return cached
    analysis = analyze_file(path, video_id)
    # bpm_manual é relido só agora: o DJ pode ter corrigido o BPM durante a análise.
    with _cache_lock:
        analysis.bpm_manual = _stored_manual_bpm(video_id, cache_dir)
        save_analysis(analysis, cache_dir)
    return analysis


def set_manual_bpm(video_id: str, bpm: float | None, cache_dir: Path | None = None) -> Analysis | None:
    """Grava (ou limpa, com None/0) a correção manual de BPM no JSON do cache e devolve a análise.

    Devolve None se a faixa ainda não tem análise válida no cache. BPM negativo,
    infinito ou não numérico gera ValueError.
    """
    value = _clean_manual_bpm(bpm)
    with _cache_lock:
        analysis = load_analysis(video_id, cache_dir)
        if analysis is None:
            return None
        analysis.bpm_manual = value
        save_analysis(analysis, cache_dir)
    return analysis
