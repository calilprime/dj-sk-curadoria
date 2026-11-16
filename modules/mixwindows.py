"""Janelas de mixagem a partir do stem de voz (seção 4.8).

CONTRATO (implementar mantendo as assinaturas):

- Com o stem de voz (stems/{videoId}/vocals.mp3), calcula RMS em janelas de VOICE_HOP_S,
  converte para dBFS, marca voz onde passa de VOICE_THRESHOLD_DB, une lacunas menores que
  VOICE_MERGE_GAP_S e descarta trechos menores que VOICE_MIN_SEGMENT_S.
- Cache em `stems/{videoId}/voice.json` (VoiceActivity.to_dict()).
- Transição A→B: sobrepõe a janela de saída de A (últimos N s) com a de entrada de B
  (primeiros N s), N = TRANSITION_WINDOW_S (ou menor, se a faixa for curta).
  Rótulos: "limpa" (nenhum lado tem voz na janela), "voz de um lado" (só um lado),
  "choque de voz" (fração do tempo com voz NOS DOIS LADOS ao mesmo tempo >= VOICE_CLASH_MIN_OVERLAP),
  "sem stems" (falta o stem de algum lado). Se os dois lados têm voz mas a sobreposição
  simultânea fica abaixo do limite, o rótulo é "voz de um lado".

Detalhes da implementação:
- O stem é lido com soundfile (mp3/wav/flac, sem reamostrar; estéreo vira mono pela média);
  se falhar, librosa.load. O RMS em dBFS é independente da taxa de amostragem.
- N = min(janela, duração_A/2, duração_B/2): o mesmo N dos dois lados. Na mixagem, o instante
  t ∈ [0, N] toca A em dur_A − N + t e B em t.
- voice.json guarda, além de VoiceActivity.to_dict(), os parâmetros da detecção e o tamanho/
  data do stem: se o stem for trocado ou os parâmetros mudarem, a detecção é refeita. Sem o
  stem, um voice.json válido continua sendo usado.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

import config
from modules import util
from modules.models import VoiceActivity

logger = logging.getLogger(__name__)

LABEL_CLEAN = "limpa"
LABEL_ONE_SIDE = "voz de um lado"
LABEL_CLASH = "choque de voz"
LABEL_NO_STEMS = "sem stems"

VOICE_CACHE_VERSION = 1
_VOCALS_EXTENSIONS = (".mp3", ".wav", ".flac", *config.LOCAL_AUDIO_EXTENSIONS)
_SILENCE_DB = -120.0        # piso do dBFS (evita log de zero)


class VoiceDetectionError(Exception):
    """Não foi possível ler o stem de voz (mensagem em português)."""


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------

def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return y.mean(axis=1), int(sr)
    except Exception as sf_error:  # formato que o libsndfile não abre: tenta o librosa
        try:
            import librosa

            y, sr = librosa.load(str(path), sr=None, mono=True)
            return np.asarray(y, dtype=np.float32), int(sr)
        except Exception as lr_error:
            logger.debug("Leitura do stem falhou: %s / %s", sf_error, lr_error)
            raise VoiceDetectionError(
                f"Não consegui ler o stem de voz {path.name} (arquivo vazio ou corrompido).") from lr_error


def _frames_db(y: np.ndarray, hop: int) -> np.ndarray:
    """RMS em dBFS por janela de `hop` amostras (a última janela pode ser menor)."""
    starts = np.arange(0, len(y), hop)
    sums = np.add.reduceat(y.astype(np.float64) ** 2, starts)
    counts = np.diff(np.append(starts, len(y)))
    rms = np.sqrt(sums / counts)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms)
    return np.maximum(db, _SILENCE_DB)


def _mask_to_intervals(mask: np.ndarray, hop_s: float, duration_s: float) -> list[tuple[float, float]]:
    if not mask.any():
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.nonzero(edges == 1)[0]
    ends = np.nonzero(edges == -1)[0]
    return [(float(s * hop_s), float(min(e * hop_s, duration_s))) for s, e in zip(starts, ends)]


def _merge_and_filter(intervals: list[tuple[float, float]], max_gap_s: float,
                      min_len_s: float) -> list[tuple[float, float]]:
    """Une lacunas menores que `max_gap_s` e depois descarta trechos menores que `min_len_s`."""
    merged: list[list[float]] = []
    for a, b in sorted(intervals):
        if merged and a - merged[-1][1] < max_gap_s:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in merged if b - a >= min_len_s]


def detect_voice(vocals_path: str | Path, video_id: str) -> VoiceActivity:
    """Detecta trechos com voz no stem (sem cache)."""
    path = Path(vocals_path)
    y, sr = _load_mono(path)
    duration = len(y) / sr if sr > 0 else 0.0
    if len(y) == 0:
        return VoiceActivity(video_id=video_id, duration_s=0.0, intervals=[])
    hop = max(1, int(round(config.VOICE_HOP_S * sr)))
    voiced = _frames_db(y, hop) > config.VOICE_THRESHOLD_DB
    intervals = _mask_to_intervals(voiced, hop / sr, duration)
    intervals = _merge_and_filter(intervals, config.VOICE_MERGE_GAP_S, config.VOICE_MIN_SEGMENT_S)
    return VoiceActivity(video_id=video_id, duration_s=round(duration, 3), intervals=intervals)


# ---------------------------------------------------------------------------
# Cache em disco
# ---------------------------------------------------------------------------

def voice_path(video_id: str, stems_dir: Path | None = None) -> Path:
    """`stems/{videoId}/voice.json`."""
    base = Path(stems_dir) if stems_dir is not None else config.STEMS_DIR
    # Mesmo nome de pasta que o modules/stems.py usa.
    from modules import stems  # mesma pasta que o Estúdio de Vocal usa
    return stems.stems_dir(video_id, base) / "voice.json"


def _find_vocals(folder: Path) -> Path | None:
    for ext in dict.fromkeys(_VOCALS_EXTENSIONS):
        p = folder / f"vocals{ext}"
        try:
            if p.is_file() and p.stat().st_size > 0:   # stem vazio = gravação interrompida
                return p
        except OSError:
            continue
    return None


def _detection_params() -> dict[str, float]:
    return {"hop_s": float(config.VOICE_HOP_S), "threshold_db": float(config.VOICE_THRESHOLD_DB),
            "merge_gap_s": float(config.VOICE_MERGE_GAP_S),
            "min_segment_s": float(config.VOICE_MIN_SEGMENT_S)}


def _source_signature(vocals: Path) -> dict[str, object]:
    st = vocals.stat()
    return {"name": vocals.name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _read_cache(path: Path, video_id: str) -> tuple[VoiceActivity, dict] | None:
    data = util.read_json(path)
    if not isinstance(data, dict):
        return None
    try:
        va = VoiceActivity.from_dict(data)
    except (KeyError, TypeError, ValueError):
        logger.warning("Cache de voz inválido em %s; a detecção será refeita.", path)
        return None
    if va.video_id != video_id:
        return None
    return va, data


def load_or_detect(video_id: str, stems_dir: Path | None = None) -> VoiceActivity | None:
    """Usa o cache; senão detecta a partir de stems/{videoId}/vocals.*; None se não há stem."""
    cache_path = voice_path(video_id, stems_dir)
    vocals = _find_vocals(cache_path.parent)
    cached = _read_cache(cache_path, video_id)

    if vocals is None:
        return cached[0] if cached else None

    signature = _source_signature(vocals)
    params = _detection_params()
    if cached:
        va, data = cached
        if (data.get("version") == VOICE_CACHE_VERSION and data.get("source") == signature
                and data.get("params") == params):
            return va

    try:
        va = detect_voice(vocals, video_id)
    except Exception:
        logger.warning("Falha ao detectar voz no stem de %s; a faixa fica sem janela de voz.",
                       video_id, exc_info=True)
        return None
    try:
        util.atomic_write_json(cache_path, {**va.to_dict(), "version": VOICE_CACHE_VERSION,
                                            "params": params, "source": signature})
    except OSError:
        logger.warning("Não consegui gravar o cache de voz em %s.", cache_path, exc_info=True)
    return va


# ---------------------------------------------------------------------------
# Classificação da transição
# ---------------------------------------------------------------------------

def _clipped_union(intervals: list[tuple[float, float]], start: float, end: float,
                   shift: float = 0.0) -> list[tuple[float, float]]:
    """Intervalos deslocados por `shift`, recortados a [start, end] e unidos (ordenados)."""
    out: list[list[float]] = []
    for a, b in sorted((a + shift, b + shift) for a, b in intervals):
        a, b = max(a, start), min(b, end)
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _total(intervals: list[tuple[float, float]]) -> float:
    return float(sum(b - a for a, b in intervals))


def _intersection(x: list[tuple[float, float]], y: list[tuple[float, float]]) -> float:
    """Comprimento da interseção de duas listas ordenadas de intervalos disjuntos."""
    total, i, j = 0.0, 0, 0
    while i < len(x) and j < len(y):
        lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if hi > lo:
            total += hi - lo
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return total


def voice_fraction(va: VoiceActivity, start: float, end: float) -> float:
    """Fração de [start, end] coberta por voz (0..1)."""
    if not end > start:
        return 0.0
    covered = _total(_clipped_union(va.intervals, start, end))
    return min(max(covered / (end - start), 0.0), 1.0)


def classify_pair(a: VoiceActivity | None, b: VoiceActivity | None,
                  window_s: float | None = None) -> tuple[str, float]:
    """(rótulo, fração de choque 0..1) para a transição A→B."""
    if a is None or b is None:
        return LABEL_NO_STEMS, 0.0
    window = config.TRANSITION_WINDOW_S if window_s is None else float(window_s)
    n = min(window, a.duration_s / 2.0, b.duration_s / 2.0)
    if not n > 0:      # stem vazio ou janela nula: não dá para avaliar
        return LABEL_NO_STEMS, 0.0

    # Linha do tempo da mixagem: t ∈ [0, N]; A toca em dur_A − N + t, B em t.
    a_side = _clipped_union(a.intervals, 0.0, n, shift=-(a.duration_s - n))
    b_side = _clipped_union(b.intervals, 0.0, n)
    if not a_side and not b_side:
        return LABEL_CLEAN, 0.0
    clash = min(_intersection(a_side, b_side) / n, 1.0)
    if a_side and b_side and clash >= config.VOICE_CLASH_MIN_OVERLAP:
        return LABEL_CLASH, clash
    return LABEL_ONE_SIDE, clash


def make_voice_lookup(stems_dir: Path | None = None) -> Callable[[str, str], tuple[str, float]]:
    """Devolve `lookup(from_id, to_id) -> (rótulo, choque)` com cache em memória,
    no formato esperado por `order.order_set(voice_lookup=...)`."""
    activities: dict[str, VoiceActivity | None] = {}

    def get(video_id: str) -> VoiceActivity | None:
        if video_id not in activities:
            activities[video_id] = load_or_detect(video_id, stems_dir)
        return activities[video_id]

    def lookup(from_id: str, to_id: str) -> tuple[str, float]:
        return classify_pair(get(from_id), get(to_id))

    lookup.cache_clear = activities.clear  # type: ignore[attr-defined]  # após extrair novos stems
    return lookup
