"""Estruturas de dados compartilhadas entre os módulos.

Todas são dataclasses simples, serializáveis com `to_dict()` / `from_dict()`,
para caber em `st.session_state` e em arquivos JSON de cache.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


def _from_dict(cls, data: dict[str, Any]):
    """Cria a dataclass ignorando chaves desconhecidas (cache de versões antigas)."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class Track:
    """Uma faixa candidata (sugestão, referência ou arquivo local)."""

    video_id: str                       # id do YouTube; para arquivos locais: "local_<hash>"
    title: str
    artists: list[str] = field(default_factory=list)
    duration_s: float | None = None
    album: str | None = None
    thumbnail_url: str | None = None
    score: int = 0                      # quantas referências sugeriram esta faixa
    best_rank: int = 0                  # melhor posição em qualquer rádio (desempate; menor é melhor)
    source: str = "ytmusic"             # "ytmusic" | "youtube_api" | "local" | "spotify"
    local_path: str | None = None       # caminho do arquivo quando source == "local"
    isrc: str | None = None

    @property
    def artist_str(self) -> str:
        return ", ".join(self.artists) if self.artists else "Desconhecido"

    @property
    def display_name(self) -> str:
        return f"{self.artist_str} - {self.title}"

    @property
    def is_local(self) -> bool:
        return self.local_path is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        return _from_dict(cls, data)


@dataclass
class Analysis:
    """Resultado da análise de áudio de uma faixa inteira (4.4)."""

    video_id: str
    duration_s: float
    bpm: float                          # BPM estimado, já normalizado para [BPM_MIN, BPM_MAX]
    bpm_raw: float                      # BPM bruto devolvido pelo estimador
    bpm_manual: float | None = None     # correção manual do DJ (tem prioridade sobre `bpm`)
    energy: list[float] = field(default_factory=list)   # RMS por janela de 1 s, normalizada 0..1
    intro_s: float = 0.0                # tempo até a energia passar de INTRO_THRESHOLD
    outro_s: float = 0.0                # tempo entre a última passagem do limiar e o fim
    energy_in: float = 0.0              # energia média dos primeiros TRANSITION_WINDOW_S
    energy_out: float = 0.0             # energia média dos últimos TRANSITION_WINDOW_S
    energy_mean: float = 0.0
    loudness_db: float = 0.0            # RMS médio em dBFS (antes da normalização)
    waveform: list[float] = field(default_factory=list)  # envelope de pico 0..1 (WAVEFORM_POINTS)
    key: str | None = None              # ex.: "A minor"
    camelot: str | None = None          # ex.: "8A"
    key_confidence: float = 0.0
    version: int = 1                    # ANALYSIS_VERSION usada

    @property
    def effective_bpm(self) -> float:
        return self.bpm_manual if self.bpm_manual else self.bpm

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Analysis":
        return _from_dict(cls, data)


@dataclass
class VoiceActivity:
    """Trechos com voz de MC detectados no stem de voz (4.8)."""

    video_id: str
    duration_s: float
    intervals: list[tuple[float, float]] = field(default_factory=list)  # (início, fim) em segundos

    def to_dict(self) -> dict[str, Any]:
        return {"video_id": self.video_id, "duration_s": self.duration_s,
                "intervals": [list(iv) for iv in self.intervals]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceActivity":
        return cls(video_id=data["video_id"], duration_s=float(data["duration_s"]),
                   intervals=[(float(a), float(b)) for a, b in data.get("intervals", [])])


@dataclass
class Transition:
    """Uma transição A→B do set ordenado (4.6 e 4.8)."""

    from_id: str
    to_id: str
    cost: float                         # custo ponderado total
    bpm_diff: float                     # |ΔBPM| usando effective_bpm
    energy_diff: float                  # |energia_saída(A) − energia_entrada(B)|
    camelot_dist: int | None            # None quando algum tom é desconhecido
    voice: str = "sem stems"            # "limpa" | "voz de um lado" | "choque de voz" | "sem stems"
    voice_clash: float = 0.0            # fração da janela com voz dos dois lados (0..1)
    label: str = ""                     # "suave" | "boa" | "atenção" | "difícil"
    explanation: str | None = None      # frase do Gemini (4.11), opcional

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transition":
        return _from_dict(cls, data)


@dataclass
class StemResult:
    """Saída do Estúdio de Vocal (4.7)."""

    video_id: str
    stems: dict[str, str]               # nome do stem ("vocals", "drums", "bass", "other") -> caminho local
    backend: str                        # "replicate" | "local" | "cache"
    predict_time_s: float | None = None
    cost_usd: float = 0.0
    cost_brl: float = 0.0
    prediction_id: str | None = None

    @property
    def vocals_path(self) -> Path | None:
        p = self.stems.get("vocals")
        return Path(p) if p else None
