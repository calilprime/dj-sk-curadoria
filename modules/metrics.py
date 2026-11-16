"""Medição de tempo para provar o valor (seção 6).

CONTRATO (implementar mantendo as assinaturas):

Por sessão registra: início (clique em "Buscar sugestões"), fim (exportação), faixas
aprovadas e revisadas. Tempo médio por faixa aprovada = (fim − início) / aprovadas.
Compara com BASELINE_SECONDS_PER_TRACK (60–90 s por música no processo manual) e projeta
para um set de TYPICAL_SET_SIZE músicas. Histórico em `data/metricas.json` (lista de sessões).
Usa relógio injetável (`clock`) para os testes.

Além do cronômetro, a sessão guarda cada decisão da triagem (aprovada/reprovada, com a hora)
para mostrar ao vivo: ritmo de decisão, taxa de aprovação, tempo economizado, custo por
faixa e o funil "sugeridas → baixadas → decididas → aprovadas → exportadas" (`live_stats`).
"""
from __future__ import annotations

import json
import logging
import math
import statistics
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from modules.util import atomic_write_json

logger = logging.getLogger(__name__)

METRICS_FILENAME = "metricas.json"
APPROVED = "aprovada"
REJECTED = "reprovada"
DECISION_VALUES = (APPROVED, REJECTED)
FUNNEL_LABELS = ("Sugeridas", "Baixadas", "Decididas", "Aprovadas", "Exportadas")
_lock = threading.Lock()


@dataclass
class Comparison:
    seconds_per_track: float            # tempo médio no painel
    baseline_min_s: float               # 60
    baseline_max_s: float               # 90
    speedup_min: float                  # baseline_min / painel
    speedup_max: float                  # baseline_max / painel
    saved_per_track_min_s: float        # baseline_min − painel (pode ser negativo)
    saved_per_track_max_s: float
    set_size: int
    set_minutes_panel: float            # projeção do set inteiro no painel
    set_minutes_baseline_min: float
    set_minutes_baseline_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _baseline() -> tuple[float, float]:
    base_min, base_max = (float(v) for v in config.BASELINE_SECONDS_PER_TRACK)
    return base_min, base_max


def compare_to_baseline(seconds_per_track: float, set_size: int | None = None) -> Comparison:
    """Compara o tempo médio por faixa com o processo manual.

    Sem medição (tempo <= 0, ex.: nenhuma faixa aprovada) não há ganho a calcular:
    ganho, economia e projeção do painel ficam 0 (a linha de base continua preenchida).
    """
    base_min, base_max = _baseline()
    size = max(0, int(config.TYPICAL_SET_SIZE if set_size is None else set_size))
    spt = float(seconds_per_track or 0.0)
    if not math.isfinite(spt) or spt <= 0:
        spt = 0.0
    measured = spt > 0
    return Comparison(
        seconds_per_track=spt,
        baseline_min_s=base_min,
        baseline_max_s=base_max,
        speedup_min=base_min / spt if measured else 0.0,
        speedup_max=base_max / spt if measured else 0.0,
        saved_per_track_min_s=base_min - spt if measured else 0.0,
        saved_per_track_max_s=base_max - spt if measured else 0.0,
        set_size=size,
        set_minutes_panel=spt * size / 60.0,
        set_minutes_baseline_min=base_min * size / 60.0,
        set_minutes_baseline_max=base_max * size / 60.0,
    )


def _iso(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):  # relógio fora da faixa do Windows (ex.: negativo)
        return str(ts)


def _num(value: Any) -> float | None:
    """Número finito ou None (aceita int/float/str numérica; bool não conta)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _count(value: Any) -> int:
    """Contagem não negativa (valores inválidos viram 0)."""
    v = _num(value)
    return max(0, int(v)) if v is not None else 0


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def _normalize_decision(decision: Any) -> str | None:
    text = str(decision or "").strip().lower()
    return text if text in DECISION_VALUES else None


class SessionMetrics:
    def __init__(self, path: Path | None = None, clock: Callable[[], float] = time.time):
        """path None = DATA_DIR/'metricas.json'."""
        self._path = Path(path) if path is not None else None
        self._clock = clock
        self._session_id: str | None = None
        self._start: float | None = None
        self._end: float | None = None
        # video_id -> (decisão, hora); a ordem do dicionário segue a hora da última decisão.
        self._decisions: dict[str, tuple[str, float]] = {}

    @property
    def path(self) -> Path:
        # Lido na hora do uso, para seguir config.DATA_DIR (os testes trocam a pasta).
        return self._path if self._path is not None else Path(config.DATA_DIR) / METRICS_FILENAME

    def start(self) -> None:
        """Marca o início (chamado ao buscar sugestões). Recomeça se já havia início sem fim.

        Uma sessão nova começa sem decisões (as da sessão anterior já foram para o histórico).
        """
        # Depois de uma exportação, buscar sugestões de novo abre uma sessão nova no histórico.
        self._session_id = uuid.uuid4().hex[:12]
        self._start = float(self._clock())
        self._end = None
        self._decisions = {}

    @property
    def started(self) -> bool:
        return self._start is not None

    @property
    def finished(self) -> bool:
        return self._end is not None

    def elapsed_s(self) -> float:
        """Tempo desde o início (até o fim, se já exportou). 0 se não começou."""
        if self._start is None:
            return 0.0
        end = self._end if self._end is not None else float(self._clock())
        return max(0.0, end - self._start)

    # ------------------------------------------------------------------
    # Decisões da triagem
    # ------------------------------------------------------------------
    def record_decision(self, video_id: str, decision: str) -> None:
        """Registra "aprovada" ou "reprovada" para a faixa, com a hora do relógio.

        Uma nova decisão da mesma faixa substitui a anterior (e passa a contar a partir de agora).
        ValueError para faixa vazia ou decisão desconhecida.
        """
        vid = str(video_id or "").strip()
        if not vid:
            raise ValueError("Faixa sem identificador.")
        value = _normalize_decision(decision)
        if value is None:
            raise ValueError(f"Decisão desconhecida: {decision!r} (use 'aprovada' ou 'reprovada').")
        self._decisions.pop(vid, None)          # reinsere no fim: a ordem segue a hora
        self._decisions[vid] = (value, float(self._clock()))

    def undo_decision(self, video_id: str) -> bool:
        """Desfaz a decisão da faixa. True se havia uma decisão para desfazer."""
        return self._decisions.pop(str(video_id or "").strip(), None) is not None

    @property
    def decisions(self) -> dict[str, tuple[str, float]]:
        """Cópia de video_id -> (decisão, hora da decisão)."""
        return dict(self._decisions)

    def _decision_counts(self) -> tuple[int, int]:
        approved = sum(1 for d, _ in self._decisions.values() if d == APPROVED)
        return approved, len(self._decisions) - approved

    def _decision_intervals(self) -> list[float]:
        """Segundos entre decisões consecutivas; a 1ª conta a partir do início da sessão."""
        times = sorted(ts for _, ts in self._decisions.values())
        if not times:
            return []
        anchors = ([self._start] if self._start is not None else []) + times
        return [max(0.0, b - a) for a, b in zip(anchors, anchors[1:])]

    def live_stats(self, n_suggested: int, n_downloaded: int, n_exported: int | None = None,
                   spent_brl: float = 0.0, set_size: int | None = None) -> dict[str, Any]:
        """Números ao vivo da sessão para o painel de métricas.

        Chaves: elapsed_s, decisions, approved, rejected, approval_rate (0..1),
        avg_decision_s, median_decision_s, s_per_approved, baseline_lo_s, baseline_hi_s,
        speedup_lo, speedup_hi, saved_min_lo, saved_min_hi (nunca negativos),
        saved_min_lo_raw, saved_min_hi_raw (podem ser negativos), set_projection
        {size, panel_min, manual_lo_min, manual_hi_min}, cost_brl, cost_per_approved_brl,
        funnel [("Sugeridas", n), ("Baixadas", n), ("Decididas", n), ("Aprovadas", n),
        ("Exportadas", n)], started, finished.
        Valores que não fazem sentido ainda (ex.: nenhuma aprovada) vêm como None.
        """
        elapsed = self.elapsed_s()
        approved, rejected = self._decision_counts()
        n_decisions = approved + rejected
        intervals = self._decision_intervals()
        base_lo, base_hi = _baseline()

        spa = elapsed / approved if approved and elapsed > 0 else None
        saved_lo_raw = (approved * base_lo - elapsed) / 60.0
        saved_hi_raw = (approved * base_hi - elapsed) / 60.0

        size = _count(config.TYPICAL_SET_SIZE if set_size is None else set_size)
        cost = _num(spent_brl)
        cost = max(0.0, cost) if cost is not None else 0.0

        return {
            "started": self.started,
            "finished": self.finished,
            "elapsed_s": elapsed,
            "decisions": n_decisions,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": _ratio(approved, n_decisions),
            "avg_decision_s": statistics.fmean(intervals) if intervals else None,
            "median_decision_s": float(statistics.median(intervals)) if intervals else None,
            "s_per_approved": spa,
            "baseline_lo_s": base_lo,
            "baseline_hi_s": base_hi,
            "speedup_lo": base_lo / spa if spa else None,
            "speedup_hi": base_hi / spa if spa else None,
            "saved_min_lo": max(0.0, saved_lo_raw),
            "saved_min_hi": max(0.0, saved_hi_raw),
            "saved_min_lo_raw": saved_lo_raw,
            "saved_min_hi_raw": saved_hi_raw,
            "set_projection": {
                "size": size,
                "panel_min": spa * size / 60.0 if spa else None,
                "manual_lo_min": base_lo * size / 60.0,
                "manual_hi_min": base_hi * size / 60.0,
            },
            "cost_brl": cost,
            "cost_per_approved_brl": cost / approved if approved else None,
            "funnel": list(zip(FUNNEL_LABELS, (
                _count(n_suggested), _count(n_downloaded), n_decisions, approved,
                _count(n_exported),
            ))),
        }

    # ------------------------------------------------------------------
    def finish(self, n_approved: int, n_reviewed: int) -> dict:
        """Marca a exportação, grava a sessão no histórico e devolve o registro
        {"inicio", "fim", "duracao_s", "aprovadas", "revisadas", "s_por_faixa", "comparacao": {...},
         "decisoes", "reprovadas", "taxa_aprovacao", "s_por_decisao"}.
        Exportar de novo na mesma sessão atualiza o mesmo registro (não duplica).

        "decisoes"/"reprovadas"/"taxa_aprovacao" vêm das decisões registradas
        (`record_decision`); sem elas, de `n_reviewed` e `n_approved`. "taxa_aprovacao"
        (0..1) e "s_por_decisao" ficam None quando não há decisão.

        Sem `start()` antes, devolve o registro com duração 0 e não grava no histórico.
        """
        now = float(self._clock())
        approved = max(0, int(n_approved or 0))
        reviewed = max(0, int(n_reviewed or 0))
        if self._start is None:
            logger.warning("Exportação sem início de sessão marcado; tempo não registrado.")
            return self._record(None, now, now, approved, reviewed)

        self._end = now
        record = self._record(self._session_id, self._start, now, approved, reviewed)
        try:
            self._upsert(record)
        except OSError as exc:
            logger.warning("Não foi possível gravar o histórico de métricas em %s: %s", self.path, exc)
        return record

    def history(self) -> list[dict]:
        # No Windows, ler durante a troca atômica do arquivo faz a gravação falhar: espera o lock.
        with _lock:
            return self._read_history(quarantine=False)

    # ------------------------------------------------------------------
    # Persistência dentro da sessão salva (modules/state.py, chave "metrics")
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self._session_id,
            "start": self._start,
            "end": self._end,
            "decisions": {vid: [d, ts] for vid, (d, ts) in self._decisions.items()},
        }

    def restore(self, data: dict[str, Any] | None) -> None:
        """Retoma o cronômetro e as decisões salvos por `to_dict()` (ignora dados inválidos).

        Dados antigos, sem "decisions", continuam válidos (a sessão volta sem decisões).
        """
        if not isinstance(data, dict):
            return
        try:
            start = data.get("start")
            end = data.get("end")
            self._start = float(start) if start is not None else None
            self._end = float(end) if end is not None and start is not None else None
            sid = data.get("session_id")
            self._session_id = str(sid) if sid else (uuid.uuid4().hex[:12] if start is not None else None)
        except (TypeError, ValueError):
            logger.warning("Métricas salvas inválidas; cronômetro zerado.")
            self._session_id = self._start = self._end = None
        self._decisions = _parse_decisions(data.get("decisions")) if self._start is not None else {}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, path: Path | None = None,
                  clock: Callable[[], float] = time.time) -> "SessionMetrics":
        m = cls(path=path, clock=clock)
        m.restore(data)
        return m

    # ------------------------------------------------------------------
    def _record(self, session_id: str | None, start: float, end: float,
                approved: int, reviewed: int) -> dict:
        duration = max(0.0, end - start)
        spt = duration / approved if approved > 0 else 0.0
        if self._decisions:
            own_approved, rejected = self._decision_counts()
            n_decisions = own_approved + rejected
            rate = _ratio(own_approved, n_decisions)
        else:  # decisões não registradas aqui: usa as contagens recebidas
            n_decisions = reviewed
            rejected = max(0, reviewed - approved)
            rate = _ratio(min(approved, reviewed), reviewed)
        intervals = self._decision_intervals() if session_id is not None else []
        if intervals:
            s_per_decision: float | None = statistics.fmean(intervals)
        elif n_decisions and duration > 0:
            s_per_decision = duration / n_decisions
        else:
            s_per_decision = None
        return {
            "id": session_id,
            "inicio": _iso(start),
            "fim": _iso(end),
            "duracao_s": round(duration, 2),
            "aprovadas": approved,
            "revisadas": reviewed,
            "s_por_faixa": round(spt, 2),
            "comparacao": compare_to_baseline(spt).to_dict(),
            "decisoes": n_decisions,
            "reprovadas": rejected,
            "taxa_aprovacao": round(rate, 4) if rate is not None else None,
            "s_por_decisao": round(s_per_decision, 2) if s_per_decision is not None else None,
        }

    def _read_history(self, quarantine: bool) -> list[dict]:
        return _read_history_file(self.path, quarantine)

    def _upsert(self, record: dict) -> None:
        with _lock:
            entries = self._read_history(quarantine=True)
            for i, e in enumerate(entries):
                if record["id"] is not None and e.get("id") == record["id"]:
                    entries[i] = record
                    break
            else:
                entries.append(record)
            atomic_write_json(self.path, entries)


def _parse_decisions(raw: Any) -> dict[str, tuple[str, float]]:
    """Lê as decisões salvas ({vid: [decisão, hora]}), descartando itens inválidos."""
    if not isinstance(raw, dict):
        return {}
    items: list[tuple[str, str, float]] = []
    for vid, value in raw.items():
        if isinstance(value, dict):
            decision, ts = value.get("decision"), value.get("ts")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            decision, ts = value
        else:
            continue
        decision = _normalize_decision(decision)
        ts_num = _num(ts)
        if not vid or decision is None or ts_num is None:
            continue
        items.append((str(vid), decision, ts_num))
    items.sort(key=lambda it: it[2])
    return {vid: (decision, ts) for vid, decision, ts in items}


# ---------------------------------------------------------------------------
# Histórico (data/metricas.json)
# ---------------------------------------------------------------------------
def _history_path(path: Path | None) -> Path:
    return Path(path) if path is not None else Path(config.DATA_DIR) / METRICS_FILENAME


def _read_history_file(path: Path, quarantine: bool) -> list[dict]:
    """Lê o histórico. Com `quarantine`, um arquivo corrompido é renomeado para
    `.corrompido` (para não ser sobrescrito) e o histórico recomeça vazio."""
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except OSError as exc:
        if quarantine:
            raise  # gravar agora apagaria as sessões antigas que não conseguimos ler
        logger.warning("Não foi possível ler %s: %s", path, exc)
        return []
    except ValueError:
        data = None
    if isinstance(data, list):
        return [e for e in data if isinstance(e, dict)]
    logger.warning("Histórico de métricas corrompido em %s.", path)
    if quarantine:
        try:
            path.replace(path.with_name(path.name + ".corrompido"))
        except OSError:
            pass
    return []


def load_history(path: Path | None = None) -> list[dict]:
    """Sessões gravadas (path None = DATA_DIR/'metricas.json' na hora da chamada)."""
    with _lock:
        return _read_history_file(_history_path(path), quarantine=False)


def clear_history(path: Path | None = None) -> int:
    """Apaga o histórico de sessões e devolve quantas sessões havia nele.

    path None = DATA_DIR/'metricas.json' na hora da chamada. Sem arquivo, devolve 0.
    Levanta OSError se o arquivo estiver travado por outro programa.
    """
    target = _history_path(path)
    with _lock:
        n = len(_read_history_file(target, quarantine=False))
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Não foi possível apagar o histórico de métricas em %s: %s", target, exc)
            raise
    logger.info("Histórico de métricas apagado (%d sessão(ões)).", n)
    return n


def history_stats(history: Iterable[dict] | None) -> dict[str, Any]:
    """Resumo do histórico para o painel.

    Chaves: sessions, total_approved, total_saved_min_lo, total_saved_min_hi (minutos
    economizados somando todas as sessões, nunca negativos), best_s_per_track (menor tempo
    por faixa aprovada; None sem medição), avg_s_per_track (tempo total ÷ aprovadas),
    total_minutes, total_decisions e trend: [(nº da sessão, s por faixa)] na ordem do
    histórico (só sessões com faixa aprovada).
    """
    base_lo, base_hi = _baseline()
    entries = [e for e in (history or []) if isinstance(e, dict)]
    total_approved = 0
    total_decisions = 0
    total_duration = 0.0
    saved_lo = saved_hi = 0.0
    trend: list[tuple[int, float]] = []
    for n, entry in enumerate(entries, start=1):
        approved = _count(entry.get("aprovadas"))
        duration = max(0.0, _num(entry.get("duracao_s")) or 0.0)
        decisions = entry.get("decisoes", entry.get("revisadas"))
        total_approved += approved
        total_decisions += _count(decisions)
        total_duration += duration
        saved_lo += approved * base_lo - duration
        saved_hi += approved * base_hi - duration
        spt = _num(entry.get("s_por_faixa"))
        if spt is None and approved and duration > 0:
            spt = duration / approved
        if approved and spt is not None and spt > 0:
            trend.append((n, spt))
    return {
        "sessions": len(entries),
        "total_approved": total_approved,
        "total_decisions": total_decisions,
        "total_minutes": total_duration / 60.0,
        "total_saved_min_lo": max(0.0, saved_lo / 60.0),
        "total_saved_min_hi": max(0.0, saved_hi / 60.0),
        "best_s_per_track": min((s for _, s in trend), default=None),
        "avg_s_per_track": (total_duration / total_approved
                            if total_approved and total_duration > 0 else None),
        "trend": trend,
    }
