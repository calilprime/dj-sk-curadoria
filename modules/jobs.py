"""Fila em segundo plano: baixa (até 4 em paralelo) e depois analisa cada faixa.

CONTRATO (implementar mantendo as assinaturas):

A interface Streamlit roda de novo a cada clique, então o `TrackJobs` vive em
`st.session_state` e é consultado por um `st.fragment(run_every=1)` para a barra de
progresso. Precisa ser thread-safe (lock em volta do dicionário de status).
Usa `download.download_audio` / `download.audio_path_for` e `analyze.load_or_analyze`.
Faixas locais (Track.local_path) pulam o download e vão direto para a análise.

Implementação: dois ThreadPoolExecutor (downloads e análise). Cada faixa passa por
QUEUED -> DOWNLOADING -> ANALYZING -> READY, ou termina em ERROR com mensagem.
Faixas locais ou já em cache entram direto em ANALYZING (contam como "baixadas"
enquanto esperam a vez na análise). As funções dos outros módulos são chamadas pelo
módulo (`download.x`, `analyze.x`) para os testes poderem trocá-las.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import config
from modules import analyze, download
from modules.models import Analysis, Track

log = logging.getLogger(__name__)

QUEUED = "na fila"
DOWNLOADING = "baixando"
ANALYZING = "analisando"
READY = "pronto"
ERROR = "erro"

_ACTIVE_STATES = (QUEUED, DOWNLOADING, ANALYZING)

_MSG_CANCELLED = "Cancelado: a fila foi encerrada."
_MSG_CLOSED = "A fila já foi encerrada. Recarregue a sessão para baixar de novo."
_MSG_UNEXPECTED = "Erro inesperado ao processar a faixa."
_MSG_ANALYSIS_UNAVAILABLE = "A análise de áudio ainda não está disponível."


@dataclass
class JobStatus:
    state: str = QUEUED                 # QUEUED | DOWNLOADING | ANALYZING | READY | ERROR
    progress: float = 0.0               # 0..1 do download
    error: str | None = None            # mensagem em português quando state == ERROR


def _download_message(exc: Exception) -> str:
    if isinstance(exc, download.DownloadError):
        return str(exc) or "Falha ao baixar a faixa."
    return str(download._translate_error(exc))


def _analysis_message(exc: Exception) -> str:
    if isinstance(exc, NotImplementedError):
        return _MSG_ANALYSIS_UNAVAILABLE
    return f"Falha ao analisar o áudio ({download._short_detail(exc)})."


class TrackJobs:
    def __init__(self, max_downloads: int | None = None, analyze_workers: int = 1,
                 cache_dir=None):
        """max_downloads=None usa config.MAX_PARALLEL_DOWNLOADS."""
        if max_downloads is None:
            max_downloads = config.MAX_PARALLEL_DOWNLOADS
        # None é repassado como None: download/analyze leem config.CACHE_DIR na hora.
        self._cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        self._lock = threading.Lock()
        self._status: dict[str, JobStatus] = {}
        self._tracks: dict[str, Track] = {}
        self._analyses: dict[str, Analysis] = {}
        self._closed = False
        self._download_pool = ThreadPoolExecutor(max_workers=max(1, int(max_downloads)),
                                                 thread_name_prefix="djsk-download")
        self._analyze_pool = ThreadPoolExecutor(max_workers=max(1, int(analyze_workers)),
                                                thread_name_prefix="djsk-analise")

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------
    def submit(self, tracks: list[Track]) -> None:
        """Enfileira faixas. Idempotente: não reenfileira o que já está na fila, rodando ou pronto.

        Faixas com ERROR são reenfileiradas (permite "tentar de novo").
        """
        for track in tracks:
            vid = track.video_id
            with self._lock:
                current = self._status.get(vid)
                if current is not None and current.state != ERROR:
                    continue
                self._tracks[vid] = track
                self._analyses.pop(vid, None)
                if self._closed:
                    self._status[vid] = JobStatus(ERROR, 0.0, _MSG_CLOSED)
                    continue
                self._status[vid] = JobStatus(QUEUED)
            # Fora do lock: mexe em disco e nos executores.
            self._dispatch(track)

    def status(self, video_id: str) -> JobStatus:
        """Cópia do status atual (JobStatus() padrão se a faixa nunca foi enviada)."""
        with self._lock:
            st = self._status.get(video_id)
            return replace(st) if st is not None else JobStatus()

    def summary(self) -> dict[str, int]:
        """Contagem: {"total", "na fila", "baixando", "analisando", "pronto", "erro", "baixadas"}.

        "baixadas" = faixas com áudio disponível (analisando + pronto).
        """
        counts = {"total": 0, QUEUED: 0, DOWNLOADING: 0, ANALYZING: 0, READY: 0, ERROR: 0}
        with self._lock:
            for st in self._status.values():
                counts[st.state] += 1
            counts["total"] = len(self._status)
        counts["baixadas"] = counts[ANALYZING] + counts[READY]
        return counts

    def get_analysis(self, video_id: str) -> Analysis | None:
        """Análise pronta (ou None)."""
        with self._lock:
            return self._analyses.get(video_id)

    def update_analysis(self, analysis: Analysis) -> None:
        """Substitui a análise em memória (ex.: após correção manual de BPM)."""
        with self._lock:
            self._analyses[analysis.video_id] = analysis

    def all_done(self) -> bool:
        """True quando nada está na fila, baixando ou analisando."""
        with self._lock:
            return not any(st.state in _ACTIVE_STATES for st in self._status.values())

    def shutdown(self, wait: bool = False) -> None:
        """Encerra as filas. O que ainda não começou vira ERROR ("Cancelado").

        Downloads e análises já em andamento terminam (threads não são interrompidas);
        `wait=True` espera por eles. Downloads que terminarem depois disso não são analisados.
        """
        with self._lock:
            self._closed = True
        # Fora do lock: o cancelamento chama _on_done, que pega o lock.
        self._download_pool.shutdown(wait=wait, cancel_futures=True)
        self._analyze_pool.shutdown(wait=wait, cancel_futures=True)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------
    def _dispatch(self, track: Track) -> None:
        vid = track.video_id
        try:
            path = download.audio_path_for(track, self._cache_dir)
        except Exception:
            log.debug("audio_path_for falhou para %s", vid, exc_info=True)
            path = None
        if track.is_local and path is None:
            name = Path(track.local_path or "").name or vid
            self._fail(vid, f"Arquivo local não encontrado: {name}")
            return
        if path is not None:
            self._set(vid, ANALYZING, progress=1.0)
            self._start(self._analyze_pool, vid, self._run_analysis, track, Path(path))
        else:
            self._start(self._download_pool, vid, self._run_download, track)

    def _start(self, pool: ThreadPoolExecutor, vid: str, fn: Callable, *args) -> None:
        with self._lock:
            closed = self._closed
        if closed:
            self._fail(vid, _MSG_CANCELLED)
            return
        try:
            fut = pool.submit(fn, *args)
        except RuntimeError:            # executor encerrado entre a checagem e o submit
            self._fail(vid, _MSG_CANCELLED)
            return
        fut.add_done_callback(lambda f, vid=vid: self._on_done(vid, f))

    def _on_done(self, vid: str, fut: Future) -> None:
        if fut.cancelled():
            self._fail(vid, _MSG_CANCELLED)
            return
        exc = fut.exception()
        if exc is not None:             # rede de segurança: os workers já tratam os erros
            log.error("Erro inesperado na fila (%s)", vid, exc_info=exc)
            self._fail(vid, _MSG_UNEXPECTED)

    def _run_download(self, track: Track) -> None:
        vid = track.video_id
        with self._lock:
            closed = self._closed
        if closed:
            self._fail(vid, _MSG_CANCELLED)
            return
        self._set(vid, DOWNLOADING, progress=0.0)
        try:
            path = download.download_audio(vid, progress_cb=lambda f: self._progress(vid, f),
                                           cache_dir=self._cache_dir)
        except Exception as exc:
            msg = _download_message(exc)
            log.warning("Download de %s falhou: %s", vid, msg)
            self._fail(vid, msg)
            return
        self._set(vid, ANALYZING, progress=1.0)
        self._start(self._analyze_pool, vid, self._run_analysis, track, Path(path))

    def _run_analysis(self, track: Track, path: Path) -> None:
        vid = track.video_id
        try:
            result = analyze.load_or_analyze(vid, path, cache_dir=self._cache_dir)
        except Exception as exc:
            msg = _analysis_message(exc)
            log.warning("Análise de %s falhou: %s", vid, download._short_detail(exc))
            self._fail(vid, msg)
            return
        with self._lock:
            self._analyses[vid] = result
            self._status[vid] = JobStatus(READY, 1.0, None)

    def _set(self, vid: str, state: str, progress: float | None = None) -> None:
        with self._lock:
            st = self._status.setdefault(vid, JobStatus())
            st.state = state
            st.error = None
            if progress is not None:
                st.progress = progress

    def _progress(self, vid: str, fraction: float) -> None:
        try:
            value = min(max(float(fraction), 0.0), 1.0)
        except (TypeError, ValueError):
            return
        with self._lock:
            st = self._status.get(vid)
            if st is not None and st.state == DOWNLOADING:
                st.progress = value

    def _fail(self, vid: str, message: str) -> None:
        with self._lock:
            st = self._status.setdefault(vid, JobStatus())
            if st.state in (READY, ERROR):     # não sobrescreve um fim já registrado
                return
            st.state = ERROR
            st.error = message
