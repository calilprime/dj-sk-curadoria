"""Testes da fila em segundo plano (modules/jobs.py) com download e análise falsos.

O analyze.py é escrito em paralelo por outro agente: aqui `analyze.load_or_analyze` e
`download.download_audio` são sempre trocados por fakes.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import config
from modules import analyze, download
from modules.jobs import ANALYZING, DOWNLOADING, ERROR, QUEUED, READY, JobStatus, TrackJobs
from modules.models import Analysis, Track


def wait_until(cond, timeout: float = 10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


def fake_analysis(video_id: str) -> Analysis:
    return Analysis(video_id=video_id, duration_s=180.0, bpm=130.0, bpm_raw=130.0)


class Fakes:
    """download_audio / load_or_analyze falsos, com contadores de concorrência."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.download_calls: list[str] = []
        self.download_cache_dirs: list = []
        self.analyze_calls: list[tuple[str, Path]] = []
        self.analyze_cache_dirs: list = []
        self.active_downloads = 0
        self.peak_downloads = 0
        self.active_analyses = 0
        self.peak_analyses = 0
        self.download_delay = 0.0
        self.analyze_delay = 0.0
        self.download_errors: dict[str, list[BaseException]] = {}
        self.analyze_errors: dict[str, list[BaseException]] = {}
        self.download_gate: threading.Event | None = None   # downloads esperam este evento
        self.analyze_gate: threading.Event | None = None

    def release(self) -> None:
        for gate in (self.download_gate, self.analyze_gate):
            if gate is not None:
                gate.set()

    def download_audio(self, video_id, progress_cb=None, cache_dir=None):
        with self.lock:
            self.download_calls.append(video_id)
            self.download_cache_dirs.append(cache_dir)
            self.active_downloads += 1
            self.peak_downloads = max(self.peak_downloads, self.active_downloads)
            errors = self.download_errors.get(video_id)
            err = errors.pop(0) if errors else None
        try:
            if progress_cb:
                progress_cb(0.45)
            if self.download_gate is not None:
                self.download_gate.wait(5)
            if self.download_delay:
                time.sleep(self.download_delay)
            if err is not None:
                raise err
            path = Path(cache_dir or config.CACHE_DIR) / f"{video_id}.mp3"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ID3")
            if progress_cb:
                progress_cb(1.0)
            return path
        finally:
            with self.lock:
                self.active_downloads -= 1

    def load_or_analyze(self, video_id, path, force=False, cache_dir=None):
        with self.lock:
            self.analyze_calls.append((video_id, Path(path)))
            self.analyze_cache_dirs.append(cache_dir)
            self.active_analyses += 1
            self.peak_analyses = max(self.peak_analyses, self.active_analyses)
            errors = self.analyze_errors.get(video_id)
            err = errors.pop(0) if errors else None
        try:
            if self.analyze_gate is not None:
                self.analyze_gate.wait(5)
            if self.analyze_delay:
                time.sleep(self.analyze_delay)
            if err is not None:
                raise err
            return fake_analysis(video_id)
        finally:
            with self.lock:
                self.active_analyses -= 1


@pytest.fixture
def fakes(monkeypatch, tmp_dirs):
    f = Fakes()
    monkeypatch.setattr(download, "download_audio", f.download_audio)
    monkeypatch.setattr(analyze, "load_or_analyze", f.load_or_analyze)
    yield f
    f.release()


@pytest.fixture
def make_jobs(fakes):
    created: list[TrackJobs] = []

    def factory(**kw) -> TrackJobs:
        jobs = TrackJobs(**kw)
        created.append(jobs)
        return jobs

    yield factory
    fakes.release()
    for jobs in created:
        jobs.shutdown(wait=True)


def yt(video_id: str) -> Track:
    return Track(video_id=video_id, title=f"Faixa {video_id}", artists=["MC Teste"])


# ---------------------------------------------------------------------------
# Básico
# ---------------------------------------------------------------------------
def test_unknown_track_and_empty_queue(make_jobs):
    jobs = make_jobs()
    assert jobs.status("nunca_enviada") == JobStatus()
    assert jobs.get_analysis("nunca_enviada") is None
    assert jobs.all_done()
    assert jobs.summary() == {"total": 0, QUEUED: 0, DOWNLOADING: 0, ANALYZING: 0,
                              READY: 0, ERROR: 0, "baixadas": 0}


def test_pipeline_downloads_then_analyzes(fakes, make_jobs, tmp_dirs):
    jobs = make_jobs()
    jobs.submit([yt("a"), yt("b"), yt("c")])
    assert wait_until(jobs.all_done)

    for vid in ("a", "b", "c"):
        assert jobs.status(vid) == JobStatus(READY, 1.0, None)
        assert jobs.get_analysis(vid) == fake_analysis(vid)
    assert sorted(fakes.download_calls) == ["a", "b", "c"]
    # a análise recebe o mp3 devolvido pelo download
    assert sorted(fakes.analyze_calls) == [(v, tmp_dirs["CACHE_DIR"] / f"{v}.mp3") for v in "abc"]
    # cache_dir=None é repassado como None (os módulos leem o config na hora)
    assert fakes.download_cache_dirs == [None] * 3
    assert fakes.analyze_cache_dirs == [None] * 3
    assert jobs.summary() == {"total": 3, QUEUED: 0, DOWNLOADING: 0, ANALYZING: 0,
                              READY: 3, ERROR: 0, "baixadas": 3}


def test_cache_dir_is_passed_through(fakes, make_jobs, tmp_path):
    other = tmp_path / "outro_cache"
    jobs = make_jobs(cache_dir=other)
    jobs.submit([yt("a")])
    assert wait_until(jobs.all_done)
    assert jobs.status("a").state == READY
    assert fakes.download_cache_dirs == [other]
    assert fakes.analyze_cache_dirs == [other]
    assert fakes.analyze_calls == [("a", other / "a.mp3")]


# ---------------------------------------------------------------------------
# Concorrência
# ---------------------------------------------------------------------------
def test_twenty_tracks_at_most_four_parallel_downloads(fakes, make_jobs, monkeypatch):
    monkeypatch.setattr(config, "MAX_PARALLEL_DOWNLOADS", 4)
    fakes.download_delay = 0.05
    fakes.analyze_delay = 0.005
    jobs = make_jobs()                      # max_downloads=None -> config
    tracks = [yt(f"v{i:02d}") for i in range(20)]
    jobs.submit(tracks)

    observed_downloading = 0
    end = time.monotonic() + 20
    while not jobs.all_done() and time.monotonic() < end:
        observed_downloading = max(observed_downloading, jobs.summary()[DOWNLOADING])
        time.sleep(0.005)

    assert jobs.all_done()
    assert fakes.peak_downloads == 4
    assert observed_downloading <= 4
    assert fakes.peak_analyses == 1         # analyze_workers=1 por padrão
    assert sorted(fakes.download_calls) == sorted(t.video_id for t in tracks)
    s = jobs.summary()
    assert s["total"] == 20 and s[READY] == 20 and s["baixadas"] == 20 and s[ERROR] == 0


def test_max_downloads_parameter(fakes, make_jobs):
    fakes.download_delay = 0.05
    jobs = make_jobs(max_downloads=2, analyze_workers=2)
    jobs.submit([yt(f"v{i}") for i in range(8)])
    assert wait_until(jobs.all_done)
    assert fakes.peak_downloads == 2
    assert fakes.peak_analyses <= 2


def test_concurrent_submits_download_each_track_once(fakes, make_jobs):
    fakes.download_delay = 0.01
    jobs = make_jobs()
    tracks = [yt(f"v{i:02d}") for i in range(20)]
    threads = [threading.Thread(target=jobs.submit, args=(list(tracks),)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert wait_until(jobs.all_done)
    assert sorted(fakes.download_calls) == sorted(t.video_id for t in tracks)
    assert jobs.summary()[READY] == 20


# ---------------------------------------------------------------------------
# Estados intermediários e progresso
# ---------------------------------------------------------------------------
def test_states_progress_and_summary_while_running(fakes, make_jobs, tmp_path):
    fakes.download_gate = threading.Event()
    fakes.analyze_gate = threading.Event()
    local_ok = tmp_path / "MC Local - Faixa.wav"
    local_ok.write_bytes(b"RIFF")
    jobs = make_jobs(max_downloads=1)
    jobs.submit([
        yt("a"),                                                           # baixando
        yt("b"),                                                           # na fila
        Track(video_id="local_ok", title="x", local_path=str(local_ok)),   # analisando
        Track(video_id="local_sumiu", title="y", local_path=str(tmp_path / "sumiu.wav")),
    ])

    assert wait_until(lambda: jobs.status("a").progress == pytest.approx(0.45))
    assert jobs.status("a").state == DOWNLOADING
    assert jobs.status("b") == JobStatus(QUEUED, 0.0, None)
    assert wait_until(lambda: fakes.analyze_calls == [("local_ok", local_ok)])
    assert jobs.status("local_ok").state == ANALYZING
    assert jobs.status("local_sumiu").state == ERROR
    assert jobs.summary() == {"total": 4, QUEUED: 1, DOWNLOADING: 1, ANALYZING: 1,
                              READY: 0, ERROR: 1, "baixadas": 1}
    assert not jobs.all_done()

    fakes.download_gate.set()
    # os downloads terminam e esperam a análise (travada): contam como baixadas
    assert wait_until(lambda: jobs.summary()[ANALYZING] == 3)
    assert jobs.status("a") == JobStatus(ANALYZING, 1.0, None)
    assert jobs.summary()["baixadas"] == 3
    assert not jobs.all_done()

    fakes.analyze_gate.set()
    assert wait_until(jobs.all_done)
    assert jobs.summary() == {"total": 4, QUEUED: 0, DOWNLOADING: 0, ANALYZING: 0,
                              READY: 3, ERROR: 1, "baixadas": 3}


def test_status_returns_a_copy(fakes, make_jobs):
    jobs = make_jobs()
    jobs.submit([yt("a")])
    assert wait_until(jobs.all_done)
    st = jobs.status("a")
    st.state, st.progress, st.error = ERROR, 0.0, "mexi"
    assert jobs.status("a") == JobStatus(READY, 1.0, None)


# ---------------------------------------------------------------------------
# Erros
# ---------------------------------------------------------------------------
def test_download_error_marks_track_and_skips_analysis(fakes, make_jobs):
    msg = "Vídeo indisponível ou privado no YouTube."
    fakes.download_errors["ruim"] = [download.DownloadError(msg)]
    jobs = make_jobs()
    jobs.submit([yt("bom"), yt("ruim")])
    assert wait_until(jobs.all_done)

    st = jobs.status("ruim")
    assert st.state == ERROR and st.error == msg
    assert jobs.get_analysis("ruim") is None
    assert [vid for vid, _ in fakes.analyze_calls] == ["bom"]
    assert jobs.status("bom").state == READY
    s = jobs.summary()
    assert s[ERROR] == 1 and s[READY] == 1 and s["baixadas"] == 1


def test_unexpected_download_exception_gets_portuguese_message(fakes, make_jobs):
    fakes.download_errors["a"] = [RuntimeError("explodiu")]
    jobs = make_jobs()
    jobs.submit([yt("a")])
    assert wait_until(jobs.all_done)
    st = jobs.status("a")
    assert st.state == ERROR
    assert st.error.startswith("Falha ao baixar a faixa") and "explodiu" in st.error


def test_network_exception_is_translated(fakes, make_jobs):
    fakes.download_errors["a"] = [ConnectionResetError("reset")]
    jobs = make_jobs()
    jobs.submit([yt("a")])
    assert wait_until(jobs.all_done)
    assert "rede" in jobs.status("a").error


def test_analysis_error_marks_track(fakes, make_jobs):
    fakes.analyze_errors["a"] = [ValueError("arquivo corrompido")]
    fakes.analyze_errors["b"] = [NotImplementedError()]
    jobs = make_jobs()
    jobs.submit([yt("a"), yt("b"), yt("c")])
    assert wait_until(jobs.all_done)

    a = jobs.status("a")
    assert a.state == ERROR and "analisar" in a.error and "arquivo corrompido" in a.error
    b = jobs.status("b")
    assert b.state == ERROR and "análise" in b.error.lower()
    assert jobs.get_analysis("a") is None and jobs.get_analysis("b") is None
    assert jobs.status("c").state == READY
    assert jobs.summary()["baixadas"] == 1      # áudio de a/b existe, mas a faixa está em erro


# ---------------------------------------------------------------------------
# Idempotência e nova tentativa
# ---------------------------------------------------------------------------
def test_submit_is_idempotent(fakes, make_jobs):
    fakes.download_gate = threading.Event()
    jobs = make_jobs()
    jobs.submit([yt("a"), yt("a")])
    assert wait_until(lambda: jobs.status("a").state == DOWNLOADING)
    jobs.submit([yt("a")])                 # rodando: ignora
    fakes.download_gate.set()
    assert wait_until(jobs.all_done)
    jobs.submit([yt("a")])                 # pronto: ignora
    assert wait_until(jobs.all_done)
    assert fakes.download_calls == ["a"]
    assert len(fakes.analyze_calls) == 1
    assert jobs.summary()["total"] == 1


def test_error_is_requeued_on_submit(fakes, make_jobs):
    fakes.download_errors["a"] = [download.DownloadError("Sem internet ou erro de rede ao baixar.")]
    jobs = make_jobs()
    jobs.submit([yt("a")])
    assert wait_until(jobs.all_done)
    assert jobs.status("a").state == ERROR

    jobs.submit([yt("a")])                 # "tentar de novo"
    assert wait_until(jobs.all_done)
    assert jobs.status("a") == JobStatus(READY, 1.0, None)
    assert fakes.download_calls == ["a", "a"]
    assert jobs.get_analysis("a") == fake_analysis("a")


# ---------------------------------------------------------------------------
# Faixas locais e cache
# ---------------------------------------------------------------------------
def test_local_tracks_skip_download(fakes, make_jobs, tmp_path):
    f = tmp_path / "MC Local - Faixa.mp3"
    f.write_bytes(b"ID3")
    ok = Track(video_id="local_0123456789ab", title="Faixa", source="local", local_path=str(f))
    missing = Track(video_id="local_ba9876543210", title="Sumiu", source="local",
                    local_path=str(tmp_path / "sumiu.mp3"))
    jobs = make_jobs()
    jobs.submit([ok, missing])
    assert wait_until(jobs.all_done)

    assert fakes.download_calls == []
    assert fakes.analyze_calls == [("local_0123456789ab", f)]
    assert jobs.status("local_0123456789ab") == JobStatus(READY, 1.0, None)
    st = jobs.status("local_ba9876543210")
    assert st.state == ERROR and "não encontrado" in st.error and "sumiu.mp3" in st.error


def test_cached_audio_skips_download(fakes, make_jobs, tmp_dirs):
    cached = tmp_dirs["CACHE_DIR"] / "abc.mp3"
    cached.write_bytes(b"ID3 pronto")
    jobs = make_jobs()
    jobs.submit([yt("abc")])
    assert wait_until(jobs.all_done)
    assert fakes.download_calls == []
    assert fakes.analyze_calls == [("abc", cached)]
    assert jobs.status("abc").state == READY


# ---------------------------------------------------------------------------
# Análise em memória
# ---------------------------------------------------------------------------
def test_update_analysis_replaces_in_memory(fakes, make_jobs):
    jobs = make_jobs()
    jobs.submit([yt("a")])
    assert wait_until(jobs.all_done)
    corrected = fake_analysis("a")
    corrected.bpm_manual = 128.0
    jobs.update_analysis(corrected)
    assert jobs.get_analysis("a").bpm_manual == 128.0
    assert jobs.get_analysis("a").effective_bpm == 128.0
    assert jobs.status("a").state == READY


# ---------------------------------------------------------------------------
# Encerramento
# ---------------------------------------------------------------------------
def test_shutdown_without_wait_cancels_pending(fakes, make_jobs):
    fakes.download_gate = threading.Event()
    jobs = make_jobs(max_downloads=1)
    jobs.submit([yt("a"), yt("b"), yt("c")])
    assert wait_until(lambda: jobs.status("a").state == DOWNLOADING)

    t0 = time.monotonic()
    jobs.shutdown(wait=False)
    assert time.monotonic() - t0 < 1.0     # não espera o download travado
    for vid in ("b", "c"):
        st = jobs.status(vid)
        assert st.state == ERROR and "Cancelado" in st.error

    fakes.download_gate.set()
    assert wait_until(jobs.all_done)
    # o download em andamento termina, mas não é mais analisado
    assert jobs.status("a").state == ERROR
    assert fakes.download_calls == ["a"] and fakes.analyze_calls == []

    jobs.submit([yt("d")])                 # depois de encerrada a fila não aceita mais nada
    st = jobs.status("d")
    assert st.state == ERROR and "encerrada" in st.error
    assert fakes.download_calls == ["a"]


def test_shutdown_with_wait_finishes_running_jobs(fakes, make_jobs):
    fakes.download_delay = 0.05
    jobs = make_jobs(max_downloads=4)
    jobs.submit([yt("a")])
    assert wait_until(lambda: fakes.download_calls == ["a"])
    jobs.shutdown(wait=True)
    assert jobs.all_done()
    assert jobs.status("a").state in (READY, ERROR)
    jobs.shutdown(wait=True)               # pode chamar de novo
