"""Testes do cache de áudio (modules/download.py) com um yt-dlp falso; um teste real de rede."""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yt_dlp
import yt_dlp.utils

import config
from modules import download
from modules.models import Track

NET_ERROR = "ERROR: [youtube] abc: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>"


class FakeController:
    """Controla o YoutubeDL falso: fila de ações por chamada de download()."""

    def __init__(self) -> None:
        self.actions: list = []
        self.opts: list[dict] = []
        self.urls: list[str] = []
        self.calls = 0
        self.lock = threading.Lock()


def _out_base(opts: dict) -> Path:
    """Caminho sem extensão pedido pelo outtmpl (respeitando paths['home'])."""
    tmpl = opts["outtmpl"]
    if isinstance(tmpl, dict):
        tmpl = tmpl["default"]
    p = Path(tmpl.replace(".%(ext)s", ""))
    if not p.is_absolute():
        p = Path(opts.get("paths", {}).get("home", ".")) / p
    return p


def write_ok(opts: dict, size: int = 4000) -> int:
    """Simula download (com progress_hooks) e conversão para mp3."""
    base = _out_base(opts)
    partial = base.with_suffix(".webm.part")
    partial.write_bytes(b"x" * 10)
    for hook in opts["progress_hooks"]:
        hook({"status": "downloading", "downloaded_bytes": 25, "total_bytes": 100})
        hook({"status": "downloading", "downloaded_bytes": 50, "total_bytes": None,
              "total_bytes_estimate": 100})
        hook({"status": "downloading", "downloaded_bytes": 100, "total_bytes": 100})
        hook({"status": "finished", "downloaded_bytes": 100, "total_bytes": 100})
    partial.unlink()
    base.with_suffix(".mp3").write_bytes(b"ID3" + b"\x00" * size)
    return 0


@pytest.fixture
def fake_ydl(monkeypatch, tmp_dirs):
    ctl = FakeController()

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts
            with ctl.lock:
                ctl.opts.append(opts)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def download(self, urls):
            with ctl.lock:
                ctl.calls += 1
                ctl.urls.extend(urls)
                action = ctl.actions.pop(0) if ctl.actions else "ok"
            if isinstance(action, BaseException):
                raise action
            if callable(action):
                return action(self.opts)
            return write_ok(self.opts)

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    monkeypatch.setattr(download, "_RETRY_BACKOFF_S", 0.0)
    monkeypatch.setattr(download, "_ffmpeg_available", lambda: True)
    return ctl


def _tmp_leftovers(cache_dir: Path) -> list[Path]:
    tmp_root = cache_dir / download.TMP_DIRNAME
    return list(tmp_root.iterdir()) if tmp_root.exists() else []


# ---------------------------------------------------------------------------
# cache_path / is_cached / audio_path_for
# ---------------------------------------------------------------------------
def test_cache_path_uses_config_at_call_time(tmp_dirs):
    assert download.cache_path("abc") == tmp_dirs["CACHE_DIR"] / "abc.mp3"
    assert download.cache_path("abc", cache_dir=Path("X")) == Path("X") / "abc.mp3"


def test_is_cached_requires_non_empty_file(tmp_dirs):
    p = tmp_dirs["CACHE_DIR"] / "abc.mp3"
    assert not download.is_cached("abc")
    p.write_bytes(b"")
    assert not download.is_cached("abc")
    p.write_bytes(b"ID3data")
    assert download.is_cached("abc")


def test_audio_path_for_local_and_cached(tmp_dirs, tmp_path):
    local_file = tmp_path / "faixa.wav"
    local_file.write_bytes(b"RIFF")
    local = Track(video_id="local_123", title="x", local_path=str(local_file), source="local")
    assert download.audio_path_for(local) == local_file
    gone = Track(video_id="local_456", title="y", local_path=str(tmp_path / "sumiu.wav"))
    assert download.audio_path_for(gone) is None

    yt = Track(video_id="abc", title="z")
    assert download.audio_path_for(yt) is None
    (tmp_dirs["CACHE_DIR"] / "abc.mp3").write_bytes(b"ID3")
    assert download.audio_path_for(yt) == tmp_dirs["CACHE_DIR"] / "abc.mp3"


# ---------------------------------------------------------------------------
# download_audio
# ---------------------------------------------------------------------------
def test_download_success_moves_mp3_and_reports_progress(fake_ydl, tmp_dirs):
    progress: list[float] = []
    path = download.download_audio("abc123", progress_cb=progress.append)

    assert path == tmp_dirs["CACHE_DIR"] / "abc123.mp3"
    assert path.read_bytes().startswith(b"ID3")
    assert _tmp_leftovers(tmp_dirs["CACHE_DIR"]) == []
    assert fake_ydl.calls == 1
    assert fake_ydl.urls == ["https://music.youtube.com/watch?v=abc123"]

    # durante o download a barra vai até 0.9; 1.0 só depois da conversão
    assert progress[-1] == 1.0
    assert progress[:-1] and max(progress[:-1]) <= 0.9 + 1e-9
    assert progress == sorted(progress)
    assert progress[0] == pytest.approx(0.25 * 0.9)

    opts = fake_ydl.opts[0]
    assert opts["format"] == "bestaudio/best"
    assert opts["noplaylist"] and opts["quiet"] and opts["no_warnings"] and opts["noprogress"]
    pp = opts["postprocessors"][0]
    assert pp["key"] == "FFmpegExtractAudio"
    assert pp["preferredcodec"] == config.AUDIO_CODEC
    assert pp["preferredquality"] == str(config.AUDIO_QUALITY)
    # a pasta de trabalho é temporária e exclusiva, dentro de cache/.tmp
    work = Path(opts["paths"]["home"])
    assert work.parent == tmp_dirs["CACHE_DIR"] / download.TMP_DIRNAME
    assert work.name.startswith("abc123-")


def test_download_returns_cache_without_calling_ytdlp(fake_ydl, tmp_dirs):
    cached = tmp_dirs["CACHE_DIR"] / "abc.mp3"
    cached.write_bytes(b"ID3 pronto")
    progress: list[float] = []
    assert download.download_audio("abc", progress_cb=progress.append) == cached
    assert fake_ydl.calls == 0
    assert progress == [1.0]


def test_progress_callback_errors_do_not_break_download(fake_ydl, tmp_dirs):
    def bad_cb(_fraction):
        raise RuntimeError("barra quebrou")

    path = download.download_audio("abc", progress_cb=bad_cb)
    assert path.exists()


def test_retry_after_network_error_then_success(fake_ydl, tmp_dirs, monkeypatch):
    monkeypatch.setattr(config, "DOWNLOAD_RETRIES", 2)
    fake_ydl.actions = [yt_dlp.utils.DownloadError(NET_ERROR), "ok"]
    path = download.download_audio("abc")
    assert path.exists()
    assert fake_ydl.calls == 2
    assert _tmp_leftovers(tmp_dirs["CACHE_DIR"]) == []


def test_network_error_exhausts_retries(fake_ydl, tmp_dirs, monkeypatch):
    monkeypatch.setattr(config, "DOWNLOAD_RETRIES", 3)
    fake_ydl.actions = [yt_dlp.utils.DownloadError(NET_ERROR) for _ in range(5)]
    with pytest.raises(download.DownloadError) as info:
        download.download_audio("abc")
    assert "rede" in str(info.value)
    assert fake_ydl.calls == 3
    assert not download.cache_path("abc").exists()
    assert _tmp_leftovers(tmp_dirs["CACHE_DIR"]) == []


@pytest.mark.parametrize("raw, expected", [
    ("ERROR: [youtube] abc: Video unavailable. This content isn't available.", "indisponível"),
    ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", "privado"),
    ("ERROR: [youtube] abc: Sign in to confirm your age. This video may be inappropriate "
     "for some users.", "idade"),
    ("ERROR: Postprocessing: ffprobe and ffmpeg not found. Please install or provide the "
     "path using --ffmpeg-location", "ffmpeg"),
])
def test_permanent_errors_are_translated_and_not_retried(fake_ydl, tmp_dirs, monkeypatch, raw, expected):
    monkeypatch.setattr(config, "DOWNLOAD_RETRIES", 3)
    monkeypatch.setattr(download, "cookie_options", lambda: {})   # sem login do YouTube
    fake_ydl.actions = [yt_dlp.utils.DownloadError(raw) for _ in range(3)]
    with pytest.raises(download.DownloadError) as info:
        download.download_audio("abc")
    assert expected in str(info.value)
    assert fake_ydl.calls == 1


def test_age_restricted_retries_once_with_youtube_login(fake_ydl, tmp_dirs, monkeypatch):
    monkeypatch.setattr(download, "cookie_options", lambda: {"cookiefile": "cookies.txt"})
    fake_ydl.actions = [yt_dlp.utils.DownloadError("ERROR: Sign in to confirm your age")]
    path = download.download_audio("abc")
    assert path.is_file() and fake_ydl.calls == 2


@pytest.mark.parametrize("raw, expected, forbidden", [
    # sem formato de áudio = yt-dlp desatualizado/YouTube mudou, não vídeo privado
    ("ERROR: [youtube] abc: Requested format is not available. Use --list-formats for a list "
     "of available formats", "yt-dlp", "privado"),
    ("ERROR: [youtube] abc: Only images are available for download. use --list-formats to see them",
     "yt-dlp", "privado"),
    ("ERROR: [youtube] abc: The uploader has not made this video available in your country",
     "indisponível", "Falha ao baixar a faixa ("),
])
def test_format_and_geo_errors_are_not_misreported(fake_ydl, tmp_dirs, monkeypatch, raw, expected,
                                                   forbidden):
    monkeypatch.setattr(config, "DOWNLOAD_RETRIES", 3)
    fake_ydl.actions = [yt_dlp.utils.DownloadError(raw) for _ in range(3)]
    with pytest.raises(download.DownloadError) as info:
        download.download_audio("abc")
    msg = str(info.value)
    assert expected in msg
    assert forbidden not in msg
    assert "Falha ao baixar a faixa (" not in msg      # nada de mensagem crua em inglês
    assert fake_ydl.calls == 1                          # nova tentativa não resolve


def test_unusable_cache_dir_raises_download_error(fake_ydl, tmp_path):
    not_a_dir = tmp_path / "cache_e_um_arquivo"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(download.DownloadError, match="pasta do cache"):
        download.download_audio("abc", cache_dir=not_a_dir)
    assert fake_ydl.calls == 0


def test_translate_error_messages():
    t = download._translate_error
    assert "rede" in str(t(ConnectionResetError("reset")))
    assert t(ConnectionResetError("reset")).retryable
    assert "rede" in str(t(yt_dlp.utils.DownloadError("ERROR: Read timed out.")))
    generic = t(ValueError("algo estranho"))
    assert str(generic).startswith("Falha ao baixar a faixa")
    assert "algo estranho" in str(generic)
    assert isinstance(generic.__cause__, ValueError)


def test_missing_ffmpeg_fails_before_download(fake_ydl, tmp_dirs, monkeypatch):
    monkeypatch.setattr(download, "_ffmpeg_available", lambda: False)
    with pytest.raises(download.DownloadError, match="ffmpeg"):
        download.download_audio("abc")
    assert fake_ydl.calls == 0


def test_failure_never_leaves_partial_file_with_final_name(fake_ydl, tmp_dirs, monkeypatch):
    monkeypatch.setattr(config, "DOWNLOAD_RETRIES", 1)

    def partial_then_crash(opts):
        _out_base(opts).with_suffix(".webm.part").write_bytes(b"meio arquivo")
        _out_base(opts).with_suffix(".mp3").write_bytes(b"mp3 pela metade")
        raise yt_dlp.utils.DownloadError("ERROR: Postprocessing: Conversion failed!")

    fake_ydl.actions = [partial_then_crash]
    with pytest.raises(download.DownloadError):
        download.download_audio("abc")
    assert not download.cache_path("abc").exists()
    assert _tmp_leftovers(tmp_dirs["CACHE_DIR"]) == []


def test_missing_output_file_is_an_error(fake_ydl, tmp_dirs, monkeypatch):
    monkeypatch.setattr(config, "DOWNLOAD_RETRIES", 1)
    fake_ydl.actions = [lambda opts: 0]
    with pytest.raises(download.DownloadError, match="mp3"):
        download.download_audio("abc")


def test_concurrent_downloads_of_same_id_run_once(fake_ydl, tmp_dirs):
    def slow_ok(opts):
        time.sleep(0.2)
        return write_ok(opts)

    fake_ydl.actions = [slow_ok, slow_ok]
    results: list[Path] = []
    threads = [threading.Thread(target=lambda: results.append(download.download_audio("same")))
               for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert fake_ydl.calls == 1
    assert results == [download.cache_path("same")] * 2


def test_invalid_and_local_ids_are_rejected(fake_ydl, tmp_dirs):
    for bad in ("../../etc", "a b", "", "x/y", None):
        with pytest.raises(download.DownloadError):
            download.download_audio(bad)
    with pytest.raises(download.DownloadError, match="local"):
        download.download_audio("local_0123456789ab")
    assert fake_ydl.calls == 0


def test_clear_tmp_removes_only_old_leftovers(tmp_dirs):
    tmp_root = tmp_dirs["CACHE_DIR"] / download.TMP_DIRNAME
    old = tmp_root / "abc-old"
    new = tmp_root / "abc-new"
    old.mkdir(parents=True)
    new.mkdir()
    (old / "abc.webm.part").write_bytes(b"x")
    past = time.time() - 7200
    import os
    os.utime(old, (past, past))
    assert download.clear_tmp() == 1
    assert not old.exists() and new.exists()


# ---------------------------------------------------------------------------
# Pasta local
# ---------------------------------------------------------------------------
def _write_wav(path: Path, seconds: float = 2.0, sr: int = 22050) -> Path:
    sf.write(path, np.zeros(int(seconds * sr), dtype=np.float32), sr)
    return path


def test_local_tracks_from_folder(tmp_path):
    folder = tmp_path / "Músicas do SK"
    folder.mkdir()
    _write_wav(folder / "MC Teste - Faixa Um.wav", 2.0)
    _write_wav(folder / "01 - MC Dois & MC Três - Faixa Dois.WAV", 3.0)
    _write_wav(folder / "sem separador.wav", 1.0)
    _write_wav(folder / ".oculto.wav", 1.0)
    (folder / "notas.txt").write_text("não é áudio", encoding="utf-8")
    (folder / "sub").mkdir()
    _write_wav(folder / "sub" / "MC Sub - Dentro.wav", 1.0)

    tracks = download.local_tracks_from_folder(folder)
    by_title = {t.title: t for t in tracks}
    assert set(by_title) == {"Faixa Um", "Faixa Dois", "sem separador"}

    um = by_title["Faixa Um"]
    assert um.artists == ["MC Teste"]
    assert um.source == "local" and um.is_local
    assert Path(um.local_path).is_absolute() and Path(um.local_path).exists()
    assert re.fullmatch(r"local_[0-9a-f]{12}", um.video_id)
    assert um.duration_s == pytest.approx(2.0, abs=0.05)

    assert by_title["Faixa Dois"].artists == ["MC Dois", "MC Três"]
    assert by_title["Faixa Dois"].duration_s == pytest.approx(3.0, abs=0.05)
    assert by_title["sem separador"].artists == []

    # ids estáveis e únicos
    again = download.local_tracks_from_folder(str(folder))
    assert [t.video_id for t in again] == [t.video_id for t in tracks]
    assert len({t.video_id for t in tracks}) == len(tracks)

    deep = download.local_tracks_from_folder(folder, recursive=True)
    assert "Dentro" in {t.title for t in deep}


def test_local_folder_missing_returns_empty(tmp_path):
    assert download.local_tracks_from_folder(tmp_path / "não existe") == []


def test_local_folder_accepts_pasted_windows_path(tmp_path, monkeypatch):
    folder = tmp_path / "Músicas do SK"
    folder.mkdir()
    _write_wav(folder / "MC Teste - Faixa Um.wav", 1.0)
    expected = [t.video_id for t in download.local_tracks_from_folder(folder)]
    assert len(expected) == 1
    # "Copiar como caminho" do Explorer põe aspas (o caminho tem espaços); colar traz espaços
    for pasted in (f'"{folder}"', f"  {folder}  ", f"'{folder}'", f' "{folder}" \n'):
        assert [t.video_id for t in download.local_tracks_from_folder(pasted)] == expected, pasted
    # campo vazio não pode listar a pasta atual do processo
    monkeypatch.chdir(folder)
    assert download.local_tracks_from_folder("") == []
    assert download.local_tracks_from_folder("   ") == []


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------
def test_purge_removes_mp3_and_analysis_json(tmp_dirs, tmp_path):
    cache = tmp_dirs["CACHE_DIR"]
    for name in ("a.mp3", "a.json", "b.mp3", "keep.mp3", "keep.json"):
        (cache / name).write_bytes(b"x")
    local_file = tmp_path / "local.mp3"
    local_file.write_bytes(b"x")

    removed = download.purge(["a", "b", "c", "a", "local_0123456789ab", "../keep"])
    assert removed == 3
    assert sorted(p.name for p in cache.iterdir()) == ["keep.json", "keep.mp3"]
    assert local_file.exists()


# ---------------------------------------------------------------------------
# Rede real (rodar com: pytest -m network)
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_real_download(tmp_path):
    progress: list[float] = []
    t0 = time.monotonic()
    path = download.download_audio("bN8S3cRF9X0", progress_cb=progress.append, cache_dir=tmp_path)
    elapsed = time.monotonic() - t0
    size = path.stat().st_size
    print(f"\n[rede] bN8S3cRF9X0: {elapsed:.1f} s, {size / 1e6:.2f} MB, {len(progress)} callbacks")

    assert path == tmp_path / "bN8S3cRF9X0.mp3"
    assert size > 1_000_000
    assert progress and progress[-1] == 1.0
    assert any(0 < p < 0.9 for p in progress)
    assert _tmp_leftovers(tmp_path) == []

    import mutagen

    audio = mutagen.File(path)
    assert audio is not None and audio.info.length > 60

    # segunda chamada vem do cache, sem rede
    t1 = time.monotonic()
    assert download.download_audio("bN8S3cRF9X0", cache_dir=tmp_path) == path
    assert time.monotonic() - t1 < 1.0
