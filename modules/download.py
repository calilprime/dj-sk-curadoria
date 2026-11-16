"""Cache de áudio da faixa inteira com yt-dlp (seção 4.3).

CONTRATO (implementar mantendo as assinaturas):

- VALIDADO em 22/10/2026: yt-dlp (API Python) com format="bestaudio/best" e
  postprocessor FFmpegExtractAudio (mp3, 320) baixou 5 faixas em 24 s com 4 em paralelo
  (~6 MB por faixa). URL usada: https://music.youtube.com/watch?v={videoId}.
- Chamar `util.setup_ssl()` antes de baixar (há interceptação TLS nesta máquina).
- Grava em `cache/{videoId}.mp3`. Nunca deixa arquivo parcial com o nome final.
- Também deve funcionar sem internet, com arquivos de áudio de uma pasta local.

Implementação: cada tentativa baixa e converte numa subpasta exclusiva
`cache/.tmp/{videoId}-{uuid}/` e só no fim move o mp3 com `os.replace` (atômico no
mesmo disco). Um lock por arquivo final impede que dois downloads do mesmo id rodem
juntos; o segundo encontra o cache pronto e volta na hora.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

import yt_dlp

import config
from modules import util
from modules.models import Track

log = logging.getLogger(__name__)

URL_TEMPLATE = "https://music.youtube.com/watch?v={}"
TMP_DIRNAME = ".tmp"                # subpasta do cache para downloads em andamento
DOWNLOAD_SHARE = 0.9                # fração da barra reservada ao download; o resto é a conversão
_RETRY_BACKOFF_S = 2.0              # espera antes da nova tentativa (× nº da tentativa)
_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_MSG_FFMPEG = "ffmpeg não encontrado. Instale o ffmpeg e coloque-o no PATH."
_MSG_NETWORK = "Sem internet ou erro de rede ao baixar."

# (trechos da mensagem do yt-dlp em minúsculas, mensagem ao usuário, vale tentar de novo?)
# A ordem importa: a primeira regra que casar vence.
_ERROR_RULES: list[tuple[tuple[str, ...], str, bool]] = [
    (("ffmpeg not found", "ffprobe not found", "ffprobe and ffmpeg not found"), _MSG_FFMPEG, False),
    (("confirm your age", "age-restricted", "age restricted", "inappropriate for some users"),
     "Vídeo com restrição de idade (exige login no YouTube).", False),
    (("not a bot",), "O YouTube pediu verificação anti-robô. Tente de novo mais tarde.", False),
    # Antes da regra de "indisponível": "Requested format is not available" não é vídeo privado.
    (("requested format is not available", "only images are available", "no video formats found"),
     "O YouTube não ofereceu o áudio desta faixa (o yt-dlp pode estar desatualizado).", False),
    (("private video", "video is private", "video unavailable", "is not available",
      "isn't available", "has been removed", "copyright", "members-only", "members only",
      "not available in your country", "not made this video available in your country",
      "geo restrict", "this live event", "premieres in",
      "incomplete youtube id", "unsupported url"),
     "Vídeo indisponível ou privado no YouTube.", False),
    (("certificate verify failed", "certificate_verify_failed", "sslerror", "ssl:"),
     "Erro de certificado SSL (antivírus ou proxy interceptando a conexão).", False),
    (("getaddrinfo", "name resolution", "nodename nor servname", "timed out", "timeout",
      "connection", "network is unreachable", "unreachable", "temporary failure",
      "unable to download", "urlopen error", "remote end closed", "http error 5",
      "http error 429", "http error 403", "incompleteread", "content too short"),
     _MSG_NETWORK, True),
]

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class DownloadError(Exception):
    """Falha ao baixar/converter uma faixa (mensagem curta em português)."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable      # uso interno: decide se vale uma nova tentativa


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _resolve_cache_dir(cache_dir: Path | None) -> Path:
    return Path(cache_dir) if cache_dir is not None else Path(config.CACHE_DIR)


def cache_path(video_id: str, cache_dir: Path | None = None) -> Path:
    """Caminho `cache/{videoId}.mp3` (cache_dir=None usa config.CACHE_DIR no momento da chamada)."""
    return _resolve_cache_dir(cache_dir) / f"{video_id}.{config.AUDIO_CODEC}"


def is_cached(video_id: str, cache_dir: Path | None = None) -> bool:
    """True se o mp3 existe e não está vazio."""
    try:
        return cache_path(video_id, cache_dir).stat().st_size > 0
    except OSError:
        return False


def audio_path_for(track: Track, cache_dir: Path | None = None) -> Path | None:
    """Arquivo de áudio da faixa: `local_path` para faixas locais, senão o cache (se existir)."""
    if track.local_path:
        p = Path(track.local_path)
        return p if p.is_file() else None
    if is_cached(track.video_id, cache_dir):
        return cache_path(track.video_id, cache_dir)
    return None


def clear_tmp(cache_dir: Path | None = None, max_age_s: float = 3600.0) -> int:
    """Apaga sobras de downloads interrompidos (app fechado no meio). Devolve quantas pastas.

    Só remove subpastas de `cache/.tmp` mais velhas que `max_age_s`, para não atrapalhar
    downloads em andamento.
    """
    tmp_root = _resolve_cache_dir(cache_dir) / TMP_DIRNAME
    if not tmp_root.is_dir():
        return 0
    removed = 0
    now = time.time()
    for d in tmp_root.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _lock_for(path: Path) -> threading.Lock:
    key = os.path.normcase(str(path.absolute()))
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = _locks[key] = threading.Lock()
        return lock


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _notify(progress_cb: Callable[[float], None] | None, fraction: float) -> None:
    """Chama o callback sem deixar uma exceção dele derrubar o download."""
    if progress_cb is None:
        return
    try:
        progress_cb(min(max(float(fraction), 0.0), 1.0))
    except Exception:
        log.debug("progress_cb levantou exceção (ignorada)", exc_info=True)


def _short_detail(exc: BaseException, limit: int = 100) -> str:
    text = _ANSI_RE.sub("", str(exc)).strip()
    text = text.splitlines()[0] if text else type(exc).__name__
    if text.upper().startswith("ERROR:"):
        text = text[6:].strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _translate_error(exc: BaseException) -> DownloadError:
    """Converte erros do yt-dlp/ffmpeg/rede em DownloadError com mensagem em português."""
    if isinstance(exc, DownloadError):
        return exc
    text = _ANSI_RE.sub("", str(exc)).lower()
    for needles, message, retryable in _ERROR_RULES:
        if any(n in text for n in needles):
            err = DownloadError(message, retryable=retryable)
            break
    else:
        if isinstance(exc, (ConnectionError, TimeoutError)):
            err = DownloadError(_MSG_NETWORK, retryable=True)
        else:
            err = DownloadError(f"Falha ao baixar a faixa ({_short_detail(exc)}).", retryable=True)
    err.__cause__ = exc
    return err


class _YdlLogger:
    """Encaminha as mensagens do yt-dlp para o logging (nada vai para o terminal)."""

    def debug(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def info(self, msg: str) -> None:
        log.debug("yt-dlp: %s", msg)

    def warning(self, msg: str) -> None:
        log.debug("yt-dlp aviso: %s", msg)

    def error(self, msg: str) -> None:
        log.debug("yt-dlp erro: %s", msg)


_AGE_MSG = "Vídeo com restrição de idade (exige login no YouTube)."
COOKIES_FILENAME = "cookies.txt"


def cookie_options() -> dict:
    """Login do YouTube para vídeos com restrição de idade (conta do próprio DJ, maior de idade).

    1º `cookies.txt` na pasta do projeto (exportado do navegador); senão o navegador indicado
    em YOUTUBE_COOKIES_BROWSER no .env (ex.: firefox). Sem nada configurado, devolve {}.
    """
    path = Path(config.BASE_DIR) / COOKIES_FILENAME
    if path.is_file() and path.stat().st_size > 0:
        return {"cookiefile": str(path)}
    browser = config.get_secret("YOUTUBE_COOKIES_BROWSER")
    return {"cookiesfrombrowser": (browser.lower(),)} if browser else {}


def _ydl_options(work_dir: Path, video_id: str,
                 progress_cb: Callable[[float], None] | None, extra: dict | None = None) -> dict:
    def hook(d: dict) -> None:
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes")
            if total and done is not None:
                _notify(progress_cb, min(done / total, 1.0) * DOWNLOAD_SHARE)
        elif status == "finished":
            _notify(progress_cb, DOWNLOAD_SHARE)

    return {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # A pasta vai em "paths" (não no outtmpl) para um "%" no caminho não virar campo do modelo.
        "paths": {"home": str(work_dir), "temp": str(work_dir)},
        "outtmpl": f"{video_id}.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": config.AUDIO_CODEC,
            "preferredquality": str(config.AUDIO_QUALITY),
        }],
        "progress_hooks": [hook],
        "logger": _YdlLogger(),
        "overwrites": True,
        "socket_timeout": 30,
        **(extra or {}),
    }


def _find_output(work_dir: Path, video_id: str) -> Path | None:
    expected = work_dir / f"{video_id}.{config.AUDIO_CODEC}"
    if expected.is_file() and expected.stat().st_size > 0:
        return expected
    for p in work_dir.glob(f"*.{config.AUDIO_CODEC}"):
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _download_once(video_id: str, final: Path,
                   progress_cb: Callable[[float], None] | None, extra: dict | None = None) -> None:
    work_dir = final.parent / TMP_DIRNAME / f"{video_id}-{uuid.uuid4().hex[:12]}"
    work_dir.mkdir(parents=True)
    try:
        with yt_dlp.YoutubeDL(_ydl_options(work_dir, video_id, progress_cb, extra)) as ydl:
            retcode = ydl.download([URL_TEMPLATE.format(video_id)])
        if retcode:
            raise DownloadError("Falha ao baixar a faixa.", retryable=True)
        produced = _find_output(work_dir, video_id)
        if produced is None:
            raise DownloadError("A conversão para mp3 não gerou arquivo.", retryable=True)
        os.replace(produced, final)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def download_audio(video_id: str, progress_cb: Callable[[float], None] | None = None,
                   cache_dir: Path | None = None) -> Path:
    """Baixa só o áudio e converte para mp3 com ffmpeg. Devolve o caminho final.

    Se já estiver em cache, devolve na hora. `progress_cb(fração 0..1)` é chamado durante o
    download (a partir dos progress_hooks do yt-dlp). Tenta DOWNLOAD_RETRIES vezes.
    Levanta DownloadError com mensagem amigável em caso de falha.
    """
    if not isinstance(video_id, str):
        raise DownloadError("Identificador de vídeo inválido.")
    if video_id.startswith("local_"):
        raise DownloadError("Faixa local não precisa de download.")
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise DownloadError("Identificador de vídeo inválido.")

    cache_dir = _resolve_cache_dir(cache_dir)
    final = cache_path(video_id, cache_dir)
    if is_cached(video_id, cache_dir):
        _notify(progress_cb, 1.0)
        return final

    with _lock_for(final):
        if is_cached(video_id, cache_dir):   # outro download do mesmo id terminou enquanto esperávamos
            _notify(progress_cb, 1.0)
            return final
        if not _ffmpeg_available():
            raise DownloadError(_MSG_FFMPEG)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:      # caminho é um arquivo, sem permissão, unidade inexistente
            raise DownloadError(f"Não foi possível usar a pasta do cache: {cache_dir}") from exc
        util.setup_ssl()

        attempts = max(1, int(config.DOWNLOAD_RETRIES))
        last_error: DownloadError | None = None
        started = time.monotonic()
        cookies: dict | None = None   # só usa o login quando o vídeo exige (restrição de idade)
        for attempt in range(1, attempts + 1):
            try:
                _download_once(video_id, final, progress_cb, cookies)
            except Exception as exc:  # yt-dlp, ffmpeg, rede ou disco
                last_error = _translate_error(exc)
                if str(last_error) == _AGE_MSG and cookies is None:
                    cookies = cookie_options()
                    if cookies:
                        log.info("%s tem restrição de idade; tentando com o login do YouTube.", video_id)
                        try:
                            _download_once(video_id, final, progress_cb, cookies)
                            _notify(progress_cb, 1.0)
                            return final
                        except Exception as exc2:  # noqa: BLE001
                            last_error = _translate_error(exc2)
                    else:
                        last_error = DownloadError(
                            "Vídeo com restrição de idade: coloque o cookies.txt do YouTube na pasta do "
                            "painel (veja o guia) e clique em tentar de novo.")
                    break
                log.warning("Download de %s falhou (tentativa %d/%d): %s",
                            video_id, attempt, attempts, _short_detail(exc))
                if not last_error.retryable:
                    break
                if attempt < attempts and _RETRY_BACKOFF_S > 0:
                    time.sleep(_RETRY_BACKOFF_S * attempt)
                continue
            _notify(progress_cb, 1.0)
            log.info("Baixado %s em %.1f s", video_id, time.monotonic() - started)
            return final

    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Pasta local e limpeza
# ---------------------------------------------------------------------------
def _clean_folder_text(folder: str | Path) -> str:
    """Tira espaços e um par de aspas em volta de um caminho digitado/colado."""
    if isinstance(folder, os.PathLike):
        return os.fspath(folder)
    text = str(folder).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def _local_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8", "surrogatepass")).hexdigest()
    return "local_" + digest[:12]


def _parse_filename(stem: str) -> tuple[list[str], str]:
    """"Artista - Título" -> (["Artista"], "Título"); aceita prefixo numérico "01 - "."""
    name = re.sub(r"\s+[–—]\s+", " - ", stem)
    name = re.sub(r"\s+", " ", name).strip()
    parts = name.split(" - ")
    if len(parts) >= 3 and re.fullmatch(r"\d{1,3}", parts[0]):
        parts = parts[1:]
    if len(parts) >= 2 and parts[0].strip() and " - ".join(parts[1:]).strip():
        return _split_artists(parts[0]), " - ".join(parts[1:]).strip()
    return [], name or stem


def _split_artists(text: str) -> list[str]:
    pieces = re.split(r"\s*,\s*|\s+&\s+|\s+x\s+|\s+feat\.?\s+|\s+ft\.?\s+", text, flags=re.IGNORECASE)
    return [p.strip() for p in pieces if p.strip()]


def _read_tags(path: Path) -> tuple[str | None, list[str], float | None]:
    """(título, artistas, duração) pelas tags do arquivo; vazio se o mutagen falhar."""
    try:
        import mutagen

        audio = mutagen.File(path, easy=True)
    except Exception:
        log.debug("mutagen não leu %s", path.name, exc_info=True)
        return None, [], None
    if audio is None:
        return None, [], None
    duration = None
    try:
        length = float(audio.info.length)
        duration = length if length > 0 else None
    except Exception:
        pass
    title, artists = None, []
    try:
        tags = audio.tags or {}
        if tags.get("title"):
            title = str(tags["title"][0]).strip() or None
        for a in tags.get("artist", []) or []:
            artists.extend(_split_artists(str(a)))
    except Exception:
        pass
    return title, artists, duration


def local_tracks_from_folder(folder: str | Path, recursive: bool = False) -> list[Track]:
    """Lista arquivos de áudio (LOCAL_AUDIO_EXTENSIONS) de uma pasta como Tracks.

    video_id = "local_" + 12 primeiros hex do sha1 do caminho absoluto; source="local";
    local_path preenchido; título/artista lidos de "Artista - Título.ext" quando possível
    (senão título = nome do arquivo); duração via mutagen se disponível.
    Pasta inexistente (ou texto vazio) devolve lista vazia. `recursive=True` inclui subpastas.
    Aceita o caminho colado com aspas ("Copiar como caminho" do Explorer) e espaços nas pontas.
    """
    text = _clean_folder_text(folder)
    if not text:
        log.warning("Nenhuma pasta de áudio informada.")
        return []
    base = Path(text).expanduser()
    if not base.is_dir():
        log.warning("Pasta de áudio não encontrada: %s", base)
        return []
    exts = {e.lower() for e in config.LOCAL_AUDIO_EXTENSIONS}
    candidates = base.rglob("*") if recursive else base.iterdir()
    files = sorted(
        (p for p in candidates
         if p.suffix.lower() in exts and not p.name.startswith(".") and p.is_file()),
        key=lambda p: str(p).casefold(),
    )
    tracks: list[Track] = []
    for p in files:
        abs_path = p.resolve()
        artists, title = _parse_filename(p.stem)
        tag_title, tag_artists, duration = _read_tags(abs_path)
        if not artists and tag_artists:
            artists = tag_artists
            title = tag_title or title
        tracks.append(Track(
            video_id=_local_id(abs_path),
            title=title,
            artists=artists,
            duration_s=duration,
            source="local",
            local_path=str(abs_path),
        ))
    return tracks


def purge(video_ids: Iterable[str], cache_dir: Path | None = None) -> int:
    """Apaga do cache o mp3 e o JSON de análise das faixas indicadas (reprovadas).

    Nunca apaga arquivos de faixas locais (ids "local_..." não têm mp3 no cache).
    Devolve quantos arquivos foram apagados.
    """
    base = _resolve_cache_dir(cache_dir)
    removed = 0
    for vid in dict.fromkeys(video_ids):
        if not vid or vid.startswith("local_") or not _VIDEO_ID_RE.fullmatch(vid):
            continue
        for p in (cache_path(vid, base), base / f"{vid}.json"):
            try:
                p.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as exc:   # no Windows: arquivo aberto pelo player
                log.warning("Não foi possível apagar %s: %s", p.name, exc)
    return removed
