"""Estúdio de Vocal: separação de stems com Demucs (seção 4.7).

CONTRATO (implementar mantendo as assinaturas):

- VALIDADO em 22/10/2026 lendo a aba API de cjwbw/demucs (versão em config.REPLICATE_VERSION):
  entrada: audio (uri/arquivo, obrigatório), model_name (enum, padrão "htdemucs"),
           stem (enum opcional: separa só esse stem + resto; NÃO usar, queremos os 4),
           shifts (int, 1), overlap (0.25), clip_mode ("rescale"), mp3_bitrate (320),
           float32 (bool), output_format ("mp3" | "wav" | "flac").
  saída: objeto {"vocals", "drums", "bass", "other", "guitar", "piano"} com URLs
         (guitar/piano só no htdemucs_6s; podem vir None).
  hardware: Nvidia T4, US$ 0,000225/s, ~US$ 0,016 por execução, termina em ~70 s.
- Usar `replicate.Client(api_token=...)` + `client.predictions.create(version=..., input=...)`
  e `prediction.wait()` para ler `prediction.metrics["predict_time"]` (custo real).
  Enviar o arquivo aberto em "rb" no campo audio (o cliente faz upload).
- Saldo insuficiente: a Replicate responde HTTP 402 (ReplicateError com status 402 ou
  mensagem com "insufficient credit"/"billing") -> `budget.mark_insufficient_credit` e
  levantar InsufficientCredit.
- Antes de rodar: `budget.can_run()`; se não puder, levantar BudgetExceeded (sem chamar a API).
- Baixar as URLs com `requests` (após `util.setup_ssl()`) para stems/{videoId}/{stem}.mp3.
- Plano B sem custo: Demucs local por linha de comando
  (`python -m demucs -n htdemucs --mp3 -o <tmp> <arquivo>`), se o pacote `demucs` existir.

Notas de implementação:
- Conferido no replicate 1.0.7: um handle aberto em "rb" (io.IOBase) é enviado por
  `client.files.create` dentro de `predictions.create`; erros HTTP viram ReplicateError
  com `.status`. A espera usa `prediction.reload()` em laço (equivale a `wait()`, mas com
  o limite config.REPLICATE_TIMEOUT_S); sem `reload`, cai em `wait()`.
- Se a separação paga terminou mas o download falhou, as URLs ficam em meta.json e a
  próxima chamada só baixa de novo, sem pagar outra vez (a Replicate guarda por ~1 h).
"""
from __future__ import annotations

import base64
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

import config
from modules import util
from modules.budget import Budget
from modules.models import StemResult

log = logging.getLogger(__name__)

ProgressFn = Callable[[str], None]

_TERMINAL = ("succeeded", "failed", "canceled")
_POLL_INTERVAL_S = 1.0
_MAX_RELOAD_FAILURES = 3
_DOWNLOAD_TIMEOUT = (15, 120)          # (conexão, leitura) em segundos
_LOCAL_TIMEOUT_S = 3600                # Demucs na CPU leva minutos por faixa
_STEM_EXTENSIONS = (".mp3", ".wav", ".flac")
_META_NAME = "meta.json"
_INSUFFICIENT_HINTS = ("insufficient", "billing", "credit", "payment required")
_AUTH_HOSTS = ("api.replicate.com",)  # só esses hosts recebem o token no download
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

_STATUS_MESSAGES = {
    "starting": "Na fila da Replicate (ligando a GPU)…",
    "processing": "Separando os stems na GPU da Replicate…",
}

_LOCAL_INSTALL_MSG = (
    "O Demucs local não está instalado neste Python. Instale com "
    "`.venv\\Scripts\\python.exe -m pip install demucs` (é pesado: baixa o PyTorch, "
    "cerca de 2 GB, e na CPU cada faixa leva alguns minutos)."
)


class StemsError(Exception):
    """Falha na separação (mensagem em português)."""


class BudgetExceeded(StemsError):
    """O teto de gasto foi atingido; nada foi enviado à Replicate."""


class InsufficientCredit(StemsError):
    """A Replicate informou saldo insuficiente."""


class MissingToken(StemsError):
    """REPLICATE_API_TOKEN ausente no .env."""


class _ExpiredOutput(StemsError):
    """As URLs de saída da Replicate não valem mais (HTTP 401/403/404/410)."""


# Uma trava por pasta de stems: dois cliques na mesma faixa não pagam duas vezes.
_VIDEO_LOCKS: dict[str, threading.Lock] = {}
_VIDEO_LOCKS_GUARD = threading.Lock()


def _video_lock(folder: Path) -> threading.Lock:
    key = os.path.normcase(str(folder.resolve()))
    with _VIDEO_LOCKS_GUARD:
        lock = _VIDEO_LOCKS.get(key)
        if lock is None:
            lock = _VIDEO_LOCKS[key] = threading.Lock()
        return lock


def _notify(on_progress: ProgressFn | None, message: str) -> None:
    if on_progress is None:
        return
    try:
        on_progress(message)
    except Exception:  # a interface não pode derrubar uma execução já paga
        log.debug("Falha no callback de progresso", exc_info=True)


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _output_ext() -> str:
    return "." + str(config.DEMUCS_OUTPUT_FORMAT).lower().lstrip(".")


def _silent_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.debug("Não foi possível apagar %s", path, exc_info=True)


# ----------------------------------------------------------------------
# Cache em disco
# ----------------------------------------------------------------------
NAMES_FILE = "_nomes.json"   # video_id -> nome da pasta ("Artista - Título")


def register_name(video_id: str, name: str, base_dir: Path | None = None) -> None:
    """Faz os stems da faixa irem para `stems/Artista - Título/` em vez de `stems/{videoId}/`.

    Se já existe a pasta antiga com o id, ela é renomeada.
    """
    base = Path(base_dir) if base_dir is not None else Path(config.STEMS_DIR)
    names = util.read_json(base / NAMES_FILE, {}) or {}
    folder = util.safe_filename(name, max_len=120)
    taken = {v for k, v in names.items() if k != video_id}
    if folder in taken:
        folder = util.safe_filename(f"{name} [{video_id}]", max_len=140)
    if names.get(video_id) == folder:
        return
    old = stems_dir(video_id, base)
    names[video_id] = folder
    util.atomic_write_json(base / NAMES_FILE, names)
    new = base / folder
    if old.is_dir() and old != new and not new.exists():
        old.rename(new)


def stems_dir(video_id: str, base_dir: Path | None = None) -> Path:
    """`stems/Artista - Título/` (se registrado com `register_name`), senão `stems/{videoId}/`."""
    base = Path(base_dir) if base_dir is not None else Path(config.STEMS_DIR)
    names = util.read_json(base / NAMES_FILE, {}) or {}
    return base / (names.get(str(video_id)) or util.safe_filename(str(video_id)))


def _find_stem(folder: Path, stem: str) -> Path | None:
    for ext in dict.fromkeys((_output_ext(), *_STEM_EXTENSIONS)):
        p = folder / f"{stem}{ext}"
        try:
            if p.is_file() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def _read_meta(folder: Path) -> dict[str, Any]:
    meta = util.read_json(folder / _META_NAME, default={})
    return meta if isinstance(meta, dict) else {}


def load_stems(video_id: str, base_dir: Path | None = None) -> StemResult | None:
    """StemResult(backend="cache") se os 4 stems (STEM_NAMES) já existem; senão None."""
    folder = stems_dir(video_id, base_dir)
    found: dict[str, str] = {}
    for stem in config.STEM_NAMES:
        p = _find_stem(folder, stem)
        if p is None:
            return None
        found[stem] = str(p)
    meta = _read_meta(folder)
    return StemResult(video_id=video_id, stems=found, backend="cache",
                      prediction_id=meta.get("prediction_id"))


def has_stems(video_id: str, base_dir: Path | None = None) -> bool:
    return load_stems(video_id, base_dir) is not None


# ----------------------------------------------------------------------
# Replicate
# ----------------------------------------------------------------------
def _error_text(exc: BaseException) -> str:
    parts = [getattr(exc, "detail", None), getattr(exc, "title", None)]
    text = next((str(p) for p in parts if p), "") or str(exc) or exc.__class__.__name__
    return " ".join(text.split())[:300]


def _mentions_credit(text: str | None) -> bool:
    low = (text or "").lower()
    return any(h in low for h in _INSUFFICIENT_HINTS)


def _insufficient(budget: Budget, detail: str) -> InsufficientCredit:
    budget.mark_insufficient_credit(detail)
    msg = "A Replicate recusou por saldo insuficiente"
    msg += f" ({detail})." if detail else "."
    msg += (" Adicione crédito em replicate.com/account/billing e libere a trava no painel"
            " antes de tentar de novo.")
    return InsufficientCredit(msg)


def _is_network_error(exc: BaseException) -> bool:
    try:
        import httpx
        net: tuple[type[BaseException], ...] = (httpx.TransportError,)
    except ImportError:
        net = ()
    return isinstance(exc, (*net, requests.ConnectionError, requests.Timeout,
                            ConnectionError, TimeoutError))


def _translate_api_error(exc: BaseException, budget: Budget) -> StemsError:
    status = getattr(exc, "status", None)
    detail = _error_text(exc)
    if status == 402 or _mentions_credit(detail):
        return _insufficient(budget, detail)
    if status in (401, 403):
        return StemsError("A Replicate recusou o token (REPLICATE_API_TOKEN inválido ou sem "
                          "permissão). Gere outro em replicate.com/account/api-tokens e "
                          "atualize o .env.")
    if status in (404, 422):
        return StemsError(f"A Replicate não aceitou o modelo/versão configurado ({detail}). "
                          "Confira REPLICATE_VERSION no config.py.")
    if status == 429:
        return StemsError("A Replicate pediu para esperar (muitas requisições). "
                          "Tente de novo em um minuto.")
    if _is_network_error(exc):
        return StemsError(f"Não foi possível falar com a Replicate (sem internet?): {detail}")
    return StemsError(f"Falha ao chamar a Replicate: {detail}")


def _predict_time(prediction: Any) -> float | None:
    metrics = getattr(prediction, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("predict_time")
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _create_prediction(client: Any, audio: Path, budget: Budget,
                       on_progress: ProgressFn | None) -> Any:
    try:
        handle = open(audio, "rb")
    except OSError as exc:
        raise StemsError(f"Não foi possível ler o arquivo de áudio {audio.name}: "
                         f"{exc.strerror or exc}") from exc
    _notify(on_progress, "Enviando o áudio para a Replicate…")
    with handle:
        try:
            return client.predictions.create(
                version=config.REPLICATE_VERSION,
                input={"audio": handle,
                       "model_name": config.DEMUCS_MODEL_NAME,
                       "output_format": config.DEMUCS_OUTPUT_FORMAT},
            )
        except Exception as exc:
            raise _translate_api_error(exc, budget) from exc


def _wait(prediction: Any, video_id: str, budget: Budget,
          on_progress: ProgressFn | None) -> None:
    """Espera o fim da previsão. Timeout ou perda de contato contam o custo e levantam."""
    timeout_s = float(config.REPLICATE_TIMEOUT_S)
    deadline = time.monotonic() + timeout_s
    last_status: str | None = None
    failures = 0
    while True:
        status = getattr(prediction, "status", None)
        if status != last_status:
            last_status = status
            if status in _STATUS_MESSAGES:
                _notify(on_progress, _STATUS_MESSAGES[status])
        if status in _TERMINAL:
            return
        pid = getattr(prediction, "id", None)
        if time.monotonic() >= deadline:
            try:
                prediction.cancel()
                prediction.reload()
            except Exception:
                log.warning("Não foi possível cancelar a previsão %s", pid, exc_info=True)
            budget.record(video_id, pid, _predict_time(prediction), status="canceled")
            raise StemsError(f"A Replicate não terminou em {timeout_s / 60:.0f} min; a execução "
                             "foi cancelada e o custo contabilizado. Tente de novo mais tarde.")
        time.sleep(_POLL_INTERVAL_S)
        reload = getattr(prediction, "reload", None)
        try:
            if reload is None:
                prediction.wait()
            else:
                reload()
            failures = 0
        except Exception as exc:
            failures += 1
            if failures < _MAX_RELOAD_FAILURES:
                log.warning("Falha ao consultar a previsão %s (%s); tentando de novo", pid,
                            _error_text(exc))
                continue
            # A previsão pode ter rodado: conta a estimativa para não estourar o teto.
            budget.record(video_id, pid, _predict_time(prediction), status="unknown")
            raise StemsError("Perdemos o contato com a Replicate durante a separação "
                             f"({_error_text(exc)}). O custo estimado foi contabilizado por "
                             "segurança; tente de novo.") from exc


def _as_url(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    url = getattr(value, "url", None)  # replicate.helpers.FileOutput
    if isinstance(url, str) and url:
        return url
    text = str(value).strip()
    return text or None


def _output_urls(output: Any) -> dict[str, str]:
    urls: dict[str, str] = {}
    if isinstance(output, dict):
        for key, value in output.items():
            url = _as_url(value)
            if url:
                urls[str(key)] = url
    elif isinstance(output, (list, tuple)):
        for value in output:
            url = _as_url(value)
            if url:
                urls.setdefault(Path(urlparse(url).path).stem, url)
    return urls


def _auth_headers(url: str, token: str | None) -> dict[str, str]:
    host = (urlparse(url).hostname or "").lower()
    if token and host in _AUTH_HOSTS:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _download(url: str, dest: Path, headers: dict[str, str]) -> None:
    """Baixa para `dest.part` e só então renomeia (download parcial não vira stem)."""
    part = dest.with_name(dest.name + ".part")
    try:
        if url.startswith("data:"):
            part.write_bytes(base64.b64decode(url.split(",", 1)[1]))
        else:
            with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT,
                              headers=headers) as resp:
                resp.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 16):
                        if chunk:
                            fh.write(chunk)
        if part.stat().st_size == 0:
            raise StemsError(f"download de {dest.stem} veio vazio")
        os.replace(part, dest)
    except BaseException:
        _silent_unlink(part)
        raise


def _download_all(urls: dict[str, str], folder: Path, token: str | None,
                  on_progress: ProgressFn | None) -> dict[str, str]:
    util.setup_ssl()
    folder.mkdir(parents=True, exist_ok=True)
    ext = _output_ext()
    names = config.STEM_NAMES
    out: dict[str, str] = {}
    for i, stem in enumerate(names, 1):
        _notify(on_progress, f"Baixando os stems ({i}/{len(names)}): {stem}…")
        url = urls[stem]
        dest = folder / f"{stem}{ext}"
        try:
            _download(url, dest, _auth_headers(url, token))
        except requests.HTTPError as exc:
            code = getattr(exc.response, "status_code", None)
            cls = _ExpiredOutput if code in (401, 403, 404, 410) else StemsError
            raise cls(f"download de {stem} falhou (HTTP {code})") from exc
        except StemsError:
            raise
        except (requests.RequestException, OSError, ValueError) as exc:
            raise StemsError(f"download de {stem} falhou: {_error_text(exc)}") from exc
        out[stem] = str(dest)
    return out


def _write_meta(folder: Path, data: dict[str, Any]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    util.atomic_write_json(folder / _META_NAME, data)


def _resume_download(video_id: str, folder: Path, token: str | None,
                     on_progress: ProgressFn | None) -> StemResult | None:
    """Baixa de novo uma separação já paga cujo download falhou. None se não há o que retomar."""
    meta = _read_meta(folder)
    urls = meta.get("output_urls")
    if meta.get("backend") != "replicate" or meta.get("complete") or not isinstance(urls, dict):
        return None
    if any(not isinstance(urls.get(s), str) for s in config.STEM_NAMES):
        return None
    log.info("Retomando o download dos stems de %s (separação já paga)", video_id)
    try:
        paths = _download_all(urls, folder, token, on_progress)
    except _ExpiredOutput:
        log.warning("As URLs da separação anterior de %s expiraram; será preciso rodar de novo",
                    video_id)
        return None
    except StemsError as exc:
        raise StemsError(f"A separação desta faixa já foi paga, mas o {exc}. "
                         "Tente de novo em instantes (não haverá nova cobrança).") from exc
    meta.pop("output_urls", None)
    meta.update(complete=True, completed_at=_now_iso(),
                stems={k: Path(v).name for k, v in paths.items()})
    _write_meta(folder, meta)
    return StemResult(video_id=video_id, stems=paths, backend="replicate",
                      predict_time_s=meta.get("predict_time_s"),
                      cost_usd=float(meta.get("cost_usd") or 0.0),
                      cost_brl=float(meta.get("cost_brl") or 0.0),
                      prediction_id=meta.get("prediction_id"))


def separate_replicate(video_id: str, audio_path: str | Path, budget: Budget,
                       token: str | None = None, base_dir: Path | None = None,
                       client=None, on_progress: ProgressFn | None = None) -> StemResult:
    """Separa via Replicate e registra o custo no `budget`.

    Se os stems já existem, devolve o cache sem gastar. `client` permite injetar um
    cliente falso nos testes. token None lê REPLICATE_API_TOKEN do .env (MissingToken se faltar).
    Registra no budget também execuções que falharam após rodar (status "failed") quando
    houver predict_time.
    `on_progress` (opcional) recebe mensagens curtas em português para a interface.
    """
    folder = stems_dir(video_id, base_dir)
    with _video_lock(folder):
        cached = load_stems(video_id, base_dir)
        if cached is not None:
            return cached

        token = token or config.get_secret("REPLICATE_API_TOKEN")
        resumed = _resume_download(video_id, folder, token, on_progress)
        if resumed is not None:
            return resumed

        if client is None and not token:
            raise MissingToken("Configure REPLICATE_API_TOKEN no arquivo .env (copie de "
                               ".env.example) para usar o Estúdio de Vocal. Gere o token em "
                               "replicate.com/account/api-tokens.")
        ok, reason = budget.can_run()
        if not ok:
            raise BudgetExceeded(reason)
        audio = Path(audio_path)
        if not audio.is_file():
            raise StemsError(f"Arquivo de áudio não encontrado: {audio.name}. "
                             "Baixe a faixa de novo antes de extrair a acapella.")

        util.setup_ssl()
        if client is None:
            import replicate

            client = replicate.Client(api_token=token)

        log.info("Separando %s na Replicate (%s)", video_id, config.DEMUCS_MODEL_NAME)
        prediction = _create_prediction(client, audio, budget, on_progress)
        _wait(prediction, video_id, budget, on_progress)

        pid = getattr(prediction, "id", None)
        status = getattr(prediction, "status", None)
        predict_time = _predict_time(prediction)
        if status != "succeeded":
            if predict_time is not None:
                budget.record(video_id, pid, predict_time, status=str(status))
            detail = " ".join(str(getattr(prediction, "error", None) or "").split())[:300]
            if _mentions_credit(detail):
                raise _insufficient(budget, detail)
            verb = "foi cancelada" if status == "canceled" else "falhou"
            raise StemsError(f"A separação {verb} na Replicate"
                             + (f": {detail}" if detail else "."))

        entry = budget.record(video_id, pid, predict_time, status="succeeded")
        urls = _output_urls(getattr(prediction, "output", None))
        missing = [s for s in config.STEM_NAMES if s not in urls]
        if missing:
            raise StemsError("A Replicate terminou, mas não devolveu os stems "
                             f"{', '.join(missing)} (o custo já foi contabilizado).")

        meta: dict[str, Any] = {
            "video_id": video_id,
            "backend": "replicate",
            "model": config.REPLICATE_MODEL,
            "version": config.REPLICATE_VERSION,
            "demucs_model": config.DEMUCS_MODEL_NAME,
            "prediction_id": pid,
            "predict_time_s": predict_time,
            "cost_usd": entry["usd"],
            "cost_brl": entry["brl"],
            "cost_estimated": bool(entry.get("estimated")),
            "created_at": _now_iso(),
            "complete": False,
            "output_urls": {s: urls[s] for s in config.STEM_NAMES},
        }
        _write_meta(folder, meta)
        try:
            paths = _download_all(urls, folder, token, on_progress)
        except StemsError as exc:
            raise StemsError(f"A Replicate separou a faixa (custo já contabilizado), mas o {exc}. "
                             "Clique de novo em até 1 hora para baixar sem pagar outra vez.") from exc
        meta.pop("output_urls", None)
        meta.update(complete=True, completed_at=_now_iso(),
                    stems={k: Path(v).name for k, v in paths.items()})
        _write_meta(folder, meta)
        log.info("Stems de %s prontos (US$ %.5f)", video_id, entry["usd"])
        return StemResult(video_id=video_id, stems=paths, backend="replicate",
                          predict_time_s=predict_time, cost_usd=entry["usd"],
                          cost_brl=entry["brl"], prediction_id=pid)


# ----------------------------------------------------------------------
# Plano B: Demucs local
# ----------------------------------------------------------------------
def local_demucs_available() -> bool:
    """True se `import demucs` funciona neste Python.

    Usa `find_spec` para não importar o PyTorch a cada rerun do Streamlit; se a
    instalação estiver quebrada, `separate_local` informa o erro do Demucs.
    """
    try:
        return importlib.util.find_spec("demucs") is not None
    except (ImportError, ValueError):
        return False


def _tail(text: str | None, lines: int = 6) -> str:
    rows = [r.strip() for r in (text or "").replace("\r", "\n").splitlines() if r.strip()]
    return " | ".join(rows[-lines:])[-600:]


def _collect_local_outputs(tmp: Path, model: str, track: str) -> dict[str, Path]:
    expected = tmp / model / track
    found: dict[str, Path] = {}
    for stem in config.STEM_NAMES:
        for ext in _STEM_EXTENSIONS:
            p = expected / f"{stem}{ext}"
            if p.is_file():
                found[stem] = p
                break
        else:  # o Demucs pode ter normalizado o nome da pasta da faixa
            matches = sorted(m for m in tmp.rglob(f"{stem}.*")
                             if m.suffix.lower() in _STEM_EXTENSIONS and m.is_file())
            if matches:
                found[stem] = matches[0]
    return found


def separate_local(video_id: str, audio_path: str | Path, base_dir: Path | None = None,
                   on_progress: ProgressFn | None = None) -> StemResult:
    """Plano B: Demucs local, custo zero. Move as saídas para stems/{videoId}/{stem}.mp3."""
    folder = stems_dir(video_id, base_dir)
    with _video_lock(folder):
        cached = load_stems(video_id, base_dir)
        if cached is not None:
            return cached
        if not local_demucs_available():
            raise StemsError(_LOCAL_INSTALL_MSG)
        audio = Path(audio_path)
        if not audio.is_file():
            raise StemsError(f"Arquivo de áudio não encontrado: {audio.name}. "
                             "Baixe a faixa de novo antes de extrair a acapella.")

        model = config.DEMUCS_MODEL_NAME
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="djsk_demucs_", ignore_cleanup_errors=True) as tmp:
            cmd = [sys.executable, "-m", "demucs", "-n", model, "--mp3", "-o", tmp, str(audio)]
            _notify(on_progress, "Separando com o Demucs local (na CPU pode levar alguns minutos)…")
            log.info("Separando %s com o Demucs local (%s)", video_id, model)
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                      errors="replace", timeout=_LOCAL_TIMEOUT_S, env=env,
                                      creationflags=_NO_WINDOW)
            except subprocess.TimeoutExpired as exc:
                raise StemsError(f"O Demucs local passou de {_LOCAL_TIMEOUT_S // 60} min "
                                 "e foi interrompido.") from exc
            except OSError as exc:
                raise StemsError(f"Não foi possível iniciar o Demucs local: {exc}") from exc
            if proc.returncode != 0:
                tail = _tail(proc.stderr or proc.stdout)
                if "No module named demucs" in tail:
                    raise StemsError(_LOCAL_INSTALL_MSG)
                raise StemsError(f"O Demucs local falhou (código {proc.returncode}): "
                                 f"{tail or 'sem mensagem'}")

            produced = _collect_local_outputs(Path(tmp), model, audio.stem)
            missing = [s for s in config.STEM_NAMES if s not in produced]
            if missing:
                raise StemsError("O Demucs local terminou sem gerar os stems "
                                 f"{', '.join(missing)}.")
            folder.mkdir(parents=True, exist_ok=True)
            stems: dict[str, str] = {}
            for stem in config.STEM_NAMES:
                src = produced[stem]
                dest = folder / f"{stem}{src.suffix.lower()}"
                part = dest.with_name(dest.name + ".part")
                try:
                    shutil.move(str(src), str(part))
                    os.replace(part, dest)
                except OSError as exc:
                    _silent_unlink(part)
                    raise StemsError(f"Não foi possível salvar o stem {stem}: {exc}") from exc
                stems[stem] = str(dest)

        _write_meta(folder, {
            "video_id": video_id, "backend": "local", "demucs_model": model,
            "prediction_id": None, "predict_time_s": None, "cost_usd": 0.0, "cost_brl": 0.0,
            "elapsed_s": round(time.monotonic() - started, 1), "created_at": _now_iso(),
            "complete": True, "stems": {k: Path(v).name for k, v in stems.items()},
        })
        return StemResult(video_id=video_id, stems=stems, backend="local")
