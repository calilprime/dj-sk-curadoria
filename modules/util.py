"""Utilitários compartilhados: SSL, nomes de arquivo, JSON atômico, texto."""
from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

_ssl_ready = False

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def setup_ssl() -> None:
    """Faz o Python usar o repositório de certificados do sistema.

    Nesta máquina há interceptação TLS (antivírus/proxy com CA própria): o
    `certifi` não conhece essa CA e todas as chamadas HTTPS falham. O
    `truststore` resolve usando os certificados do Windows. Idempotente.
    """
    global _ssl_ready
    if _ssl_ready:
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    _ssl_ready = True


def safe_filename(name: str, max_len: int = 150) -> str:
    """Nome de arquivo válido no Windows, preservando acentos."""
    name = unicodedata.normalize("NFC", name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
    name = re.sub(r"\s+", " ", name).strip().rstrip(". ")
    if not name:
        name = "sem_nome"
    if name.split(".")[0].upper() in _WINDOWS_RESERVED:
        name = f"_{name}"
    if len(name) > max_len:
        name = name[:max_len].rstrip(". ")
    return name


def atomic_write_json(path: str | os.PathLike, obj: Any) -> None:
    """Grava JSON em arquivo temporário e renomeia (não corrompe se o app cair)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    """Lê JSON; devolve `default` se o arquivo não existe ou está corrompido."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def parse_duration(text: str | None) -> float | None:
    """Converte "2:38" ou "1:02:03" em segundos."""
    if not text:
        return None
    try:
        parts = [int(p) for p in str(text).strip().split(":")]
    except ValueError:
        return None
    seconds = 0
    for p in parts:
        seconds = seconds * 60 + p
    return float(seconds)


def format_duration(seconds: float | None) -> str:
    """Converte segundos em "m:ss"."""
    if seconds is None:
        return "--:--"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def normalize_text(text: str) -> str:
    """Minúsculas, sem acentos, sem pontuação; para comparar títulos e artistas."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return text.strip()


def format_brl(value: float) -> str:
    """Formata 1234.5 como "R$ 1.234,50"."""
    s = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"
