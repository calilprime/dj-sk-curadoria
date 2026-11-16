"""Sessão salva em disco: protege a curadoria se o navegador recarregar ou o app cair.

CONTRATO (implementar mantendo as assinaturas):

Snapshot em `data/sessao_atual.json` (util.atomic_write_json) com:
{"references": [Track.to_dict()], "suggestions": [Track.to_dict()], "page": int,
 "decisions": {video_id: "aprovada" | "reprovada"}, "order": [video_id],
 "weights": {...}, "metrics": {...}, "saved_at": iso}.
"""
from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from modules.util import atomic_write_json

logger = logging.getLogger(__name__)

SESSION_FILENAME = "sessao_atual.json"
DECISION_VALUES = ("aprovada", "reprovada")
_lock = threading.Lock()


def session_path(data_dir: Path | None = None) -> Path:
    """`data_dir/sessao_atual.json` (data_dir None = config.DATA_DIR na hora da chamada)."""
    base = Path(data_dir) if data_dir is not None else Path(config.DATA_DIR)
    return base / SESSION_FILENAME


def save_session(snapshot: dict[str, Any], data_dir: Path | None = None) -> None:
    """Grava o snapshot (acrescenta "saved_at").

    Não altera o dicionário recebido. Erro de disco é registrado no log e não derruba o app
    (a sessão salva é só uma proteção extra).
    """
    data = _to_jsonable(dict(snapshot or {}))
    data["saved_at"] = datetime.now().isoformat(timespec="seconds")
    path = session_path(data_dir)
    try:
        with _lock:
            atomic_write_json(path, data)
    except OSError as exc:
        logger.warning("Não foi possível salvar a sessão em %s: %s", path, exc)


def load_session(data_dir: Path | None = None) -> dict[str, Any] | None:
    """Snapshot salvo ou None (arquivo ausente ou corrompido).

    As chaves do contrato sempre vêm presentes e com o tipo certo (itens inválidos são
    descartados). Um arquivo corrompido é renomeado para `.corrompido`, para não se perder.
    """
    path = session_path(data_dir)
    with _lock:
        if not path.exists():
            return None
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except OSError as exc:
            logger.warning("Não foi possível ler a sessão salva em %s: %s", path, exc)
            return None
        except ValueError:
            data = None
        if not isinstance(data, dict):
            logger.warning("Sessão salva corrompida em %s; começando do zero.", path)
            _quarantine(path)
            return None
    return _normalize(data)


def clear_session(data_dir: Path | None = None) -> None:
    """Apaga a sessão salva (sem erro se não existir)."""
    path = session_path(data_dir)
    try:
        with _lock:
            path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Não foi possível apagar a sessão salva em %s: %s", path, exc)


# ---------------------------------------------------------------------------
def _quarantine(path: Path) -> None:
    try:
        path.replace(path.with_name(path.name + ".corrompido"))
    except OSError:
        pass


def _to_jsonable(obj: Any) -> Any:
    """Converte o que o json não grava sozinho (dataclasses com to_dict, Path, set, numpy)."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return _to_jsonable(to_dict())
    tolist = getattr(obj, "tolist", None)   # arrays numpy (curva de energia, waveform)
    if callable(tolist) and getattr(obj, "ndim", 0):
        return _to_jsonable(tolist())
    item = getattr(obj, "item", None)   # escalares numpy (np.float64, np.int64...)
    if callable(item):
        try:
            return _to_jsonable(item())
        except (TypeError, ValueError):
            pass
    logger.warning("Valor de tipo %s gravado como texto na sessão salva.", type(obj).__name__)
    return str(obj)


def _track_dicts(value: Any) -> list[dict[str, Any]]:
    """Só dicionários que viram Track (precisam de video_id e title)."""
    if not isinstance(value, list):
        return []
    return [t for t in value if isinstance(t, dict) and t.get("video_id") and t.get("title") is not None]


def _normalize(data: dict[str, Any]) -> dict[str, Any]:
    """Garante as chaves do contrato com tipos válidos; chaves extras passam intactas."""
    out = dict(data)
    out["references"] = _track_dicts(data.get("references"))
    out["suggestions"] = _track_dicts(data.get("suggestions"))

    page = data.get("page")
    out["page"] = page if isinstance(page, int) and not isinstance(page, bool) and page >= 0 else 0

    decisions = data.get("decisions")
    out["decisions"] = ({str(k): v for k, v in decisions.items() if v in DECISION_VALUES}
                        if isinstance(decisions, dict) else {})

    order = data.get("order")
    out["order"] = [v for v in order if isinstance(v, str) and v] if isinstance(order, list) else []

    weights = data.get("weights")
    saved_weights = ({str(k): float(v) for k, v in weights.items()
                      if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}
                     if isinstance(weights, dict) else {})
    # Fora da faixa do slider, o st.slider recusaria o valor e derrubaria o app a cada recarga.
    saved_weights = {k: min(max(v, 0.0), float(config.WEIGHT_SLIDER_MAX)) for k, v in saved_weights.items()}
    # Pesos que faltarem voltam ao padrão (ex.: peso novo criado depois do snapshot).
    out["weights"] = {**config.DEFAULT_WEIGHTS, **saved_weights}

    metrics = data.get("metrics")
    out["metrics"] = metrics if isinstance(metrics, dict) else {}

    saved_at = data.get("saved_at")
    out["saved_at"] = saved_at if isinstance(saved_at, str) else None
    return out
