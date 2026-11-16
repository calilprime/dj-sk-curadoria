"""Medidor e teto de gasto (seção 7).

CONTRATO (implementar mantendo as assinaturas):

- Registro em JSON (padrão `data/orcamento.json`) de cada execução paga:
  {"timestamp", "video_id", "prediction_id", "predict_time_s", "usd", "brl", "status"}.
- Custo real = predict_time × REPLICATE_USD_PER_SECOND, convertido com USD_BRL.
- `can_run` trava quando: gasto_brl + estimativa >= teto; ou gasto_usd + estimativa > crédito
  da Replicate; ou a Replicate já informou saldo insuficiente (flag persistida).
- Thread-safe e tolerante a JSON corrompido (começa vazio e não perde o arquivo antigo:
  renomeia para .corrompido).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import config
from modules import util

log = logging.getLogger(__name__)

# Folga para comparações de ponto flutuante (gasto que chega "exatamente" no teto trava).
_EPS = 1e-9

# Um lock por arquivo: instâncias diferentes de Budget no mesmo processo (reruns do
# Streamlit, abas do navegador) gravam o mesmo JSON sem perder entradas.
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path.resolve()))
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = _PATH_LOCKS[key] = threading.RLock()
        return lock


def _num(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.0
    return x if x == x else 0.0  # descarta NaN


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _empty() -> dict[str, Any]:
    return {"entries": [], "insufficient_credit": False,
            "insufficient_credit_message": "", "insufficient_credit_at": None}


class Budget:
    def __init__(self, path: Path | None = None, cap_brl: float | None = None,
                 usd_per_second: float | None = None, usd_brl: float | None = None,
                 credit_usd: float | None = None):
        """None usa os valores de config (DATA_DIR/'orcamento.json', BUDGET_CAP_BRL, ...)."""
        self.path = Path(path) if path is not None else Path(config.DATA_DIR) / "orcamento.json"
        self.cap_brl = float(config.BUDGET_CAP_BRL if cap_brl is None else cap_brl)
        self.usd_per_second = float(config.REPLICATE_USD_PER_SECOND
                                    if usd_per_second is None else usd_per_second)
        self.usd_brl = float(config.USD_BRL if usd_brl is None else usd_brl)
        self.credit_usd = float(config.REPLICATE_CREDIT_USD if credit_usd is None else credit_usd)
        self._lock = _lock_for(self.path)
        self._stamp: tuple[int, int] | None = None
        self._data: dict[str, Any] = _empty()
        with self._lock:
            self._reload()

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------
    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _quarantine(self, reason: str) -> None:
        """Renomeia o arquivo ilegível para .corrompido (nunca apaga)."""
        target = self.path.with_name(self.path.name + ".corrompido")
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            target = self.path.with_name(f"{self.path.name}.{stamp}.corrompido")
        try:
            os.replace(self.path, target)
            log.warning("Orçamento ilegível (%s); arquivo antigo guardado em %s", reason, target.name)
        except OSError:
            log.exception("Orçamento ilegível (%s) e não foi possível renomeá-lo", reason)

    def _reload(self) -> None:
        stamp = self._file_stamp()
        if stamp is None:
            self._data, self._stamp = _empty(), None
            return
        try:
            blob = self.path.read_bytes()
        except FileNotFoundError:
            self._data, self._stamp = _empty(), None
            return
        except OSError as exc:
            # Arquivo momentaneamente travado (antivírus): mantém o que está em memória
            # e tenta de novo na próxima leitura.
            log.warning("Não foi possível ler o orçamento agora: %s", exc)
            return
        try:
            raw = json.loads(blob.decode("utf-8-sig"))
        except ValueError:  # inclui UnicodeDecodeError
            self._quarantine("JSON inválido")
            self._data, self._stamp = _empty(), None
            return

        data = _empty()
        if isinstance(raw, list):  # formato antigo: só a lista de entradas
            data["entries"] = raw
        elif isinstance(raw, dict) and isinstance(raw.get("entries", []), list):
            data.update({k: raw[k] for k in data if k in raw})
        else:
            self._quarantine("estrutura inesperada")
            self._data, self._stamp = _empty(), None
            return
        data["entries"] = [e for e in data["entries"] if isinstance(e, dict)]
        data["insufficient_credit"] = bool(data.get("insufficient_credit"))
        self._data, self._stamp = data, stamp

    def _refresh(self) -> None:
        """Relê o arquivo se outra instância o alterou."""
        if self._file_stamp() != self._stamp:
            self._reload()

    def _save(self) -> None:
        attempts = 5
        for attempt in range(attempts):
            try:
                util.atomic_write_json(self.path, self._data)
                self._stamp = self._file_stamp()
                return
            except PermissionError:  # Windows: antivírus segurando o arquivo por instantes
                if attempt == attempts - 1:
                    raise
                time.sleep(0.05 * (attempt + 1))

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------
    @property
    def entries(self) -> list[dict]:
        with self._lock:
            self._refresh()
            return [dict(e) for e in self._data["entries"]]

    @property
    def spent_usd(self) -> float:
        with self._lock:
            self._refresh()
            return sum(_num(e.get("usd")) for e in self._data["entries"])

    @property
    def spent_brl(self) -> float:
        with self._lock:
            self._refresh()
            return sum(_num(e.get("brl")) for e in self._data["entries"])

    @property
    def remaining_brl(self) -> float:
        return max(0.0, self.cap_brl - self.spent_brl)

    @property
    def remaining_credit_usd(self) -> float:
        return max(0.0, self.credit_usd - self.spent_usd)

    @property
    def fraction_used(self) -> float:
        """gasto_brl / teto, entre 0 e 1 (para a barra do topo do painel)."""
        if self.cap_brl <= 0:
            return 1.0
        return min(1.0, max(0.0, self.spent_brl / self.cap_brl))

    @property
    def insufficient_credit(self) -> bool:
        with self._lock:
            self._refresh()
            return bool(self._data.get("insufficient_credit"))

    @property
    def insufficient_credit_message(self) -> str:
        """Mensagem que a Replicate devolveu ao recusar por saldo (pode ser vazia)."""
        with self._lock:
            self._refresh()
            return str(self._data.get("insufficient_credit_message") or "")

    # ------------------------------------------------------------------
    # Custos e trava
    # ------------------------------------------------------------------
    def cost_from_predict_time(self, predict_time_s: float) -> tuple[float, float]:
        """(usd, brl)."""
        usd = max(0.0, _num(predict_time_s)) * self.usd_per_second
        return usd, usd * self.usd_brl

    def estimate_cost(self, estimate_usd: float | None = None) -> tuple[float, float]:
        """(usd, brl) estimados de uma execução. None = STEM_COST_ESTIMATE_USD."""
        usd = float(config.STEM_COST_ESTIMATE_USD if estimate_usd is None else estimate_usd)
        usd = max(0.0, usd)
        return usd, usd * self.usd_brl

    def can_run(self, estimate_usd: float | None = None) -> tuple[bool, str]:
        """(pode rodar?, motivo em português quando não pode). estimate None = STEM_COST_ESTIMATE_USD."""
        est_usd, est_brl = self.estimate_cost(estimate_usd)
        with self._lock:
            self._refresh()
            if self._data.get("insufficient_credit"):
                detail = str(self._data.get("insufficient_credit_message") or "").strip()
                msg = "A Replicate informou saldo insuficiente"
                msg += f" ({detail})." if detail else "."
                msg += (" Adicione crédito em replicate.com/account/billing e libere a trava"
                        " no painel antes de tentar de novo.")
                return False, msg
            spent_brl = sum(_num(e.get("brl")) for e in self._data["entries"])
            spent_usd = sum(_num(e.get("usd")) for e in self._data["entries"])
        if spent_brl + est_brl >= self.cap_brl - _EPS:
            return False, (f"Teto de gasto atingido: {util.format_brl(spent_brl)} usados de "
                           f"{util.format_brl(self.cap_brl)}. A próxima separação custaria cerca de "
                           f"{util.format_brl(est_brl)} e passaria do limite.")
        if spent_usd + est_usd > self.credit_usd + _EPS:
            return False, (f"O crédito da Replicate não cobre outra separação: US$ {spent_usd:.4f} "
                           f"usados de US$ {self.credit_usd:.2f} (a próxima custa cerca de "
                           f"US$ {est_usd:.4f}).")
        return True, ""

    def record(self, video_id: str, prediction_id: str | None, predict_time_s: float | None,
               status: str = "succeeded") -> dict:
        """Registra uma execução e devolve a entrada. predict_time None conta a estimativa."""
        if predict_time_s is None:
            usd, brl = self.estimate_cost()
            estimated = True
        else:
            usd, brl = self.cost_from_predict_time(predict_time_s)
            estimated = False
        entry = {
            "timestamp": _now_iso(),
            "video_id": video_id,
            "prediction_id": prediction_id,
            "predict_time_s": None if predict_time_s is None else _num(predict_time_s),
            "usd": usd,
            "brl": brl,
            "status": status,
            "estimated": estimated,
        }
        with self._lock:
            self._refresh()
            self._data["entries"].append(entry)
            self._save()
        log.info("Gasto registrado: %s (%s) US$ %.5f", video_id, status, usd)
        return dict(entry)

    def mark_insufficient_credit(self, message: str = "") -> None:
        """Persiste a trava de saldo insuficiente informada pela Replicate."""
        with self._lock:
            self._refresh()
            self._data["insufficient_credit"] = True
            self._data["insufficient_credit_message"] = str(message or "")[:500]
            self._data["insufficient_credit_at"] = _now_iso()
            self._save()
        log.warning("Replicate informou saldo insuficiente; Estúdio de Vocal travado")

    def reset_insufficient_credit(self) -> None:
        with self._lock:
            self._refresh()
            self._data["insufficient_credit"] = False
            self._data["insufficient_credit_message"] = ""
            self._data["insufficient_credit_at"] = None
            self._save()
