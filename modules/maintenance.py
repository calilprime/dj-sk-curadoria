"""Zerar painel: mostra o espaço usado e apaga, por categoria, o que o painel guardou.

Categorias (pastas lidas do config na hora da chamada):
- "cache":     CACHE_DIR inteiro (músicas baixadas, análises, prévias).
- "sessao":    DATA_DIR/sessao_atual.json (e cópias .corrompido da sessão).
- "metricas":  DATA_DIR/metricas.json (histórico de sessões e cópias .corrompido).
- "stems":     STEMS_DIR inteiro (vozes e instrumentais separados na Replicate).
- "setlists":  SETLISTS_DIR e a pasta antiga EXPORT_DIR (playlists exportadas).
- "orcamento": DATA_DIR/orcamento.json (medidor de gasto da Replicate).

Segurança: só apaga dentro dessas pastas/arquivos, e só se estiverem dentro de
config.BASE_DIR; nunca segue atalhos (links) para fora; nunca toca em .env, cookies.txt,
data/.spotify_token, .venv nem em código. As pastas apagadas são recriadas vazias.
Arquivo travado (aberto em outro programa) vira um item em `failures`, não um erro.
"""
from __future__ import annotations

import fnmatch
import logging
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config

logger = logging.getLogger(__name__)

CATEGORIES = ("cache", "sessao", "metricas", "stems", "setlists", "orcamento")
DEFAULT_RESET = frozenset({"cache", "sessao", "metricas"})

LABELS = {
    "cache": "Músicas baixadas e análises",
    "sessao": "Sessão atual (triagem em andamento)",
    "metricas": "Histórico de métricas",
    "stems": "Vozes e instrumentais separados",
    "setlists": "Setlists exportados",
    "orcamento": "Medidor de gasto",
}

DESCRIPTIONS = {
    "cache": "Apaga as músicas baixadas, as análises de velocidade, energia e tom e as prévias. "
             "Elas são baixadas e analisadas de novo quando você precisar.",
    "sessao": "Esquece as sugestões, as aprovações e a ordem do set da sessão atual. "
              "O painel volta para a tela inicial.",
    "metricas": "Apaga o histórico de sessões e de tempo economizado. As próximas sessões "
                "começam a contar do zero.",
    "stems": "Apaga as vozes e os instrumentais separados no Estúdio de Vocal. "
             "Separar de novo gasta crédito na Replicate.",
    "setlists": "Apaga as pastas de setlist exportadas (músicas numeradas e playlist .m3u8). "
                "Se já importou no Rekordbox, as faixas vão sumir de lá.",
    "orcamento": "Zera o medidor de gasto. Atenção: o dinheiro gasto na Replicate foi real.",
}

# Nunca apagados, onde quer que estejam.
PROTECTED_NAMES = frozenset({".env", ".env.example", "cookies.txt", ".spotify_token", ".cache",
                             ".gitignore", "requirements.txt"})
PROTECTED_SUFFIXES = frozenset({".py", ".pyw", ".bat", ".cmd", ".ps1"})
_PROTECTED_LOWER = frozenset(n.lower() for n in PROTECTED_NAMES)
PROTECTED_DIRS = (".venv", "modules", "tests", "scripts", ".streamlit", ".git")

# Arquivos de DATA_DIR de cada categoria de arquivos soltos (inclui cópias e temporários).
_DATA_FILES = {
    "sessao": "sessao_atual.json",
    "metricas": "metricas.json",
    "orcamento": "orcamento.json",
}


@dataclass
class ResetReport:
    removed_files: int = 0
    freed_bytes: int = 0
    per_category: dict[str, dict[str, int]] = field(default_factory=dict)  # {"files", "bytes"}
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        """Frase curta para o painel, ex.: "12 arquivo(s) apagado(s), 45,3 MB liberados."."""
        text = f"{self.removed_files} arquivo(s) apagado(s), {format_bytes(self.freed_bytes)} liberados."
        if self.failures:
            text += f" {len(self.failures)} item(ns) não puderam ser apagados."
        return text


def format_bytes(n: int | float) -> str:
    """Tamanho legível em português: 0 B, 512 B, 1,5 KB, 45,3 MB, 1,2 GB."""
    value = float(max(0, n or 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}".replace(".", ",")
        value /= 1024
    return f"{value:.1f} GB".replace(".", ",")  # inalcançável; mantém o tipo de retorno


# ---------------------------------------------------------------------------
# Alvos de cada categoria
# ---------------------------------------------------------------------------
@dataclass
class _Target:
    path: Path                    # pasta (apaga o conteúdo) ou pasta onde ficam os arquivos soltos
    patterns: tuple[str, ...] = ()  # vazio = pasta inteira; senão, nomes (fnmatch) dentro de `path`

    @property
    def is_tree(self) -> bool:
        return not self.patterns


def _file_patterns(name: str) -> tuple[str, ...]:
    # arquivo, cópias de arquivo corrompido e temporários da gravação atômica (util.atomic_write_json)
    return (name, f"{name}*.corrompido", f".{name}.*.tmp")


def _targets(category: str) -> list[_Target]:
    if category == "cache":
        return [_Target(Path(config.CACHE_DIR))]
    if category == "stems":
        return [_Target(Path(config.STEMS_DIR))]
    if category == "setlists":
        # Uma pasta dentro da outra (ou a mesma) é varrida uma vez só, pela de fora.
        kept: list[Path] = []
        for d in (Path(config.SETLISTS_DIR), Path(config.EXPORT_DIR)):
            if any(_inside(_resolve(d), _resolve(k)) for k in kept):
                continue
            kept = [k for k in kept if not _inside(_resolve(k), _resolve(d))] + [d]
        return [_Target(d) for d in kept]
    if category in _DATA_FILES:
        return [_Target(Path(config.DATA_DIR), _file_patterns(_DATA_FILES[category]))]
    raise ValueError(f"Categoria desconhecida: {category!r}. Use uma de: {', '.join(CATEGORIES)}.")


def _check_categories(categories: Iterable[str]) -> list[str]:
    if isinstance(categories, str):
        categories = [categories]
    cats = list(dict.fromkeys(categories or []))
    unknown = [c for c in cats if c not in CATEGORIES]
    if unknown:
        raise ValueError(f"Categoria desconhecida: {', '.join(map(repr, unknown))}. "
                         f"Use uma de: {', '.join(CATEGORIES)}.")
    return [c for c in CATEGORIES if c in cats]   # ordem fixa


# ---------------------------------------------------------------------------
# Regras de segurança
# ---------------------------------------------------------------------------
def _resolve(path: Path) -> Path:
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(path))


def _inside(path: Path, base: Path) -> bool:
    try:
        return path.is_relative_to(base)
    except ValueError:
        return False


def _unsafe_reason(target: _Target) -> str | None:
    """Motivo para recusar o alvo, ou None se é seguro apagar dentro dele."""
    base = _resolve(Path(config.BASE_DIR))
    path = _resolve(target.path)
    if not _inside(path, base):
        return f"pasta fora do projeto ({target.path})"
    if target.is_tree:
        if path == base or _inside(base, path):
            return f"a pasta {target.path} é a própria pasta do projeto"
        for name in PROTECTED_DIRS:
            protected = base / name
            if _inside(path, protected) or _inside(protected, path):
                return f"a pasta {target.path} contém arquivos do programa"
    return None


def _is_protected_file(path: Path) -> bool:
    return path.name.lower() in _PROTECTED_LOWER or path.suffix.lower() in PROTECTED_SUFFIXES


def _is_link(path: Path) -> bool:
    try:
        return path.is_symlink() or path.is_junction()
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Varredura (usada por usage e reset)
# ---------------------------------------------------------------------------
def _lstat_size(path: Path) -> int:
    try:
        return int(path.lstat().st_size)
    except OSError:
        return 0


def _walk_files(target: _Target) -> Iterable[Path]:
    """Arquivos (e links) que a categoria apagaria, sem seguir links nem entrar em pastas protegidas."""
    root = Path(target.path)
    if not root.is_dir() or _is_link(root):
        return
    if not target.is_tree:
        try:
            entries = sorted(root.iterdir())
        except OSError:
            return
        for p in entries:
            if (any(fnmatch.fnmatchcase(p.name, pat) for pat in target.patterns)
                    and not _is_protected_file(p) and (_is_link(p) or p.is_file())):
                yield p
        return
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            entries = sorted(folder.iterdir())
        except OSError:
            continue
        for p in entries:
            if _is_link(p):
                yield p                 # o link é apagado; o destino, nunca
            elif p.is_dir():
                stack.append(p)
            elif not _is_protected_file(p):
                yield p


def _subdirs_deepest_first(root: Path) -> list[Path]:
    out: list[Path] = []
    stack = [root]
    while stack:
        folder = stack.pop()
        try:
            for p in folder.iterdir():
                if not _is_link(p) and p.is_dir():
                    out.append(p)
                    stack.append(p)
        except OSError:
            continue
    return sorted(out, key=lambda p: len(p.parts), reverse=True)


def usage() -> dict[str, dict[str, Any]]:
    """Espaço usado por categoria: {"files": int, "bytes": int, "path": str, "paths": [str],
    "safe": bool}. "path" é o local principal; "safe" False = a categoria seria recusada."""
    out: dict[str, dict[str, Any]] = {}
    for cat in CATEGORIES:
        targets = _targets(cat)
        files = size = 0
        safe = True
        for t in targets:
            safe = safe and _unsafe_reason(t) is None
            for p in _walk_files(t):
                files += 1
                size += _lstat_size(p)
        main = targets[0]
        path = main.path / _DATA_FILES[cat] if cat in _DATA_FILES else main.path
        out[cat] = {"files": files, "bytes": size, "path": str(path),
                    "paths": [str(t.path) for t in targets], "safe": safe}
    return out


# ---------------------------------------------------------------------------
# Apagar
# ---------------------------------------------------------------------------
def _remove_file(path: Path) -> None:
    """Apaga um arquivo ou link (sem seguir). Tira o "somente leitura" do Windows se preciso."""
    try:
        _unlink(path)
    except PermissionError:
        if _is_link(path) or not path.exists():
            raise
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
        _unlink(path)


def _unlink(path: Path) -> None:
    try:
        os.unlink(path)
    except (IsADirectoryError, PermissionError):
        if path.is_junction() or (path.is_symlink() and path.is_dir()):
            os.rmdir(path)        # link para pasta (Windows): remove só o link
        else:
            raise


def _lock_for(category: str):
    """Trava usada pelo módulo dono do arquivo, para não apagar no meio de uma gravação."""
    try:
        if category == "metricas":
            from modules import metrics
            return metrics._lock
        if category == "sessao":
            from modules import state
            return state._lock
        if category == "orcamento":
            from modules import budget
            return budget._lock_for(Path(config.DATA_DIR) / _DATA_FILES[category])
    except (ImportError, AttributeError):
        logger.debug("Sem trava para a categoria %s", category, exc_info=True)
    return None


def _reset_target(target: _Target, label: str, report: ResetReport, totals: dict[str, int]) -> None:
    reason = _unsafe_reason(target)
    if reason is not None:
        msg = f"{label}: não apagado, {reason}."
        logger.warning("Zerar painel recusado: %s", msg)
        report.failures.append(msg)
        return
    for p in list(_walk_files(target)):
        size = _lstat_size(p)
        try:
            _remove_file(p)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("Não foi possível apagar %s: %s", p, exc)
            report.failures.append(f"{label}: {p.name} está aberto em outro programa ou "
                                   f"protegido ({exc.strerror or exc}).")
            continue
        totals["files"] += 1
        totals["bytes"] += size
    if target.is_tree:
        for d in _subdirs_deepest_first(Path(target.path)):
            try:
                d.rmdir()
            except OSError:
                pass            # sobrou algo travado (já listado em failures) ou protegido
    try:
        Path(target.path).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report.failures.append(f"{label}: não foi possível recriar a pasta {target.path} ({exc}).")


def reset(categories: Iterable[str]) -> ResetReport:
    """Apaga as categorias pedidas e recria as pastas vazias.

    ValueError para categoria desconhecida (nada é apagado nesse caso).
    """
    cats = _check_categories(categories)
    report = ResetReport()
    for cat in cats:
        totals = {"files": 0, "bytes": 0}
        lock = _lock_for(cat)
        if lock is not None:
            with lock:
                for t in _targets(cat):
                    _reset_target(t, LABELS[cat], report, totals)
        else:
            for t in _targets(cat):
                _reset_target(t, LABELS[cat], report, totals)
        report.per_category[cat] = totals
        report.removed_files += totals["files"]
        report.freed_bytes += totals["bytes"]
    logger.info("Zerar painel (%s): %s", ", ".join(cats) or "nada", report.summary())
    return report
