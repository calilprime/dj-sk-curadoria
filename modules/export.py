"""Exportação do set (seção 4.9).

CONTRATO (implementar mantendo as assinaturas):

- Pasta `Setlist_Atual/` com CÓPIAS numeradas `NN - Artista - Título.mp3`
  (NN com 2 dígitos; 3 dígitos se o set passar de 99). Usar util.safe_filename.
  Arquivos locais que não são mp3 mantêm a extensão original.
- Antes de exportar, limpa da pasta só o que uma exportação anterior criou (arquivos de
  áudio numerados "NN - ...", o .m3u8 e o .csv). Nunca apaga outras coisas.
- `Setlist_Atual.m3u8` em UTF-8 (sem BOM): "#EXTM3U", depois por faixa
  "#EXTINF:<segundos inteiros>,<Artista> - <Título>" e o caminho (absoluto se
  M3U_ABSOLUTE_PATHS, senão relativo à pasta). Quebras de linha "\r\n" (Windows).
- `Setlist_Atual.csv` (CSV_ENCODING, CSV_DELIMITER) com colunas:
  ordem, arquivo, artista, titulo, video_id, duracao, bpm, bpm_manual, energia_media,
  energia_entrada, energia_saida, tom, camelot, transicao_para_proxima (rótulo),
  custo_transicao, janela_voz, explicacao. Números com ponto decimal? NÃO: use vírgula
  decimal (Excel pt-BR), com 1 casa para BPM e 2 para energia/custo.
- Se WRITE_ID3_TAGS e mutagen disponível: grava TIT2 (título), TPE1 (artista),
  TBPM (inteiro), TKEY (tom curto, ex. "Am") e TRCK (posição) nas CÓPIAS mp3. Falha de tag não aborta.
"""
from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path

import config
from modules.models import Analysis, Track, Transition
from modules.util import format_duration, safe_filename

logger = logging.getLogger(__name__)


@dataclass
class ExportResult:
    folder: Path
    m3u8_path: Path
    csv_path: Path
    files: list[Path] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)   # display_name das faixas sem áudio
    portable_m3u8_path: Path | None = None   # caminhos relativos: funciona após copiar a pasta (pen drive)


# Playlist com caminhos relativos, gravada ao lado da absoluta: a pasta pode ir para outro
# computador (pen drive) e o Rekordbox de lá ainda acha as músicas.
PORTABLE_SUFFIX = " (pen drive)"


CSV_COLUMNS = [
    "ordem", "arquivo", "artista", "titulo", "video_id", "duracao", "bpm", "bpm_manual",
    "energia_media", "energia_entrada", "energia_saida", "tom", "camelot",
    "transicao_para_proxima", "custo_transicao", "janela_voz", "explicacao",
]

_KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b♯♭]?)\s*(.*)$")
_MAX_FILENAME = 150
_MIN_FILENAME = 40
_WINDOWS_MAX_PATH = 259


def build_filename(position: int, track: Track, total: int, ext: str = ".mp3",
                   max_len: int = _MAX_FILENAME) -> str:
    """"01 - Artista - Título.mp3".

    `max_len` limita o nome sem a extensão (caminhos longos no Windows).
    """
    width = 3 if total > 99 else 2
    ext = (ext or ".mp3").lower()
    if not ext.startswith("."):
        ext = "." + ext
    stem = safe_filename(f"{position:0{width}d} - {track.artist_str} - {track.title}", max_len=max_len)
    return stem + ext


def short_key(key: str | None) -> str | None:
    """"A minor" -> "Am", "C# major" -> "C#" (formato do TKEY). None se não reconhecer."""
    if not key:
        return None
    m = _KEY_RE.match(key)
    if not m:
        return None
    root = m.group(1).upper() + m.group(2).replace("♯", "#").replace("♭", "b")
    mode = m.group(3).strip().lower()
    if mode in ("", "major", "maj", "maior"):
        return root
    if mode in ("m", "minor", "min", "menor"):
        return root + "m"
    return None


def _fmt(value: float | None, decimals: int) -> str:
    """Número com vírgula decimal (Excel pt-BR); vazio se ausente."""
    if value is None:
        return ""
    return f"{float(value):.{decimals}f}".replace(".", ",")


def _resolve_audio(track: Track, cache_dir: Path) -> Path | None:
    """`local_path` se existir; senão `cache_dir/{video_id}.mp3`. None se não há áudio."""
    candidates = []
    if track.local_path:
        candidates.append(Path(track.local_path))
    candidates.append(cache_dir / f"{track.video_id}.mp3")
    for p in candidates:
        try:
            if p.is_file() and p.stat().st_size > 0:
                return p
        except OSError:
            continue
    return None


def _exported_audio_re() -> re.Pattern:
    """Áudio numerado "NN - ..." com as extensões do config (lidas na hora da chamada)."""
    exts = {e.lower().lstrip(".") for e in (*config.LOCAL_AUDIO_EXTENSIONS, ".mp3")}
    return re.compile(r"^\d{2,3} - .+\.(" + "|".join(sorted(map(re.escape, exts))) + r")$",
                      re.IGNORECASE)


def _is_previous_export(path: Path, out_dir: Path, playlist_name: str) -> bool:
    """True se `path` é um arquivo que a limpeza de exportação anterior apagaria."""
    try:
        if path.parent.resolve() != out_dir.resolve() or not path.is_file():
            return False
    except OSError:
        return False
    return (path.name in (f"{playlist_name}.m3u8", f"{playlist_name}{PORTABLE_SUFFIX}.m3u8",
                          f"{playlist_name}.csv")
            or bool(_exported_audio_re().match(path.name)))


def _duration_s(track: Track, analysis: Analysis | None, path: Path) -> float | None:
    if analysis is not None and analysis.duration_s:
        return float(analysis.duration_s)
    if track.duration_s:
        return float(track.duration_s)
    try:
        import mutagen

        info = mutagen.File(path)
        if info is not None and info.info is not None and info.info.length:
            return float(info.info.length)
    except Exception:  # noqa: BLE001 - duração é só informativa
        pass
    return None


def _write_tags(path: Path, track: Track, analysis: Analysis | None, position: int, total: int) -> bool:
    """Grava as tags ID3 na cópia. Nunca levanta exceção (devolve False em caso de falha)."""
    try:
        from mutagen.id3 import ID3, TBPM, TIT2, TKEY, TPE1, TRCK, ID3NoHeaderError
    except ImportError:
        return False
    try:
        try:
            tags = ID3(path)
        except ID3NoHeaderError:
            tags = ID3()
        tags.update_to_v23()  # converte quadros v2.4 da origem (ex.: TDRC -> TYER) antes de salvar como v2.3
        tags.setall("TIT2", [TIT2(encoding=3, text=track.title)])
        tags.setall("TPE1", [TPE1(encoding=3, text=track.artist_str)])
        tags.setall("TRCK", [TRCK(encoding=3, text=f"{position}/{total}")])
        if analysis is not None and analysis.effective_bpm:
            tags.setall("TBPM", [TBPM(encoding=3, text=str(int(round(analysis.effective_bpm))))])
        key = short_key(analysis.key) if analysis is not None else None
        if key:
            tags.setall("TKEY", [TKEY(encoding=3, text=key)])
        tags.save(path, v2_version=3)  # ID3v2.3: o mais compatível (Rekordbox e Windows)
        return True
    except Exception as exc:  # noqa: BLE001 - falha de tag não aborta a exportação
        logger.warning("Não foi possível gravar as tags em %s: %s", path.name, exc)
        return False


def _transition_for(idx: int, next_idx: int | None, tracks: list[Track],
                    transitions: list[Transition], by_pair: dict[tuple[str, str], Transition]) -> Transition | None:
    """Transição da faixa exportada `idx` para a próxima exportada `next_idx` (índices de `tracks`)."""
    if next_idx is None:
        return None
    if next_idx == idx + 1 and idx < len(transitions):
        t = transitions[idx]
        if t is not None and (t.from_id, t.to_id) == (tracks[idx].video_id, tracks[next_idx].video_id):
            return t
    # A vizinha original foi pulada (ou a lista veio fora de ordem): procura pelo par de ids.
    return by_pair.get((tracks[idx].video_id, tracks[next_idx].video_id))


def _clean_line(text: str) -> str:
    return re.sub(r"[\r\n]+", " ", text).strip()


def _text_cell(text: str) -> str:
    """Texto vindo do YouTube não pode virar fórmula no Excel ("=", "+", "-", "@")."""
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _csv_row(position: int, name: str, track: Track, analysis: Analysis | None,
             duration: float | None, tr: Transition | None) -> list[str]:
    """Linha do CSV. "bpm" é o detectado; "bpm_manual" é a correção do DJ (vazio se não houver)."""
    a = analysis
    return [
        str(position), _text_cell(name), _text_cell(track.artist_str), _text_cell(track.title),
        _text_cell(track.video_id),
        format_duration(duration) if duration else "",
        _fmt(a.bpm if a else None, 1),
        _fmt(a.bpm_manual if a and a.bpm_manual else None, 1),
        _fmt(a.energy_mean if a else None, 2),
        _fmt(a.energy_in if a else None, 2),
        _fmt(a.energy_out if a else None, 2),
        (a.key or "") if a else "",
        (a.camelot or "") if a else "",
        tr.label if tr else "",
        _fmt(tr.cost, 2) if tr else "",
        tr.voice if tr else "",
        _text_cell(_clean_line(tr.explanation or "")) if tr else "",
    ]


def _unique_name(name: str, used: set[str]) -> str:
    """O número já diferencia faixas repetidas; isto só blinda contra sobrescrita (Windows ignora caixa)."""
    suffix = Path(name).suffix
    stem = name[: len(name) - len(suffix)]
    candidate, dup = name, 2
    while candidate.lower() in used:
        candidate = f"{stem} ({dup}){suffix}"
        dup += 1
    used.add(candidate.lower())
    return candidate


_export_lock = threading.Lock()   # duas abas exportando ao mesmo tempo não misturam as pastas


def _friendly_swap_error(path: Path, exc: OSError) -> OSError:
    if isinstance(exc, PermissionError):
        return OSError(f'Não consegui substituir "{path.name}": o arquivo está aberto em outro programa '
                       "(Excel, Rekordbox ou player). Feche-o e exporte de novo; a exportação anterior "
                       "foi mantida.")
    return OSError(f'Não consegui substituir "{path.name}" ({exc}). A exportação anterior foi mantida.')


def _move_all(pairs: list[tuple[Path, Path]]) -> list[tuple[Path, Path]]:
    """Move cada (origem, destino); em erro desfaz o que já moveu e levanta a exceção original."""
    done: list[tuple[Path, Path]] = []
    try:
        for src, dst in pairs:
            os.replace(src, dst)
            done.append((src, dst))
    except OSError:
        for src, dst in reversed(done):
            try:
                os.replace(dst, src)
            except OSError:
                logger.error("Não consegui desfazer a troca de %s", dst, exc_info=True)
        raise
    return done


def export_setlist(tracks: list[Track], analyses: dict[str, Analysis],
                   transitions: list[Transition] | None = None,
                   out_dir: Path | None = None, playlist_name: str | None = None,
                   cache_dir: Path | None = None) -> ExportResult:
    """`tracks` já na ordem do set. `transitions[i]` é a transição de tracks[i] para tracks[i+1].

    Faixas sem arquivo de áudio são puladas e listadas em `skipped` (a numeração segue contínua).
    A exportação é montada numa pasta de preparo e só substitui a anterior no fim: se algo
    falhar (disco cheio, CSV aberto no Excel), a exportação anterior fica intacta.
    Faixas locais que moravam na exportação anterior passam a apontar para as cópias novas.
    """
    with _export_lock:
        return _export_locked(tracks, analyses, transitions, out_dir, playlist_name, cache_dir)


def _export_locked(tracks, analyses, transitions, out_dir, playlist_name, cache_dir) -> ExportResult:
    out_dir = Path(out_dir) if out_dir is not None else Path(config.EXPORT_DIR)
    playlist_name = safe_filename(playlist_name or config.PLAYLIST_NAME)
    cache_dir = Path(cache_dir) if cache_dir is not None else Path(config.CACHE_DIR)
    transitions = list(transitions or [])
    analyses = analyses or {}
    write_tags = bool(config.WRITE_ID3_TAGS)
    absolute = bool(config.M3U_ABSOLUTE_PATHS)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = ExportResult(folder=out_dir, m3u8_path=out_dir / f"{playlist_name}.m3u8",
                          csv_path=out_dir / f"{playlist_name}.csv")
    if absolute:
        result.portable_m3u8_path = out_dir / f"{playlist_name}{PORTABLE_SUFFIX}.m3u8"

    sources: list[tuple[int, Path]] = []   # (índice em tracks, arquivo de origem)
    for i, track in enumerate(tracks):
        src = _resolve_audio(track, cache_dir)
        if src is None:
            result.skipped.append(track.display_name)
        else:
            sources.append((i, src))
    total = len(sources)
    by_pair = {(t.from_id, t.to_id): t for t in transitions if t is not None}
    # Nome sem extensão limitado para o caminho completo caber no MAX_PATH do Windows.
    max_len = max(_MIN_FILENAME, min(_MAX_FILENAME, _WINDOWS_MAX_PATH - len(str(out_dir.resolve())) - 6))

    staging = Path(tempfile.mkdtemp(prefix=".djsk_export_", dir=out_dir))
    backup: Path | None = None
    try:
        m3u_lines = ["#EXTM3U"]
        portable_lines = ["#EXTM3U"]
        rows: list[list[str]] = []
        used_names: set[str] = set()
        new_local: dict[int, Path] = {}        # faixas locais que estavam na exportação anterior
        for n, (i, src) in enumerate(sources):
            position = n + 1
            track = tracks[i]
            analysis = analyses.get(track.video_id)
            name = _unique_name(build_filename(position, track, total, ext=src.suffix or ".mp3",
                                               max_len=max_len), used_names)
            staged = staging / name
            shutil.copy2(src, staged)
            if write_tags and staged.suffix.lower() == ".mp3":
                _write_tags(staged, track, analysis, position, total)
            dst = out_dir / name
            result.files.append(dst)
            if track.local_path and _is_previous_export(src, out_dir, playlist_name):
                new_local[i] = dst

            duration = _duration_s(track, analysis, staged)
            extinf = f"#EXTINF:{int(round(duration)) if duration else -1},{_clean_line(track.display_name)}"
            m3u_lines += [extinf, str(dst.resolve()) if absolute else name]
            portable_lines += [extinf, name]

            next_idx = sources[n + 1][0] if n + 1 < len(sources) else None
            tr = _transition_for(i, next_idx, tracks, transitions, by_pair)
            rows.append(_csv_row(position, name, track, analysis, duration, tr))

        # UTF-8 sem BOM e CRLF, como o Rekordbox no Windows espera.
        (staging / result.m3u8_path.name).write_bytes(("\r\n".join(m3u_lines) + "\r\n").encode("utf-8"))
        if result.portable_m3u8_path is not None:
            (staging / result.portable_m3u8_path.name).write_bytes(
                ("\r\n".join(portable_lines) + "\r\n").encode("utf-8"))
        with open(staging / result.csv_path.name, "w", encoding=config.CSV_ENCODING, newline="") as f:
            writer = csv.writer(f, delimiter=config.CSV_DELIMITER, lineterminator="\r\n")
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)

        # Troca: a exportação anterior vai para um backup; a nova entra no lugar.
        backup = Path(tempfile.mkdtemp(prefix=".djsk_backup_", dir=out_dir))
        previous = [p for p in out_dir.iterdir() if _is_previous_export(p, out_dir, playlist_name)]
        try:
            moved_out = _move_all([(p, backup / p.name) for p in previous])
        except OSError as exc:
            raise _friendly_swap_error(Path(exc.filename or previous[0]), exc) from exc
        try:
            _move_all([(p, out_dir / p.name) for p in staging.iterdir()])
        except OSError as exc:
            _move_all([(dst, src) for src, dst in moved_out])
            raise _friendly_swap_error(Path(exc.filename or result.m3u8_path), exc) from exc
    except BaseException:
        if backup is not None:
            try:
                backup.rmdir()          # só some se a restauração devolveu tudo
            except OSError:
                logger.error("Arquivos da exportação anterior guardados em %s", backup)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)

    for i, dst in new_local.items():
        tracks[i].local_path = str(dst)
    if result.skipped:
        logger.warning("%d faixa(s) sem áudio ficaram de fora da exportação.", len(result.skipped))
    logger.info("Set exportado em %s: %d faixa(s).", out_dir, len(result.files))
    return result
