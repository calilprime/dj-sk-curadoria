"""Validações iniciais da especificação (seção 9), pela linha de comando.

Uso, na pasta do projeto:

    .venv\\Scripts\\python.exe scripts\\validar.py ytmusic ["Artista - Faixa" ...]
    .venv\\Scripts\\python.exe scripts\\validar.py ytdlp [--saida PASTA] [--lotes 5 20]
    .venv\\Scripts\\python.exe scripts\\validar.py librosa [PASTA] [--csv ARQUIVO]
    .venv\\Scripts\\python.exe scripts\\validar.py m3u8 PASTA [--saida PASTA]
    .venv\\Scripts\\python.exe scripts\\validar.py replicate
    .venv\\Scripts\\python.exe scripts\\validar.py tudo

Nenhum subcomando gasta dinheiro: o "replicate" só lê o esquema do modelo (não cria previsão).
Os módulos do projeto são importados dentro de cada subcomando, para que um módulo
quebrado ou incompleto não impeça as outras validações.
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from modules import util  # noqa: E402

logger = logging.getLogger("validar")

# Faixa de mandelão usada na validação de 22/10/2026 (videoId bN8S3cRF9X0).
DEFAULT_REFERENCES = ["Mc Paiva ZS - Casei Com a Putaria"]
DEFAULT_BATCHES = [5, 20]
# Palavras que indicam funk no título/artista (mc, dj, funk, mandelão, montagem/mtg, automotivo, beat).
FUNK_RE = re.compile(r"\b(mc|dj|funk\w*|mandel\w*|montage\w*|mtg|automotiv\w*|beat\w*)\b")
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
REPLICATE_API_URL = "https://api.replicate.com/v1"
HTTP_TIMEOUT_S = 20
REKORDBOX_SAMPLE = 5            # faixas analisadas/exportadas no "tudo" (validação 3 e 4)

OK = "OK"
ATTENTION = "ATENÇÃO"
FAILED = "FALHOU"
PENDING = "PENDENTE"
SKIPPED = "PULADO"

VALIDATIONS = {
    1: "ytmusicapi: rádio sem login",
    2: "yt-dlp: 5 e 20 faixas em paralelo",
    3: "librosa x Rekordbox",
    4: "Rekordbox: importar .m3u8",
    5: "Replicate: parâmetros do Demucs",
    6: "Conversa com o DJ SK",
}


@dataclass
class Check:
    number: int
    status: str
    detail: str

    @property
    def name(self) -> str:
        return VALIDATIONS[self.number]


@dataclass
class RadioRun:
    references: list[Any]               # list[Track]
    radios: dict[str, list[dict]]
    suggestions: list[Any]              # list[Track], já ranqueadas e filtradas
    radio_seconds: dict[str, float]
    first_is_reference: dict[str, bool]


@dataclass
class DownloadOutcome:
    track: Any                          # Track
    seconds: float
    path: Path | None
    error: str | None


@dataclass
class AudioItem:
    label: str
    path: Path
    video_id: str


# ---------------------------------------------------------------------------
# Saída no terminal
# ---------------------------------------------------------------------------
def say(text: str = "") -> None:
    print(text, flush=True)


def header(number: int) -> None:
    say()
    say(f"== {number}. {VALIDATIONS[number]} ==")


def num(value: float | None, decimals: int = 1) -> str:
    """Número com vírgula decimal (padrão brasileiro)."""
    if value is None:
        return "-"
    return f"{value:.{decimals}f}".replace(".", ",")


def clip(text: str, width: int) -> str:
    text = text or ""
    return text if len(text) <= width else text[: width - 1] + "…"


def label_of(track: Any) -> str:
    """"Artista - Título"; só o título quando não há artista (ex.: --ids)."""
    return track.display_name if track.artists else track.title


def friendly_error(exc: BaseException, module: str) -> str:
    if isinstance(exc, NotImplementedError):
        return f"o módulo {module} ainda não foi implementado"
    msg = str(exc).strip()
    return f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__


# ---------------------------------------------------------------------------
# 1. ytmusicapi
# ---------------------------------------------------------------------------
def looks_like_funk(track: Any) -> bool:
    text = util.normalize_text(f"{track.title} {' '.join(track.artists)} {track.album or ''}")
    return bool(FUNK_RE.search(text))


def collect_suggestions(queries: list[str], yt: Any = None, verbose: bool = True) -> RadioRun:
    """Resolve as referências, busca o rádio cru de cada uma e ranqueia com o módulo suggest."""
    from modules import suggest

    util.setup_ssl()
    if len(queries) > config.MAX_REFERENCES:
        say(f"  Aviso: o painel aceita até {config.MAX_REFERENCES} referências; aqui vou usar todas.")
    yt = yt or suggest.get_ytmusic()
    run = RadioRun([], {}, [], {}, {})
    for query in queries:
        try:
            ref = suggest.resolve_reference(query, yt)
        except suggest.SuggestError as exc:
            say(f"  Referência \"{query}\": {exc}")
            continue
        if ref is None:
            say(f"  Referência \"{query}\": nada encontrado no YouTube Music.")
            continue
        run.references.append(ref)
        started = time.perf_counter()
        try:
            data = yt.get_watch_playlist(videoId=ref.video_id, radio=True)
        except Exception as exc:  # noqa: BLE001 - qualquer falha do ytmusicapi vira mensagem
            say(f"  Rádio de \"{ref.display_name}\" falhou: {friendly_error(exc, 'suggest')}")
            continue
        elapsed = time.perf_counter() - started
        items = list((data or {}).get("tracks") or [])
        run.radios[ref.video_id] = items
        run.radio_seconds[ref.video_id] = elapsed
        run.first_is_reference[ref.video_id] = bool(items) and items[0].get("videoId") == ref.video_id
        if verbose:
            say(f"  Referência \"{query}\" -> {ref.display_name} [{ref.video_id}]")
            say(f"    rádio sem login: {len(items)} faixas em {num(elapsed)} s; 1ª faixa é a própria "
                f"referência: {'sim' if run.first_is_reference[ref.video_id] else 'não'}")
    if run.radios:
        run.suggestions = suggest.rank_suggestions(run.radios, run.references)
    return run


def validate_ytmusic(queries: list[str], n_show: int, yt: Any = None) -> tuple[Check, RadioRun | None]:
    header(1)
    try:
        run = collect_suggestions(queries, yt=yt)
    except Exception as exc:  # noqa: BLE001
        return Check(1, FAILED, friendly_error(exc, "suggest")), None
    if not run.references:
        return Check(1, FAILED, "nenhuma referência foi encontrada no YouTube Music"), run
    if not run.radios:
        return Check(1, FAILED, "o rádio não devolveu nada (sem login)"), run

    shown = run.suggestions[:n_show]
    funk = [t for t in shown if looks_like_funk(t)]
    say()
    say(f"  {len(shown)} primeiras sugestões (* = tem cara de funk pelo título/artista):")
    for i, track in enumerate(shown, start=1):
        votes = f"  [{track.score} ref.]" if len(run.references) > 1 else ""
        mark = "*" if looks_like_funk(track) else " "
        say(f"  {i:>3}. {mark} {clip(track.display_name, 70)} ({util.format_duration(track.duration_s)}){votes}")
    pct = 100.0 * len(funk) / len(shown) if shown else 0.0
    raw = sum(len(v) for v in run.radios.values())
    mean_s = sum(run.radio_seconds.values()) / len(run.radio_seconds)
    say()
    say(f"  Com cara de funk: {len(funk)}/{len(shown)} ({num(pct, 0)}%). "
        f"Rádio médio: {num(raw / len(run.radios), 0)} faixas em {num(mean_s)} s; "
        f"{len(run.suggestions)} sugestões após filtros e duplicatas.")
    detail = (f"{len(run.radios)} rádio(s) sem login, {raw} faixas cruas, {len(run.suggestions)} sugestões; "
              f"{num(pct, 0)}% com cara de funk nas {len(shown)} primeiras")
    enough = len(shown) >= min(n_show, config.N_SUGGESTIONS)
    status = OK if enough and pct >= 50.0 else ATTENTION
    if status != OK:
        detail += " (poucas sugestões ou pouco funk: ouça e confira)"
    return Check(1, status, detail), run


# ---------------------------------------------------------------------------
# 2. yt-dlp
# ---------------------------------------------------------------------------
def download_batch(tracks: list[Any], out_dir: Path, workers: int) -> tuple[list[DownloadOutcome], float]:
    """Baixa as faixas em paralelo com download.download_audio e mede cada uma."""
    from modules import download

    def job(track: Any) -> DownloadOutcome:
        started = time.perf_counter()
        try:
            path = download.download_audio(track.video_id, cache_dir=out_dir)
        except Exception as exc:  # noqa: BLE001 - DownloadError ou falha inesperada
            msg = str(exc) if isinstance(exc, download.DownloadError) else friendly_error(exc, "download")
            return DownloadOutcome(track, time.perf_counter() - started, None, msg)
        return DownloadOutcome(track, time.perf_counter() - started, Path(path), None)

    outcomes: list[DownloadOutcome] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(job, t) for t in tracks]
        for fut in as_completed(futures):
            res = fut.result()
            outcomes.append(res)
            if res.error:
                say(f"    [falhou] {num(res.seconds):>6} s  {clip(label_of(res.track), 60)}: {res.error}")
            else:
                size_mb = res.path.stat().st_size / 1e6 if res.path and res.path.exists() else 0.0
                say(f"    [ok]     {num(res.seconds):>6} s  {num(size_mb):>5} MB  {clip(label_of(res.track), 60)}")
    return outcomes, time.perf_counter() - started


def summarize_batch(outcomes: list[DownloadOutcome], wall_s: float, workers: int) -> str:
    ok = [o for o in outcomes if o.error is None]
    size_mb = sum(o.path.stat().st_size for o in ok if o.path and o.path.exists()) / 1e6
    say(f"    Total: {len(ok)}/{len(outcomes)} baixadas em {num(wall_s)} s ({workers} em paralelo), {num(size_mb)} MB.")
    if ok:
        times = [o.seconds for o in ok]
        per_track = wall_s / len(ok)
        say(f"    Por faixa: {num(per_track)} s (tempo total ÷ faixas); individual médio "
            f"{num(sum(times) / len(times))} s (mín {num(min(times))} / máx {num(max(times))}).")
        say(f"    Projeção para {config.TYPICAL_SET_SIZE} faixas: ~{num(per_track * config.TYPICAL_SET_SIZE / 60, 0)} min.")
    return f"{len(ok)}/{len(outcomes)} em {num(wall_s)} s"


def validate_ytdlp(tracks: list[Any], out_dir: Path, batches: list[int],
                   workers: int) -> tuple[Check, list[tuple[Any, Path]]]:
    """Baixa lotes de faixas NOVAS (as que já estão no cache não entram na medição)."""
    header(2)
    from modules import download

    if shutil.which("ffmpeg") is None:
        return Check(2, FAILED, "ffmpeg não encontrado no PATH (instale com: winget install Gyan.FFmpeg)"), []
    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"  Pasta de saída: {out_dir}")
    cached = [t for t in tracks if download.is_cached(t.video_id, out_dir)]
    fresh = [t for t in tracks if not download.is_cached(t.video_id, out_dir)]
    if cached:
        say(f"  {len(cached)} faixa(s) já estavam no cache e não entram na medição.")

    available: dict[str, Path] = {t.video_id: download.cache_path(t.video_id, out_dir) for t in cached}
    parts: list[str] = []
    failures = 0
    succeeded = 0
    pos = 0
    for size in batches:
        batch = fresh[pos:pos + size]
        pos += len(batch)
        if not batch:
            say(f"  Lote de {size}: não sobrou nenhuma faixa nova para baixar "
                f"(use mais referências, --ids ou outra pasta em --saida).")
            parts.append(f"lote de {size}: sem faixas novas")
            continue
        if len(batch) < size:
            say(f"  Aviso: só havia {len(batch)} faixa(s) nova(s) para o lote de {size}.")
        parallel = min(workers, len(batch))
        say(f"  Lote de {len(batch)} faixa(s), {parallel} em paralelo:")
        outcomes, wall_s = download_batch(batch, out_dir, parallel)
        parts.append(f"lote de {len(batch)}: {summarize_batch(outcomes, wall_s, parallel)}")
        failures += sum(1 for o in outcomes if o.error)
        succeeded += sum(1 for o in outcomes if o.error is None)
        for o in outcomes:
            if o.error is None and o.path is not None:
                available[o.track.video_id] = o.path

    ordered = [(t, available[t.video_id]) for t in tracks if t.video_id in available]
    detail = "; ".join(parts) + f" (até {workers} em paralelo)"
    if succeeded == 0:
        status = FAILED if failures else ATTENTION
    else:
        status = ATTENTION if failures else OK
    return Check(2, status, detail), ordered


# ---------------------------------------------------------------------------
# 3. librosa
# ---------------------------------------------------------------------------
def audio_files(folder: Path, recursive: bool = False) -> list[Path]:
    exts = {e.lower() for e in config.LOCAL_AUDIO_EXTENSIONS}
    candidates = folder.rglob("*") if recursive else folder.iterdir()
    return sorted((p for p in candidates
                   if p.is_file() and p.suffix.lower() in exts and not p.name.startswith(".")),
                  key=lambda p: str(p).casefold())


def items_from_folder(folder: Path, recursive: bool = False, limit: int = 0) -> list[AudioItem]:
    files = audio_files(folder, recursive)
    if limit > 0:
        files = files[:limit]
    return [AudioItem(label=p.name, path=p, video_id=p.stem if YOUTUBE_ID_RE.match(p.stem) else f"local_{p.stem}")
            for p in files]


CSV_COLUMNS = [
    "faixa", "arquivo", "duracao", "bpm", "bpm_bruto", "energia_media", "energia_entrada",
    "energia_saida", "loudness_db", "tom", "camelot", "confianca_tom", "intro_s", "saida_s",
    "tempo_analise_s", "bpm_rekordbox", "tom_rekordbox", "observacao",
]


def write_librosa_csv(rows: list[list[str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding=config.CSV_ENCODING, newline="") as f:
        writer = csv.writer(f, delimiter=config.CSV_DELIMITER, lineterminator="\r\n")
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def validate_librosa(items: list[AudioItem], csv_path: Path) -> tuple[Check, dict[str, Any]]:
    header(3)
    if not items:
        return Check(3, FAILED, "nenhum arquivo de áudio para analisar"), {}
    from modules import analyze
    from modules.util import format_duration

    say(f"  Analisando {len(items)} arquivo(s) inteiros. A 1ª análise compila o numba (~12 s).")
    cols = f"  {'arquivo':<36} | {'BPM':>6} | {'BPM bruto':>9} | {'energia média':>13} | {'tom':<10} | {'Camelot':>7} | {'intro':>7} | {'saída':>7}"
    say(cols)
    say("  " + "-" * (len(cols) - 2))
    analyses: dict[str, Any] = {}
    rows: list[list[str]] = []
    errors: list[str] = []
    for item in items:
        started = time.perf_counter()
        try:
            a = analyze.analyze_file(item.path, item.video_id)
        except Exception as exc:  # noqa: BLE001 - arquivo corrompido, módulo incompleto etc.
            msg = friendly_error(exc, "analyze")
            errors.append(f"{item.label}: {msg}")
            say(f"  {clip(item.label, 36):<36} | erro: {msg}")
            if isinstance(exc, NotImplementedError):
                break
            continue
        elapsed = time.perf_counter() - started
        analyses[item.video_id] = a
        say(f"  {clip(item.label, 36):<36} | {num(a.bpm):>6} | {num(a.bpm_raw):>9} | {num(a.energy_mean, 2):>13} | "
            f"{(a.key or '-'):<10} | {(a.camelot or '-'):>7} | {num(a.intro_s) + ' s':>7} | {num(a.outro_s) + ' s':>7}")
        rows.append([
            item.label, item.path.name, format_duration(a.duration_s), num(a.bpm), num(a.bpm_raw),
            num(a.energy_mean, 2), num(a.energy_in, 2), num(a.energy_out, 2), num(a.loudness_db),
            a.key or "", a.camelot or "", num(a.key_confidence, 2), num(a.intro_s), num(a.outro_s),
            num(elapsed), "", "", "",
        ])
    if rows:
        try:
            write_librosa_csv(rows, csv_path)
        except OSError as exc:
            errors.append(f"CSV: {exc}")
            say(f"  Não consegui gravar o CSV (ele está aberto no Excel?): {exc}")
        else:
            say()
            say(f"  CSV salvo em: {csv_path}")
            say("  Preencha as colunas bpm_rekordbox e tom_rekordbox com o que o Rekordbox mostra "
                "e compare (BPM: diferença de até 1 é aceitável).")
    if not analyses:
        return Check(3, FAILED, errors[0] if errors else "nenhuma faixa analisada"), {}
    detail = f"{len(analyses)} faixa(s) analisada(s); CSV em {csv_path}; comparar com o Rekordbox junto com o DJ SK"
    if errors:
        detail += f" ({len(errors)} erro(s))"
    return Check(3, ATTENTION if errors else PENDING, detail), analyses


# ---------------------------------------------------------------------------
# 4. Playlist .m3u8 para o Rekordbox
# ---------------------------------------------------------------------------
def validate_m3u8(tracks: list[Any], analyses: dict[str, Any], out_dir: Path, name: str,
                  cache_dir: Path | None = None) -> Check:
    header(4)
    if not tracks:
        return Check(4, FAILED, "nenhuma faixa com áudio para montar a playlist")
    from modules import export

    try:
        result = export.export_setlist(tracks, analyses, transitions=None, out_dir=out_dir,
                                       playlist_name=name, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001
        return Check(4, FAILED, friendly_error(exc, "export"))
    if not result.files:
        return Check(4, FAILED, "nenhum arquivo foi exportado (faixas sem áudio)")
    say(f"  {len(result.files)} faixa(s) copiadas e numeradas em: {result.folder}")
    say(f"  Playlist: {result.m3u8_path}")
    say(f"  CSV:      {result.csv_path}")
    if result.skipped:
        say(f"  Sem áudio (ficaram de fora): {', '.join(result.skipped)}")
    try:
        lines = result.m3u8_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    if lines:
        say("  Início do .m3u8:")
        for line in lines[:5]:
            say(f"    {line}")
    say()
    say("  Para validar no Rekordbox:")
    say("    1. Menu Arquivo > Importar > Importar playlist (em inglês: File > Import > Import Playlist)")
    say("       e escolha o .m3u8 acima, ou arraste o .m3u8 para a lista de Playlists.")
    say(f"    2. Confira se as {len(result.files)} faixas aparecem na ordem 01, 02, ... e tocam.")
    say("    3. Se não funcionar (plano B): arraste a pasta numerada inteira para o Rekordbox.")
    return Check(4, PENDING, f"playlist com {len(result.files)} faixa(s) em {result.m3u8_path}; importar no Rekordbox")


def tracks_from_folder(folder: Path, recursive: bool = False, limit: int = 0) -> list[Any]:
    from modules import download

    tracks = download.local_tracks_from_folder(folder, recursive=recursive)
    return tracks[:limit] if limit > 0 else tracks


def analyses_for(tracks: list[Any], run_analysis: bool) -> dict[str, Any]:
    """Análises para gravar BPM/tom nas tags: do cache, ou calculadas com --analisar."""
    from modules import analyze

    found: dict[str, Any] = {}
    for t in tracks:
        try:
            a = analyze.load_analysis(t.video_id)
            if a is None and run_analysis and t.local_path:
                a = analyze.analyze_file(t.local_path, t.video_id)
        except NotImplementedError:
            say("  (o módulo analyze ainda não foi implementado: exportando sem BPM/tom nas tags)")
            return {}
        except Exception as exc:  # noqa: BLE001
            say(f"  Análise de {label_of(t)} falhou: {friendly_error(exc, 'analyze')}")
            continue
        if a is not None:
            found[t.video_id] = a
    return found


# ---------------------------------------------------------------------------
# 5. Replicate
# ---------------------------------------------------------------------------
def replicate_get(path: str, token: str) -> Any:
    """GET na API da Replicate. Só leitura: não cria previsão e não gasta crédito."""
    import requests

    util.setup_ssl()
    resp = requests.get(f"{REPLICATE_API_URL}{path}", timeout=HTTP_TIMEOUT_S,
                        headers={"Authorization": f"Bearer {token}"})
    if resp.status_code == 401:
        raise RuntimeError("token da Replicate inválido ou expirado (REPLICATE_API_TOKEN no .env)")
    if resp.status_code == 404:
        raise RuntimeError(f"não encontrado na Replicate: {path}")
    if resp.status_code >= 400:
        raise RuntimeError(f"a Replicate respondeu HTTP {resp.status_code}")
    return resp.json()


def _resolve_ref(node: dict, schemas: dict) -> dict:
    """Resolve "$ref" e "allOf" de um esquema OpenAPI do cog."""
    if not isinstance(node, dict):
        return {}
    if "$ref" in node:
        merged = dict(_resolve_ref(schemas.get(node["$ref"].rsplit("/", 1)[-1], {}), schemas))
        merged.update({k: v for k, v in node.items() if k != "$ref"})
        return merged
    if node.get("allOf"):
        merged: dict = {}
        for part in node["allOf"]:
            merged.update(_resolve_ref(part, schemas))
        merged.update({k: v for k, v in node.items() if k != "allOf"})
        return merged
    return node


def describe_schema(openapi: dict) -> tuple[dict[str, dict], list[str], dict]:
    """Devolve (entradas resolvidas na ordem do modelo, obrigatórias, esquema de saída)."""
    schemas = (openapi.get("components") or {}).get("schemas") or {}
    inp = _resolve_ref(schemas.get("Input") or {}, schemas)
    props = {k: _resolve_ref(v, schemas) for k, v in (inp.get("properties") or {}).items()}
    props = dict(sorted(props.items(), key=lambda kv: kv[1].get("x-order", 999)))
    output = _resolve_ref(schemas.get("Output") or {}, schemas)
    return props, list(inp.get("required") or []), output


def check_replicate_schema(props: dict[str, dict], output: dict) -> list[str]:
    """Compara o esquema com o que o módulo stems usa. Devolve as divergências."""
    problems: list[str] = []
    if "audio" not in props:
        problems.append("entrada 'audio' não existe")
    for name, expected in (("model_name", config.DEMUCS_MODEL_NAME),
                           ("output_format", config.DEMUCS_OUTPUT_FORMAT)):
        if name not in props:
            problems.append(f"entrada '{name}' não existe")
        elif props[name].get("enum") and expected not in props[name]["enum"]:
            problems.append(f"'{expected}' não está entre as opções de '{name}'")
    out_props = set((output.get("properties") or {}).keys())
    missing = [s for s in config.STEM_NAMES if s not in out_props]
    if missing:
        problems.append(f"saída sem {', '.join(missing)}")
    return problems


def validate_replicate(token: str | None = None) -> Check:
    header(5)
    model, version = config.REPLICATE_MODEL, config.REPLICATE_VERSION
    token = token if token is not None else config.get_secret("REPLICATE_API_TOKEN")
    if not token:
        say("  REPLICATE_API_TOKEN não está no .env, então não dá para ler o esquema pela API.")
        say("  Para validar à mão (sem custo):")
        say(f"    1. Abra https://replicate.com/{model}/api (versão {version[:12]}...).")
        say(f"    2. Confira as entradas: audio (arquivo), model_name = {config.DEMUCS_MODEL_NAME}, "
            f"output_format = {config.DEMUCS_OUTPUT_FORMAT}.")
        say(f"    3. Confira a saída: {', '.join(config.STEM_NAMES)} (links dos arquivos).")
        say("  Ou crie um token em https://replicate.com/account/api-tokens, coloque em "
            "REPLICATE_API_TOKEN no .env e rode de novo.")
        return Check(5, PENDING, "sem REPLICATE_API_TOKEN: conferir pela aba API do modelo")

    say(f"  Lendo o esquema de {model}, versão {version[:12]}... (só leitura, custo zero)")
    try:
        data = replicate_get(f"/models/{model}/versions/{version}", token)
    except Exception as exc:  # noqa: BLE001
        return Check(5, FAILED, friendly_error(exc, "replicate"))
    props, required, output = describe_schema(data.get("openapi_schema") or {})
    say("  Entradas (Input):")
    for name, spec in props.items():
        kind = spec.get("type") or ("enum" if spec.get("enum") else "?")
        if spec.get("format"):
            kind += f"/{spec['format']}"
        extra = f" opções={spec['enum']}" if spec.get("enum") else ""
        default = f" padrão={spec['default']!r}" if "default" in spec else ""
        req = " (obrigatória)" if name in required else ""
        say(f"    - {name}: {kind}{default}{extra}{req}")
    out_props = output.get("properties") or {}
    say(f"  Saída (Output): {output.get('type', '?')} com {', '.join(out_props) or '(sem propriedades)'}")

    try:
        latest = ((replicate_get(f"/models/{model}", token) or {}).get("latest_version") or {}).get("id")
    except Exception as exc:  # noqa: BLE001 - informação extra, não derruba a validação
        logger.info("Não consegui ler a versão mais recente: %s", exc)
        latest = None
    if latest and latest != version:
        say(f"  Aviso: há uma versão mais nova do modelo ({latest[:12]}...). "
            f"O config.py usa {version[:12]}..., que continua funcionando.")
    elif latest:
        say("  A versão do config.py é a mais recente do modelo.")

    problems = check_replicate_schema(props, output)
    if problems:
        return Check(5, FAILED, "esquema diferente do esperado: " + "; ".join(problems))
    return Check(5, OK, f"audio, model_name={config.DEMUCS_MODEL_NAME}, output_format={config.DEMUCS_OUTPUT_FORMAT}; "
                        f"saída {', '.join(config.STEM_NAMES)}")


# ---------------------------------------------------------------------------
# 6. Conversa com o DJ SK (manual)
# ---------------------------------------------------------------------------
def checklist_dj() -> Check:
    header(6)
    say("  Perguntas para o DJ SK (preencha o config.py com as respostas):")
    say("    - Como você ordena um set: por BPM, por energia, por tom? (pesos em DEFAULT_WEIGHTS)")
    say(f"    - Qual a faixa de BPM típica do mandelão? (hoje: {num(config.BPM_MIN, 0)} a {num(config.BPM_MAX, 0)})")
    say("    - O BPM do CSV bate com o do Rekordbox? E a playlist .m3u8 importou?")
    say("    - Quanto tempo leva hoje para aprovar e baixar uma música? (linha de base: 1 a 1,5 min)")
    return Check(6, PENDING, "conversa com o DJ SK (ordem do set e faixa de BPM)")


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------
def _queries(args: argparse.Namespace, attr: str) -> list[str]:
    return [q for q in (getattr(args, attr, None) or []) if q.strip()] or list(DEFAULT_REFERENCES)


def _tracks_for_download(args: argparse.Namespace) -> list[Any]:
    if args.ids:
        from modules.models import Track

        return [Track(video_id=v, title=v) for v in dict.fromkeys(args.ids)]
    say("  Buscando sugestões no rádio do YouTube Music...")
    run = collect_suggestions(_queries(args, "referencia"), verbose=True)
    say(f"  {len(run.suggestions)} sugestões do rádio para baixar.")
    return run.suggestions


def cmd_ytmusic(args: argparse.Namespace) -> list[Check]:
    check, _ = validate_ytmusic(_queries(args, "referencias"), args.quantidade)
    return [check]


def cmd_ytdlp(args: argparse.Namespace) -> list[Check]:
    out_dir = Path(args.saida) if args.saida else Path(config.CACHE_DIR)
    try:
        tracks = _tracks_for_download(args)
    except Exception as exc:  # noqa: BLE001
        return [Check(2, FAILED, f"não consegui montar a lista de faixas: {friendly_error(exc, 'suggest')}")]
    if not tracks:
        return [Check(2, FAILED, "nenhuma faixa para baixar")]
    check, _ = validate_ytdlp(tracks, out_dir, args.lotes, args.paralelo)
    return [check]


def cmd_librosa(args: argparse.Namespace) -> list[Check]:
    folder = Path(args.pasta) if args.pasta else Path(config.CACHE_DIR)
    if not folder.is_dir():
        return [Check(3, FAILED, f"pasta não encontrada: {folder}")]
    csv_path = Path(args.csv) if args.csv else Path(config.DATA_DIR) / "validacao_librosa.csv"
    check, _ = validate_librosa(items_from_folder(folder, args.recursivo, args.limite), csv_path)
    return [check]


def cmd_m3u8(args: argparse.Namespace) -> list[Check]:
    folder = Path(args.pasta)
    if not folder.is_dir():
        return [Check(4, FAILED, f"pasta não encontrada: {folder}")]
    out_dir = Path(args.saida) if args.saida else Path(config.DATA_DIR) / "Setlist_Teste"
    if out_dir.resolve() == folder.resolve():
        return [Check(4, FAILED, "a pasta de saída deve ser diferente da pasta com as músicas")]
    try:
        tracks = tracks_from_folder(folder, args.recursivo, args.limite)
    except Exception as exc:  # noqa: BLE001
        return [Check(4, FAILED, friendly_error(exc, "download"))]
    if not tracks:
        return [Check(4, FAILED, f"nenhum arquivo de áudio em {folder}")]
    analyses = analyses_for(tracks, args.analisar)
    return [validate_m3u8(tracks, analyses, out_dir, args.nome)]


def cmd_replicate(args: argparse.Namespace) -> list[Check]:
    return [validate_replicate()]


def cmd_tudo(args: argparse.Namespace) -> list[Check]:
    skip = set(args.pular or [])
    checks: list[Check] = []
    out_dir = Path(args.saida) if args.saida else Path(config.CACHE_DIR)

    run: RadioRun | None = None
    if "ytmusic" in skip:
        checks.append(Check(1, SKIPPED, "pulado (--pular ytmusic)"))
    else:
        check, run = validate_ytmusic(_queries(args, "referencia"), config.N_SUGGESTIONS)
        checks.append(check)
    suggestions = run.suggestions if run else []

    available: list[tuple[Any, Path]] = []
    if "ytdlp" in skip:
        checks.append(Check(2, SKIPPED, "pulado (--pular ytdlp)"))
    elif not suggestions:
        checks.append(Check(2, FAILED, "sem sugestões do rádio para baixar"))
    else:
        try:
            check, available = validate_ytdlp(suggestions, out_dir, args.lotes, args.paralelo)
        except Exception as exc:  # noqa: BLE001
            check = Check(2, FAILED, friendly_error(exc, "download"))
        checks.append(check)
    if not available and suggestions:
        # Sem download nesta execução: usa o que já estiver no cache.
        try:
            from modules import download

            available = [(t, download.cache_path(t.video_id, out_dir)) for t in suggestions
                         if download.is_cached(t.video_id, out_dir)]
        except Exception:  # noqa: BLE001
            available = []
    sample = available[:REKORDBOX_SAMPLE]

    analyses: dict[str, Any] = {}
    if "librosa" in skip:
        checks.append(Check(3, SKIPPED, "pulado (--pular librosa)"))
    elif not sample:
        checks.append(Check(3, FAILED, "nenhum áudio baixado para analisar"))
    else:
        items = [AudioItem(label=label_of(t), path=p, video_id=t.video_id) for t, p in sample]
        csv_path = Path(args.csv) if args.csv else Path(config.DATA_DIR) / "validacao_librosa.csv"
        check, analyses = validate_librosa(items, csv_path)
        checks.append(check)

    if "m3u8" in skip:
        checks.append(Check(4, SKIPPED, "pulado (--pular m3u8)"))
    elif not sample:
        checks.append(Check(4, FAILED, "nenhum áudio baixado para montar a playlist"))
    else:
        m3u_dir = Path(args.saida_m3u8) if args.saida_m3u8 else Path(config.DATA_DIR) / "Setlist_Teste"
        checks.append(validate_m3u8([t for t, _ in sample], analyses, m3u_dir, args.nome, cache_dir=out_dir))

    if "replicate" in skip:
        checks.append(Check(5, SKIPPED, "pulado (--pular replicate)"))
    else:
        checks.append(validate_replicate())
    checks.append(checklist_dj())
    return checks


def print_summary(checks: list[Check]) -> None:
    say()
    say("== Resumo ==")
    for c in checks:
        say(f"  {c.number}. {c.name:<36} {c.status:<9} {c.detail}")


# ---------------------------------------------------------------------------
# Linha de comando
# ---------------------------------------------------------------------------
class _Formatter(argparse.RawDescriptionHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):  # noqa: ANN001 - assinatura do argparse
        return super().add_usage(usage, actions, groups, prefix if prefix is not None else "uso: ")


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"número inteiro inválido: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("use um número maior que zero")
    return value


def _translate(parser: argparse.ArgumentParser) -> None:
    parser._positionals.title = "argumentos"
    parser._optionals.title = "opções"
    parser.add_argument("-h", "--help", action="help", help="mostra esta ajuda e sai")


def _download_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("-r", "--referencia", action="append", metavar='"ARTISTA - FAIXA"',
                   help=f"referência (repita até {config.MAX_REFERENCES} vezes; padrão: {DEFAULT_REFERENCES[0]!r})")
    p.add_argument("--saida", metavar="PASTA", help="pasta dos mp3 baixados (padrão: cache/ do projeto)")
    p.add_argument("--lotes", type=_positive_int, nargs="+", default=list(DEFAULT_BATCHES), metavar="N",
                   help="tamanhos dos lotes baixados em sequência (padrão: 5 20)")
    p.add_argument("--paralelo", type=_positive_int, default=config.MAX_PARALLEL_DOWNLOADS, metavar="N",
                   help=f"downloads simultâneos (padrão: {config.MAX_PARALLEL_DOWNLOADS})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="validar.py", formatter_class=_Formatter, add_help=False,
        description="Validações iniciais do painel de curadoria do DJ SK (seção 9 da especificação).",
        epilog="Exemplo: .venv\\Scripts\\python.exe scripts\\validar.py tudo")
    _translate(parser)
    parser.add_argument("-v", "--verbose", action="store_true", help="mostra os registros (logs) dos módulos")
    subs = parser.add_subparsers(dest="comando", required=True, metavar="COMANDO", title="comandos")

    def sub(name: str, help_text: str) -> argparse.ArgumentParser:
        p = subs.add_parser(name, help=help_text, description=help_text.replace("%%", "%"), formatter_class=_Formatter, add_help=False)
        _translate(p)
        return p

    p = sub("ytmusic", "1. rádio do YouTube Music sem login: 20 primeiras sugestões e %% com cara de funk")
    p.add_argument("referencias", nargs="*", metavar='"ARTISTA - FAIXA"',
                   help=f"1 a {config.MAX_REFERENCES} referências, link ou videoId (padrão: {DEFAULT_REFERENCES[0]!r})")
    p.add_argument("-n", "--quantidade", type=_positive_int, default=config.N_SUGGESTIONS, metavar="N",
                   help=f"quantas sugestões mostrar (padrão: {config.N_SUGGESTIONS})")
    p.set_defaults(func=cmd_ytmusic)

    p = sub("ytdlp", "2. baixa 5 e depois 20 sugestões do rádio em paralelo e mede o tempo")
    _download_options(p)
    p.add_argument("--ids", nargs="+", metavar="VIDEOID", help="baixa estes videoIds em vez das sugestões do rádio")
    p.set_defaults(func=cmd_ytdlp)

    p = sub("librosa", "3. analisa os áudios de uma pasta (BPM, energia, tom, intro/saída) e salva CSV")
    p.add_argument("pasta", nargs="?", help="pasta com os áudios (padrão: cache/ do projeto)")
    p.add_argument("--csv", metavar="ARQUIVO", help="onde salvar o CSV (padrão: data/validacao_librosa.csv)")
    p.add_argument("-n", "--limite", type=int, default=0, metavar="N", help="analisa só os N primeiros arquivos")
    p.add_argument("--recursivo", action="store_true", help="inclui subpastas")
    p.set_defaults(func=cmd_librosa)

    p = sub("m3u8", "4. gera uma playlist de teste (.m3u8 + pasta numerada) para importar no Rekordbox")
    p.add_argument("pasta", help="pasta com os mp3")
    p.add_argument("--saida", metavar="PASTA",
                   help="pasta da playlist de teste (padrão: data/Setlist_Teste). "
                        "Exportações anteriores nessa pasta são substituídas")
    p.add_argument("--nome", default="Setlist_Teste", help="nome da playlist (padrão: Setlist_Teste)")
    p.add_argument("--analisar", action="store_true", help="analisa as faixas para gravar BPM e tom nas tags")
    p.add_argument("-n", "--limite", type=int, default=0, metavar="N", help="usa só os N primeiros arquivos")
    p.add_argument("--recursivo", action="store_true", help="inclui subpastas")
    p.set_defaults(func=cmd_m3u8)

    p = sub("replicate", "5. confirma os parâmetros do cjwbw/demucs na Replicate (só leitura, custo zero)")
    p.set_defaults(func=cmd_replicate)

    p = sub("tudo", "roda as validações 1 a 5 em sequência e mostra o checklist da 6")
    _download_options(p)
    p.add_argument("--csv", metavar="ARQUIVO", help="CSV da análise (padrão: data/validacao_librosa.csv)")
    p.add_argument("--saida-m3u8", metavar="PASTA", help="pasta da playlist de teste (padrão: data/Setlist_Teste)")
    p.add_argument("--nome", default="Setlist_Teste", help="nome da playlist de teste")
    p.add_argument("--pular", nargs="+", choices=["ytmusic", "ytdlp", "librosa", "m3u8", "replicate"],
                   metavar="ETAPA", help="etapas para pular: ytmusic, ytdlp, librosa, m3u8, replicate")
    p.set_defaults(func=cmd_tudo)
    return parser


def _setup_console() -> None:
    # Acentos não podem derrubar o script quando a saída vai para arquivo ou pipe.
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="replace")
            else:
                stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    util.setup_ssl()
    try:
        checks = args.func(args)
    except KeyboardInterrupt:
        say("\nInterrompido.")
        return 130
    if args.comando == "tudo" or len(checks) > 1:
        print_summary(checks)
    else:
        say()
        for c in checks:
            say(f"Resultado: {c.status} - {c.detail}")
    return 1 if any(c.status == FAILED for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
