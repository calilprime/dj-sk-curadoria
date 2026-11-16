"""Referências → sugestões (seções 4.1 e 4.2).

CONTRATO (implementar mantendo as assinaturas):

- Rádio do YouTube Music via `ytmusicapi`, sem login. VALIDADO em 22/10/2026:
  `get_watch_playlist(videoId, radio=True)` devolve ~50 faixas de funk sem login;
  a 1ª faixa do rádio é a própria referência. Cada item do rádio tem as chaves
  'videoId', 'title', 'artists' [{'name','id'}], 'length' ("2:38"), 'album', 'thumbnail', 'videoType'.
  O resultado de `search(..., filter="songs")` tem 'duration_seconds', 'artists', 'thumbnails', 'album'.
- Chamar `util.setup_ssl()` antes de qualquer rede (há interceptação TLS nesta máquina).
- Plano B: YouTube Data API v3 (`YOUTUBE_API_KEY`), busca por artistas das referências + estilo.

Convenção dos filtros: `exclude_keywords=None` usa config.EXCLUDE_KEYWORDS e `[]` desliga
o filtro; `min_duration_s`/`max_duration_s` None usam MIN/MAX_TRACK_DURATION_S do config.
"""
from __future__ import annotations

import difflib
import html
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse

import requests

import config
from modules import util
from modules.models import Track

logger = logging.getLogger(__name__)

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
HTTP_TIMEOUT_S = 15
MAX_PARALLEL_RADIOS = 3

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_ISO_DURATION_RE = re.compile(
    r"^P(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+(?:\.\d+)?)S)?)?$"
)
_PAREN_RE = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")
# Parênteses com estas palavras não mudam a música: participação, produção, "clipe oficial" etc.
_NOISE_WORDS = {
    "feat", "ft", "featuring", "part", "participacao", "participacoes", "prod", "producao",
    "official", "oficial", "clipe", "videoclipe", "video", "audio", "visualizer",
    "lyric", "lyrics", "letra", "explicit", "explicito",
}
_TRAILING_CREDIT_RE = re.compile(
    r"\s+(?:feat\.?|ft\.?|featuring|part\.?|participa[cç][aã]o|prod\.?)\s+.*$", re.IGNORECASE
)
_DASH_RE = re.compile(r"\s+[-–—]\s+")


class SuggestError(Exception):
    """Falha ao resolver referências ou buscar sugestões (mensagem em português)."""


@dataclass
class SuggestionResult:
    tracks: list[Track]                 # lista COMPLETA ranqueada (a interface mostra páginas de N_SUGGESTIONS)
    source: str                         # "ytmusic" | "youtube_api"
    warnings: list[str] = field(default_factory=list)  # avisos para mostrar ao DJ (em português)


# ---------------------------------------------------------------------------
# Cliente do YouTube Music
# ---------------------------------------------------------------------------
_yt_lock = threading.Lock()
_yt_client = None


def get_ytmusic():
    """Cliente `YTMusic()` sem login, reutilizado entre chamadas (singleton por processo)."""
    global _yt_client
    with _yt_lock:
        if _yt_client is None:
            util.setup_ssl()
            from ytmusicapi import YTMusic

            _yt_client = YTMusic()
        return _yt_client


# ---------------------------------------------------------------------------
# Auxiliares de texto e conversão (também usados por spotify_import)
# ---------------------------------------------------------------------------
def strip_title_noise(title: str) -> str:
    """Remove "(feat ...)", "(part. ...)", "(Clipe Oficial)" e créditos no fim do título."""
    def _drop(m: re.Match) -> str:
        words = set(util.normalize_text(m.group(1)).split())
        return " " if words & _NOISE_WORDS else m.group(0)

    cleaned = _PAREN_RE.sub(_drop, title or "")
    cleaned = _TRAILING_CREDIT_RE.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def title_core(title: str) -> str:
    """Título normalizado sem participações nem rótulos de vídeo, para comparar faixas."""
    return util.normalize_text(strip_title_noise(title))


def title_variants(title: str) -> set[str]:
    """Núcleos possíveis do título; em "MC X - Faixa" inclui também só "Faixa"."""
    variants = {title_core(title)}
    parts = _DASH_RE.split(title or "", maxsplit=1)
    if len(parts) == 2:
        variants.add(title_core(parts[1]))
    return {v for v in variants if v}


def titles_similar(a: str, b: str, threshold: float = 0.75) -> bool:
    """True se algum núcleo de `a` é parecido com algum núcleo de `b`."""
    for x in title_variants(a):
        for y in title_variants(b):
            if x == y:
                return True
            if difflib.SequenceMatcher(None, x, y).ratio() >= threshold:
                return True
            short, long_ = sorted((x, y), key=len)
            if len(short) >= 4 and f" {short} " in f" {long_} ":
                return True
    return False


def _artist_names(title: str, artists: Iterable[str]) -> tuple[set[str], str]:
    """Nomes normalizados e um "texto corrido" com todos os artistas (inclui o que vem antes do " - ")."""
    names = {util.normalize_text(a) for a in artists if a}
    parts = _DASH_RE.split(title or "", maxsplit=1)
    if len(parts) == 2:
        names.add(util.normalize_text(parts[0]))
    names.discard("")
    return names, " ".join(sorted(names))


def artists_overlap(title_a: str, artists_a: Iterable[str],
                    title_b: str, artists_b: Iterable[str]) -> bool | None:
    """True/False se há algum artista em comum; None quando um dos lados não tem artista."""
    names_a, blob_a = _artist_names(title_a, artists_a)
    names_b, blob_b = _artist_names(title_b, artists_b)
    if not names_a or not names_b:
        return None
    for name in names_a:
        if f" {name} " in f" {blob_b} ":
            return True
    for name in names_b:
        if f" {name} " in f" {blob_a} ":
            return True
    return False


def same_song(a: Track, b: Track) -> bool:
    """Mesma música (talvez outro upload): núcleo do título igual e artista em comum."""
    if not (title_variants(a.title) & title_variants(b.title)):
        return False
    return artists_overlap(a.title, a.artists, b.title, b.artists) is not False


def best_thumbnail(thumbs: Any) -> str | None:
    """URL da maior miniatura de uma lista [{'url','width','height'}]."""
    if not isinstance(thumbs, list) or not thumbs:
        return None
    valid = [t for t in thumbs if isinstance(t, dict) and t.get("url")]
    if not valid:
        return None
    best = max(enumerate(valid), key=lambda it: ((it[1].get("width") or 0) * (it[1].get("height") or 0), it[0]))
    return best[1]["url"]


def item_duration(item: dict) -> float | None:
    """Duração de um item cru: 'duration_seconds', 'length' ("2:38") ou 'duration'."""
    ds = item.get("duration_seconds")
    if isinstance(ds, (int, float)) and not isinstance(ds, bool) and ds > 0:
        return float(ds)
    for key in ("length", "duration"):
        d = util.parse_duration(item.get(key))
        if d is not None:
            return d
    return None


def track_from_item(item: dict, source: str = "ytmusic") -> Track | None:
    """Converte um item cru do ytmusicapi (rádio ou busca) em Track. None se não houver videoId."""
    if not isinstance(item, dict):
        return None
    video_id = item.get("videoId")
    if not video_id:
        return None
    artists = [a.get("name") for a in item.get("artists") or []
               if isinstance(a, dict) and a.get("name")]
    album = item.get("album")
    if isinstance(album, dict):
        album = album.get("name")
    thumbs = item.get("thumbnails") or item.get("thumbnail")
    return Track(
        video_id=video_id,
        title=(item.get("title") or "").strip(),
        artists=artists,
        duration_s=item_duration(item),
        album=album if isinstance(album, str) and album else None,
        thumbnail_url=best_thumbnail(thumbs),
        source=source,
    )


def extract_video_id(text: str) -> str | None:
    """videoId de um link do YouTube/YouTube Music ou de um id puro de 11 caracteres."""
    text = (text or "").strip()
    if _VIDEO_ID_RE.match(text):
        return text
    if "youtu" not in text.lower():
        return None
    url = text if "://" in text else f"https://{text}"
    try:
        parsed = urlparse(url)
    except ValueError:
        return None
    host = (parsed.hostname or "").lower()
    candidate = None
    if host.endswith("youtu.be"):
        candidate = parsed.path.strip("/").split("/")[0]
    elif "youtube" in host:
        candidate = (parse_qs(parsed.query).get("v") or [None])[0]
        if not candidate:
            segs = [s for s in parsed.path.split("/") if s]
            if len(segs) >= 2 and segs[0] in ("shorts", "embed", "live", "v"):
                candidate = segs[1]
    if candidate and _VIDEO_ID_RE.match(candidate):
        return candidate
    return None


# ---------------------------------------------------------------------------
# 4.1 Referências
# ---------------------------------------------------------------------------
def _track_from_video_id(video_id: str, yt) -> Track | None:
    try:
        data = yt.get_watch_playlist(videoId=video_id)
        tracks = (data or {}).get("tracks") or []
        for item in tracks:
            if isinstance(item, dict) and item.get("videoId") == video_id:
                track = track_from_item(item)
                if track and track.title:
                    return track
    except Exception as e:  # noqa: BLE001 - tenta o get_song a seguir
        logger.info("get_watch_playlist falhou para %s: %s", video_id, e)
    try:
        song = yt.get_song(video_id) or {}
    except Exception as e:  # noqa: BLE001
        logger.info("get_song falhou para %s: %s", video_id, e)
        return None
    details = song.get("videoDetails") or {}
    if not details.get("title"):
        return None
    try:
        duration = float(details.get("lengthSeconds")) or None
    except (TypeError, ValueError):
        duration = None
    author = details.get("author")
    return Track(
        video_id=details.get("videoId") or video_id,
        title=details["title"].strip(),
        artists=[author] if author else [],
        duration_s=duration,
        thumbnail_url=best_thumbnail((details.get("thumbnail") or {}).get("thumbnails")),
        source="ytmusic",
    )


def resolve_reference(query: str, yt=None) -> Track | None:
    """Resolve "Artista - Faixa" para a melhor faixa (filter="songs"). None se nada encontrado.

    Aceita também um link do YouTube/YouTube Music ou um videoId de 11 caracteres:
    nesse caso busca os metadados com `get_watch_playlist(videoId)` / `get_song`.
    Levanta SuggestError se a busca falhar (rede, bloqueio do YouTube).
    """
    query = (query or "").strip()
    if not query:
        return None
    yt = yt or get_ytmusic()
    video_id = extract_video_id(query)
    if video_id:
        track = _track_from_video_id(video_id, yt)
        if track is not None or "youtu" in query.lower():
            return track
        # Uma palavra de 11 letras pode não ser um id: segue para a busca por texto.
    for search_filter in ("songs", "videos"):
        try:
            results = yt.search(query, filter=search_filter, limit=10) or []
        except Exception as e:  # noqa: BLE001
            raise SuggestError(f"Não consegui buscar \"{query}\" no YouTube Music. "
                               f"Verifique a internet e tente de novo. ({type(e).__name__})") from e
        for item in results:
            track = track_from_item(item)
            if track is not None:
                return track
    return None


# ---------------------------------------------------------------------------
# 4.2 Sugestões
# ---------------------------------------------------------------------------
def _filter_settings(exclude_keywords, min_duration_s, max_duration_s):
    kws = config.EXCLUDE_KEYWORDS if exclude_keywords is None else exclude_keywords
    kws = [k for k in (util.normalize_text(k) for k in kws) if k]
    min_d = config.MIN_TRACK_DURATION_S if min_duration_s is None else min_duration_s
    max_d = config.MAX_TRACK_DURATION_S if max_duration_s is None else max_duration_s
    return kws, min_d, max_d


def _has_keyword(title: str, keywords: list[str]) -> bool:
    padded = f" {util.normalize_text(title)} "
    return any(f" {kw} " in padded for kw in keywords)


def _duration_ok(duration: float | None, min_d: float, max_d: float) -> bool:
    return duration is None or (min_d <= duration <= max_d)


def _rank(lists: dict[str, list[dict]], references: list[Track], exclude_keywords,
          min_duration_s, max_duration_s, source: str,
          item_to_track=track_from_item) -> list[Track]:
    """Junta listas cruas (rádios ou buscas), filtra e ordena por score desc, best_rank asc."""
    kws, min_d, max_d = _filter_settings(exclude_keywords, min_duration_s, max_duration_s)
    ref_ids = {r.video_id for r in references}
    kept: dict[str, Track] = {}
    voters: dict[str, set[str]] = {}
    rejected: set[str] = set()
    for list_key, items in lists.items():
        for pos, item in enumerate(items or [], start=1):
            track = item_to_track(item, source)
            if track is None:
                continue
            vid = track.video_id
            if vid in ref_ids or vid in rejected:
                continue
            if vid in kept:
                voters[vid].add(list_key)
                kept[vid].best_rank = min(kept[vid].best_rank, pos)
                continue
            if (any(same_song(track, ref) for ref in references)
                    or _has_keyword(track.title, kws)
                    or not _duration_ok(track.duration_s, min_d, max_d)):
                rejected.add(vid)
                continue
            track.best_rank = pos
            kept[vid] = track
            voters[vid] = {list_key}
    result = list(kept.values())
    for track in result:
        track.score = len(voters[track.video_id])
    result.sort(key=lambda t: (-t.score, t.best_rank))  # sort estável: empate mantém ordem das referências
    return result


def rank_suggestions(radios: dict[str, list[dict]], references: list[Track],
                     exclude_keywords: list[str] | None = None,
                     min_duration_s: float | None = None,
                     max_duration_s: float | None = None) -> list[Track]:
    """Função PURA (sem rede) que junta os rádios e ranqueia.

    `radios` mapeia videoId da referência -> itens crus do rádio (formato do ytmusicapi).
    Regras: remove duplicatas por videoId; remove as próprias referências (por videoId e
    também por título+artista normalizados, para pegar outro upload da mesma faixa);
    aplica filtro de palavras e de duração (faixas sem duração passam); `score` = quantas
    referências sugeriram; `best_rank` = melhor posição em qualquer rádio; ordena por
    score desc, depois best_rank asc. Devolve Tracks com source="ytmusic".
    """
    return _rank(radios, references, exclude_keywords, min_duration_s, max_duration_s,
                 source="ytmusic")


def _fetch_radio(yt, video_id: str) -> list[dict]:
    data = yt.get_watch_playlist(videoId=video_id, radio=True)
    return list((data or {}).get("tracks") or [])


def _merge_extra(tracks: list[Track], extra: list[Track]) -> int:
    """Acrescenta ao fim as faixas de `extra` que ainda não estão em `tracks`. Devolve quantas entraram."""
    seen = {t.video_id for t in tracks}
    added = 0
    for t in extra:
        if t.video_id in seen or any(same_song(t, other) for other in tracks):
            continue
        tracks.append(t)
        seen.add(t.video_id)
        added += 1
    return added


def get_suggestions(references: list[Track], exclude_keywords: list[str] | None = None,
                    yt=None, youtube_api_key: str | None = None) -> SuggestionResult:
    """Busca o rádio de cada referência e devolve a lista ranqueada completa.

    Se o rádio falhar ou devolver menos que N_SUGGESTIONS e houver `youtube_api_key`
    (ou YOUTUBE_API_KEY no .env), completa com `suggest_from_youtube_api` e registra aviso.
    Levanta SuggestError só se nenhuma fonte devolver nada.
    """
    if not references:
        raise SuggestError("Informe pelo menos uma referência no formato \"Artista - Faixa\".")
    warnings: list[str] = []
    radio_refs = [r for r in references if r.video_id and not r.is_local]
    if len(radio_refs) < len(references):
        warnings.append("Referências de arquivo local não têm rádio no YouTube Music e foram ignoradas na busca.")

    radios: dict[str, list[dict]] = {}
    radio_failed = False
    if radio_refs:
        try:
            yt = yt or get_ytmusic()
        except Exception as e:  # noqa: BLE001
            logger.warning("Falha ao criar o cliente do YouTube Music: %s", e)
            yt = None
            radio_failed = True
            warnings.append("Não consegui conectar ao YouTube Music.")
    if yt is not None and radio_refs:
        workers = max(1, min(MAX_PARALLEL_RADIOS, len(radio_refs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [(ref, pool.submit(_fetch_radio, yt, ref.video_id)) for ref in radio_refs]
            for ref, fut in futures:
                try:
                    radios[ref.video_id] = fut.result()
                except Exception as e:  # noqa: BLE001
                    logger.warning("Rádio falhou para %s: %s", ref.video_id, e)
                    warnings.append(f"Não consegui buscar o rádio de \"{ref.display_name}\" no YouTube Music.")
        radio_failed = not radios

    tracks = rank_suggestions(radios, references, exclude_keywords)
    source = "ytmusic"
    n_wanted = config.N_SUGGESTIONS
    if len(tracks) < n_wanted:
        api_key = youtube_api_key or config.get_secret("YOUTUBE_API_KEY")
        if api_key:
            try:
                extra = suggest_from_youtube_api(references, api_key, exclude_keywords=exclude_keywords)
            except SuggestError as e:
                warnings.append(f"Plano B (YouTube Data API) falhou: {e}")
            else:
                added = _merge_extra(tracks, extra)
                if radio_failed or not radios:
                    warnings.append(f"Usei a YouTube Data API (plano B): {added} sugestões.")
                else:
                    warnings.append(f"O rádio do YouTube Music trouxe poucas faixas; completei com "
                                    f"{added} sugestões da YouTube Data API (plano B).")
                if added and not any(t.source == "ytmusic" for t in tracks):
                    source = "youtube_api"
        elif tracks:
            warnings.append(f"Só encontrei {len(tracks)} sugestões. Para completar, adicione mais "
                            f"referências ou configure YOUTUBE_API_KEY no .env (plano B).")
    if not tracks:
        detail = " ".join(warnings)
        raise SuggestError(("Nenhuma sugestão encontrada. " + detail).strip()
                           + ("" if (youtube_api_key or config.get_secret("YOUTUBE_API_KEY"))
                              else " Configure YOUTUBE_API_KEY no .env para usar o plano B."))
    return SuggestionResult(tracks=tracks, source=source, warnings=warnings)


# ---------------------------------------------------------------------------
# Plano B: YouTube Data API v3
# ---------------------------------------------------------------------------
def parse_iso_duration(text: str | None) -> float | None:
    """Converte duração ISO 8601 ("PT3M12S", "PT1H2M") em segundos."""
    if not text:
        return None
    m = _ISO_DURATION_RE.match(text.strip())
    if not m:
        return None
    d, h, mi, s = (m.group(k) for k in ("d", "h", "m", "s"))
    return float(int(d or 0) * 86400 + int(h or 0) * 3600 + int(mi or 0) * 60) + float(s or 0)


def _api_error_message(resp) -> str:
    reasons: set[str] = set()
    try:
        err = (resp.json() or {}).get("error") or {}
    except ValueError:
        err = {}
    for e in err.get("errors") or []:
        if isinstance(e, dict) and e.get("reason"):
            reasons.add(str(e["reason"]))
    for d in err.get("details") or []:
        if isinstance(d, dict) and d.get("reason"):
            reasons.add(str(d["reason"]))
    if err.get("status"):
        reasons.add(str(err["status"]))
    text = str(err.get("message") or "").lower()
    if reasons & {"quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded", "RESOURCE_EXHAUSTED"}:
        return "a cota diária da YouTube Data API acabou (ela renova à meia-noite do horário do Pacífico)."
    if reasons & {"keyInvalid", "API_KEY_INVALID"} or "api key not valid" in text:
        return "a YOUTUBE_API_KEY do .env é inválida."
    if reasons & {"accessNotConfigured", "SERVICE_DISABLED"}:
        return "a YouTube Data API v3 não está ativada no projeto do Google Cloud dessa chave."
    return f"a YouTube Data API recusou a busca (HTTP {resp.status_code})."


def _youtube_get(endpoint: str, params: dict, api_key: str) -> dict:
    # A chave vai no cabeçalho, nunca na URL: assim não aparece em logs nem em mensagens de erro.
    try:
        resp = requests.get(f"{YOUTUBE_API_URL}/{endpoint}", params=params,
                            headers={"X-Goog-Api-Key": api_key}, timeout=HTTP_TIMEOUT_S)
    except requests.RequestException as e:
        raise SuggestError(f"sem conexão com a YouTube Data API ({type(e).__name__}).") from None
    if resp.status_code != 200:
        raise SuggestError(_api_error_message(resp))
    try:
        return resp.json() or {}
    except ValueError:
        raise SuggestError("resposta inválida da YouTube Data API.") from None


def _youtube_queries(references: list[Track]) -> list[str]:
    style = config.YOUTUBE_STYLE_QUERY
    queries: list[str] = []
    seen: set[str] = set()
    for ref in references:
        artist = ref.artists[0] if ref.artists else ""
        if not artist:
            parts = _DASH_RE.split(ref.title or "", maxsplit=1)
            artist = parts[0] if len(parts) == 2 else ""
        key = util.normalize_text(artist)
        if key and key not in seen:
            seen.add(key)
            queries.append(f"{artist} {style}".strip())
    if not queries:
        queries.append(style)
    return queries[: config.MAX_REFERENCES]


def _clean_channel(name: str) -> str:
    name = re.sub(r"\s+-\s+Topic$", "", name or "", flags=re.IGNORECASE)
    name = re.sub(r"VEVO$", "", name).strip()
    return name


def split_artist_title(title: str, channel: str = "") -> tuple[list[str], str]:
    """"MC X - Faixa (Clipe Oficial)" -> (["MC X"], "Faixa"); sem " - ", o artista é o canal."""
    title = html.unescape(title or "").strip()
    parts = _DASH_RE.split(title, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        artists, song = [parts[0].strip()], parts[1]
    else:
        channel = _clean_channel(html.unescape(channel or ""))
        artists, song = ([channel] if channel else []), title
    return artists, strip_title_noise(song) or song.strip()


def _youtube_item(search_item: dict, video: dict) -> dict:
    """Item da YouTube Data API no formato dos itens do ytmusicapi (para reaproveitar `_rank`)."""
    snippet = video.get("snippet") or search_item.get("snippet") or {}
    artists, title = split_artist_title(snippet.get("title", ""), snippet.get("channelTitle", ""))
    duration = parse_iso_duration((video.get("contentDetails") or {}).get("duration"))
    thumbs = [dict(t) for t in (snippet.get("thumbnails") or {}).values() if isinstance(t, dict)]
    item = {
        "videoId": video.get("id") or (search_item.get("id") or {}).get("videoId"),
        "title": title,
        "artists": [{"name": a} for a in artists],
        "thumbnail": thumbs,
    }
    if duration is not None:
        item["duration_seconds"] = duration if duration > 0 else 0.0
    return item


def _yt_api_item_to_track(item: dict, source: str) -> Track | None:
    track = track_from_item(item, source)
    if track is not None and item.get("duration_seconds") == 0.0:
        track.duration_s = 0.0  # transmissão ao vivo: duração zero cai no filtro de duração
    return track


def suggest_from_youtube_api(references: list[Track], api_key: str,
                             max_results: int = 25,
                             exclude_keywords: list[str] | None = None,
                             min_duration_s: float | None = None,
                             max_duration_s: float | None = None) -> list[Track]:
    """Plano B: YouTube Data API v3 (search.list + videos.list para duração).

    Busca pelos artistas das referências combinados com YOUTUBE_STYLE_QUERY.
    Devolve Tracks com source="youtube_api". Use `requests` (não precisa de SDK do Google).
    `max_results` vale por busca (máximo 50); aplica os mesmos filtros de `rank_suggestions`.
    """
    if not api_key:
        raise SuggestError("YOUTUBE_API_KEY não configurada no .env.")
    util.setup_ssl()
    per_query = max(1, min(50, int(max_results)))
    lists: dict[str, list[dict]] = {}
    for query in _youtube_queries(references):
        data = _youtube_get("search", {
            "part": "snippet", "q": query, "type": "video", "videoCategoryId": "10",
            "regionCode": "BR", "relevanceLanguage": "pt", "maxResults": per_query,
        }, api_key)
        items = []
        for it in data.get("items") or []:
            vid = (it.get("id") or {}).get("videoId")
            live = (it.get("snippet") or {}).get("liveBroadcastContent", "none")
            if vid and live in (None, "none"):
                items.append(it)
        lists[query] = items

    ids = list(dict.fromkeys(it["id"]["videoId"] for items in lists.values() for it in items))
    details: dict[str, dict] = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        data = _youtube_get("videos", {"part": "contentDetails,snippet", "id": ",".join(chunk),
                                       "maxResults": len(chunk)}, api_key)
        for v in data.get("items") or []:
            if v.get("id"):
                details[v["id"]] = v

    raw = {q: [_youtube_item(it, details[it["id"]["videoId"]]) for it in items
               if it["id"]["videoId"] in details]  # fora do videos.list = vídeo indisponível
           for q, items in lists.items()}
    return _rank(raw, references, exclude_keywords, min_duration_s, max_duration_s,
                 source="youtube_api", item_to_track=_yt_api_item_to_track)
