"""Importar playlist do Spotify (seção 4.10, opcional).

CONTRATO (implementar mantendo as assinaturas):

- OAuth com código de autorização (spotipy.SpotifyOAuth), conta do DJ SK.
  Redirect: SPOTIFY_REDIRECT_URI (http://127.0.0.1:8888/callback; o Spotify não aceita
  mais "localhost"). Escopos: SPOTIFY_SCOPES. Token em DATA_DIR/'.spotify_token'
  (spotipy.cache_handler.CacheFileHandler). open_browser=True: o spotipy sobe um servidor
  local na porta 8888 para receber o código.
- Modo de desenvolvimento do Spotify: só dá para ler itens de playlists que o usuário
  possui ou colabora; busca devolve no máximo 10; recomendações, audio features e previews
  NÃO existem. Usar só listagem de playlists e leitura de itens.
- spotipy 2.26 instalado: `current_user_playlists`, `playlist_items`, `current_user`.
  Itens podem vir com a faixa em "track" ou "item"; episódios/locais/None devem ser ignorados.
- Resolver cada faixa no YouTube Music: 1º tenta `search(isrc, filter="songs")`; aceita se
  a duração bate (± SPOTIFY_DURATION_TOLERANCE_S). Senão busca "Artista - Título" e escolhe
  o candidato com menor diferença de duração entre os que têm título parecido.

`get_client(open_browser=False)` só usa o token salvo: se não houver login válido, levanta
SpotifyError em vez de pedir a URL no terminal (o que travaria o Streamlit). Se o token salvo
foi criado com menos permissões do que SPOTIFY_SCOPES (ex.: antes de o painel criar playlists),
ele não serve mais: sem navegador levanta SpotifyError; com navegador apaga e pede login de novo.

Busca e exportação (acrescentadas depois):
- `search_tracks(query)` busca faixas SEM login do usuário (client credentials, cliente em cache).
- `find_on_spotify(client, track)` acha uma faixa do set no Spotify (ISRC, título, artista, duração).
- `export_set_to_spotify(sp, name, tracks)` cria a playlist na conta logada (POST /me/playlists) e
  adiciona as faixas na ordem do set (POST /playlists/{id}/items, em lotes de 100). O spotipy 2.26
  já usa esses endpoints novos em `current_user_playlist_create` e `playlist_add_items`.
"""
from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

import config
from modules import suggest, util
from modules.models import Track

logger = logging.getLogger(__name__)
# No nível DEBUG o spotipy registra o cabeçalho de autorização (chaves do app): nunca deixar sair no log.
logging.getLogger("spotipy").setLevel(logging.INFO)

TOKEN_FILENAME = ".spotify_token"
HTTP_TIMEOUT_S = 15
PAGE_LIMIT = 50                    # máximo aceito por current_user_playlists e playlist_items
MAX_PARALLEL_RESOLVE = 4           # buscas simultâneas no YouTube Music ao importar
SEARCH_LIMIT = 10
SPOTIFY_SEARCH_MAX = 10            # modo de desenvolvimento: a busca do Spotify devolve no máximo 10
MAX_PARALLEL_SPOTIFY = 3           # buscas simultâneas no Spotify ao exportar o set
ADD_BATCH_SIZE = 100               # máximo de faixas por chamada ao adicionar na playlist
RETRY_AFTER_MAX_S = 10             # espera máxima quando o Spotify pede "espere" (429), para não travar o painel
HTTP_RETRIES = 3
PLAYLIST_NAME_MAX = 100
PLAYLIST_DESCRIPTION_MAX = 300
DEFAULT_PLAYLIST_NAME = "Set DJ SK"


class SpotifyError(Exception):
    """Falha de configuração/login/leitura (mensagem em português)."""


@dataclass
class SpotifyTrack:
    title: str
    artists: list[str]
    duration_s: float
    isrc: str | None = None
    spotify_id: str | None = None

    @property
    def display_name(self) -> str:
        return f"{', '.join(self.artists) or 'Desconhecido'} - {self.title}"


# ---------------------------------------------------------------------------
# Configuração e login
# ---------------------------------------------------------------------------
def is_configured() -> bool:
    """True se SPOTIFY_CLIENT_ID e SPOTIFY_CLIENT_SECRET existem no .env."""
    return bool(config.get_secret("SPOTIFY_CLIENT_ID") and config.get_secret("SPOTIFY_CLIENT_SECRET"))


def token_path() -> Path:
    """Arquivo do token OAuth (lido de config.DATA_DIR no momento da chamada)."""
    return Path(config.DATA_DIR) / TOKEN_FILENAME


def logout() -> bool:
    """Apaga o token salvo (o próximo `get_client` pede login de novo). True se havia token."""
    try:
        token_path().unlink()
        return True
    except FileNotFoundError:
        return False


def _scope_set(scope: Any) -> set[str]:
    """"a b" / "a,b" / ["a", "b"] -> {"a", "b"}."""
    if not scope:
        return set()
    items = scope if isinstance(scope, (list, tuple, set)) else str(scope).replace(",", " ").split()
    return {str(s).strip() for s in items if str(s).strip()}


def token_has_scopes(token_info: dict | None, scopes: str | None = None) -> bool:
    """True se o token tem todas as permissões pedidas (padrão: config.SPOTIFY_SCOPES na hora da chamada)."""
    if not isinstance(token_info, dict):
        return False
    required = _scope_set(config.SPOTIFY_SCOPES if scopes is None else scopes)
    return required <= _scope_set(token_info.get("scope"))


def saved_login_state() -> str:
    """Situação do login salvo, sem acessar a rede.

    "none" = nunca entrou (ou saiu); "outdated" = o token salvo não tem as permissões novas
    (precisa entrar de novo); "saved" = há token com as permissões certas.
    """
    data = util.read_json(token_path(), default=None)
    if not isinstance(data, dict) or not data.get("access_token"):
        return "none"
    return "saved" if token_has_scopes(data) else "outdated"


def _http_session() -> requests.Session:
    """Sessão HTTP com novas tentativas, mas sem esperar horas quando o Spotify manda "espere" (429).

    O spotipy, por padrão, obedece o Retry-After sem limite (pode travar o painel por horas).
    """
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    kwargs = dict(total=HTTP_RETRIES, connect=None, read=False,
                  allowed_methods=frozenset(["GET", "POST", "PUT", "DELETE"]),
                  status=HTTP_RETRIES, backoff_factor=0.3, status_forcelist=(429, 500, 502, 503, 504))
    try:
        retry = Retry(retry_after_max=RETRY_AFTER_MAX_S, **kwargs)
    except TypeError:  # urllib3 antigo, sem retry_after_max
        retry = Retry(respect_retry_after_header=False, **kwargs)
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _friendly_error(e: BaseException, context: str = "") -> SpotifyError:
    """Converte exceções do spotipy/requests em SpotifyError com mensagem para o DJ.

    Nunca repassa o texto cru da exceção (pode conter URLs e detalhes de autenticação).
    """
    try:
        from spotipy.exceptions import SpotifyException, SpotifyOauthError
    except ImportError:  # sem spotipy, isinstance(e, ()) é sempre False
        SpotifyException = SpotifyOauthError = ()  # type: ignore[assignment]

    if isinstance(e, SpotifyError):
        return e
    if isinstance(e, SpotifyOauthError):
        logger.info("Erro de OAuth do Spotify: %s", getattr(e, "error", None))
        code = getattr(e, "error", None)
        if code == "invalid_client":
            return SpotifyError("O Spotify recusou o app: confira SPOTIFY_CLIENT_ID e SPOTIFY_CLIENT_SECRET no .env.")
        if code == "access_denied":
            return SpotifyError("O login no Spotify foi cancelado.")
        if code == "invalid_grant":
            return SpotifyError("O login do Spotify expirou. Saia e entre de novo.")
        if context == "search":
            return SpotifyError("Não consegui conectar o painel ao Spotify para buscar. Confira "
                                "SPOTIFY_CLIENT_ID e SPOTIFY_CLIENT_SECRET no .env.")
        return SpotifyError("O login no Spotify falhou. Confira as chaves no .env e se o redirect "
                            f"{config.SPOTIFY_REDIRECT_URI} está cadastrado no painel do app.")
    if isinstance(e, SpotifyException):
        status = getattr(e, "http_status", None)
        logger.info("Spotify respondeu HTTP %s (%s) em %s", status, getattr(e, "reason", None), context)
        if status == 401:
            if context == "search":
                return SpotifyError("O Spotify recusou a busca: confira SPOTIFY_CLIENT_ID e "
                                    "SPOTIFY_CLIENT_SECRET no .env.")
            return SpotifyError("O login do Spotify expirou ou foi revogado. Saia e entre de novo.")
        if status == 403:
            if context == "playlist":
                return SpotifyError("O Spotify não deixou ler esta playlist: no modo de desenvolvimento só dá "
                                    "para ler playlists que você criou ou em que colabora.")
            if context in ("create", "add"):
                return SpotifyError("O Spotify não deixou criar a playlist. Saia e entre de novo na conta do "
                                    "Spotify para autorizar a criação de playlists.")
            return SpotifyError("O Spotify recusou o acesso. No modo de desenvolvimento, a conta precisa estar "
                                "cadastrada como usuária no painel do app (máximo de 5) e o dono do app precisa ser Premium.")
        if status == 404:
            return SpotifyError("Playlist não encontrada no Spotify (confira o link ou se ela ainda existe)."
                                if context in ("playlist", "add") else "O Spotify não encontrou o recurso pedido.")
        if status == 429:
            return SpotifyError("O Spotify limitou as requisições por agora. Espere um pouco e tente de novo.")
        if status == 400 and context == "playlist":
            return SpotifyError("Link ou id de playlist inválido.")
        if status == 400 and context == "search":
            return SpotifyError("O Spotify não entendeu a busca. Tente outras palavras.")
        if status == 400 and context == "create":
            return SpotifyError("O Spotify recusou o nome ou a descrição da playlist. Tente um nome mais simples.")
        return SpotifyError(f"O Spotify devolveu um erro (HTTP {status}). Tente de novo em instantes.")
    if isinstance(e, requests.RequestException):
        return SpotifyError("Sem conexão com o Spotify. Verifique a internet e tente de novo.")
    if isinstance(e, OSError):
        if context != "login":
            return SpotifyError("Sem conexão com o Spotify. Verifique a internet e tente de novo.")
        return SpotifyError("Não consegui abrir o servidor local de login (porta 8888 ocupada?). "
                            "Feche outros programas que usem essa porta e tente de novo.")
    logger.warning("Erro inesperado no Spotify (%s): %s", context, type(e).__name__)
    return SpotifyError(f"Erro inesperado ao falar com o Spotify ({type(e).__name__}).")


def get_client(open_browser: bool = True):
    """spotipy.Spotify autenticado. Levanta SpotifyError se não configurado.

    Token salvo com menos permissões do que SPOTIFY_SCOPES não serve (o spotipy também o ignora):
    sem navegador levanta SpotifyError pedindo para entrar de novo; com navegador apaga o token
    antigo e abre o login, que pede as permissões novas.
    """
    if not is_configured():
        raise SpotifyError("Spotify não configurado: preencha SPOTIFY_CLIENT_ID e SPOTIFY_CLIENT_SECRET no .env.")
    try:
        import spotipy
        from spotipy.cache_handler import CacheFileHandler
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError:
        raise SpotifyError("O pacote spotipy não está instalado (pip install spotipy).") from None
    util.setup_ssl()
    path = token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = CacheFileHandler(cache_path=str(path))
    cached = handler.get_cached_token()
    if cached is not None and not token_has_scopes(cached):
        logger.info("Token salvo do Spotify sem as permissões novas: é preciso entrar de novo.")
        if not open_browser:
            raise SpotifyError("O painel agora também cria playlists no Spotify e precisa de uma nova "
                               "autorização. Clique em entrar para fazer login de novo.")
        logout()
    auth = SpotifyOAuth(
        client_id=config.get_secret("SPOTIFY_CLIENT_ID"),
        client_secret=config.get_secret("SPOTIFY_CLIENT_SECRET"),
        redirect_uri=config.SPOTIFY_REDIRECT_URI,
        scope=config.SPOTIFY_SCOPES,
        cache_handler=handler,
        open_browser=open_browser,
        requests_timeout=HTTP_TIMEOUT_S,
    )
    try:
        if not open_browser and auth.validate_token(handler.get_cached_token()) is None:
            raise SpotifyError("Você ainda não entrou na conta do Spotify. Clique em entrar para fazer login no navegador.")
        sp = spotipy.Spotify(auth_manager=auth, requests_timeout=HTTP_TIMEOUT_S,
                             requests_session=_http_session())
        sp.current_user()  # força o login agora (abre o navegador se não houver token válido)
    except Exception as e:  # noqa: BLE001 - tudo vira mensagem amigável
        raise _friendly_error(e, "login") from None
    return sp


# ---------------------------------------------------------------------------
# Leitura de playlists
# ---------------------------------------------------------------------------
def _iter_pages(sp, first: dict | None):
    page = first
    while page:
        for item in page.get("items") or []:
            yield item
        page = sp.next(page) if page.get("next") else None


def list_playlists(sp) -> list[dict]:
    """[{"id", "name", "owner", "owner_id", "tracks_total", "collaborative", "readable",
    "image_url", "url"}] (image_url = capa; url = link da playlist; podem ser None).

    readable = o usuário é dono ou a playlist é colaborativa (limite do modo dev).
    Pagina até o fim.
    """
    try:
        user_id = (sp.current_user() or {}).get("id")
        out: list[dict] = []
        for p in _iter_pages(sp, sp.current_user_playlists(limit=PAGE_LIMIT)):
            if not isinstance(p, dict) or not p.get("id"):
                continue
            owner = p.get("owner") or {}
            # A API nova chama de "items" o que antes era "tracks" ({"href", "total"}).
            counts = p.get("items") if isinstance(p.get("items"), dict) else p.get("tracks")
            collaborative = bool(p.get("collaborative"))
            out.append({
                "id": p["id"],
                "name": p.get("name") or "(sem nome)",
                "owner": owner.get("display_name") or owner.get("id") or "",
                "owner_id": owner.get("id"),
                "tracks_total": int((counts or {}).get("total") or 0),
                "collaborative": collaborative,
                "readable": collaborative or (user_id is not None and owner.get("id") == user_id),
                "image_url": suggest.best_thumbnail(p.get("images")),
                "url": (p.get("external_urls") or {}).get("spotify"),
            })
    except Exception as e:  # noqa: BLE001
        raise _friendly_error(e, "playlists") from None
    return out


def _parse_item(item: Any) -> SpotifyTrack | None:
    """Item de playlist -> SpotifyTrack; None para vazio, episódio, arquivo local ou sem nome."""
    if not isinstance(item, dict) or item.get("is_local"):
        return None
    track = item.get("item") if isinstance(item.get("item"), dict) else item.get("track")
    if not isinstance(track, dict) or track.get("is_local"):
        return None
    if track.get("type", "track") != "track" or track.get("episode"):
        return None
    title = (track.get("name") or "").strip()
    if not title:
        return None
    artists = [a["name"] for a in track.get("artists") or [] if isinstance(a, dict) and a.get("name")]
    isrc = ((track.get("external_ids") or {}).get("isrc") or "").strip().upper() or None
    return SpotifyTrack(
        title=title,
        artists=artists,
        duration_s=float(track.get("duration_ms") or 0) / 1000.0,
        isrc=isrc,
        spotify_id=track.get("id"),
    )


def playlist_tracks(sp, playlist_id: str) -> list[SpotifyTrack]:
    """Lê todos os itens (paginando). Levanta SpotifyError amigável em 403/404."""
    try:
        first = sp.playlist_items(playlist_id, limit=PAGE_LIMIT, additional_types=("track",))
        return [t for t in (_parse_item(it) for it in _iter_pages(sp, first)) if t is not None]
    except Exception as e:  # noqa: BLE001
        raise _friendly_error(e, "playlist") from None


# ---------------------------------------------------------------------------
# Resolução no YouTube Music
# ---------------------------------------------------------------------------
def _diff(candidate: Track, sp_track: SpotifyTrack) -> float:
    if candidate.duration_s is None or not sp_track.duration_s:
        return float("inf")
    return abs(candidate.duration_s - sp_track.duration_s)


def _search(yt, query: str, search_filter: str) -> list[Track]:
    results = yt.search(query, filter=search_filter, limit=SEARCH_LIMIT) or []
    return [t for t in (suggest.track_from_item(r) for r in results) if t is not None]


def _by_isrc(sp_track: SpotifyTrack, yt, tol: float) -> Track | None:
    candidates = [c for c in _search(yt, sp_track.isrc, "songs") if _diff(c, sp_track) <= tol]
    if not candidates:
        return None
    # Entre os que batem na duração, prefere o de título parecido e depois o mais próximo.
    return min(candidates, key=lambda c: (not suggest.titles_similar(c.title, sp_track.title),
                                          _diff(c, sp_track)))


def _by_text(sp_track: SpotifyTrack, yt, tol: float) -> Track | None:
    artist = sp_track.artists[0] if sp_track.artists else ""
    title = suggest.strip_title_noise(sp_track.title) or sp_track.title
    query = f"{artist} - {title}" if artist else title
    pool: list[Track] = []
    best: Track | None = None
    for search_filter in ("songs", "videos"):
        pool += [c for c in _search(yt, query, search_filter) if suggest.titles_similar(c.title, sp_track.title)]
        if not pool:
            continue
        # Se algum candidato tem artista em comum, descarta os de outro artista (mesmo nome, outra música).
        same_artist = [c for c in pool if suggest.artists_overlap(
            c.title, c.artists, sp_track.title, sp_track.artists) is not False]
        best = min(same_artist or pool, key=lambda c: _diff(c, sp_track))
        if _diff(best, sp_track) <= tol:
            break  # bom o bastante: não precisa buscar entre os vídeos
    return best


def resolve_on_ytmusic(sp_track: SpotifyTrack, yt=None) -> Track | None:
    """Encontra o upload certo no YouTube Music (ISRC + duração). Track com source="spotify"
    e isrc preenchido, ou None."""
    yt = yt or suggest.get_ytmusic()
    tol = float(config.SPOTIFY_DURATION_TOLERANCE_S)
    found = _by_isrc(sp_track, yt, tol) if sp_track.isrc else None
    if found is None:
        found = _by_text(sp_track, yt, tol)
    if found is None:
        return None
    # Título e artistas do Spotify (mais limpos); id, duração e capa do upload do YouTube.
    return Track(
        video_id=found.video_id,
        title=sp_track.title,
        artists=list(sp_track.artists) or list(found.artists),
        duration_s=found.duration_s if found.duration_s is not None else (sp_track.duration_s or None),
        album=found.album,
        thumbnail_url=found.thumbnail_url,
        source="spotify",
        isrc=sp_track.isrc,
    )


def import_playlist(sp, playlist_id: str, yt=None, limit: int | None = None,
                    progress: Callable[[int, int], None] | None = None) -> tuple[list[Track], list[str]]:
    """(faixas resolvidas, nomes que não foram encontrados).

    `progress(feitas, total)` é chamado a cada faixa resolvida (opcional, para a barra da interface).
    Faixas repetidas na playlist (mesmo upload do YouTube) entram uma vez só.
    """
    sp_tracks = playlist_tracks(sp, playlist_id)
    if limit is not None:
        sp_tracks = sp_tracks[:max(0, int(limit))]
    if not sp_tracks:
        return [], []
    yt = yt or suggest.get_ytmusic()
    total = len(sp_tracks)
    done = 0
    errors = 0

    def _resolve(st: SpotifyTrack):
        try:
            return resolve_on_ytmusic(st, yt=yt), False
        except Exception as e:  # noqa: BLE001 - uma faixa com erro não derruba a importação
            logger.warning("Falha ao resolver \"%s\" no YouTube Music: %s", st.display_name, type(e).__name__)
            return None, True

    results: list[Track | None] = []
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_RESOLVE, total)) as pool:
        for track, failed in pool.map(_resolve, sp_tracks):  # map preserva a ordem da playlist
            results.append(track)
            errors += failed
            done += 1
            if progress is not None:
                progress(done, total)
    if errors == total:
        raise SpotifyError("Não consegui buscar as faixas no YouTube Music. Verifique a internet e tente de novo.")

    tracks: list[Track] = []
    not_found: list[str] = []
    seen: set[str] = set()
    for st, track in zip(sp_tracks, results):
        if track is None:
            not_found.append(st.display_name)
        elif track.video_id not in seen:
            seen.add(track.video_id)
            tracks.append(track)
    return tracks, not_found


# ---------------------------------------------------------------------------
# Busca sem login (client credentials)
# ---------------------------------------------------------------------------
_search_lock = threading.Lock()
_search_client = None
_search_client_key: str | None = None


def _reset_search_client() -> None:
    """Descarta o cliente de busca em cache (usado nos testes e se as chaves do .env mudarem)."""
    global _search_client, _search_client_key
    with _search_lock:
        _search_client = None
        _search_client_key = None


def _get_search_client():
    """spotipy.Spotify com as chaves do app (sem conta de usuário), reutilizado entre chamadas.

    O token fica só na memória (MemoryCacheHandler): o padrão do spotipy gravaria um arquivo
    ".cache" na pasta em que o painel foi aberto.
    """
    global _search_client, _search_client_key
    if not is_configured():
        raise SpotifyError("Spotify não configurado: preencha SPOTIFY_CLIENT_ID e SPOTIFY_CLIENT_SECRET no .env.")
    client_id = config.get_secret("SPOTIFY_CLIENT_ID")
    with _search_lock:
        if _search_client is not None and _search_client_key == client_id:
            return _search_client
        try:
            import spotipy
            from spotipy.cache_handler import MemoryCacheHandler
            from spotipy.oauth2 import SpotifyClientCredentials
        except ImportError:
            raise SpotifyError("O pacote spotipy não está instalado (pip install spotipy).") from None
        util.setup_ssl()
        try:
            auth = SpotifyClientCredentials(
                client_id=client_id,
                client_secret=config.get_secret("SPOTIFY_CLIENT_SECRET"),
                requests_timeout=HTTP_TIMEOUT_S,
                cache_handler=MemoryCacheHandler(),
            )
            client = spotipy.Spotify(auth_manager=auth, requests_timeout=HTTP_TIMEOUT_S,
                                     requests_session=_http_session())
        except Exception as e:  # noqa: BLE001
            raise _friendly_error(e, "search") from None
        _search_client, _search_client_key = client, client_id
        return client


def _track_dict(raw: Any) -> dict | None:
    """Objeto "track" da API do Spotify -> dicionário simples. None se não for uma faixa utilizável."""
    if not isinstance(raw, dict) or raw.get("is_local") or raw.get("type", "track") != "track":
        return None
    title = (raw.get("name") or "").strip()
    spotify_id = raw.get("id")
    if not title or not spotify_id:
        return None
    album = raw.get("album") if isinstance(raw.get("album"), dict) else {}
    isrc = ((raw.get("external_ids") or {}).get("isrc") or "").strip().upper() or None
    ms = raw.get("duration_ms")
    return {
        "title": title,
        "artists": [a["name"] for a in raw.get("artists") or [] if isinstance(a, dict) and a.get("name")],
        "duration_s": float(ms) / 1000.0 if isinstance(ms, (int, float)) and ms > 0 else None,
        "isrc": isrc,
        "spotify_id": spotify_id,
        "cover_url": suggest.best_thumbnail(album.get("images")),
        "spotify_url": (raw.get("external_urls") or {}).get("spotify")
        or f"https://open.spotify.com/track/{spotify_id}",
        "album": (album.get("name") or "").strip() or None,
        "uri": raw.get("uri") or f"spotify:track:{spotify_id}",
    }


def _parse_search(result: Any) -> list[dict]:
    """Resposta de `search(type="track")` -> lista de dicionários, sem repetidos."""
    tracks = result.get("tracks") if isinstance(result, dict) else None
    items = tracks.get("items") if isinstance(tracks, dict) else None
    out: list[dict] = []
    seen: set[str] = set()
    for raw in items or []:
        d = _track_dict(raw)
        if d is not None and d["spotify_id"] not in seen:
            seen.add(d["spotify_id"])
            out.append(d)
    return out


_QUERY_SYNTAX_RE = re.compile(r"[\"“”:]")


def _clean_query(text: str) -> str:
    """Tira aspas e dois-pontos (sintaxe de filtro da busca) e espaços repetidos."""
    return " ".join(_QUERY_SYNTAX_RE.sub(" ", text or "").split())


def search_tracks(query: str, limit: int = 10) -> list[dict]:
    """Busca faixas no Spotify sem login do usuário.

    Devolve [{"title", "artists": list[str], "duration_s", "isrc", "spotify_id", "cover_url",
    "spotify_url", "album", "uri"}] (no máximo 10: limite do modo de desenvolvimento).
    `duration_s`, `isrc`, `cover_url` e `album` podem ser None. Erros viram SpotifyError amigável.
    """
    q = " ".join((query or "").split())
    if not q:
        return []
    limit = max(1, min(int(limit or SPOTIFY_SEARCH_MAX), SPOTIFY_SEARCH_MAX))
    client = _get_search_client()
    try:
        result = client.search(q=q, limit=limit, type="track")
    except Exception as e:  # noqa: BLE001
        raise _friendly_error(e, "search") from None
    return _parse_search(result)[:limit]


def to_spotify_track(d: dict) -> SpotifyTrack:
    """Resultado de `search_tracks` (ou objeto "track" cru da API) -> SpotifyTrack.

    Serve para usar um resultado da busca como referência com `resolve_on_ytmusic`.
    Levanta ValueError se o dicionário não tem título.
    """
    if not isinstance(d, dict):
        raise ValueError("Faixa do Spotify inválida.")
    if "title" not in d and "name" in d:  # objeto cru da API
        parsed = _track_dict(d)
        if parsed is None:
            raise ValueError("Faixa do Spotify inválida.")
        d = parsed
    title = str(d.get("title") or "").strip()
    if not title:
        raise ValueError("Faixa do Spotify sem título.")
    duration = d.get("duration_s")
    isrc = str(d.get("isrc") or "").strip().upper() or None
    return SpotifyTrack(
        title=title,
        artists=[str(a).strip() for a in d.get("artists") or [] if str(a or "").strip()],
        duration_s=float(duration) if isinstance(duration, (int, float)) and duration > 0 else 0.0,
        isrc=isrc,
        spotify_id=d.get("spotify_id") or None,
    )


# ---------------------------------------------------------------------------
# Conta logada e exportação do set
# ---------------------------------------------------------------------------
def current_user_info(sp) -> dict:
    """{"id", "name", "image_url", "product"} da conta logada (product: "premium", "free"... ou None)."""
    try:
        me = sp.current_user() or {}
    except Exception as e:  # noqa: BLE001
        raise _friendly_error(e, "user") from None
    return {
        "id": me.get("id"),
        "name": me.get("display_name") or me.get("id") or "",
        "image_url": suggest.best_thumbnail(me.get("images")),
        "product": me.get("product"),
    }


def _search_pairs(track: Track) -> list[tuple[str, str]]:
    """Pares (artista, título) para buscar uma faixa do set.

    Em vídeos do YouTube o título costuma ser "MC X - Faixa (Clipe Oficial)" e o "artista" é o canal:
    nesse caso também tenta o artista e o título que estão no próprio título.
    """
    artists = [a.strip() for a in track.artists if a and a.strip()]
    title_clean = suggest.strip_title_noise(track.title) or (track.title or "").strip()
    dash_artists, dash_title = suggest.split_artist_title(track.title or "")
    pairs: list[tuple[str, str]] = []
    if dash_artists:
        left = dash_artists[0]
        if not artists or suggest.artists_overlap("", [left], "", artists):
            pairs.append((artists[0] if artists else left, dash_title))
        else:  # o lado esquerdo pode ser parte do nome da música: tenta as duas leituras
            pairs += [(artists[0], title_clean), (left, dash_title)]
    else:
        pairs.append((artists[0] if artists else "", title_clean))
    out: list[tuple[str, str]] = []
    for artist, title in pairs:
        pair = (_clean_query(artist), _clean_query(title))
        if pair[1] and pair not in out:
            out.append(pair)
    return out


def _duration_diff(candidate: dict, track: Track) -> float | None:
    d = candidate.get("duration_s")
    if not d or not track.duration_s:
        return None
    return abs(float(d) - float(track.duration_s))


def _pick_match(candidates: list[dict], track: Track, tol: float, by_isrc: bool = False) -> dict | None:
    """Melhor candidato: título parecido (ou ISRC igual) e duração dentro da tolerância.

    Sem duração dos dois lados, exige artista em comum. Entre os aceitos, prefere quem tem artista
    em comum, depois a menor diferença de duração, depois a ordem da busca.
    """
    accepted = []
    for rank, c in enumerate(candidates):
        if not by_isrc and not suggest.titles_similar(c["title"], track.title):
            continue
        diff = _duration_diff(c, track)
        if diff is not None and diff > tol:
            continue
        overlap = suggest.artists_overlap(c["title"], c["artists"], track.title, track.artists)
        if diff is None and overlap is False:
            continue
        accepted.append((overlap is False, diff if diff is not None else float("inf"), rank))
    if not accepted:
        return None
    return candidates[min(accepted)[2]]


def _find_raw(client, track: Track) -> dict | None:
    """Como `find_on_spotify`, mas deixa passar as exceções do spotipy (a exportação decide o que fazer)."""
    tol = float(config.SPOTIFY_DURATION_TOLERANCE_S)
    queries: list[tuple[str, bool]] = []
    isrc = _clean_query(track.isrc or "").replace(" ", "")
    if isrc:
        queries.append((f"isrc:{isrc}", True))
    for artist, title in _search_pairs(track):
        if artist:
            queries += [(f"track:{title} artist:{artist}", False), (f"{artist} {title}", False)]
        else:
            queries.append((title, False))
    tried: set[str] = set()
    for q, by_isrc in queries:
        if q in tried:
            continue
        tried.add(q)
        candidates = _parse_search(client.search(q=q, limit=SPOTIFY_SEARCH_MAX, type="track"))
        best = _pick_match(candidates, track, tol, by_isrc=by_isrc)
        if best is not None:
            return best
    return None


def find_on_spotify(client, track: Track) -> dict | None:
    """Acha a faixa do set no Spotify. Devolve o mesmo dicionário de `search_tracks` ou None.

    Ordem: ISRC (se houver); "track:<título limpo> artist:<1º artista>"; "<artista> <título>".
    Só aceita título parecido e, quando os dois lados têm duração, diferença de até
    SPOTIFY_DURATION_TOLERANCE_S. Erros de rede/login viram SpotifyError.
    """
    try:
        return _find_raw(client, track)
    except Exception as e:  # noqa: BLE001
        raise _friendly_error(e, "find") from None


def _is_skippable(e: BaseException) -> bool:
    """Erro de uma busca só (consulta estranha): a faixa conta como não encontrada e a exportação segue."""
    try:
        from spotipy.exceptions import SpotifyException
    except ImportError:
        return False
    return isinstance(e, SpotifyException) and getattr(e, "http_status", None) in (400, 404)


def _clean_playlist_text(text: str, max_len: int) -> str:
    """Uma linha só, sem < >, cortada no limite do Spotify."""
    return " ".join(re.sub(r"[<>]", " ", text or "").split())[:max_len].strip()


def export_set_to_spotify(sp, name: str, tracks: list[Track], description: str = "",
                          public: bool = False,
                          progress: Callable[[int, int], None] | None = None) -> dict:
    """Cria uma playlist na conta logada com as faixas do set, na ordem do set.

    Devolve {"playlist_id", "playlist_url", "added": int, "missing": list[str]} (missing = nomes das
    faixas do set que não foram achadas no Spotify). `progress(feitas, total)` é chamado a cada faixa
    procurada (na thread de quem chamou). Repetidas (mesma faixa do Spotify) entram uma vez só.
    Levanta SpotifyError se o set está vazio, se nenhuma faixa foi achada (aí NÃO cria a playlist)
    ou se o Spotify recusar a criação.
    """
    tracks = list(tracks or [])
    if not tracks:
        raise SpotifyError("O set está vazio: não há músicas para mandar ao Spotify.")
    total = len(tracks)

    def _find(track: Track) -> dict | None:
        try:
            return _find_raw(sp, track)
        except Exception as e:  # noqa: BLE001
            if _is_skippable(e):
                logger.info("Busca no Spotify recusada para \"%s\": conta como não encontrada.", track.display_name)
                return None
            raise

    found: list[dict | None] = []
    pool = ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_SPOTIFY, total))
    try:
        for done, hit in enumerate(pool.map(_find, tracks), start=1):  # map preserva a ordem do set
            found.append(hit)
            if progress is not None:
                progress(done, total)
    except Exception as e:  # noqa: BLE001 - login expirado, sem internet, limite: para tudo
        raise _friendly_error(e, "find") from None
    finally:  # se parou no meio (erro ou o Streamlit interrompeu), não deixa buscas pendentes rodando
        pool.shutdown(wait=False, cancel_futures=True)

    uris: list[str] = []
    missing: list[str] = []
    seen: set[str] = set()
    for track, hit in zip(tracks, found):
        if hit is None:
            missing.append(track.display_name)
        elif hit["uri"] not in seen:
            seen.add(hit["uri"])
            uris.append(hit["uri"])
    if not uris:
        raise SpotifyError("Nenhuma música do set foi encontrada no Spotify, então a playlist não foi criada.")

    playlist_name = _clean_playlist_text(name, PLAYLIST_NAME_MAX) or DEFAULT_PLAYLIST_NAME
    try:
        created = sp.current_user_playlist_create(
            playlist_name, public=bool(public), collaborative=False,
            description=_clean_playlist_text(description, PLAYLIST_DESCRIPTION_MAX),
        ) or {}
    except Exception as e:  # noqa: BLE001
        raise _friendly_error(e, "create") from None
    playlist_id = created.get("id")
    if not playlist_id:
        raise SpotifyError("O Spotify não confirmou a criação da playlist. Tente de novo.")
    playlist_url = ((created.get("external_urls") or {}).get("spotify")
                    or f"https://open.spotify.com/playlist/{playlist_id}")

    added = 0
    for start in range(0, len(uris), ADD_BATCH_SIZE):
        batch = uris[start:start + ADD_BATCH_SIZE]
        try:
            sp.playlist_add_items(playlist_id, batch)
        except Exception as e:  # noqa: BLE001
            err = _friendly_error(e, "add")
            raise SpotifyError(f"A playlist \"{playlist_name}\" foi criada no Spotify, mas só {added} de "
                               f"{len(uris)} músicas entraram. {err}") from None
        added += len(batch)
    logger.info("Playlist exportada para o Spotify: %d faixas, %d não encontradas.", added, len(missing))
    return {"playlist_id": playlist_id, "playlist_url": playlist_url, "added": added, "missing": missing}
