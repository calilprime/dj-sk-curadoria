"""Testes da importação do Spotify (modules/spotify_import.py), 100% com fakes (sem rede)."""
from __future__ import annotations

import json
import logging
import threading
import time

import pytest
import requests
import spotipy
import spotipy.oauth2
from spotipy.cache_handler import MemoryCacheHandler
from spotipy.exceptions import SpotifyException, SpotifyOauthError

import config
from modules import spotify_import as si
from modules import suggest, util
from modules.models import Track
from modules.spotify_import import SpotifyTrack

SECRET = "segredo-de-teste-123"


@pytest.fixture
def configured(monkeypatch):
    secrets = {"SPOTIFY_CLIENT_ID": "id-de-teste", "SPOTIFY_CLIENT_SECRET": SECRET}
    monkeypatch.setattr(config, "get_secret", lambda name: secrets.get(name))
    monkeypatch.setattr(util, "setup_ssl", lambda: None)
    return secrets


@pytest.fixture
def not_configured(monkeypatch):
    monkeypatch.setattr(config, "get_secret", lambda name: None)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def sp_track(name: str, artists: list[str], ms: int, isrc: str | None = None, tid: str = "t1",
             type_: str = "track", is_local: bool = False) -> dict:
    t = {"name": name, "artists": [{"name": a, "id": f"ar_{a}"} for a in artists], "duration_ms": ms,
         "id": tid, "type": type_, "is_local": is_local, "uri": f"spotify:{type_}:{tid}"}
    if isrc is not None:
        t["external_ids"] = {"isrc": isrc}
    return t


class FakeSpotify:
    """spotipy.Spotify falso com paginação via `next`."""

    def __init__(self, playlist_pages: list[dict] | None = None, item_pages: list[dict] | None = None,
                 error: Exception | None = None, user_id: str = "djsk"):
        self.playlist_pages = playlist_pages or [{"items": [], "next": None}]
        self.item_pages = item_pages or [{"items": [], "next": None}]
        self.error = error
        self.user_id = user_id
        self.calls: list[tuple] = []

    def current_user(self):
        self.calls.append(("me",))
        return {"id": self.user_id, "display_name": "DJ SK"}

    def current_user_playlists(self, limit=50, offset=0):
        self.calls.append(("playlists", limit, offset))
        if self.error:
            raise self.error
        return self.playlist_pages[0]

    def playlist_items(self, playlist_id, fields=None, limit=50, offset=0, market=None,
                       additional_types=("track", "episode")):
        self.calls.append(("items", playlist_id, limit, tuple(additional_types)))
        if self.error:
            raise self.error
        return self.item_pages[0]

    def next(self, page):
        self.calls.append(("next", page["next"]))
        pages = self.playlist_pages if page["next"].startswith("pl") else self.item_pages
        return pages[int(page["next"].split(":")[1])]


def yt_song(vid: str, title: str, artists: list[str], seconds: float | None) -> dict:
    item = {"resultType": "song", "videoId": vid, "title": title,
            "artists": [{"name": a, "id": None} for a in artists],
            "album": {"name": title, "id": "MPREb"},
            "thumbnails": [{"url": f"https://img/{vid}.jpg", "width": 120, "height": 120}]}
    if seconds is not None:
        item["duration_seconds"] = seconds
    return item


class FakeYT:
    def __init__(self, results: dict | None = None, fail: bool = False):
        self.results = results or {}
        self.fail = fail
        self.calls: list[tuple] = []
        self.lock = threading.Lock()

    def search(self, query, filter=None, limit=20, **kw):
        with self.lock:
            self.calls.append((query, filter))
        if self.fail:
            raise ConnectionError("rede caiu")
        return self.results.get((query, filter), [])


# ---------------------------------------------------------------------------
# Configuração e login
# ---------------------------------------------------------------------------
def test_is_configured(monkeypatch, configured):
    assert si.is_configured()
    monkeypatch.setattr(config, "get_secret", lambda name: "x" if name == "SPOTIFY_CLIENT_ID" else None)
    assert not si.is_configured()


def test_get_client_not_configured(not_configured):
    with pytest.raises(si.SpotifyError, match="SPOTIFY_CLIENT_ID"):
        si.get_client()


def test_get_client_builds_oauth(monkeypatch, configured, tmp_dirs):
    captured = {}

    class FakeOAuth:
        def __init__(self, **kw):
            captured["oauth"] = kw

    class FakeClient:
        def __init__(self, **kw):
            captured["client"] = kw

        def current_user(self):
            captured["me"] = True
            return {"id": "djsk"}

    monkeypatch.setattr(spotipy.oauth2, "SpotifyOAuth", FakeOAuth)
    monkeypatch.setattr(spotipy, "Spotify", FakeClient)
    sp = si.get_client()
    assert isinstance(sp, FakeClient)
    kw = captured["oauth"]
    assert kw["client_id"] == "id-de-teste" and kw["client_secret"] == SECRET
    assert kw["redirect_uri"] == config.SPOTIFY_REDIRECT_URI == "http://127.0.0.1:8888/callback"
    assert kw["scope"] == config.SPOTIFY_SCOPES
    assert kw["open_browser"] is True
    assert kw["cache_handler"].cache_path == str(tmp_dirs["DATA_DIR"] / ".spotify_token")
    assert captured["client"]["auth_manager"] is not None
    assert captured["me"]  # login feito já na criação


def test_get_client_without_browser_needs_saved_token(configured, tmp_dirs):
    # SpotifyOAuth real: sem token salvo, não acessa a rede nem pede input().
    with pytest.raises(si.SpotifyError, match="entrou na conta"):
        si.get_client(open_browser=False)


def test_get_client_uses_saved_token(monkeypatch, configured, tmp_dirs):
    import json
    import time

    token = {"access_token": "a", "token_type": "Bearer", "expires_in": 3600, "refresh_token": "r",
             "scope": config.SPOTIFY_SCOPES, "expires_at": int(time.time()) + 3600}
    (tmp_dirs["DATA_DIR"] / ".spotify_token").write_text(json.dumps(token), encoding="utf-8")
    monkeypatch.setattr(spotipy.Spotify, "current_user", lambda self: {"id": "djsk"})
    sp = si.get_client(open_browser=False)
    assert isinstance(sp, spotipy.Spotify)
    assert si.logout() is True
    assert not (tmp_dirs["DATA_DIR"] / ".spotify_token").exists()
    assert si.logout() is False


@pytest.mark.parametrize("error, expected", [
    (SpotifyException(403, -1, "https://api.spotify.com/v1/me:\n User not registered", reason=None), "cadastrada"),
    (SpotifyOauthError("Received error from auth server: invalid_client", error="invalid_client"), "SPOTIFY_CLIENT_ID"),
    (SpotifyOauthError("x", error="access_denied"), "cancelado"),
    (OSError("Address already in use"), "8888"),
])
def test_get_client_friendly_errors(monkeypatch, configured, tmp_dirs, error, expected):
    class FakeOAuth:
        def __init__(self, **kw):
            pass

    class FakeClient:
        def __init__(self, **kw):
            pass

        def current_user(self):
            raise error

    monkeypatch.setattr(spotipy.oauth2, "SpotifyOAuth", FakeOAuth)
    monkeypatch.setattr(spotipy, "Spotify", FakeClient)
    with pytest.raises(si.SpotifyError) as e:
        si.get_client()
    assert expected in str(e.value)
    assert SECRET not in str(e.value)


# ---------------------------------------------------------------------------
# Playlists
# ---------------------------------------------------------------------------
def test_list_playlists_paginates_and_marks_readable():
    pages = [
        {"items": [
            {"id": "p1", "name": "Mandelão", "collaborative": False,
             "owner": {"id": "djsk", "display_name": "DJ SK"}, "tracks": {"href": "h", "total": 42}},
            None,
            {"id": "p2", "name": "Dos amigos", "collaborative": True,
             "owner": {"id": "amigo", "display_name": "Amigo"}, "items": {"href": "h", "total": 7}},
        ], "next": "pl:1"},
        {"items": [
            {"id": "p3", "name": "Seguida", "collaborative": False, "owner": {"id": "spotify"},
             "tracks": {"total": 50}},
        ], "next": None},
    ]
    sp = FakeSpotify(playlist_pages=pages)
    out = si.list_playlists(sp)
    assert [p["id"] for p in out] == ["p1", "p2", "p3"]
    assert out[0] == {"id": "p1", "name": "Mandelão", "owner": "DJ SK", "owner_id": "djsk",
                      "tracks_total": 42, "collaborative": False, "readable": True,
                      "image_url": None, "url": None}
    assert (out[1]["readable"], out[1]["tracks_total"], out[1]["collaborative"]) == (True, 7, True)
    assert (out[2]["readable"], out[2]["owner"]) == (False, "spotify")
    assert ("playlists", 50, 0) in sp.calls and ("next", "pl:1") in sp.calls


def test_list_playlists_error():
    sp = FakeSpotify(error=SpotifyException(401, -1, "expired"))
    with pytest.raises(si.SpotifyError, match="expirou"):
        si.list_playlists(sp)


def test_playlist_tracks_parses_both_formats_and_skips_junk():
    pages = [
        {"items": [
            {"added_at": "x", "is_local": False,
             "track": sp_track("Casei Com a Putaria", ["MC Ryan SP", "Kotim"], 158000, "brxyz2400001", "t1")},
            {"added_at": "x", "item": sp_track("Fui Mlk", ["Nilo"], 175500, None, "t2")},  # formato novo
            None,
            {"track": None},
            {"track": sp_track("Podcast", ["Alguém"], 3600000, tid="e1", type_="episode")},
            {"is_local": True, "track": sp_track("Arquivo local", ["Eu"], 120000, tid="")},
            {"track": sp_track("Local 2", ["Eu"], 120000, tid="", is_local=True)},
            {"track": sp_track("", ["Sem nome"], 120000)},
        ], "next": "it:1"},
        {"items": [{"item": sp_track("Felina", ["WIU", "MC Ryan SP"], 226000, " BRabc ", "t3")}], "next": None},
    ]
    sp = FakeSpotify(item_pages=pages)
    out = si.playlist_tracks(sp, "https://open.spotify.com/playlist/abc")
    assert [t.title for t in out] == ["Casei Com a Putaria", "Fui Mlk", "Felina"]
    assert out[0] == SpotifyTrack("Casei Com a Putaria", ["MC Ryan SP", "Kotim"], 158.0, "BRXYZ2400001", "t1")
    assert out[1].isrc is None and out[1].duration_s == 175.5
    assert out[2].isrc == "BRABC"
    assert sp.calls[0] == ("items", "https://open.spotify.com/playlist/abc", 50, ("track",))


@pytest.mark.parametrize("status, expected", [(403, "criou ou em que colabora"), (404, "não encontrada"),
                                              (429, "Espere"), (500, "HTTP 500")])
def test_playlist_tracks_friendly_errors(status, expected):
    sp = FakeSpotify(error=SpotifyException(status, -1, "https://api.spotify.com/v1/playlists/x/items:\n err"))
    with pytest.raises(si.SpotifyError) as e:
        si.playlist_tracks(sp, "x")
    assert expected in str(e.value)
    assert "api.spotify.com" not in str(e.value)


def test_playlist_tracks_network_error():
    import requests

    sp = FakeSpotify(error=requests.ConnectionError("sem rede"))
    with pytest.raises(si.SpotifyError, match="Sem conexão"):
        si.playlist_tracks(sp, "x")


# ---------------------------------------------------------------------------
# Resolução no YouTube Music
# ---------------------------------------------------------------------------
CASEI = SpotifyTrack("Casei Com a Putaria", ["MC Ryan SP", "Kotim"], 158.0, "BRXYZ2400001", "t1")


def test_resolve_by_isrc_when_duration_matches():
    yt = FakeYT({("BRXYZ2400001", "songs"): [
        yt_song("longaversao", "Casei Com a Putaria", ["MC Ryan SP"], 240),
        yt_song("bN8S3cRF9X0", "Casei Com a Putaria", ["Mc Paiva ZS", "MC Ryan SP", "Kotim"], 160),
    ]})
    t = si.resolve_on_ytmusic(CASEI, yt=yt)
    assert t.video_id == "bN8S3cRF9X0"
    assert (t.source, t.isrc, t.title, t.artists) == ("spotify", "BRXYZ2400001", "Casei Com a Putaria",
                                                      ["MC Ryan SP", "Kotim"])
    assert t.duration_s == 160.0 and t.thumbnail_url == "https://img/bN8S3cRF9X0.jpg"
    assert yt.calls == [("BRXYZ2400001", "songs")]


def test_resolve_falls_back_to_text_and_picks_closest_duration():
    q = "MC Ryan SP - Casei Com a Putaria"
    yt = FakeYT({
        ("BRXYZ2400001", "songs"): [yt_song("isrcerrado1", "Casei Com a Putaria", ["MC Ryan SP"], 200)],
        (q, "songs"): [
            yt_song("outramusica", "Felina", ["MC Ryan SP"], 158),                 # título diferente
            yt_song("remixlongo1", "Casei Com a Putaria (Remix)", ["MC Ryan SP"], 190),
            yt_song("certo000001", "Casei Com a Putaria", ["Mc Paiva ZS", "MC Ryan SP"], 157),
            yt_song("semduracao1", "Casei Com a Putaria", ["MC Ryan SP"], None),
        ],
    })
    t = si.resolve_on_ytmusic(CASEI, yt=yt)
    assert t.video_id == "certo000001"
    assert yt.calls == [("BRXYZ2400001", "songs"), (q, "songs")]  # achou nos songs: não busca vídeos


def test_resolve_text_strips_feat_and_prefers_same_artist():
    st = SpotifyTrack("Baile (feat. MC Y)", ["MC X"], 150.0, None)
    q = "MC X - Baile"
    yt = FakeYT({(q, "songs"): [
        yt_song("outroartist", "Baile", ["Banda Qualquer"], 150),
        yt_song("mesmoartist", "Baile", ["MC X", "MC Y"], 165),
    ]})
    assert si.resolve_on_ytmusic(st, yt=yt).video_id == "mesmoartist"


def test_resolve_uses_videos_when_songs_do_not_match():
    st = SpotifyTrack("Fui Mlk", ["Nilo"], 175.0, None)
    q = "Nilo - Fui Mlk"
    yt = FakeYT({
        (q, "songs"): [yt_song("versaolonga", "Fui Mlk", ["Nilo"], 230)],
        (q, "videos"): [yt_song("clipe000001", "Nilo - Fui Mlk (Clipe Oficial)", ["Nilo"], 177)],
    })
    t = si.resolve_on_ytmusic(st, yt=yt)
    assert t.video_id == "clipe000001"
    assert t.isrc is None and t.source == "spotify"
    # Sem nada dentro da tolerância, fica com o mais próximo entre os parecidos.
    yt2 = FakeYT({(q, "songs"): [yt_song("versaolonga", "Fui Mlk", ["Nilo"], 230)]})
    assert si.resolve_on_ytmusic(st, yt=yt2).video_id == "versaolonga"


def test_resolve_not_found():
    assert si.resolve_on_ytmusic(SpotifyTrack("Nada", ["Ninguém"], 100.0, "ISRC0"), yt=FakeYT()) is None
    yt = FakeYT({("Ninguém - Nada", "songs"): [yt_song("abcdefghijk", "Outra Coisa", ["Ninguém"], 100)]})
    assert si.resolve_on_ytmusic(SpotifyTrack("Nada", ["Ninguém"], 100.0), yt=yt) is None


def test_resolve_uses_config_tolerance(monkeypatch):
    monkeypatch.setattr(config, "SPOTIFY_DURATION_TOLERANCE_S", 50.0)
    yt = FakeYT({("BRXYZ2400001", "songs"): [yt_song("isrcok00001", "Casei Com a Putaria", ["MC Ryan SP"], 200)]})
    assert si.resolve_on_ytmusic(CASEI, yt=yt).video_id == "isrcok00001"


# ---------------------------------------------------------------------------
# import_playlist
# ---------------------------------------------------------------------------
def _items(*tracks: dict) -> list[dict]:
    return [{"track": t} for t in tracks]


def test_import_playlist_order_dedup_not_found_and_progress():
    pages = [{"items": _items(
        sp_track("Fui Mlk", ["Nilo"], 175000, None, "a"),
        sp_track("Inexistente", ["MC Fantasma"], 120000, None, "b"),
        sp_track("Felina", ["WIU", "MC Ryan SP"], 226000, None, "c"),
        sp_track("Fui Mlk", ["Nilo"], 175000, None, "d"),  # repetida
    ), "next": None}]
    yt = FakeYT({
        ("Nilo - Fui Mlk", "songs"): [yt_song("4cJgIfrO8jM", "Fui Mlk", ["Nilo"], 175)],
        ("WIU - Felina", "songs"): [yt_song("felina00001", "Felina", ["WIU"], 226)],
    })
    progress = []
    tracks, missing = si.import_playlist(FakeSpotify(item_pages=pages), "pl", yt=yt,
                                         progress=lambda d, t: progress.append((d, t)))
    assert [t.video_id for t in tracks] == ["4cJgIfrO8jM", "felina00001"]
    assert all(t.source == "spotify" for t in tracks)
    assert missing == ["MC Fantasma - Inexistente"]
    assert progress == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_import_playlist_limit_and_empty():
    pages = [{"items": _items(sp_track("Fui Mlk", ["Nilo"], 175000, None, "a"),
                              sp_track("Felina", ["WIU"], 226000, None, "c")), "next": None}]
    yt = FakeYT({("Nilo - Fui Mlk", "songs"): [yt_song("4cJgIfrO8jM", "Fui Mlk", ["Nilo"], 175)]})
    tracks, missing = si.import_playlist(FakeSpotify(item_pages=pages), "pl", yt=yt, limit=1)
    assert [t.video_id for t in tracks] == ["4cJgIfrO8jM"] and missing == []
    assert si.import_playlist(FakeSpotify(), "vazia", yt=FakeYT()) == ([], [])


def test_import_playlist_all_searches_fail():
    pages = [{"items": _items(sp_track("Fui Mlk", ["Nilo"], 175000, None, "a")), "next": None}]
    with pytest.raises(si.SpotifyError, match="YouTube Music"):
        si.import_playlist(FakeSpotify(item_pages=pages), "pl", yt=FakeYT(fail=True))


def test_import_playlist_propagates_playlist_error():
    sp = FakeSpotify(error=SpotifyException(403, -1, "forbidden"))
    with pytest.raises(si.SpotifyError, match="modo de desenvolvimento"):
        si.import_playlist(sp, "pl", yt=FakeYT())


# ---------------------------------------------------------------------------
# Teste real (rede): só a parte do YouTube Music, sem login no Spotify
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_real_resolve_on_ytmusic_by_text():
    st = SpotifyTrack("Casei Com a Putaria", ["MC Ryan SP"], 158.0, None)
    t = si.resolve_on_ytmusic(st)
    assert t is not None and t.source == "spotify"
    assert t.duration_s is not None and abs(t.duration_s - 158.0) <= config.SPOTIFY_DURATION_TOLERANCE_S


# ===========================================================================
# Permissões novas (criar playlist) e token salvo antigo
# ===========================================================================
OLD_SCOPES = "playlist-read-private playlist-read-collaborative"


def _token(scope: str | None = OLD_SCOPES, expires_in: int = 3600) -> dict:
    t = {"access_token": "a", "token_type": "Bearer", "expires_in": expires_in, "refresh_token": "r",
         "expires_at": int(time.time()) + expires_in}
    if scope is not None:
        t["scope"] = scope
    return t


def _save_token(tmp_dirs, token: dict):
    path = tmp_dirs["DATA_DIR"] / ".spotify_token"
    path.write_text(json.dumps(token), encoding="utf-8")
    return path


def test_scopes_include_playlist_creation_and_profile():
    scopes = set(config.SPOTIFY_SCOPES.split())
    assert scopes == {"playlist-read-private", "playlist-read-collaborative", "playlist-modify-private",
                      "playlist-modify-public", "user-read-private"}


def test_token_has_scopes():
    assert si.token_has_scopes(_token(config.SPOTIFY_SCOPES))
    # Ordem diferente, vírgulas e escopos a mais não atrapalham.
    assert si.token_has_scopes(_token(",".join(reversed(config.SPOTIFY_SCOPES.split())) + ",user-top-read"))
    assert not si.token_has_scopes(_token(OLD_SCOPES))
    assert not si.token_has_scopes(_token(None))
    assert not si.token_has_scopes(None)
    assert si.token_has_scopes(_token(OLD_SCOPES), scopes=OLD_SCOPES)


def test_spotipy_itself_ignores_token_with_fewer_scopes(configured):
    # Garante o comportamento do spotipy 2.26: escopo diferente no cache = token inválido
    # (get_access_token cai no fluxo de login em vez de usar o token antigo). Sem rede.
    auth = spotipy.oauth2.SpotifyOAuth(client_id="id", client_secret="s", redirect_uri=config.SPOTIFY_REDIRECT_URI,
                                       scope=config.SPOTIFY_SCOPES, cache_handler=MemoryCacheHandler(),
                                       open_browser=False)
    assert auth.validate_token(_token(OLD_SCOPES)) is None
    assert auth.validate_token(_token(config.SPOTIFY_SCOPES))["access_token"] == "a"


def test_saved_login_state(configured, tmp_dirs):
    assert si.saved_login_state() == "none"
    _save_token(tmp_dirs, _token(OLD_SCOPES))
    assert si.saved_login_state() == "outdated"
    _save_token(tmp_dirs, _token(config.SPOTIFY_SCOPES))
    assert si.saved_login_state() == "saved"
    (tmp_dirs["DATA_DIR"] / ".spotify_token").write_text("corrompido", encoding="utf-8")
    assert si.saved_login_state() == "none"


def test_get_client_without_browser_refuses_outdated_token(configured, tmp_dirs):
    path = _save_token(tmp_dirs, _token(OLD_SCOPES))
    with pytest.raises(si.SpotifyError, match="nova autorização"):
        si.get_client(open_browser=False)
    assert path.exists()  # sem navegador não apaga nada


def test_get_client_with_browser_drops_outdated_token_and_logs_in(monkeypatch, configured, tmp_dirs):
    path = _save_token(tmp_dirs, _token(OLD_SCOPES))
    seen = {}

    class FakeOAuth:
        def __init__(self, **kw):
            seen["token_exists_at_login"] = path.exists()
            seen["scope"] = kw["scope"]

    class FakeClient:
        def __init__(self, **kw):
            seen["session"] = kw.get("requests_session")

        def current_user(self):
            seen["login"] = True
            return {"id": "djsk"}

    monkeypatch.setattr(spotipy.oauth2, "SpotifyOAuth", FakeOAuth)
    monkeypatch.setattr(spotipy, "Spotify", FakeClient)
    si.get_client()
    assert seen["token_exists_at_login"] is False
    assert seen["scope"] == config.SPOTIFY_SCOPES and seen["login"] is True
    assert isinstance(seen["session"], requests.Session)


def test_get_client_keeps_token_with_new_scopes(monkeypatch, configured, tmp_dirs):
    path = _save_token(tmp_dirs, _token(config.SPOTIFY_SCOPES))
    monkeypatch.setattr(spotipy.Spotify, "current_user", lambda self: {"id": "djsk"})
    assert isinstance(si.get_client(open_browser=False), spotipy.Spotify)
    assert path.exists()


def test_http_session_caps_rate_limit_wait():
    retry = si._http_session().get_adapter("https://api.spotify.com/v1/search").max_retries
    assert 429 in retry.status_forcelist and 503 in retry.status_forcelist
    assert retry.retry_after_max == si.RETRY_AFTER_MAX_S
    assert retry.total == si.HTTP_RETRIES


def test_spotipy_debug_log_is_silenced():
    # O DEBUG do spotipy registra o cabeçalho Authorization (chaves do app).
    assert logging.getLogger("spotipy").getEffectiveLevel() >= logging.INFO


# ===========================================================================
# Fakes da busca / exportação
# ===========================================================================
def raw_track(name: str, artists: list[str], ms: int | None, tid: str, isrc: str | None = None,
              images: list | None = None, album: str = "Álbum", **kw) -> dict:
    t = sp_track(name, artists, ms or 0, isrc, tid, **kw)
    if ms is None:
        t.pop("duration_ms")
    t["album"] = {"name": album, "images": images if images is not None else [
        {"url": f"https://i.scdn.co/{tid}_64", "width": 64, "height": 64},
        {"url": f"https://i.scdn.co/{tid}_640", "width": 640, "height": 640},
    ]}
    t["external_urls"] = {"spotify": f"https://open.spotify.com/track/{tid}"}
    return t


def page(*raws: dict) -> dict:
    return {"tracks": {"items": list(raws), "next": None, "total": len(raws)}}


class FakeSearchSpotify:
    """Cliente spotipy falso para busca, criação de playlist e inclusão de faixas."""

    def __init__(self, results: dict | None = None, handler=None, error: Exception | None = None,
                 errors: dict | None = None):
        self.results = results or {}
        self.handler = handler
        self.error = error
        self.errors = errors or {}
        self.calls: list[tuple] = []
        self.lock = threading.Lock()
        self.created: list[dict] = []
        self.added: list[tuple[str, list[str]]] = []
        self.create_error: Exception | None = None
        self.add_error: Exception | None = None
        self.add_error_at: int | None = None
        self.me = {"id": "djsk", "display_name": "DJ SK", "product": "premium",
                   "images": [{"url": "https://i.scdn.co/me_64", "width": 64, "height": 64},
                              {"url": "https://i.scdn.co/me_300", "width": 300, "height": 300}]}

    def search(self, q, limit=10, offset=0, type="track", market=None):
        with self.lock:
            self.calls.append((q, limit, type))
        if self.error:
            raise self.error
        if q in self.errors:
            raise self.errors[q]
        if self.handler is not None:
            return self.handler(q)
        return self.results.get(q, page())

    def current_user(self):
        if self.error:
            raise self.error
        return self.me

    def current_user_playlist_create(self, name, public=True, collaborative=False, description=""):
        self.created.append({"name": name, "public": public, "collaborative": collaborative,
                             "description": description})
        if self.create_error:
            raise self.create_error
        return {"id": "pl123", "name": name, "external_urls": {"spotify": "https://open.spotify.com/playlist/pl123"}}

    def playlist_add_items(self, playlist_id, items, position=None):
        if self.add_error is not None and len(self.added) == self.add_error_at:
            raise self.add_error
        self.added.append((playlist_id, list(items)))
        return {"snapshot_id": "s"}

    @property
    def queries(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture
def search_env(monkeypatch, configured):
    """Troca SpotifyClientCredentials e spotipy.Spotify por fakes e zera o cliente em cache."""
    si._reset_search_client()
    fake = FakeSearchSpotify()
    built = {"cc": [], "client": []}

    class FakeCC:
        def __init__(self, **kw):
            time.sleep(0.02)  # alarga a janela de corrida no teste de threads
            built["cc"].append(kw)

    def fake_spotify(**kw):
        built["client"].append(kw)
        return fake

    monkeypatch.setattr(spotipy.oauth2, "SpotifyClientCredentials", FakeCC)
    monkeypatch.setattr(spotipy, "Spotify", fake_spotify)
    yield fake, built
    si._reset_search_client()


# ===========================================================================
# search_tracks / to_spotify_track
# ===========================================================================
def test_search_tracks_parses_results(search_env):
    fake, built = search_env
    q = "DJ Redy This Is Submundo"
    fake.results[q] = page(
        raw_track("Submundo", ["DJ Redy", "MC Lipi"], 165500, "sp1", isrc=" brabc2600001 ", album="This Is"),
        raw_track("Submundo", ["DJ Redy", "MC Lipi"], 165500, "sp1"),        # repetida
        raw_track("Arquivo local", ["Eu"], 120000, "loc", is_local=True),
        raw_track("Sem duração", ["DJ Redy"], None, "sp2", images=[]),
        {"name": "Sem id", "artists": [], "type": "track"},
        None,
    )
    out = si.search_tracks("  DJ Redy   This Is Submundo ")
    assert out[0] == {
        "title": "Submundo", "artists": ["DJ Redy", "MC Lipi"], "duration_s": 165.5, "isrc": "BRABC2600001",
        "spotify_id": "sp1", "cover_url": "https://i.scdn.co/sp1_640",
        "spotify_url": "https://open.spotify.com/track/sp1", "album": "This Is", "uri": "spotify:track:sp1",
    }
    assert [t["spotify_id"] for t in out] == ["sp1", "sp2"]
    assert (out[1]["duration_s"], out[1]["isrc"], out[1]["cover_url"]) == (None, None, None)
    assert fake.calls == [(q, 10, "track")]


def test_search_tracks_uses_cached_client_credentials_without_disk_cache(search_env):
    fake, built = search_env
    si.search_tracks("a")
    si.search_tracks("b", limit=3)
    assert len(built["cc"]) == 1 and len(built["client"]) == 1  # cliente reaproveitado
    cc = built["cc"][0]
    assert cc["client_id"] == "id-de-teste" and cc["client_secret"] == SECRET
    assert isinstance(cc["cache_handler"], MemoryCacheHandler)  # não grava ".cache" no disco
    assert isinstance(built["client"][0]["requests_session"], requests.Session)
    assert built["client"][0]["auth_manager"] is not None
    assert [c[1] for c in fake.calls] == [10, 3]


def test_search_tracks_limit_is_clamped_and_empty_query_skips_network(search_env):
    fake, built = search_env
    assert si.search_tracks("   ") == [] and si.search_tracks("") == []
    assert built["cc"] == [] and fake.calls == []
    si.search_tracks("x", limit=50)
    si.search_tracks("y", limit=0)
    si.search_tracks("z", limit=-5)
    assert [c[1] for c in fake.calls] == [10, 10, 1]


def test_search_client_is_built_once_across_threads(search_env):
    fake, built = search_env
    clients = []
    threads = [threading.Thread(target=lambda: clients.append(si._get_search_client())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built["cc"]) == 1 and all(c is fake for c in clients) and len(clients) == 8


def test_search_client_rebuilt_when_client_id_changes(monkeypatch, search_env):
    fake, built = search_env
    si.search_tracks("a")
    monkeypatch.setattr(config, "get_secret", lambda name: {"SPOTIFY_CLIENT_ID": "outro-id",
                                                            "SPOTIFY_CLIENT_SECRET": "s2"}.get(name))
    si.search_tracks("b")
    assert [cc["client_id"] for cc in built["cc"]] == ["id-de-teste", "outro-id"]


def test_search_tracks_not_configured(not_configured):
    si._reset_search_client()
    with pytest.raises(si.SpotifyError, match="não configurado"):
        si.search_tracks("funk")


@pytest.mark.parametrize("error, expected", [
    (SpotifyOauthError("Received error from auth server: invalid_client", error="invalid_client"), "SPOTIFY_CLIENT_ID"),
    (SpotifyOauthError("x", error="server_error"), "Confira"),
    (SpotifyException(401, -1, "https://api.spotify.com/v1/search:\n token", reason=None), "SPOTIFY_CLIENT_SECRET"),
    (SpotifyException(400, -1, "https://api.spotify.com/v1/search:\n bad", reason=None), "outras palavras"),
    (SpotifyException(429, -1, "https://api.spotify.com/v1/search:\n Max Retries", reason=None), "Espere"),
    (requests.ConnectionError("sem rede"), "Sem conexão"),
    (OSError("rede"), "Sem conexão"),
])
def test_search_tracks_friendly_errors(search_env, error, expected):
    fake, _ = search_env
    fake.error = error
    with pytest.raises(si.SpotifyError) as e:
        si.search_tracks("funk")
    assert expected in str(e.value)
    assert SECRET not in str(e.value) and "api.spotify.com" not in str(e.value)


def test_to_spotify_track_from_search_result_and_raw():
    d = {"title": " Submundo ", "artists": ["DJ Redy", "", None, "MC Lipi"], "duration_s": 165.5,
         "isrc": "brabc2600001", "spotify_id": "sp1", "cover_url": None, "spotify_url": "u", "album": None}
    assert si.to_spotify_track(d) == SpotifyTrack("Submundo", ["DJ Redy", "MC Lipi"], 165.5, "BRABC2600001", "sp1")
    no_duration = si.to_spotify_track({"title": "X", "artists": [], "duration_s": None})
    assert (no_duration.duration_s, no_duration.isrc, no_duration.spotify_id) == (0.0, None, None)
    raw = raw_track("Felina", ["WIU"], 226000, "sp9", isrc="BRX")
    assert si.to_spotify_track(raw) == SpotifyTrack("Felina", ["WIU"], 226.0, "BRX", "sp9")
    for bad in ({"title": ""}, {"name": "Local", "id": "x", "is_local": True}, None):
        with pytest.raises(ValueError):
            si.to_spotify_track(bad)


def test_search_result_works_as_reference_on_ytmusic():
    d = {"title": "Casei Com a Putaria", "artists": ["MC Ryan SP", "Kotim"], "duration_s": 158.0,
         "isrc": "BRXYZ2400001", "spotify_id": "t1"}
    yt = FakeYT({("BRXYZ2400001", "songs"): [yt_song("bN8S3cRF9X0", "Casei Com a Putaria", ["MC Ryan SP"], 160)]})
    t = si.resolve_on_ytmusic(si.to_spotify_track(d), yt=yt)
    assert (t.video_id, t.source, t.isrc) == ("bN8S3cRF9X0", "spotify", "BRXYZ2400001")


# ===========================================================================
# current_user_info
# ===========================================================================
def test_current_user_info():
    sp = FakeSearchSpotify()
    assert si.current_user_info(sp) == {"id": "djsk", "name": "DJ SK", "image_url": "https://i.scdn.co/me_300",
                                        "product": "premium"}
    sp.me = {"id": "djsk2", "display_name": None, "images": []}
    assert si.current_user_info(sp) == {"id": "djsk2", "name": "djsk2", "image_url": None, "product": None}


@pytest.mark.parametrize("error, expected", [
    (SpotifyException(401, -1, "https://api.spotify.com/v1/me:\n expired"), "expirou"),
    (SpotifyException(403, -1, "https://api.spotify.com/v1/me:\n User not registered"), "cadastrada"),
    (requests.ConnectionError("x"), "Sem conexão"),
])
def test_current_user_info_errors(error, expected):
    with pytest.raises(si.SpotifyError, match=expected):
        si.current_user_info(FakeSearchSpotify(error=error))


# ===========================================================================
# find_on_spotify
# ===========================================================================
def yt_track(title: str, artists: list[str], duration: float | None, vid: str = "abcdefghijk",
             isrc: str | None = None) -> Track:
    return Track(video_id=vid, title=title, artists=artists, duration_s=duration, isrc=isrc)


def test_find_uses_field_query_first_and_checks_duration():
    q1 = "track:Casei Com a Putaria artist:MC Ryan SP"
    sp = FakeSearchSpotify({q1: page(
        raw_track("Casei Com a Putaria", ["MC Ryan SP"], 240000, "longa"),       # duração muito diferente
        raw_track("Felina", ["MC Ryan SP"], 158000, "outra"),                     # título diferente
        raw_track("Casei Com a Putaria", ["MC Ryan SP", "Kotim"], 157000, "certa"),
    )})
    hit = si.find_on_spotify(sp, yt_track("Casei Com a Putaria (Clipe Oficial)", ["MC Ryan SP"], 158.0))
    assert hit["spotify_id"] == "certa" and hit["uri"] == "spotify:track:certa"
    assert sp.queries == [q1]


def test_find_falls_back_to_plain_query():
    q1 = "track:Fui Mlk artist:Nilo"
    q2 = "Nilo Fui Mlk"
    sp = FakeSearchSpotify({
        q1: page(raw_track("Fui Mlk", ["Nilo"], 230000, "longa")),
        q2: page(raw_track("Fui Mlk", ["Nilo"], 176000, "certa")),
    })
    assert si.find_on_spotify(sp, yt_track("Fui Mlk", ["Nilo"], 175.0))["spotify_id"] == "certa"
    assert sp.queries == [q1, q2]


def test_find_tries_isrc_first():
    sp = FakeSearchSpotify({"isrc:BRXYZ2400001": page(raw_track("Casei c/ a Putaria", ["Kotim"], 159000, "isrc"))})
    hit = si.find_on_spotify(sp, yt_track("Casei Com a Putaria", ["MC Ryan SP"], 158.0, isrc="BRXYZ2400001"))
    assert hit["spotify_id"] == "isrc" and sp.queries == ["isrc:BRXYZ2400001"]
    # ISRC com duração fora da tolerância não vale: segue para a busca por texto.
    sp2 = FakeSearchSpotify({"isrc:BRXYZ2400001": page(raw_track("Casei Com a Putaria", ["MC Ryan SP"], 300000, "x"))})
    assert si.find_on_spotify(sp2, yt_track("Casei Com a Putaria", ["MC Ryan SP"], 158.0, isrc="BRXYZ2400001")) is None
    assert sp2.queries[0] == "isrc:BRXYZ2400001" and len(sp2.queries) == 3


def test_find_youtube_video_title_with_artist_in_title():
    # Vídeo do canal "KondZilla": o artista de verdade está no título.
    track = yt_track("MC Ryan SP - Casei Com a Putaria (Videoclipe Oficial)", ["KondZilla"], 158.0)
    q = "track:Casei Com a Putaria artist:MC Ryan SP"
    sp = FakeSearchSpotify({q: page(raw_track("Casei Com a Putaria", ["MC Ryan SP"], 158500, "certa"))})
    assert si.find_on_spotify(sp, track)["spotify_id"] == "certa"
    assert q in sp.queries
    # Artista do título igual ao da faixa: uma leitura só.
    track2 = yt_track("Nilo - Fui Mlk", ["Nilo"], 175.0)
    sp2 = FakeSearchSpotify({"track:Fui Mlk artist:Nilo": page(raw_track("Fui Mlk", ["Nilo"], 175000, "ok"))})
    assert si.find_on_spotify(sp2, track2)["spotify_id"] == "ok"
    assert sp2.queries == ["track:Fui Mlk artist:Nilo"]


def test_find_prefers_same_artist_and_needs_artist_without_duration():
    q = "track:Baile artist:MC X"
    sp = FakeSearchSpotify({q: page(
        raw_track("Baile", ["Banda Qualquer"], 150000, "outro"),
        raw_track("Baile", ["MC X"], 152000, "mesmo"),
    )})
    assert si.find_on_spotify(sp, yt_track("Baile", ["MC X"], 150.0))["spotify_id"] == "mesmo"
    # Sem duração na faixa do set, só aceita com artista em comum.
    sp2 = FakeSearchSpotify({q: page(raw_track("Baile", ["Banda Qualquer"], 150000, "outro"))})
    assert si.find_on_spotify(sp2, yt_track("Baile", ["MC X"], None)) is None
    sp3 = FakeSearchSpotify({q: page(raw_track("Baile", ["MC X"], 150000, "mesmo"))})
    assert si.find_on_spotify(sp3, yt_track("Baile", ["MC X"], None))["spotify_id"] == "mesmo"


def test_find_cleans_query_syntax_and_handles_no_artist():
    sp = FakeSearchSpotify()
    assert si.find_on_spotify(sp, yt_track('Montagem "Submundo": parte 2', [], 120.0)) is None
    assert sp.queries == ["Montagem Submundo parte 2"]
    sp2 = FakeSearchSpotify()
    si.find_on_spotify(sp2, yt_track("Faixa", ['DJ "K": SK'], 120.0))
    assert sp2.queries == ["track:Faixa artist:DJ K SK", "DJ K SK Faixa"]


def test_find_on_spotify_errors_are_friendly():
    sp = FakeSearchSpotify(error=SpotifyException(401, -1, "https://api.spotify.com/v1/search:\n expired"))
    with pytest.raises(si.SpotifyError) as e:
        si.find_on_spotify(sp, yt_track("Faixa", ["MC"], 100.0))
    assert "api.spotify.com" not in str(e.value)


# ===========================================================================
# export_set_to_spotify
# ===========================================================================
def _catalog_handler(catalog: dict[str, dict]):
    """Busca falsa: responde "track:<título> artist:<artista>" com a faixa do catálogo (por título)."""
    def handler(q):
        if q.startswith("track:"):
            title = q[len("track:"):].split(" artist:")[0]
            raw = catalog.get(title)
            return page(raw) if raw else page()
        return page()
    return handler


def test_export_creates_playlist_in_set_order():
    catalog = {
        "Fui Mlk": raw_track("Fui Mlk", ["Nilo"], 175000, "a"),
        "Felina": raw_track("Felina", ["WIU"], 226000, "b"),
        "Casei Com a Putaria": raw_track("Casei Com a Putaria", ["MC Ryan SP"], 158000, "c"),
    }
    sp = FakeSearchSpotify(handler=_catalog_handler(catalog))
    tracks = [
        yt_track("Casei Com a Putaria", ["MC Ryan SP"], 158.0, "v1"),
        yt_track("Inexistente", ["MC Fantasma"], 120.0, "v2"),
        yt_track("Fui Mlk", ["Nilo"], 175.0, "v3"),
        yt_track("Fui Mlk (Clipe Oficial)", ["Nilo"], 176.0, "v4"),  # outro upload da mesma faixa
        yt_track("Felina", ["WIU"], 226.0, "v5"),
    ]
    progress = []
    out = si.export_set_to_spotify(sp, "  Set   Submundo\n23/10 ", tracks, description="Feito no <painel>\ndo DJ SK",
                                   progress=lambda d, t: progress.append((d, t)))
    assert out == {"playlist_id": "pl123", "playlist_url": "https://open.spotify.com/playlist/pl123",
                   "added": 3, "missing": ["MC Fantasma - Inexistente"]}
    assert sp.created == [{"name": "Set Submundo 23/10", "public": False, "collaborative": False,
                           "description": "Feito no painel do DJ SK"}]
    assert sp.added == [("pl123", ["spotify:track:c", "spotify:track:a", "spotify:track:b"])]
    assert progress == [(i, 5) for i in range(1, 6)]


def test_export_adds_in_batches_of_100_and_public_flag():
    catalog = {f"Faixa {i:03d}": raw_track(f"Faixa {i:03d}", ["DJ SK"], 150000, f"id{i:03d}") for i in range(230)}
    sp = FakeSearchSpotify(handler=_catalog_handler(catalog))
    tracks = [yt_track(f"Faixa {i:03d}", ["DJ SK"], 150.0, f"v{i:03d}") for i in range(230)]
    out = si.export_set_to_spotify(sp, "", tracks, public=True)
    assert out["added"] == 230 and out["missing"] == []
    assert [len(items) for _, items in sp.added] == [100, 100, 30]
    assert [u for _, items in sp.added for u in items] == [f"spotify:track:id{i:03d}" for i in range(230)]
    assert sp.created[0]["name"] == si.DEFAULT_PLAYLIST_NAME and sp.created[0]["public"] is True


def test_export_name_is_truncated():
    sp = FakeSearchSpotify(handler=_catalog_handler({"X": raw_track("X", ["Y"], 100000, "x")}))
    si.export_set_to_spotify(sp, "N" * 300, [yt_track("X", ["Y"], 100.0)], description="d" * 900)
    assert len(sp.created[0]["name"]) == si.PLAYLIST_NAME_MAX
    assert len(sp.created[0]["description"]) == si.PLAYLIST_DESCRIPTION_MAX


def test_export_empty_or_nothing_found_does_not_create_playlist():
    sp = FakeSearchSpotify()
    with pytest.raises(si.SpotifyError, match="vazio"):
        si.export_set_to_spotify(sp, "Set", [])
    with pytest.raises(si.SpotifyError, match="Nenhuma música"):
        si.export_set_to_spotify(sp, "Set", [yt_track("Nada", ["Ninguém"], 100.0)])
    assert sp.created == [] and sp.added == []


def test_export_aborts_on_login_error_before_creating():
    sp = FakeSearchSpotify(error=SpotifyException(401, -1, "https://api.spotify.com/v1/search:\n expired"))
    tracks = [yt_track(f"Faixa {i}", ["DJ"], 100.0, f"v{i}") for i in range(10)]
    with pytest.raises(si.SpotifyError, match="expirou"):
        si.export_set_to_spotify(sp, "Set", tracks)
    assert sp.created == []
    sp2 = FakeSearchSpotify(error=requests.ConnectionError("sem rede"))
    with pytest.raises(si.SpotifyError, match="Sem conexão"):
        si.export_set_to_spotify(sp2, "Set", tracks)


def test_export_skips_track_whose_search_is_rejected():
    catalog = {"Boa": raw_track("Boa", ["DJ"], 100000, "boa")}
    sp = FakeSearchSpotify(handler=_catalog_handler(catalog),
                           errors={"track:Ruim artist:DJ": SpotifyException(400, -1, "bad query")})
    out = si.export_set_to_spotify(sp, "Set", [yt_track("Ruim", ["DJ"], 100.0, "v1"),
                                               yt_track("Boa", ["DJ"], 100.0, "v2")])
    assert out["added"] == 1 and out["missing"] == ["DJ - Ruim"]


def test_export_create_forbidden_is_friendly():
    sp = FakeSearchSpotify(handler=_catalog_handler({"X": raw_track("X", ["Y"], 100000, "x")}))
    sp.create_error = SpotifyException(403, -1, "https://api.spotify.com/v1/me/playlists:\n Insufficient client scope")
    with pytest.raises(si.SpotifyError, match="autorizar a criação de playlists") as e:
        si.export_set_to_spotify(sp, "Set", [yt_track("X", ["Y"], 100.0)])
    assert "api.spotify.com" not in str(e.value)


def test_export_add_failure_reports_partial_playlist():
    catalog = {f"F{i}": raw_track(f"F{i}", ["DJ"], 100000, f"id{i}") for i in range(150)}
    sp = FakeSearchSpotify(handler=_catalog_handler(catalog))
    sp.add_error = SpotifyException(500, -1, "https://api.spotify.com/v1/playlists/pl123/items:\n boom")
    sp.add_error_at = 1  # o segundo lote falha
    with pytest.raises(si.SpotifyError) as e:
        si.export_set_to_spotify(sp, "Meu set", [yt_track(f"F{i}", ["DJ"], 100.0, f"v{i}") for i in range(150)])
    msg = str(e.value)
    assert "Meu set" in msg and "foi criada" in msg and "100 de 150" in msg
    assert "api.spotify.com" not in msg


def test_export_create_without_id():
    sp = FakeSearchSpotify(handler=_catalog_handler({"X": raw_track("X", ["Y"], 100000, "x")}))
    sp.current_user_playlist_create = lambda *a, **kw: {}
    with pytest.raises(si.SpotifyError, match="não confirmou"):
        si.export_set_to_spotify(sp, "Set", [yt_track("X", ["Y"], 100.0)])


# ===========================================================================
# Testes reais (rede), só com as chaves do app (sem login de usuário)
# ===========================================================================
@pytest.fixture
def real_spotify():
    si._reset_search_client()
    if not si.is_configured():
        pytest.skip("SPOTIFY_CLIENT_ID/SECRET não configurados no .env")
    yield
    si._reset_search_client()


@pytest.mark.network
def test_real_search_tracks_submundo(real_spotify):
    out = si.search_tracks("DJ Redy This Is Submundo")
    assert 0 < len(out) <= si.SPOTIFY_SEARCH_MAX
    for d in out:
        assert d["title"] and d["spotify_id"] and d["uri"] == f"spotify:track:{d['spotify_id']}"
        assert d["spotify_url"].startswith("https://open.spotify.com/")
        assert isinstance(d["artists"], list)
    logging.getLogger(__name__).info("Busca real: %s", [(d["artists"], d["title"], d["duration_s"]) for d in out])


@pytest.mark.network
def test_real_find_on_spotify_round_trip(real_spotify):
    first = si.search_tracks("MC Ryan SP Casei Com a Putaria", limit=5)[0]
    duration = first["duration_s"] + 1.5 if first["duration_s"] else None
    track = Track(video_id="abcdefghijk", title=f"{first['title']} (Clipe Oficial)", artists=first["artists"][:1],
                  duration_s=duration)
    hit = si.find_on_spotify(si._get_search_client(), track)
    assert hit is not None and suggest.titles_similar(hit["title"], first["title"])
