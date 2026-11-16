"""Testes de referências e sugestões (modules/suggest.py) com YTMusic e requests falsos."""
from __future__ import annotations

import threading
import time

import pytest
import requests

import config
from modules import suggest, util
from modules.models import Track


# ---------------------------------------------------------------------------
# Dados no formato real do ytmusicapi (conferido em 22/10/2026)
# ---------------------------------------------------------------------------
def thumbs(*sizes: int) -> list[dict]:
    return [{"url": f"https://img/{s}.jpg", "width": s, "height": s} for s in sizes]


def radio_item(vid: str, title: str, artists: list[str], length: str | None = "2:38") -> dict:
    item = {
        "videoId": vid, "title": title,
        "artists": [{"name": a, "id": f"UC_{a}"} for a in artists],
        "thumbnail": thumbs(60, 120, 544), "videoType": "MUSIC_VIDEO_TYPE_ATV",
        "album": {"name": f"{title} (Single)", "id": "MPREb"},
    }
    if length is not None:
        item["length"] = length
    return item


def search_item(vid: str | None, title: str, artists: list[str], seconds: int | None = 158,
                duration: str | None = None) -> dict:
    item = {
        "category": "Songs", "resultType": "song", "title": title,
        "album": {"name": title, "id": "MPREb"}, "videoId": vid,
        "artists": [{"name": a, "id": f"UC_{a}"} for a in artists],
        "thumbnails": thumbs(60, 120),
    }
    if seconds is not None:
        item["duration_seconds"] = seconds
    if duration is not None:
        item["duration"] = duration
    return item


def vid(n: int) -> str:
    return f"vid{n:08d}"  # 11 caracteres, como um videoId


class FakeYT:
    """YTMusic falso: rádios por videoId, resultados de busca por (query, filter)."""

    def __init__(self, radios: dict | None = None, search_results: dict | None = None,
                 song: dict | None = None, fail_radio: set | None = None, fail_search: bool = False,
                 delay: float = 0.0) -> None:
        self.radios = radios or {}
        self.search_results = search_results or {}
        self.song = song
        self.fail_radio = fail_radio or set()
        self.fail_search = fail_search
        self.delay = delay
        self.calls: list[tuple] = []
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def search(self, query, filter=None, limit=20, **kw):
        self.calls.append(("search", query, filter))
        if self.fail_search:
            raise ConnectionError("rede caiu")
        return self.search_results.get((query, filter), [])

    def get_watch_playlist(self, videoId=None, radio=False, **kw):
        self.calls.append(("watch", videoId, radio))
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            if videoId in self.fail_radio:
                raise RuntimeError("HTTP 400")
            return {"tracks": list(self.radios.get(videoId, [])), "playlistId": "RDAMVM" + videoId}
        finally:
            with self.lock:
                self.active -= 1

    def get_song(self, videoId):
        self.calls.append(("song", videoId))
        if self.song is None:
            raise KeyError("videoDetails")
        return self.song


def ref_track(vid_: str, title: str, artists: list[str]) -> Track:
    return Track(video_id=vid_, title=title, artists=artists, duration_s=158.0)


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    """Ignora uma YOUTUBE_API_KEY real do .env durante os testes unitários."""
    real = config.get_secret
    monkeypatch.setattr(config, "get_secret",
                        lambda name: None if name == "YOUTUBE_API_KEY" else real(name))


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text, expected", [
    ("bN8S3cRF9X0", "bN8S3cRF9X0"),
    ("https://www.youtube.com/watch?v=bN8S3cRF9X0&list=RDbN8S3cRF9X0", "bN8S3cRF9X0"),
    ("https://music.youtube.com/watch?v=4cJgIfrO8jM&si=abc", "4cJgIfrO8jM"),
    ("https://youtu.be/_UKmJm3YO3c?t=30", "_UKmJm3YO3c"),
    ("youtube.com/shorts/3dgBYJNzACo", "3dgBYJNzACo"),
    ("m.youtube.com/watch?feature=share&v=5Zmjghz6bSg", "5Zmjghz6bSg"),
    ("https://www.youtube.com/watch?v=curto", None),
    ("MC Ryan SP - Casei com a Putaria", None),
    ("", None),
])
def test_extract_video_id(text, expected):
    assert suggest.extract_video_id(text) == expected


def test_title_helpers():
    assert suggest.strip_title_noise("Bandido Não Dança (feat. Love Funk)") == "Bandido Não Dança"
    assert suggest.strip_title_noise("Sereia part. Oruam") == "Sereia"
    assert suggest.strip_title_noise("Faz um Vuk Vuk (Teto Espelhado)") == "Faz um Vuk Vuk (Teto Espelhado)"
    assert suggest.title_core("Casei Com a Putaria (Clipe Oficial) [Prod. DJ X]") == "casei com a putaria"
    assert suggest.titles_similar("MC Ryan SP - Casei Com a Putaria", "Casei com a putaria")
    assert not suggest.titles_similar("Felina", "Casamento")


def test_best_thumbnail_and_item_duration():
    assert suggest.best_thumbnail(thumbs(60, 544, 120)) == "https://img/544.jpg"
    assert suggest.best_thumbnail([]) is None
    assert suggest.item_duration({"length": "2:38"}) == 158.0
    assert suggest.item_duration({"duration_seconds": 175}) == 175.0
    assert suggest.item_duration({"duration": "1:02:03"}) == 3723.0
    assert suggest.item_duration({}) is None


def test_parse_iso_duration():
    assert suggest.parse_iso_duration("PT3M12S") == 192.0
    assert suggest.parse_iso_duration("PT1H2M") == 3720.0
    assert suggest.parse_iso_duration("PT45S") == 45.0
    assert suggest.parse_iso_duration("P0D") == 0.0
    assert suggest.parse_iso_duration("3:12") is None
    assert suggest.parse_iso_duration(None) is None


def test_split_artist_title():
    assert suggest.split_artist_title("MC Ryan SP - Felina (Clipe Oficial)", "GR6 EXPLODE") == (["MC Ryan SP"], "Felina")
    assert suggest.split_artist_title("Felina", "MC Ryan SP - Topic") == (["MC Ryan SP"], "Felina")
    assert suggest.split_artist_title("MC X &amp; MC Y - Baile", "") == (["MC X & MC Y"], "Baile")


# ---------------------------------------------------------------------------
# get_ytmusic
# ---------------------------------------------------------------------------
def test_get_ytmusic_singleton_thread_safe(monkeypatch):
    import ytmusicapi

    created = []
    ssl_calls = []

    class FakeYTMusic:
        def __init__(self, *a, **kw):
            assert not a and not kw  # sem login
            time.sleep(0.05)
            created.append(self)

    monkeypatch.setattr(ytmusicapi, "YTMusic", FakeYTMusic)
    monkeypatch.setattr(util, "setup_ssl", lambda: ssl_calls.append(1))
    monkeypatch.setattr(suggest, "_yt_client", None)
    got = []
    threads = [threading.Thread(target=lambda: got.append(suggest.get_ytmusic())) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1
    assert all(g is created[0] for g in got)
    assert ssl_calls  # setup_ssl antes de criar o cliente
    assert suggest.get_ytmusic() is created[0]


# ---------------------------------------------------------------------------
# 4.1 resolve_reference
# ---------------------------------------------------------------------------
def test_resolve_reference_text_takes_first_with_video_id():
    q = "MC Ryan SP - Casei com a Putaria"
    yt = FakeYT(search_results={(q, "songs"): [
        search_item(None, "Sem vídeo", ["X"]),
        search_item("bN8S3cRF9X0", "Casei Com a Putaria", ["Mc Paiva ZS", "MC Ryan SP", "Kotim"]),
        search_item(vid(2), "Outra", ["Y"]),
    ]})
    t = suggest.resolve_reference(f"  {q}  ", yt=yt)
    assert t is not None
    assert t.video_id == "bN8S3cRF9X0"
    assert t.title == "Casei Com a Putaria"
    assert t.artists == ["Mc Paiva ZS", "MC Ryan SP", "Kotim"]
    assert t.duration_s == 158.0
    assert t.thumbnail_url == "https://img/120.jpg"
    assert t.album == "Casei Com a Putaria"
    assert t.source == "ytmusic"
    assert yt.calls[0] == ("search", q, "songs")


def test_resolve_reference_duration_from_text():
    q = "MC X - Faixa"
    yt = FakeYT(search_results={(q, "songs"): [search_item(vid(1), "Faixa", ["MC X"], seconds=None,
                                                           duration="3:05")]})
    assert suggest.resolve_reference(q, yt=yt).duration_s == 185.0


def test_resolve_reference_falls_back_to_videos_then_none():
    q = "MC Raro - Faixa Rara"
    yt = FakeYT(search_results={(q, "videos"): [search_item(vid(9), "Faixa Rara", ["MC Raro"])]})
    assert suggest.resolve_reference(q, yt=yt).video_id == vid(9)
    assert suggest.resolve_reference("MC Nada - Nada", yt=FakeYT()) is None
    assert suggest.resolve_reference("   ", yt=FakeYT()) is None


def test_resolve_reference_search_error_raises():
    with pytest.raises(suggest.SuggestError) as e:
        suggest.resolve_reference("MC X - Y", yt=FakeYT(fail_search=True))
    assert "Não consegui buscar" in str(e.value)


def test_resolve_reference_link_uses_watch_playlist():
    yt = FakeYT(radios={"bN8S3cRF9X0": [radio_item("bN8S3cRF9X0", "Casei Com a Putaria", ["MC Ryan SP"])]})
    t = suggest.resolve_reference("https://music.youtube.com/watch?v=bN8S3cRF9X0&list=RDAMVM", yt=yt)
    assert t.video_id == "bN8S3cRF9X0"
    assert t.artists == ["MC Ryan SP"]
    assert t.duration_s == 158.0
    assert t.thumbnail_url == "https://img/544.jpg"
    assert not any(c[0] == "search" for c in yt.calls)


def test_resolve_reference_video_id_falls_back_to_get_song():
    song = {"videoDetails": {"videoId": "4cJgIfrO8jM", "title": "Fui Mlk", "author": "Nilo",
                             "lengthSeconds": "175", "thumbnail": {"thumbnails": thumbs(120, 480)}}}
    yt = FakeYT(song=song, fail_radio={"4cJgIfrO8jM"})
    t = suggest.resolve_reference("4cJgIfrO8jM", yt=yt)
    assert (t.video_id, t.title, t.artists, t.duration_s) == ("4cJgIfrO8jM", "Fui Mlk", ["Nilo"], 175.0)
    assert t.thumbnail_url == "https://img/480.jpg"


def test_resolve_reference_unknown_link_returns_none():
    assert suggest.resolve_reference("https://youtu.be/bN8S3cRF9X0", yt=FakeYT()) is None


# ---------------------------------------------------------------------------
# 4.2 rank_suggestions (função pura)
# ---------------------------------------------------------------------------
REF_A = ref_track("refAAAAAAAA", "Casei Com a Putaria", ["MC Ryan SP", "Kotim"])
REF_B = ref_track("refBBBBBBBB", "Felina", ["WIU", "MC Ryan SP"])


def test_rank_dedup_score_best_rank_and_order():
    radios = {
        REF_A.video_id: [
            radio_item(REF_A.video_id, REF_A.title, REF_A.artists),       # 1ª = a própria referência
            radio_item(vid(1), "Só no A", ["MC 1"]),                       # pos 2
            radio_item(vid(2), "Nos dois", ["MC 2"]),                      # pos 3
            radio_item(vid(1), "Só no A", ["MC 1"]),                       # duplicata no mesmo rádio
            radio_item(vid(3), "Nos dois de novo", ["MC 3"]),              # pos 5
        ],
        REF_B.video_id: [
            radio_item(REF_B.video_id, REF_B.title, REF_B.artists),
            radio_item(vid(3), "Nos dois de novo", ["MC 3"]),              # pos 2 -> best_rank 2
            radio_item(vid(4), "Só no B", ["MC 4"]),                       # pos 3
            radio_item(vid(2), "Nos dois", ["MC 2"]),                      # pos 4
            radio_item(REF_A.video_id, REF_A.title, REF_A.artists),        # referência no outro rádio
        ],
    }
    out = suggest.rank_suggestions(radios, [REF_A, REF_B])
    ids = [t.video_id for t in out]
    assert ids == [vid(3), vid(2), vid(1), vid(4)]
    by_id = {t.video_id: t for t in out}
    assert (by_id[vid(3)].score, by_id[vid(3)].best_rank) == (2, 2)
    assert (by_id[vid(2)].score, by_id[vid(2)].best_rank) == (2, 3)
    assert (by_id[vid(1)].score, by_id[vid(1)].best_rank) == (1, 2)
    assert (by_id[vid(4)].score, by_id[vid(4)].best_rank) == (1, 3)
    assert all(t.source == "ytmusic" for t in out)
    assert REF_A.video_id not in ids and REF_B.video_id not in ids


def test_rank_removes_other_upload_of_reference():
    radios = {REF_A.video_id: [
        radio_item(vid(1), "Casei Com a Putaria (feat. Kotim)", ["MC Ryan SP"]),     # outro upload
        radio_item(vid(2), "MC Ryan SP - Casei com a Putaria (Clipe Oficial)", []),  # vídeo com artista no título
        radio_item(vid(3), "Casei Com a Putaria", ["Outro MC"]),                    # mesmo nome, outro artista
        radio_item(vid(4), "Casamento", ["MC Ryan SP"]),
    ]}
    ids = [t.video_id for t in suggest.rank_suggestions(radios, [REF_A])]
    assert ids == [vid(3), vid(4)]


def test_rank_keyword_filter_normalized_and_can_be_disabled():
    radios = {REF_A.video_id: [
        radio_item(vid(1), "Fui Mlk (Slowed + Reverb)", ["Nilo"]),
        radio_item(vid(2), "Felina - SPED UP", ["WIU"]),
        radio_item(vid(3), "Baile 8D Áudio", ["MC X"]),
        radio_item(vid(4), "Reverbera", ["MC Y"]),                # "reverb" só como palavra inteira
        radio_item(vid(5), "Sem Aliança no Dedo", ["mc xenon"]),
    ]}
    ids = [t.video_id for t in suggest.rank_suggestions(radios, [REF_A])]
    assert ids == [vid(4), vid(5)]
    ids_custom = [t.video_id for t in suggest.rank_suggestions(radios, [REF_A], exclude_keywords=["ALIANÇA"])]
    assert ids_custom == [vid(1), vid(2), vid(3), vid(4)]
    assert len(suggest.rank_suggestions(radios, [REF_A], exclude_keywords=[])) == 5


def test_rank_duration_filter():
    radios = {REF_A.video_id: [
        radio_item(vid(1), "Vinheta", ["DJ"], length="0:45"),
        radio_item(vid(2), "Set ao vivo", ["DJ"], length="1:02:03"),
        radio_item(vid(3), "Sem duração", ["MC"], length=None),
        radio_item(vid(4), "Normal", ["MC"], length="2:55"),
        {**radio_item(vid(5), "Pela busca", ["MC"], length=None), "duration_seconds": 30},
    ]}
    out = suggest.rank_suggestions(radios, [REF_A])
    assert [t.video_id for t in out] == [vid(3), vid(4)]
    assert out[0].duration_s is None and out[1].duration_s == 175.0
    wide = suggest.rank_suggestions(radios, [REF_A], min_duration_s=10, max_duration_s=4000)
    assert len(wide) == 5


def test_rank_uses_config_at_call_time(monkeypatch):
    monkeypatch.setattr(config, "EXCLUDE_KEYWORDS", ["remix"])
    monkeypatch.setattr(config, "MIN_TRACK_DURATION_S", 200)
    radios = {REF_A.video_id: [
        radio_item(vid(1), "Fui Mlk Remix", ["Nilo"], length="4:00"),
        radio_item(vid(2), "Curta", ["MC"], length="2:00"),
        radio_item(vid(3), "Longa slowed", ["MC"], length="4:00"),
    ]}
    assert [t.video_id for t in suggest.rank_suggestions(radios, [REF_A])] == [vid(3)]


def test_rank_ignores_bad_items_and_empty_input():
    radios = {REF_A.video_id: [None, {"title": "sem id"}, "lixo", radio_item(vid(1), "Ok", ["MC"])]}
    assert [t.video_id for t in suggest.rank_suggestions(radios, [REF_A])] == [vid(1)]
    assert suggest.rank_suggestions({}, [REF_A]) == []


# ---------------------------------------------------------------------------
# 4.2 get_suggestions
# ---------------------------------------------------------------------------
def many(prefix: int, n: int, ref: Track | None = None) -> list[dict]:
    items = [radio_item(ref.video_id, ref.title, ref.artists)] if ref else []
    items += [radio_item(vid(prefix + i), f"Faixa {prefix + i}", [f"MC {prefix + i}"]) for i in range(n)]
    return items


def test_get_suggestions_merges_radios_in_parallel():
    refs = [REF_A, REF_B, ref_track("refCCCCCCCC", "Sereia", ["Orochi"])]
    shared = radio_item(vid(999), "Comum", ["MC Comum"])
    yt = FakeYT(radios={
        REF_A.video_id: many(100, 15, REF_A) + [shared],
        REF_B.video_id: many(200, 15, REF_B) + [shared],
        "refCCCCCCCC": many(300, 15) + [shared],
    }, delay=0.1)
    res = suggest.get_suggestions(refs, yt=yt)
    assert res.source == "ytmusic"
    assert res.warnings == []
    assert len(res.tracks) == 46
    assert res.tracks[0].video_id == vid(999) and res.tracks[0].score == 3
    assert yt.max_active > 1  # rádios em paralelo
    assert sorted(c[1] for c in yt.calls if c[0] == "watch") == sorted(r.video_id for r in refs)
    assert all(c[2] is True for c in yt.calls if c[0] == "watch")


def test_get_suggestions_one_radio_fails_warns():
    yt = FakeYT(radios={REF_A.video_id: many(100, 25, REF_A)}, fail_radio={REF_B.video_id})
    res = suggest.get_suggestions([REF_A, REF_B], yt=yt)
    assert len(res.tracks) == 25
    assert any("Felina" in w and "rádio" in w for w in res.warnings)


def test_get_suggestions_requires_references():
    with pytest.raises(suggest.SuggestError):
        suggest.get_suggestions([], yt=FakeYT())


def test_get_suggestions_all_fail_without_key_raises():
    yt = FakeYT(fail_radio={REF_A.video_id})
    with pytest.raises(suggest.SuggestError) as e:
        suggest.get_suggestions([REF_A], yt=yt)
    assert "YOUTUBE_API_KEY" in str(e.value)


def test_get_suggestions_few_without_key_warns():
    yt = FakeYT(radios={REF_A.video_id: many(100, 5, REF_A)})
    res = suggest.get_suggestions([REF_A], yt=yt)
    assert len(res.tracks) == 5
    assert any("Só encontrei 5" in w for w in res.warnings)


def test_get_suggestions_exclude_keywords_passed_through():
    items = many(100, 25, REF_A) + [radio_item(vid(1), "Algo Slowed", ["MC"])]
    yt = FakeYT(radios={REF_A.video_id: items})
    assert all(t.video_id != vid(1) for t in suggest.get_suggestions([REF_A], yt=yt).tracks)
    assert any(t.video_id == vid(1) for t in suggest.get_suggestions([REF_A], exclude_keywords=[], yt=yt).tracks)


def test_get_suggestions_local_reference_ignored():
    local = Track(video_id="local_abc", title="Arquivo", local_path="C:/x.mp3", source="local")
    yt = FakeYT(radios={REF_A.video_id: many(100, 25, REF_A)})
    res = suggest.get_suggestions([REF_A, local], yt=yt)
    assert len(res.tracks) == 25
    assert any("arquivo local" in w for w in res.warnings)
    assert all(c[1] != "local_abc" for c in yt.calls)


# ---------------------------------------------------------------------------
# Plano B: YouTube Data API (requests falso)
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, status: int, payload):
        self.status_code = status
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def api_search_item(video_id: str, title: str, channel: str, live: str = "none") -> dict:
    return {"id": {"kind": "youtube#video", "videoId": video_id},
            "snippet": {"title": title, "channelTitle": channel, "liveBroadcastContent": live,
                        "thumbnails": {"default": {"url": f"https://i/{video_id}/d.jpg", "width": 120, "height": 90},
                                       "high": {"url": f"https://i/{video_id}/h.jpg", "width": 480, "height": 360}}}}


def api_video(video_id: str, title: str, channel: str, duration: str) -> dict:
    return {"id": video_id, "contentDetails": {"duration": duration},
            "snippet": {"title": title, "channelTitle": channel,
                        "thumbnails": {"high": {"url": f"https://i/{video_id}/h.jpg", "width": 480, "height": 360}}}}


class FakeYouTubeAPI:
    """Substitui requests.get: search.list e videos.list."""

    def __init__(self, search: dict[str, list[dict]], videos: dict[str, dict], error: FakeResponse | None = None):
        self.search = search
        self.videos = videos
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {}),
                           "timeout": timeout})
        if self.error is not None:
            return self.error
        if url.endswith("/search"):
            return FakeResponse(200, {"items": self.search.get(params["q"], [])})
        if url.endswith("/videos"):
            ids = params["id"].split(",")
            return FakeResponse(200, {"items": [self.videos[i] for i in ids if i in self.videos]})
        return FakeResponse(404, {})


def make_api() -> FakeYouTubeAPI:
    style = config.YOUTUBE_STYLE_QUERY
    search = {
        f"MC Ryan SP {style}": [
            api_search_item("ytA00000001", "MC Ryan SP - Felina (Clipe Oficial)", "GR6 EXPLODE"),
            api_search_item("ytA00000002", "Baile ao vivo", "Canal", live="live"),
            api_search_item("ytA00000003", "Set de 1 hora", "DJ"),
            api_search_item("ytA00000004", "MC Ryan SP - Casei Com a Putaria (Clipe Oficial)", "GR6"),
            api_search_item("ytA00000005", "Vídeo removido", "X"),
            api_search_item("ytA00000008", "MC Ryan SP - Casamento", "MC Ryan SP"),
        ],
        f"WIU {style}": [
            api_search_item("ytA00000006", "Coração de Gelo slowed reverb", "WIU"),
            api_search_item("ytA00000001", "MC Ryan SP - Felina (Clipe Oficial)", "GR6 EXPLODE"),
            api_search_item("ytA00000007", "Lágrimas &amp; Sorrisos", "WIU - Topic"),
        ],
    }
    videos = {
        "ytA00000001": api_video("ytA00000001", "MC Ryan SP - Felina (Clipe Oficial)", "GR6 EXPLODE", "PT3M46S"),
        "ytA00000002": api_video("ytA00000002", "Baile ao vivo", "Canal", "P0D"),
        "ytA00000003": api_video("ytA00000003", "Set de 1 hora", "DJ", "PT1H0M5S"),
        "ytA00000004": api_video("ytA00000004", "MC Ryan SP - Casei Com a Putaria (Clipe Oficial)", "GR6", "PT2M40S"),
        "ytA00000006": api_video("ytA00000006", "Coração de Gelo slowed reverb", "WIU", "PT3M"),
        "ytA00000007": api_video("ytA00000007", "Lágrimas &amp; Sorrisos", "WIU - Topic", "PT2M10S"),
        "ytA00000008": api_video("ytA00000008", "MC Ryan SP - Casamento", "MC Ryan SP", "PT3M12S"),
    }
    return FakeYouTubeAPI(search, videos)


def test_suggest_from_youtube_api(monkeypatch):
    api = make_api()
    monkeypatch.setattr(suggest.requests, "get", api)
    refs = [ref_track("refAAAAAAAA", "Casei Com a Putaria", ["MC Ryan SP"]),
            ref_track("refBBBBBBBB", "Sereia", ["WIU", "MC Ryan SP"])]
    out = suggest.suggest_from_youtube_api(refs, "chave-falsa", max_results=10)
    by_id = {t.video_id: t for t in out}
    # Felina aparece nas duas buscas; a referência (outro upload), o set longo, a live,
    # o vídeo sem detalhes e o "slowed reverb" saem.
    assert [t.video_id for t in out] == ["ytA00000001", "ytA00000007", "ytA00000008"]
    felina = by_id["ytA00000001"]
    assert (felina.title, felina.artists, felina.duration_s, felina.score) == ("Felina", ["MC Ryan SP"], 226.0, 2)
    assert felina.source == "youtube_api"
    assert felina.thumbnail_url == "https://i/ytA00000001/h.jpg"
    assert by_id["ytA00000007"].artists == ["WIU"]
    assert by_id["ytA00000007"].title == "Lágrimas & Sorrisos"

    searches = [c for c in api.calls if c["url"].endswith("/search")]
    assert len(searches) == 2  # "MC Ryan SP" não se repete
    p = searches[0]["params"]
    assert (p["type"], p["videoCategoryId"], p["regionCode"], p["relevanceLanguage"], p["maxResults"]) == \
        ("video", "10", "BR", "pt", 10)
    videos = [c for c in api.calls if c["url"].endswith("/videos")]
    assert videos and videos[0]["params"]["part"] == "contentDetails,snippet"
    for c in api.calls:  # a chave vai no cabeçalho, nunca na URL
        assert "chave-falsa" not in str(c["params"]) and "chave-falsa" not in c["url"]
        assert c["headers"].get("X-Goog-Api-Key") == "chave-falsa"
        assert c["timeout"]


def test_suggest_from_youtube_api_errors(monkeypatch):
    with pytest.raises(suggest.SuggestError):
        suggest.suggest_from_youtube_api([REF_A], "")
    quota = FakeResponse(403, {"error": {"code": 403, "message": "quota",
                                         "errors": [{"reason": "quotaExceeded"}]}})
    monkeypatch.setattr(suggest.requests, "get", FakeYouTubeAPI({}, {}, error=quota))
    with pytest.raises(suggest.SuggestError, match="cota"):
        suggest.suggest_from_youtube_api([REF_A], "k")
    bad_key = FakeResponse(400, {"error": {"message": "API key not valid. Please pass a valid API key.",
                                           "status": "INVALID_ARGUMENT"}})
    monkeypatch.setattr(suggest.requests, "get", FakeYouTubeAPI({}, {}, error=bad_key))
    with pytest.raises(suggest.SuggestError, match="inválida"):
        suggest.suggest_from_youtube_api([REF_A], "k")

    def offline(*a, **kw):
        raise requests.ConnectionError("sem rede")

    monkeypatch.setattr(suggest.requests, "get", offline)
    with pytest.raises(suggest.SuggestError, match="sem conexão"):
        suggest.suggest_from_youtube_api([REF_A], "k")


def test_get_suggestions_plan_b_when_radio_fails(monkeypatch):
    monkeypatch.setattr(suggest.requests, "get", make_api())
    yt = FakeYT(fail_radio={REF_A.video_id})
    res = suggest.get_suggestions([REF_A], yt=yt, youtube_api_key="k")
    assert res.source == "youtube_api"
    assert [t.video_id for t in res.tracks] == ["ytA00000001", "ytA00000008"]
    assert any("plano B" in w for w in res.warnings)


def test_get_suggestions_plan_b_completes_short_radio(monkeypatch):
    monkeypatch.setattr(suggest.requests, "get", make_api())
    monkeypatch.setattr(config, "get_secret", lambda name: "k-do-env" if name == "YOUTUBE_API_KEY" else None)
    radio = many(100, 3, REF_A) + [radio_item("ytA00000001", "Felina", ["WIU", "MC Ryan SP"])]
    res = suggest.get_suggestions([REF_A], yt=FakeYT(radios={REF_A.video_id: radio}))
    assert res.source == "ytmusic"
    ids = [t.video_id for t in res.tracks]
    assert ids[:4] == [vid(100), vid(101), vid(102), "ytA00000001"]
    assert ids.count("ytA00000001") == 1           # já veio do rádio
    assert ids[4:] == ["ytA00000008"] and res.tracks[-1].source == "youtube_api"  # completou com o plano B
    assert any("completei" in w for w in res.warnings)


def test_get_suggestions_plan_b_failure_is_warning(monkeypatch):
    quota = FakeResponse(403, {"error": {"errors": [{"reason": "quotaExceeded"}]}})
    monkeypatch.setattr(suggest.requests, "get", FakeYouTubeAPI({}, {}, error=quota))
    yt = FakeYT(radios={REF_A.video_id: many(100, 5, REF_A)})
    res = suggest.get_suggestions([REF_A], yt=yt, youtube_api_key="k")
    assert len(res.tracks) == 5
    assert any("cota" in w for w in res.warnings)
    with pytest.raises(suggest.SuggestError, match="cota"):
        suggest.get_suggestions([REF_A], yt=FakeYT(fail_radio={REF_A.video_id}), youtube_api_key="k")


def test_get_suggestions_no_api_call_when_radio_is_enough(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("não deveria chamar a YouTube Data API")

    monkeypatch.setattr(suggest.requests, "get", boom)
    yt = FakeYT(radios={REF_A.video_id: many(100, config.N_SUGGESTIONS, REF_A)})
    res = suggest.get_suggestions([REF_A], yt=yt, youtube_api_key="k")
    assert len(res.tracks) == config.N_SUGGESTIONS


# ---------------------------------------------------------------------------
# Teste real (rede)
# ---------------------------------------------------------------------------
@pytest.mark.network
def test_real_radio_mandelao():
    t0 = time.perf_counter()
    ref = suggest.resolve_reference("MC Ryan SP - Casei com a Putaria")
    assert ref is not None and ref.video_id
    assert any("ryan" in util.normalize_text(a) for a in ref.artists)
    t1 = time.perf_counter()
    res = suggest.get_suggestions([ref])
    t2 = time.perf_counter()
    assert len(res.tracks) >= 20, res.warnings
    assert res.source == "ytmusic"
    assert all(t.video_id != ref.video_id for t in res.tracks)
    assert len({t.video_id for t in res.tracks}) == len(res.tracks)
    print(f"\nresolver: {t1 - t0:.2f}s, rádio: {t2 - t1:.2f}s, {len(res.tracks)} sugestões; "
          f"1ª: {res.tracks[0].display_name}")
