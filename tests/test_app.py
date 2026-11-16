"""Testes da interface (app.py, com AppTest) e dos desenhos SVG (modules/viz.py).

Os módulos de backend são trocados por fakes (monkeypatch) ANTES de cada `at.run()`:
a interface é testada sem rede, sem áudio real e sem depender da implementação dos
outros módulos (que podem ainda levantar NotImplementedError).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import config
from modules import (analyze, budget, download, explain, export, jobs, maintenance, metrics, mixwindows,
                     order, preview, spotify_import, state, stems, suggest, viz)
from modules import setlist_insights as si
from modules.models import Analysis, StemResult, Track, Transition

APP = Path(__file__).resolve().parent.parent / "app.py"
N = config.N_SUGGESTIONS
SVG_NS = "{http://www.w3.org/2000/svg}"
_RealMetrics = metrics.SessionMetrics


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
def make_track(i: int) -> Track:
    return Track(video_id=f"vid{i:08d}", title=f"Faixa {i}", artists=[f"MC Teste {i}"], duration_s=180.0)


def make_analysis(vid: str, i: int = 0) -> Analysis:
    return Analysis(
        video_id=vid, duration_s=180.0, bpm=128.0 + (i % 5), bpm_raw=64.0 + (i % 5) / 2,
        energy=[0.2, 0.5, 0.9, 0.7, 0.3], intro_s=10.0, outro_s=12.0, energy_in=0.3,
        energy_out=0.4, energy_mean=0.5 + i / 100, waveform=[0.1, 0.8, 0.5] * 20,
        key="A minor", camelot="8A", key_confidence=0.8,
    )


def vid(i: int) -> str:
    return make_track(i).video_id


class FakeJobs:
    """Fila que deixa tudo pronto na hora."""

    def __init__(self, *args, **kwargs):
        self.submitted: list[str] = []
        self.analyses: dict[str, Analysis] = {}
        self.statuses: dict[str, jobs.JobStatus] = {}
        self.closed = False

    def submit(self, tracks):
        for t in tracks:
            if t.video_id not in self.submitted:
                self.submitted.append(t.video_id)
            self.statuses[t.video_id] = jobs.JobStatus(state=jobs.READY, progress=1.0)
            self.analyses.setdefault(t.video_id, make_analysis(t.video_id, len(self.analyses)))

    def status(self, video_id):
        st = self.statuses.get(video_id)
        return jobs.JobStatus(st.state, st.progress, st.error) if st else jobs.JobStatus()

    def summary(self):
        out = {"total": len(self.statuses), jobs.QUEUED: 0, jobs.DOWNLOADING: 0, jobs.ANALYZING: 0,
               jobs.READY: 0, jobs.ERROR: 0}
        for s in self.statuses.values():
            out[s.state] += 1
        out["baixadas"] = out[jobs.ANALYZING] + out[jobs.READY]
        return out

    def get_analysis(self, video_id):
        return self.analyses.get(video_id)

    def update_analysis(self, analysis):
        self.analyses[analysis.video_id] = analysis

    def all_done(self):
        return all(s.state in (jobs.READY, jobs.ERROR) for s in self.statuses.values())

    def shutdown(self, wait=False):
        self.closed = True


class FakeBudget:
    allowed: tuple[bool, str] = (True, "")

    def __init__(self, *args, **kwargs):
        self.cap_brl = 10.0
        self.credit_usd = 2.0
        self.usd_brl = 5.4
        self.spent = 0.0          # US$
        self.insufficient_credit = False

    @property
    def spent_usd(self):
        return self.spent

    @property
    def spent_brl(self):
        return self.spent * self.usd_brl

    @property
    def remaining_brl(self):
        return max(0.0, self.cap_brl - self.spent_brl)

    @property
    def remaining_credit_usd(self):
        return max(0.0, self.credit_usd - self.spent)

    @property
    def fraction_used(self):
        return min(1.0, self.spent_brl / self.cap_brl)

    @property
    def entries(self):
        return []

    def can_run(self, estimate_usd=None):
        return type(self).allowed

    def record(self, video_id, prediction_id, predict_time_s, status="succeeded"):
        self.spent += 0.016
        return {}

    def mark_insufficient_credit(self, message=""):
        self.insufficient_credit = True

    def reset_insufficient_credit(self):
        self.insufficient_credit = False


class LockedBudget(FakeBudget):
    allowed = (False, "Teto de gasto atingido: R$ 9,95 usados de R$ 10,00.")


class FakeMetrics(_RealMetrics):
    """Cronômetro de verdade com relógio parado: a sessão sempre tem 120 s."""

    NOW = 1_000_000.0

    def __init__(self, *args, **kwargs):
        super().__init__(clock=lambda: FakeMetrics.NOW)
        self.finish_calls: list[tuple[int, int]] = []

    def start(self):
        super().start()
        self._start = FakeMetrics.NOW - 120.0

    def finish(self, n_approved, n_reviewed):
        self.finish_calls.append((n_approved, n_reviewed))
        return super().finish(n_approved, n_reviewed)


@dataclass
class Fakes:
    audio: Path
    snapshot: dict | None = None
    saves: list[dict] = field(default_factory=list)
    cleared: int = 0
    resolve_calls: list[str] = field(default_factory=list)
    suggest_calls: list = field(default_factory=list)
    suggest_refs: list[list[str]] = field(default_factory=list)
    order_calls: list[dict] = field(default_factory=list)
    export_calls: list[dict] = field(default_factory=list)
    purge_calls: list[list[str]] = field(default_factory=list)
    bpm_calls: list[tuple] = field(default_factory=list)
    stem_calls: list[str] = field(default_factory=list)
    preview_calls: list[dict] = field(default_factory=list)
    reset_calls: list[list[str]] = field(default_factory=list)
    n_suggestions: int = 45
    suggest_error: Exception | None = None
    radio_extra: list[Track] = field(default_factory=list)


@pytest.fixture
def fakes(monkeypatch, tmp_dirs, tmp_path):
    audio = tmp_path / "faixa.mp3"
    audio.write_bytes(b"ID3" + b"\x00" * 256)
    fk = Fakes(audio=audio)

    # suggest
    def resolve_reference(query, yt=None):
        fk.resolve_calls.append(query)
        artist, _, title = query.partition(" - ")
        return Track(video_id="ref00000001", title=title or query, artists=[artist], duration_s=170.0)

    def get_suggestions(references, exclude_keywords=None, yt=None, youtube_api_key=None):
        fk.suggest_calls.append(exclude_keywords)
        fk.suggest_refs.append([r.video_id for r in references])
        if fk.suggest_error is not None:
            raise fk.suggest_error
        tracks = [make_track(i) for i in range(fk.n_suggestions)] + list(fk.radio_extra)
        return suggest.SuggestionResult(tracks=tracks, source="ytmusic", warnings=["Aviso de teste do rádio."])

    monkeypatch.setattr(suggest, "resolve_reference", resolve_reference)
    monkeypatch.setattr(suggest, "get_suggestions", get_suggestions)

    # fila, download e análise
    monkeypatch.setattr(jobs, "TrackJobs", FakeJobs)
    monkeypatch.setattr(download, "audio_path_for", lambda track, cache_dir=None: audio)
    monkeypatch.setattr(download, "local_tracks_from_folder",
                        lambda folder, recursive=False: [Track(video_id="local_abc123def456", title="Local",
                                                               artists=["DJ"], source="local",
                                                               local_path=str(audio), duration_s=150.0)])

    def purge(video_ids, cache_dir=None):
        ids = list(video_ids)
        fk.purge_calls.append(ids)
        return 2 * len(ids)

    monkeypatch.setattr(download, "purge", purge)

    def set_manual_bpm(video_id, bpm, cache_dir=None):
        fk.bpm_calls.append((video_id, bpm))
        an = make_analysis(video_id)
        an.bpm_manual = bpm
        return an

    monkeypatch.setattr(analyze, "set_manual_bpm", set_manual_bpm)

    # sessão salva
    def save_session(snapshot, data_dir=None):
        fk.saves.append(dict(snapshot))

    def clear_session(data_dir=None):
        fk.cleared += 1

    monkeypatch.setattr(state, "save_session", save_session)
    monkeypatch.setattr(state, "load_session", lambda data_dir=None: fk.snapshot)
    monkeypatch.setattr(state, "clear_session", clear_session)

    # ordenação e janelas de voz
    def evaluate_order(order_ids, analyses, weights=None, voice_lookup=None, *, energy_mode="suave"):
        kept = [v for v in order_ids if v in analyses]
        trs = [Transition(from_id=a, to_id=b, cost=0.12, bpm_diff=1.0, energy_diff=0.1, camelot_dist=0,
                          voice=voice_lookup(a, b)[0] if voice_lookup else "sem stems", label="suave")
               for a, b in zip(kept, kept[1:])]
        return order.OrderResult(order=kept, transitions=trs, total_cost=0.12 * len(trs))

    def order_set(analyses, weights=None, start_id=None, voice_lookup=None, *, energy_mode="suave"):
        fk.order_calls.append({"ids": [a.video_id for a in analyses], "weights": weights,
                               "start_id": start_id, "voice_lookup": voice_lookup, "energy_mode": energy_mode})
        ids = sorted((a.video_id for a in analyses), reverse=True)
        return evaluate_order(ids, {a.video_id: a for a in analyses}, weights, voice_lookup)

    monkeypatch.setattr(order, "order_set", order_set)
    monkeypatch.setattr(order, "evaluate_order", evaluate_order)
    monkeypatch.setattr(mixwindows, "make_voice_lookup", lambda stems_dir=None: (lambda a, b: ("limpa", 0.0)))

    # prévia da passagem (sem librosa)
    def make_transition_preview(path_a, path_b, a_id, b_id, bpm_a=None, bpm_b=None, **kwargs):
        fk.preview_calls.append({"a": a_id, "b": b_id, "bpm_a": bpm_a, "bpm_b": bpm_b,
                                 "path_a": Path(path_a), "path_b": Path(path_b)})
        return audio

    monkeypatch.setattr(preview, "make_transition_preview", make_transition_preview)
    monkeypatch.setattr(preview, "warm_up", lambda: 0.0)

    # exportação
    def export_setlist(tracks, analyses, transitions=None, out_dir=None, playlist_name=None, cache_dir=None):
        fk.export_calls.append({"ids": [t.video_id for t in tracks], "transitions": transitions})
        folder = Path(config.EXPORT_DIR)
        files = [folder / f"{i + 1:02d} - {t.display_name}.mp3" for i, t in enumerate(tracks)]
        return export.ExportResult(folder=folder, m3u8_path=folder / "Setlist_Atual.m3u8",
                                   csv_path=folder / "Setlist_Atual.csv", files=files, skipped=[])

    monkeypatch.setattr(export, "export_setlist", export_setlist)

    # zerar painel (nunca apaga nada de verdade nos testes)
    def usage():
        return {c: {"files": 2, "bytes": 2048, "path": str(tmp_path / c), "paths": [str(tmp_path / c)],
                    "safe": True} for c in maintenance.CATEGORIES}

    def reset(categories):
        cats = [categories] if isinstance(categories, str) else list(categories)
        fk.reset_calls.append(cats)
        return maintenance.ResetReport(removed_files=3 * len(cats), freed_bytes=1024 * 1024,
                                       per_category={c: {"files": 3, "bytes": 1} for c in cats})

    monkeypatch.setattr(maintenance, "usage", usage)
    monkeypatch.setattr(maintenance, "reset", reset)

    # orçamento, métricas, stems e opcionais
    monkeypatch.setattr(budget, "Budget", FakeBudget)
    monkeypatch.setattr(metrics, "SessionMetrics", FakeMetrics)
    monkeypatch.setattr(stems, "has_stems", lambda video_id, base_dir=None: False)
    monkeypatch.setattr(stems, "load_stems", lambda video_id, base_dir=None: None)
    monkeypatch.setattr(stems, "local_demucs_available", lambda: False)

    def separate_replicate(video_id, audio_path, budget_obj, token=None, base_dir=None, client=None):
        fk.stem_calls.append(video_id)
        budget_obj.record(video_id, "pred-1", 70.0)
        return StemResult(video_id=video_id, stems={"vocals": str(audio)}, backend="replicate",
                          predict_time_s=70.0, cost_usd=0.01575, cost_brl=0.085)

    monkeypatch.setattr(stems, "separate_replicate", separate_replicate)
    monkeypatch.setattr(explain, "is_available", lambda: False)
    monkeypatch.setattr(spotify_import, "is_configured", lambda: False)
    # Valor de mentira (não é segredo): só indica que o token "existe".
    monkeypatch.setattr(config, "get_secret",
                        lambda name: "token-falso-para-teste" if name == "REPLICATE_API_TOKEN" else None)
    return fk


@dataclass
class SpotifyFakes:
    login_calls: int = 0
    search_calls: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)
    export_calls: list[dict] = field(default_factory=list)
    import_calls: list[tuple] = field(default_factory=list)


@pytest.fixture
def spotify(monkeypatch, fakes):
    sf = SpotifyFakes()
    monkeypatch.setattr(spotify_import, "is_configured", lambda: True)
    monkeypatch.setattr(spotify_import, "saved_login_state", lambda: "none")

    def search_tracks(query, limit=10):
        sf.search_calls.append(query)
        return [{"title": "Casei com a <Putaria>", "artists": ["MC Ryan SP"], "duration_s": 150.0,
                 "isrc": "BR1234567890", "spotify_id": "sp1", "cover_url": None,
                 "spotify_url": "https://open.spotify.com/track/sp1", "album": "Álbum", "uri": "spotify:track:sp1"}]

    def resolve_on_ytmusic(sp_track, yt=None):
        sf.resolved.append(sp_track.title)
        return Track(video_id="spref000001", title=sp_track.title, artists=list(sp_track.artists),
                     duration_s=150.0, source="spotify", isrc=sp_track.isrc)

    def get_client(open_browser=True):
        sf.login_calls += 1
        return SimpleNamespace(name="cliente-falso")

    def export_set_to_spotify(sp, name, tracks, description="", public=False, progress=None):
        sf.export_calls.append({"name": name, "ids": [t.video_id for t in tracks]})
        if progress is not None:
            progress(len(tracks), len(tracks))
        return {"playlist_id": "pl123", "playlist_url": "https://open.spotify.com/playlist/pl123",
                "added": len(tracks) - 1, "missing": [tracks[-1].display_name]}

    def import_playlist(sp, playlist_id, yt=None, limit=None, progress=None):
        sf.import_calls.append((playlist_id, limit))
        tracks = [make_track(100 + i) for i in range(3)]
        return tracks, ["Faixa Sumida"]

    monkeypatch.setattr(spotify_import, "search_tracks", search_tracks)
    monkeypatch.setattr(spotify_import, "resolve_on_ytmusic", resolve_on_ytmusic)
    monkeypatch.setattr(spotify_import, "get_client", get_client)
    monkeypatch.setattr(spotify_import, "current_user_info",
                        lambda sp: {"id": "djsk", "name": "DJ SK", "image_url": None, "product": "premium"})
    monkeypatch.setattr(spotify_import, "list_playlists", lambda sp: [
        {"id": "pl1", "name": "Mandelão", "owner": "DJ SK", "owner_id": "djsk", "tracks_total": 12,
         "collaborative": False, "readable": True, "image_url": None, "url": None}])
    monkeypatch.setattr(spotify_import, "import_playlist", import_playlist)
    monkeypatch.setattr(spotify_import, "export_set_to_spotify", export_set_to_spotify)
    return sf


def new_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=60)


def texts(elements) -> list[str]:
    return [str(e.value) for e in elements]


def has_text(elements, fragment: str) -> bool:
    return any(fragment in t for t in texts(elements))


def html(at: AppTest) -> str:
    """Todo o HTML desenhado com st.html (página e barra lateral)."""
    return "\n".join(texts(at.html) + texts(at.sidebar.html))


def metric(at: AppTest, label: str) -> str:
    for m in at.metric:
        if m.label == label:
            return m.value
    raise AssertionError(f"métrica {label!r} não encontrada")


def keys(at: AppTest, prefix: str) -> list[str]:
    return [b.key for b in at.button if b.key and b.key.startswith(prefix)]


def search(at: AppTest, text: str = "MC Teste - Faixa Boa") -> AppTest:
    at.text_input(key="ref_0").input(text)
    at.button(key="btn_search").click()
    return at.run()


def decide(at: AppTest, i: int, approve: bool = True) -> AppTest:
    at.button(key=f"{'approve' if approve else 'reject'}_{vid(i)}").click()
    return at.run()


def with_order(fakes, n: int = 3) -> AppTest:
    at = search(new_app().run())
    for i in range(n):
        at = decide(at, i)
    at.button(key="btn_order").click()
    return at.run()


# ---------------------------------------------------------------------------
# Interface: topo, referências e triagem
# ---------------------------------------------------------------------------
def test_renderiza_sem_erro(fakes):
    at = new_app().run()
    assert not at.exception
    page = html(at)
    assert "DJ SK · Curadoria" in page
    # cabeçalho com o orçamento (R$ gasto / teto) e o crédito da Replicate
    assert "R$ 0,00" in page and "/ R$ 10,00" in page and "US$ 2,00" in page
    # barra de etapas e títulos das seções com âncoras
    assert "djsk-stepper" in page and 'href="#triagem"' in page and 'id="metricas"' in page
    assert "Referências" in page and "Exportar" in page
    assert at.button(key="btn_search").label == "🔎 Buscar sugestões"
    assert at.toggle(key="filter_slowed").value is True
    assert has_text(at.info, "Busque sugestões")
    # integrações na barra lateral, como etiquetas
    assert "Replicate" in page and "Gemini" in page and "ligado" in page and "desligado" in page
    assert at.button(key="btn_reset_open")


def test_mais_um_campo_de_referencia(fakes):
    at = new_app().run()
    for _ in range(config.MAX_REFERENCES - 1):
        at.button(key="btn_add_ref").click()
        at.run()
    assert len([t for t in at.text_input if (t.key or "").startswith("ref_")]) == config.MAX_REFERENCES
    assert at.button(key="btn_add_ref").disabled


def test_buscar_sugestoes_mostra_cards(fakes):
    at = search(new_app().run())
    assert not at.exception
    assert fakes.resolve_calls == ["MC Teste - Faixa Boa"]
    assert fakes.suggest_calls == [None]          # filtro slowed/reverb ligado = padrão do config
    assert len(keys(at, "approve_")) == N
    assert has_text(at.warning, "Aviso de teste do rádio.")
    assert has_text(at.markdown, "✔ MC Teste")
    assert at.session_state["jobs"].submitted == [vid(i) for i in range(N)]
    assert at.session_state["metrics"].started
    assert metric(at, "Pendentes") == str(N)
    # card mostra BPM (zona do funk), tom, intro/saída e os SVGs
    page = html(at)
    assert "128 BPM" in page and "zona 130 · mandelão" in page and "<b>8A</b>" in page
    assert "detectado 64 BPM: confira" in page        # BPM bruto dobrado pelo detector
    assert has_text(at.caption, "intro 10 s · saída 12 s")
    assert sum("<svg" in t for t in texts(at.markdown)) >= 2 * N
    assert fakes.saves and len(fakes.saves[-1]["suggestions"]) == fakes.n_suggestions
    # etapas: referências concluídas, triagem ativa com o contador
    assert "Triagem 0/20" in page and "djsk-step--active" in page


def test_filtro_slowed_desligado_envia_lista_vazia(fakes):
    at = new_app().run()
    at.toggle(key="filter_slowed").set_value(False)
    at = search(at)
    assert fakes.suggest_calls == [[]]


def test_carregar_mais_sugestoes(fakes):
    at = search(new_app().run())
    assert at.button(key="btn_more").label == f"Carregar mais {N} sugestões"
    at.button(key="btn_more").click()
    at.run()
    assert len(keys(at, "approve_")) == 2 * N
    assert len(at.session_state["jobs"].submitted) == 2 * N
    assert at.session_state["page"] == 2


def test_aprovar_e_reprovar_atualizam_contadores(fakes):
    at = search(new_app().run())
    at = decide(at, 0, approve=True)
    at = decide(at, 1, approve=False)
    assert not at.exception
    assert metric(at, "Aprovadas") == "1"
    assert metric(at, "Reprovadas") == "1"
    assert metric(at, "Pendentes") == str(N - 2)
    assert at.session_state["decisions"] == {vid(0): "aprovada", vid(1): "reprovada"}
    assert fakes.saves[-1]["decisions"] == {vid(0): "aprovada", vid(1): "reprovada"}
    # as decisões também vão para o cronômetro das métricas
    met = at.session_state["metrics"]
    assert {k: d for k, (d, _) in met.decisions.items()} == {vid(0): "aprovada", vid(1): "reprovada"}
    # modo rápido: o próximo pendente vira o ativo e toca do meio
    assert at.session_state["active"] == {"vid": vid(2), "pos": "meio"}
    # filtro padrão "Pendentes" esconde as decididas
    assert f"approve_{vid(0)}" not in keys(at, "approve_")
    at.segmented_control(key="triage_filter").set_value("Aprovadas")
    at.run()
    assert keys(at, "approve_") == [f"approve_{vid(0)}"]
    # desfazer tira a decisão do cronômetro também
    at.button(key=f"undo_{vid(0)}").click()
    at.run()
    assert vid(0) not in at.session_state["metrics"].decisions


def test_atalhos_so_no_card_em_foco(fakes):
    at = search(new_app().run())
    # sem nada tocando, o foco é o primeiro pendente
    assert at.session_state["focus_vid"] == vid(0)
    assert at.button(key=f"approve_{vid(0)}").proto.shortcut == "a"
    assert at.button(key=f"reject_{vid(0)}").proto.shortcut == "x"
    assert at.button(key=f"play_meio_{vid(0)}").proto.shortcut == "m"
    assert at.button(key=f"play_final_{vid(0)}").proto.shortcut == "f"
    assert at.button(key=f"play_inicio_{vid(0)}").proto.shortcut == "i"
    assert not at.button(key=f"approve_{vid(1)}").proto.shortcut
    assert "Atalhos" in html(at)
    # tocar outro card leva o foco (e os atalhos) para ele
    at.button(key=f"play_meio_{vid(3)}").click()
    at.run()
    assert at.session_state["focus_vid"] == vid(3)
    assert at.button(key=f"approve_{vid(3)}").proto.shortcut == "a"
    assert not at.button(key=f"approve_{vid(0)}").proto.shortcut


def test_ordenar_triagem_por_energia_e_bpm(fakes):
    at = search(new_app().run())
    at.segmented_control(key="triage_sort").set_value("Energia")
    at.run()
    # energia média cresce com o índice nos fakes: a mais forte primeiro
    assert keys(at, "approve_")[0] == f"approve_{vid(N - 1)}"
    at.segmented_control(key="triage_sort").set_value("BPM")
    at.run()
    first = keys(at, "approve_")[0]
    assert at.session_state["jobs"].analyses[first.removeprefix("approve_")].bpm == 128.0


def test_mais_parecidas_acrescenta_depois_da_faixa(fakes):
    at = search(new_app().run())
    fakes.radio_extra = [Track(video_id=f"new{i:08d}", title=f"Nova {i}", artists=["MC Novo"], duration_s=170.0)
                         for i in range(3)]
    at.button(key=f"similar_{vid(2)}").click()
    at.run()
    assert not at.exception
    assert fakes.suggest_refs[-1] == [vid(2)]
    ids = [t.video_id for t in at.session_state["suggestions"]]
    assert ids[3:6] == ["new00000000", "new00000001", "new00000002"]
    assert at.session_state["extra_visible"] == 3
    assert len(keys(at, "approve_")) == N + 3
    assert {"new00000000", "new00000001", "new00000002"} <= set(at.session_state["jobs"].submitted)
    assert "parecida com Faixa 2" in html(at)
    assert has_text(at.success, "3 música(s) parecida(s)")
    # de novo: nada novo
    at.button(key=f"similar_{vid(2)}").click()
    at.run()
    assert has_text(at.info, "Nenhuma música nova")


def test_botoes_do_player_ativam_o_card(fakes):
    at = search(new_app().run())
    at.button(key=f"play_final_{vid(3)}").click()
    at.run()
    assert at.session_state["active"] == {"vid": vid(3), "pos": "final"}
    assert at.button(key=f"stop_{vid(3)}")
    at.button(key=f"stop_{vid(3)}").click()
    at.run()
    assert at.session_state["active"] is None


def test_janela_de_reproducao():
    import importlib.util

    spec = importlib.util.spec_from_file_location("app_under_test", APP)
    # Só a função pura: evita executar a página inteira fora do Streamlit.
    src = APP.read_text(encoding="utf-8")
    start = src.index("def _play_window")
    end = src.index("\ndef ", start + 10)
    ns: dict = {"math": math, "FINAL_PREVIEW_S": 30}
    exec("from __future__ import annotations\n" + src[start:end], ns)
    play = ns["_play_window"]
    assert spec is not None
    assert play("inicio", 180.0) == (0, None)
    assert play("meio", 180.0) == (90, None)
    assert play("final", 180.4) == (150, 181)
    assert play("final", 20.0) == (0, 20)
    assert play("meio", None) == (0, None)


def test_corrigir_bpm(fakes):
    at = search(new_app().run())
    at.number_input(key=f"bpm_{vid(0)}").set_value(140.0)
    at.run()
    assert not at.exception
    assert fakes.bpm_calls == [(vid(0), 140.0)]
    assert at.session_state["jobs"].analyses[vid(0)].bpm_manual == 140.0
    assert "140 BPM · manual" in html(at)


# ---------------------------------------------------------------------------
# Setlist
# ---------------------------------------------------------------------------
def test_gerar_ordem_e_mover(fakes):
    at = with_order(fakes)
    assert not at.exception
    assert len(fakes.order_calls) == 1
    call = fakes.order_calls[0]
    assert sorted(call["ids"]) == [vid(0), vid(1), vid(2)]
    assert call["voice_lookup"] is not None and call["start_id"] is None
    assert call["weights"] == {k: pytest.approx(v) for k, v in config.DEFAULT_WEIGHTS.items()}
    assert call["energy_mode"] == "suave"
    assert at.session_state["order"] == [vid(2), vid(1), vid(0)]
    page = html(at)
    # fluidez, comparação com a ordem de aprovação, resumo e linha do tempo
    assert "Fluidez do set" in page and "Na ordem em que você aprovou" in page
    assert "Igual à ordem em que você aprovou" in page   # o fake dá o mesmo custo para qualquer ordem
    assert "djsk-tl-card" in page and "Cores das passagens" in page
    assert "2 passagens suaves" in page and "128–130 BPM" in page
    # cartões das passagens com o motivo em português simples
    assert "01 → 02" in page and "Passagem suave" in page and "Mesma velocidade" in page
    assert "Passagem sem voz" in page
    assert has_text(at.caption, "limpa ✔")
    # subir/descer recalcula com evaluate_order
    at.button(key=f"down_{vid(2)}").click()
    at.run()
    assert at.session_state["order"] == [vid(1), vid(2), vid(0)]
    assert at.button(key=f"up_{vid(1)}").disabled


def test_estilos_do_set_mudam_pesos_e_reordenam(fakes):
    at = with_order(fakes)
    assert at.pills(key="set_preset").value == si.DEFAULT_PRESET
    at.pills(key="set_preset").set_value("bpm")
    at.run()
    assert not at.exception
    expected = si.preset_weights("bpm")
    assert at.session_state["weights"] == {k: pytest.approx(v) for k, v in expected.items()}
    assert at.slider(key="w_bpm").value == pytest.approx(expected["bpm"])
    assert at.session_state["preset"] == "bpm"
    # já havia ordem: o novo estilo reordena sozinho
    assert len(fakes.order_calls) == 2
    assert fakes.order_calls[-1]["weights"] == {k: pytest.approx(v) for k, v in expected.items()}
    assert has_text(at.caption, "Velocidade travada")
    # "Energia subindo" liga o modo de energia
    at.pills(key="set_preset").set_value("subindo")
    at.run()
    assert at.session_state["energy_mode"] == "subindo"
    assert at.toggle(key="energy_rising").value is True
    assert fakes.order_calls[-1]["energy_mode"] == "subindo"
    assert fakes.saves[-1]["preset"] == "subindo" and fakes.saves[-1]["energy_mode"] == "subindo"


def test_ajuste_fino_explica_os_pesos(fakes):
    at = with_order(fakes)
    for k in ("bpm", "energia", "tom", "voz"):
        assert has_text(at.caption, si.WEIGHT_HELP[k])
    at.slider(key="w_tom").set_value(1.5)
    at.run()
    assert at.session_state["weights"]["tom"] == pytest.approx(1.5)
    assert has_text(at.caption, "com ajustes seus")


def test_arco_de_energia_mostra_aderencia(fakes):
    at = with_order(fakes)
    at.segmented_control(key="set_arc").set_value("subida")
    at.run()
    assert not at.exception
    assert "Segue o arco" in html(at)
    assert fakes.saves[-1]["set_arc"] == "subida"


def test_ouvir_passagem_chama_preview(fakes):
    at = with_order(fakes)
    a, b = vid(2), vid(1)
    at.button(key=f"listen_{a}__{b}").click()
    at.run()
    assert not at.exception
    assert len(fakes.preview_calls) == 1
    call = fakes.preview_calls[0]
    assert (call["a"], call["b"]) == (a, b)
    an = at.session_state["jobs"].analyses
    assert call["bpm_a"] == pytest.approx(an[a].effective_bpm)
    assert call["bpm_b"] == pytest.approx(an[b].effective_bpm)
    assert call["path_a"] == fakes.audio
    assert at.session_state["preview"]["a"] == a and at.session_state["preview"]["b"] == b
    assert at.button(key="listen_stop_0")
    assert "tocando" in html(at)                  # a linha do tempo marca as duas músicas
    at.button(key="listen_stop_0").click()
    at.run()
    assert at.session_state["preview"] is None


def test_ouvir_passagem_erro_amigavel(fakes, monkeypatch):
    def broken(*args, **kwargs):
        raise preview.PreviewError("Não consegui ler o áudio desta música.")

    monkeypatch.setattr(preview, "make_transition_preview", broken)
    at = with_order(fakes)
    at.button(key=f"listen_{vid(2)}__{vid(1)}").click()
    at.run()
    assert not at.exception
    assert has_text(at.warning, "Não consegui ler o áudio desta música.")
    assert at.session_state["preview"] is None


def test_exportar_chama_export_e_metricas(fakes):
    at = search(new_app().run())
    at = decide(at, 4)
    at = decide(at, 2)
    at = decide(at, 3, approve=False)
    at.button(key="btn_export").click()
    at.run()
    assert not at.exception
    assert fakes.export_calls == [{"ids": [vid(4), vid(2)], "transitions": None}]  # ordem de aprovação
    assert at.session_state["metrics"].finish_calls == [(2, 3)]
    assert has_text(at.success, str(config.EXPORT_DIR))
    assert has_text(at.code, "01 - MC Teste 4 - Faixa 4.mp3")
    page = html(at)
    assert "Tempo economizado" in page and "para 2 música(s) aprovada(s)" in page
    # a sessão foi para o histórico (arquivo na pasta temporária)
    assert "Sessões registradas" in page
    assert "djsk-step--done" in page


def test_exportar_usa_a_ordem_gerada(fakes):
    at = with_order(fakes, n=2)
    at.button(key="btn_export").click()
    at.run()
    call = fakes.export_calls[-1]
    assert call["ids"] == [vid(1), vid(0)]
    assert [(t.from_id, t.to_id) for t in call["transitions"]] == [(vid(1), vid(0))]


def test_orcamento_travado_desabilita_estudio_de_vocal(fakes, monkeypatch):
    monkeypatch.setattr(budget, "Budget", LockedBudget)
    at = search(new_app().run())
    at = decide(at, 0)
    btn = at.button(key=f"stem_{vid(0)}")
    assert btn.disabled
    assert "Teto de gasto atingido" in (btn.help or "")
    assert has_text(at.warning, "Teto de gasto atingido")
    assert has_text(at.caption, "Plano B: Demucs local")
    assert fakes.stem_calls == []


def test_sem_token_desabilita_estudio_de_vocal(fakes, monkeypatch):
    monkeypatch.setattr(config, "get_secret", lambda name: None)
    at = search(new_app().run())
    at = decide(at, 0)
    assert at.button(key=f"stem_{vid(0)}").disabled
    assert has_text(at.warning, "REPLICATE_API_TOKEN")


def test_extrair_acapella_atualiza_medidor(fakes):
    at = search(new_app().run())
    at = decide(at, 0)
    btn = at.button(key=f"stem_{vid(0)}")
    assert not btn.disabled
    btn.click()
    at.run()
    assert not at.exception
    assert fakes.stem_calls == [vid(0)]
    assert has_text(at.success, "Acapella pronta")
    assert "R$ 0,09" in html(at)
    assert at.session_state["stem_player"] == vid(0)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_metricas_ao_vivo(fakes):
    at = search(new_app().run())
    at = decide(at, 0)
    at = decide(at, 1)
    at = decide(at, 2, approve=False)
    assert not at.exception
    page = html(at)
    for label in ("Tempo economizado", "Tempo por música aprovada", "Ganho de velocidade",
                  "Decisões na triagem", "Custo por música", "Tempo de sessão"):
        assert label in page, label
    # 120 s para 2 aprovadas = 60 s por música; 67% de aprovação; triagem sem custo
    assert "60 s" in page and "67% aprovadas" in page and "a triagem não custa nada" in page
    assert "Nenhuma sessão registrada ainda" in " ".join(texts(at.caption))


def test_metricas_sem_busca(fakes):
    at = new_app().run()
    assert not at.exception
    page = html(at)
    assert "Tempo economizado" in page and "aprove músicas para começar a contar" in page


# ---------------------------------------------------------------------------
# Erros, sessão salva, encerrar e zerar
# ---------------------------------------------------------------------------
def test_erro_do_modulo_vira_mensagem_amigavel(fakes):
    fakes.suggest_error = suggest.SuggestError("Nenhuma sugestão encontrada. Tente outra referência.")
    at = search(new_app().run())
    assert not at.exception
    assert has_text(at.error, "Nenhuma sugestão encontrada. Tente outra referência.")
    assert keys(at, "approve_") == []


def test_erro_inesperado_nao_mostra_stack_trace(fakes, monkeypatch):
    def boom(query, yt=None):
        raise RuntimeError("falha interna")

    monkeypatch.setattr(suggest, "resolve_reference", boom)
    at = search(new_app().run())
    assert not at.exception
    # O relatório passa por _md(), que escapa ":" (diretivas do Streamlit como ":red[...]").
    assert has_text(at.markdown, "erro inesperado") and has_text(at.markdown, "RuntimeError")
    assert not has_text(at.markdown, "Traceback") and not has_text(at.markdown, "falha interna")
    assert has_text(at.error, "Nenhuma referência foi encontrada")


def test_restaura_sessao_salva_e_comeca_do_zero(fakes):
    fakes.snapshot = {
        "references": [], "suggestions": [make_track(i).to_dict() for i in range(3)], "page": 1,
        "decisions": {vid(0): "aprovada"}, "order": [], "weights": {"bpm": 1.5},
        "metrics": {"started": True}, "saved_at": "2026-10-22T09:30:00-03:00",
        "ref_texts": ["MC Antigo - Faixa"], "preset": "harmonico",
    }
    at = new_app().run()
    assert not at.exception
    assert has_text(at.info, "Sessão anterior restaurada (salva em 22/10/2026 09:30)")
    assert at.session_state["decisions"] == {vid(0): "aprovada"}
    assert at.text_input(key="ref_0").value == "MC Antigo - Faixa"
    assert at.slider(key="w_bpm").value == pytest.approx(1.5)
    assert at.pills(key="set_preset").value == "harmonico"
    assert len(keys(at, "reject_")) == 2           # filtro "Pendentes": vid0 já aprovada
    assert set(at.session_state["jobs"].submitted) == {vid(0), vid(1), vid(2)}
    at.button(key="btn_fresh_start").click()
    at.run()
    assert fakes.cleared == 1
    assert at.session_state["decisions"] == {}
    assert not has_text(at.info, "Sessão anterior restaurada")
    assert at.text_input(key="ref_0").value == ""


def test_encerrar_sessao_apaga_reprovadas(fakes):
    at = search(new_app().run())
    at = decide(at, 0, approve=False)
    at = decide(at, 1)
    at.button(key="btn_end").click()
    at.run()
    assert has_text(at.sidebar.warning, "1 faixa(s) reprovada(s)")
    at.button(key="btn_end_confirm").click()
    at.run()
    assert not at.exception
    assert fakes.purge_calls == [[vid(0)]]
    assert fakes.cleared == 1
    assert has_text(at.success, "Sessão encerrada")
    assert at.session_state["suggestions"] == []


def test_zerar_painel_apaga_o_que_foi_marcado(fakes):
    at = search(new_app().run())
    at = decide(at, 0)
    old_jobs = at.session_state["jobs"]
    at.button(key="btn_reset_open").click()
    at.run()
    assert not at.exception
    # marcadas por padrão: cache, sessão e métricas; stems, setlists e orçamento não
    for cat in maintenance.CATEGORIES:
        assert at.checkbox(key=f"reset_cat_{cat}").value is (cat in maintenance.DEFAULT_RESET), cat
    assert any("2,0 KB" in (c.label or "") for c in at.checkbox)
    # sem a confirmação explícita o botão fica travado
    assert at.button(key="btn_reset_go").disabled
    at.checkbox(key="reset_cat_orcamento").check()
    at.run()
    assert has_text(at.warning, "o dinheiro gasto na Replicate foi real")
    at.checkbox(key="reset_cat_orcamento").uncheck()
    at.checkbox(key="reset_confirm").check()
    at.run()
    at.button(key="btn_reset_go").click()
    at.run()
    assert not at.exception
    assert fakes.reset_calls == [["cache", "sessao", "metricas"]]
    assert old_jobs.closed                          # a fila parou antes de apagar
    assert at.session_state["suggestions"] == [] and at.session_state["decisions"] == {}
    assert has_text(at.success, "Painel zerado")
    assert keys(at, "btn_reset_go") == []           # a janela fechou


def test_zerar_painel_cancelar_nao_apaga(fakes):
    at = new_app().run()
    at.button(key="btn_reset_open").click()
    at.run()
    at.button(key="btn_reset_cancel").click()
    at.run()
    assert fakes.reset_calls == []
    assert at.session_state["reset_open"] is False
    assert keys(at, "btn_reset_go") == []


def test_pasta_local_vai_para_a_triagem(fakes):
    at = new_app().run()
    at.text_input(key="local_folder").input(r"C:\Musicas")
    at.button(key="btn_local").click()
    at.run()
    assert not at.exception
    assert keys(at, "approve_") == ["approve_local_abc123def456"]
    assert "arquivo local" in html(at)
    assert at.button(key="similar_local_abc123def456").disabled   # arquivo local não tem rádio


# ---------------------------------------------------------------------------
# Spotify
# ---------------------------------------------------------------------------
def test_spotify_desligado_explica_como_ligar(fakes):
    at = new_app().run()
    assert has_text(at.info, "Spotify desligado")


def test_spotify_busca_e_usa_como_referencia(fakes, spotify):
    at = new_app().run()
    at.text_input(key="sp_query").input("Casei com a")
    at.button(key="btn_sp_search").click()
    at.run()
    assert not at.exception
    assert spotify.search_calls == ["Casei com a"]
    page = html(at)
    assert "Casei com a &lt;Putaria&gt;" in page and "MC Ryan SP" in page     # texto de fora escapado
    use = keys(at, "sp_use_")
    assert len(use) == 1
    at.button(key=use[0]).click()
    at.run()
    assert not at.exception
    assert spotify.resolved == ["Casei com a <Putaria>"]
    assert at.text_input(key="ref_0").value == "MC Ryan SP - Casei com a <Putaria>"
    assert has_text(at.success, "Referência do Spotify adicionada")
    # a busca usa a faixa já resolvida (sem buscar de novo no YouTube Music)
    at.button(key="btn_search").click()
    at.run()
    assert fakes.resolve_calls == []
    assert fakes.suggest_refs[-1] == ["spref000001"]


def test_spotify_login_importa_e_cria_playlist(fakes, spotify):
    at = new_app().run()
    at.button(key="btn_spotify_login").click()
    at.run()
    assert not at.exception
    assert spotify.login_calls == 1
    page = html(at)
    assert "Conectado como DJ SK" in page and "conectado como DJ SK" in page   # aba e barra lateral
    # importar a playlist como candidatas diretas
    at.radio(key="spotify_mode").set_value("Candidatas diretas na triagem")
    at.button(key="btn_spotify_import").click()
    at.run()
    assert spotify.import_calls == [("pl1", None)]
    assert len(keys(at, "approve_")) == 3
    assert any("3 de 4 encontradas" in (p.proto.text or "") for p in at.get("progress"))
    # aprovar e criar a playlist no Spotify
    at = decide(at, 100)
    at = decide(at, 101)
    at.text_input(key="export_name").input("Set Submundo")
    at.button(key="btn_spotify_export").click()
    at.run()
    assert not at.exception
    assert spotify.export_calls == [{"name": "Set Submundo", "ids": [vid(100), vid(101)]}]
    assert has_text(at.success, "Playlist criada no Spotify com 1 música(s). 1 não foram achadas lá.")
    assert at.link_button[0].proto.url == "https://open.spotify.com/playlist/pl123"


# ---------------------------------------------------------------------------
# viz (SVG puro)
# ---------------------------------------------------------------------------
def parse_svg(svg: str) -> ET.Element:
    assert "\n" not in svg               # uma linha só: o Markdown não quebra o SVG
    root = ET.fromstring(svg)
    assert root.tag == f"{SVG_NS}svg"
    return root


def test_waveform_svg_com_marcadores_e_trechos():
    wf = [abs(math.sin(i / 7)) for i in range(400)]
    svg = viz.waveform_svg(wf, width=600, height=60, markers=[(0, "Início"), (0.5, "Meio"), (0.9, "Final")],
                           highlight=[(0, 0.1), (0.92, 1.0)])
    root = parse_svg(svg)
    assert root.get("viewBox") == "0 0 600 60"
    paths = root.findall(f"{SVG_NS}path")
    assert len(paths) == 1 and paths[0].get("fill") == viz.ACCENT
    lines = root.findall(f"{SVG_NS}line")
    assert [ln.get("x1") for ln in lines] == ["0.0", "300.0", "540.0"]
    assert [ln.find(f"{SVG_NS}title").text for ln in lines] == ["Início", "Meio", "Final"]
    assert len(root.findall(f"{SVG_NS}rect")) == 2
    # cores que funcionam nos dois temas: destaque fixo + currentColor para o neutro
    assert "currentColor" in svg and "#000" not in svg and "#fff" not in svg.lower()


def test_waveform_svg_limita_pontos_e_fica_dentro_da_caixa():
    svg = viz.waveform_svg([2.0, float("nan"), -1, None, 0.5] * 400, width=200, height=40)
    root = parse_svg(svg)
    d = root.find(f"{SVG_NS}path").get("d")
    coords = [tuple(map(float, p.split(","))) for p in d.replace("M", "").replace("L", " ").replace("Z", "").split()]
    assert len(coords) <= 2 * 200
    assert all(0 <= x <= 200 and 0 <= y <= 40 for x, y in coords)


def test_waveform_svg_vazio_e_rotulo_escapado():
    svg = viz.waveform_svg([], markers=[(1.5, "<b>A&B</b>")])
    root = parse_svg(svg)
    assert root.find(f"{SVG_NS}path") is None
    lines = root.findall(f"{SVG_NS}line")
    assert lines[0].get("stroke-dasharray") == "4 4"               # linha central tracejada
    assert lines[1].get("x1") == "600.0"                          # posição limitada a 0..1
    assert lines[1].find(f"{SVG_NS}title").text == "<b>A&B</b>"   # escapado no SVG, texto intacto


def test_sparkline_svg():
    vals = [i / 300 for i in range(300)]
    root = parse_svg(viz.sparkline_svg(vals, width=160, height=28))
    poly = root.find(f"{SVG_NS}polyline")
    pts = [tuple(map(float, p.split(","))) for p in poly.get("points").split()]
    assert 2 <= len(pts) <= 160
    assert pts[0][0] == 0.0 and pts[-1][0] == 160.0
    assert pts[0][1] > pts[-1][1]                  # energia subindo = linha subindo
    assert all(0 <= y <= 28 for _, y in pts)
    assert poly.get("stroke") == viz.ACCENT


def test_sparkline_svg_vazio_e_um_ponto():
    assert parse_svg(viz.sparkline_svg(None)).find(f"{SVG_NS}polyline") is None
    one = parse_svg(viz.sparkline_svg([0.5])).find(f"{SVG_NS}polyline")
    assert len(one.get("points").split()) == 2


def test_sparkline_text():
    assert viz.sparkline_text([0, 0.5, 1], n=3) == "▁▅█"
    assert len(viz.sparkline_text([0.3] * 100, n=12)) == 12
    assert viz.sparkline_text([]) == ""
