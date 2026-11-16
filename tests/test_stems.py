"""Testes do Estúdio de Vocal (seção 4.7) com cliente da Replicate e HTTP falsos."""
from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path

import pytest
import requests
from replicate.exceptions import ReplicateError

import config
from modules import stems, util
from modules.budget import Budget

FAKE_TOKEN = "r8_teste_falso"  # não é um token real
_REAL_GET = requests.get
_REAL_SETUP_SSL = util.setup_ssl
DELIVERY = "https://replicate.delivery/pbxt/abc123"


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakePrediction:
    """Previsão falsa: cada reload() avança para o próximo status da lista."""

    def __init__(self, statuses=("starting", "processing", "succeeded"), output=None,
                 metrics=None, error=None, pid="pred_teste_1", stuck=False):
        self.id = pid
        self._statuses = list(statuses)
        self.status = self._statuses.pop(0)
        self.output = output
        self.metrics = metrics if metrics is not None else {}
        self.error = error
        self.stuck = stuck
        self.reloads = 0
        self.canceled = False

    def reload(self):
        self.reloads += 1
        if self._statuses and not self.stuck:
            self.status = self._statuses.pop(0)

    def wait(self):
        while self.status not in ("succeeded", "failed", "canceled"):
            self.reload()

    def cancel(self):
        self.canceled = True
        self.status = "canceled"


class FakePredictions:
    def __init__(self, prediction=None, exc=None):
        self.prediction = prediction
        self.exc = exc
        self.calls: list[dict] = []
        self.by_id: dict[str, FakePrediction] = {}
        self.gets: list[str] = []

    def get(self, pid):
        self.gets.append(pid)
        if pid not in self.by_id:
            raise ReplicateError(status=404, detail="Prediction not found")
        return self.by_id[pid]

    def create(self, **kwargs):
        inp = dict(kwargs["input"])
        audio = inp.pop("audio")
        self.calls.append({"version": kwargs.get("version"), "input": inp,
                           "audio_name": getattr(audio, "name", None),
                           "audio_mode": getattr(audio, "mode", None),
                           "audio_head": audio.read(3)})
        if self.exc is not None:
            raise self.exc
        return self.prediction


class FakeClient:
    def __init__(self, prediction=None, exc=None):
        self.predictions = FakePredictions(prediction, exc)


class FakeFileOutput:
    """Imita replicate.helpers.FileOutput (tem .url e str())."""

    def __init__(self, url):
        self.url = url

    def __str__(self):
        return self.url


class FakeResponse:
    def __init__(self, url, status=200, chunks=None, fail_after=None):
        self.url = url
        self.status_code = status
        self.chunks = chunks if chunks is not None else [b"ID3", f"<{url}>".encode()]
        self.fail_after = fail_after

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk_size=1):
        for i, chunk in enumerate(self.chunks):
            if self.fail_after is not None and i >= self.fail_after:
                raise requests.ConnectionError("conexão caiu no meio do download")
            yield chunk


class FakeHTTP:
    def __init__(self, behaviors=None, default=None):
        self.behaviors = behaviors or {}
        self.default = default or {}
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, stream=False, timeout=None, headers=None):
        self.calls.append((url, dict(headers or {})))
        return FakeResponse(url, **self.behaviors.get(url, self.default))


def urls_for(base=DELIVERY, ext="mp3"):
    return {s: f"{base}/{s}.{ext}" for s in config.STEM_NAMES}


def ok_prediction(predict_time=70.0, output=None, **kw):
    if output is None:
        output = {**urls_for(), "guitar": None, "piano": None}
    return FakePrediction(output=output, metrics={"predict_time": predict_time}, **kw)


@pytest.fixture(autouse=True)
def _fast_and_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(stems, "_POLL_INTERVAL_S", 0.0)
    monkeypatch.setattr(util, "setup_ssl", lambda: None)
    # O .env lido na hora da chamada fica numa pasta temporária (nunca o .env real).
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)

    def _no_network(*a, **k):
        raise AssertionError("teste unitário tentou acessar a rede")

    monkeypatch.setattr(stems.requests, "get", _no_network)


@pytest.fixture
def http(monkeypatch):
    fake = FakeHTTP()
    monkeypatch.setattr(stems.requests, "get", fake.get)
    return fake


@pytest.fixture
def budget(tmp_dirs):
    return Budget(path=tmp_dirs["DATA_DIR"] / "orcamento.json", cap_brl=10.0,
                  usd_per_second=0.000225, usd_brl=5.40, credit_usd=2.0)


@pytest.fixture
def audio(tmp_dirs):
    p = tmp_dirs["CACHE_DIR"] / "vid123.mp3"
    p.write_bytes(b"ID3" + b"\x00" * 64)
    return p


def make_cached_stems(folder: Path, names=config.STEM_NAMES):
    folder.mkdir(parents=True, exist_ok=True)
    for s in names:
        (folder / f"{s}.mp3").write_bytes(b"ID3stem")


# ----------------------------------------------------------------------
# Cache
# ----------------------------------------------------------------------
def test_stems_dir_reads_config_at_call_time(tmp_dirs, tmp_path):
    assert stems.stems_dir("vid123") == tmp_dirs["STEMS_DIR"] / "vid123"
    assert stems.stems_dir("vid123", tmp_path / "x") == tmp_path / "x" / "vid123"
    assert stems.load_stems("vid123") is None
    assert stems.has_stems("vid123") is False


def test_load_stems_requires_all_four(tmp_dirs):
    folder = tmp_dirs["STEMS_DIR"] / "vid123"
    make_cached_stems(folder, config.STEM_NAMES[:3])
    assert stems.load_stems("vid123") is None
    (folder / "other.mp3").write_bytes(b"")  # vazio não conta
    assert stems.has_stems("vid123") is False
    (folder / "other.mp3").write_bytes(b"ID3")
    res = stems.load_stems("vid123")
    assert res is not None and res.backend == "cache"
    assert set(res.stems) == set(config.STEM_NAMES)
    assert res.vocals_path == folder / "vocals.mp3"
    assert res.cost_usd == 0.0


def test_cache_returns_without_spending(tmp_dirs, budget, audio):
    make_cached_stems(tmp_dirs["STEMS_DIR"] / "vid123")
    client = FakeClient(exc=AssertionError("não deveria chamar a Replicate"))
    res = stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN, client=client)
    assert res.backend == "cache"
    assert client.predictions.calls == []
    assert budget.entries == []


# ----------------------------------------------------------------------
# Sucesso
# ----------------------------------------------------------------------
def test_success_records_real_cost_and_saves_stems(tmp_dirs, budget, audio, http):
    pred = ok_prediction(predict_time=70.0)
    client = FakeClient(pred)
    messages: list[str] = []

    res = stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN, client=client,
                                   on_progress=messages.append)

    call = client.predictions.calls[0]
    assert call["version"] == config.REPLICATE_VERSION
    assert call["input"] == {"model_name": config.DEMUCS_MODEL_NAME,
                             "output_format": config.DEMUCS_OUTPUT_FORMAT}
    assert "stem" not in call["input"]
    assert call["audio_mode"] == "rb" and Path(call["audio_name"]) == audio
    assert call["audio_head"] == b"ID3"

    folder = tmp_dirs["STEMS_DIR"] / "vid123"
    assert res.backend == "replicate" and res.prediction_id == "pred_teste_1"
    assert res.predict_time_s == 70.0
    assert res.cost_usd == pytest.approx(70.0 * 0.000225)
    assert res.cost_brl == pytest.approx(70.0 * 0.000225 * 5.40)
    for s in config.STEM_NAMES:
        p = folder / f"{s}.mp3"
        assert res.stems[s] == str(p)
        assert p.read_bytes() == b"ID3" + f"<{DELIVERY}/{s}.mp3>".encode()
    assert not list(folder.glob("*.part"))
    assert [u for u, _ in http.calls] == [f"{DELIVERY}/{s}.mp3" for s in config.STEM_NAMES]
    assert all("Authorization" not in h for _, h in http.calls)  # token não vaza para a CDN

    entries = budget.entries
    assert len(entries) == 1
    assert entries[0]["status"] == "succeeded" and entries[0]["predict_time_s"] == 70.0
    assert entries[0]["prediction_id"] == "pred_teste_1"
    assert entries[0]["usd"] == pytest.approx(70.0 * 0.000225)

    meta = util.read_json(folder / "meta.json")
    assert meta["backend"] == "replicate" and meta["prediction_id"] == "pred_teste_1"
    assert meta["predict_time_s"] == 70.0
    assert meta["cost_usd"] == pytest.approx(70.0 * 0.000225)
    assert meta["complete"] is True and "output_urls" not in meta

    assert stems.has_stems("vid123")
    assert messages and any("Baixando" in m for m in messages)
    assert pred.reloads == 2


def test_success_without_predict_time_counts_estimate(tmp_dirs, budget, audio, http):
    pred = FakePrediction(output=urls_for(), metrics={})
    res = stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                   client=FakeClient(pred))
    assert res.cost_usd == pytest.approx(config.STEM_COST_ESTIMATE_USD)
    assert budget.entries[0]["estimated"] is True


def test_file_output_objects(tmp_dirs, budget, audio, http):
    output = {s: FakeFileOutput(u) for s, u in urls_for().items()}
    res = stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                   client=FakeClient(ok_prediction(output=output)))
    assert res.backend == "replicate"
    assert [u for u, _ in http.calls] == list(urls_for().values())


def test_list_output_is_mapped_by_filename(tmp_dirs, budget, audio, http):
    output = [FakeFileOutput(u) for u in urls_for().values()]
    res = stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                   client=FakeClient(ok_prediction(output=output)))
    assert set(res.stems) == set(config.STEM_NAMES)


def test_token_only_sent_to_replicate_api_host(tmp_dirs, budget, audio, http):
    output = urls_for(base="https://api.replicate.com/v1/files/xyz")
    stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                             client=FakeClient(ok_prediction(output=output)))
    assert all(h.get("Authorization") == f"Bearer {FAKE_TOKEN}" for _, h in http.calls)


# ----------------------------------------------------------------------
# Travas antes de chamar a API
# ----------------------------------------------------------------------
def test_missing_token_calls_nothing(tmp_dirs, budget, audio, monkeypatch):
    import replicate

    monkeypatch.delenv("REPLICATE_API_TOKEN", raising=False)

    def _boom(*a, **k):
        raise AssertionError("não deveria criar o cliente")

    monkeypatch.setattr(replicate, "Client", _boom)
    with pytest.raises(stems.MissingToken, match="REPLICATE_API_TOKEN"):
        stems.separate_replicate("vid123", audio, budget)
    assert budget.entries == []


def test_token_is_read_from_env_at_call_time(tmp_dirs, budget, audio, http, monkeypatch):
    import replicate

    seen = {}

    def _client(api_token=None, **kw):
        seen["token"] = api_token
        return FakeClient(ok_prediction())

    monkeypatch.setenv("REPLICATE_API_TOKEN", FAKE_TOKEN)
    monkeypatch.setattr(replicate, "Client", _client)
    res = stems.separate_replicate("vid123", audio, budget)
    assert seen["token"] == FAKE_TOKEN and res.backend == "replicate"


@pytest.mark.parametrize("setup, fragment", [
    (lambda b: None, "Teto de gasto"),
    (lambda b: b.mark_insufficient_credit("sem saldo"), "saldo insuficiente"),
])
def test_budget_locked_raises_without_calling_api(tmp_dirs, audio, setup, fragment):
    cap = 0.05 if fragment == "Teto de gasto" else 10.0  # estimativa (R$ 0,086) passa de 0,05
    b = Budget(path=tmp_dirs["DATA_DIR"] / "orcamento.json", cap_brl=cap,
               usd_per_second=0.000225, usd_brl=5.40, credit_usd=2.0)
    setup(b)
    client = FakeClient(ok_prediction())
    with pytest.raises(stems.BudgetExceeded, match=fragment):
        stems.separate_replicate("vid123", audio, b, token=FAKE_TOKEN, client=client)
    assert client.predictions.calls == []
    assert b.entries == []


def test_missing_audio_file(tmp_dirs, budget):
    client = FakeClient(ok_prediction())
    with pytest.raises(stems.StemsError, match="não encontrado"):
        stems.separate_replicate("vid123", tmp_dirs["CACHE_DIR"] / "nao_existe.mp3", budget,
                                 token=FAKE_TOKEN, client=client)
    assert client.predictions.calls == []


# ----------------------------------------------------------------------
# Erros da Replicate
# ----------------------------------------------------------------------
@pytest.mark.parametrize("exc", [
    ReplicateError(status=402, title="Payment Required",
                   detail="You have insufficient credit to run this model."),
    RuntimeError("Your billing details are missing"),
])
def test_insufficient_credit_on_create(tmp_dirs, budget, audio, exc):
    client = FakeClient(exc=exc)
    with pytest.raises(stems.InsufficientCredit, match="saldo insuficiente"):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN, client=client)
    assert budget.insufficient_credit is True
    assert budget.can_run()[0] is False
    assert budget.entries == []  # nada rodou, nada foi cobrado


def test_other_api_error_is_plain_stems_error(tmp_dirs, budget, audio):
    client = FakeClient(exc=ReplicateError(status=401, detail="Invalid token."))
    with pytest.raises(stems.StemsError, match="token") as info:
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN, client=client)
    assert not isinstance(info.value, stems.InsufficientCredit)
    assert budget.insufficient_credit is False and budget.entries == []


def test_failure_after_running_records_cost(tmp_dirs, budget, audio, http):
    pred = FakePrediction(statuses=("starting", "processing", "failed"),
                          metrics={"predict_time": 12.5}, error="CUDA out of memory")
    with pytest.raises(stems.StemsError, match="CUDA out of memory"):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                 client=FakeClient(pred))
    entries = budget.entries
    assert len(entries) == 1
    assert entries[0]["status"] == "failed" and entries[0]["predict_time_s"] == 12.5
    assert entries[0]["usd"] == pytest.approx(12.5 * 0.000225)
    assert http.calls == []
    assert not stems.has_stems("vid123")


def test_canceled_without_predict_time_records_nothing(tmp_dirs, budget, audio):
    pred = FakePrediction(statuses=("starting", "canceled"), metrics={})
    with pytest.raises(stems.StemsError, match="cancelada"):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                 client=FakeClient(pred))
    assert budget.entries == []


def test_failed_prediction_with_billing_error_marks_flag(tmp_dirs, budget, audio):
    pred = FakePrediction(statuses=("starting", "failed"), metrics={},
                          error="Insufficient credit: add a payment method")
    with pytest.raises(stems.InsufficientCredit):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                 client=FakeClient(pred))
    assert budget.insufficient_credit is True


def test_timeout_cancels_and_counts_cost(tmp_dirs, budget, audio, monkeypatch):
    monkeypatch.setattr(config, "REPLICATE_TIMEOUT_S", 0)
    pred = FakePrediction(statuses=("processing", "succeeded"), stuck=True, metrics={})
    with pytest.raises(stems.StemsError, match="cancelada"):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                 client=FakeClient(pred))
    assert pred.canceled is True
    entries = budget.entries
    assert len(entries) == 1 and entries[0]["status"] == "canceled"
    assert entries[0]["estimated"] is True


def test_lost_contact_counts_estimate(tmp_dirs, budget, audio):
    class Flaky(FakePrediction):
        def reload(self):
            raise requests.ConnectionError("sem rede")

    pred = Flaky(statuses=("starting",), metrics={})
    with pytest.raises(stems.StemsError, match="Perdemos o contato"):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                 client=FakeClient(pred))
    assert [e["status"] for e in budget.entries] == ["unknown"]


def test_missing_stem_in_output(tmp_dirs, budget, audio, http):
    output = {k: v for k, v in urls_for().items() if k != "bass"}
    with pytest.raises(stems.StemsError, match="bass"):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                 client=FakeClient(ok_prediction(output=output)))
    assert len(budget.entries) == 1  # rodou: o custo fica registrado


# ----------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------
def test_partial_download_failure_leaves_no_final_file(tmp_dirs, budget, audio, http):
    drums = f"{DELIVERY}/drums.mp3"
    http.behaviors[drums] = {"chunks": [b"ID3", b"meio", b"fim"], "fail_after": 1}
    with pytest.raises(stems.StemsError, match="drums") as info:
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                 client=FakeClient(ok_prediction()))
    assert "sem pagar" in str(info.value)
    folder = tmp_dirs["STEMS_DIR"] / "vid123"
    assert not (folder / "drums.mp3").exists()
    assert not list(folder.glob("*.part"))
    assert not stems.has_stems("vid123")
    assert len(budget.entries) == 1
    meta = util.read_json(folder / "meta.json")
    assert meta["complete"] is False and set(meta["output_urls"]) == set(config.STEM_NAMES)


def test_http_error_on_download(tmp_dirs, budget, audio, http):
    http.default = {"status": 500}
    with pytest.raises(stems.StemsError, match="HTTP 500"):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                 client=FakeClient(ok_prediction()))
    folder = tmp_dirs["STEMS_DIR"] / "vid123"
    assert not any((folder / f"{s}.mp3").exists() for s in config.STEM_NAMES)


def test_retry_after_download_failure_does_not_pay_again(tmp_dirs, budget, audio, http):
    http.behaviors[f"{DELIVERY}/other.mp3"] = {"fail_after": 0}
    first = FakeClient(ok_prediction())
    with pytest.raises(stems.StemsError):
        stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN, client=first)

    http.behaviors.clear()
    second = FakeClient(exc=AssertionError("não deveria rodar de novo"))
    res = stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN, client=second)
    assert res.backend == "replicate" and res.prediction_id == "pred_teste_1"
    assert second.predictions.calls == []
    assert len(budget.entries) == 1
    meta = util.read_json(tmp_dirs["STEMS_DIR"] / "vid123" / "meta.json")
    assert meta["complete"] is True and "output_urls" not in meta
    assert stems.has_stems("vid123")


def test_expired_urls_run_again(tmp_dirs, budget, audio, http):
    folder = tmp_dirs["STEMS_DIR"] / "vid123"
    old = urls_for(base="https://replicate.delivery/pbxt/velho")
    util.atomic_write_json(folder / "meta.json", {"backend": "replicate", "complete": False,
                                                  "prediction_id": "pred_velho",
                                                  "output_urls": old})
    for u in old.values():
        http.behaviors[u] = {"status": 404}
    client = FakeClient(ok_prediction(pid="pred_novo"))
    res = stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN, client=client)
    assert res.prediction_id == "pred_novo"
    assert len(client.predictions.calls) == 1 and len(budget.entries) == 1


def test_concurrent_calls_for_same_track_pay_once(tmp_dirs, budget, audio, http):
    client = FakeClient(ok_prediction())
    results, errors = [], []

    def run():
        try:
            results.append(stems.separate_replicate("vid123", audio, budget, token=FAKE_TOKEN,
                                                    client=client))
        except Exception as exc:  # pragma: no cover - só aparece se falhar
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(client.predictions.calls) == 1 and len(budget.entries) == 1
    assert sorted(r.backend for r in results) == ["cache", "cache", "replicate"]


# ----------------------------------------------------------------------
# Plano B: Demucs local
# ----------------------------------------------------------------------
def test_local_demucs_available_returns_bool():
    assert isinstance(stems.local_demucs_available(), bool)


def test_separate_local_without_demucs_explains_install(tmp_dirs, audio, monkeypatch):
    monkeypatch.setattr(stems, "local_demucs_available", lambda: False)
    with pytest.raises(stems.StemsError, match="pip install demucs"):
        stems.separate_local("vid123", audio)


def test_separate_local_moves_outputs(tmp_dirs, audio, monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        out = Path(cmd[cmd.index("-o") + 1]) / config.DEMUCS_MODEL_NAME / Path(cmd[-1]).stem
        out.mkdir(parents=True)
        for s in config.STEM_NAMES:
            (out / f"{s}.mp3").write_bytes(f"ID3{s}".encode())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(stems, "local_demucs_available", lambda: True)
    monkeypatch.setattr(stems.subprocess, "run", fake_run)
    res = stems.separate_local("vid123", audio)

    import sys
    cmd = seen["cmd"]
    assert cmd[:7] == [sys.executable, "-m", "demucs", "-n", config.DEMUCS_MODEL_NAME, "--mp3", "-o"]
    assert cmd[-1] == str(audio)
    assert not Path(cmd[7]).exists()  # pasta temporária removida
    folder = tmp_dirs["STEMS_DIR"] / "vid123"
    assert res.backend == "local" and res.cost_usd == 0.0
    for s in config.STEM_NAMES:
        assert Path(res.stems[s]) == folder / f"{s}.mp3"
        assert (folder / f"{s}.mp3").read_bytes() == f"ID3{s}".encode()
    meta = util.read_json(folder / "meta.json")
    assert meta["backend"] == "local" and meta["cost_usd"] == 0.0
    assert stems.load_stems("vid123").backend == "cache"


def test_separate_local_failure_reports_stderr(tmp_dirs, audio, monkeypatch):
    monkeypatch.setattr(stems, "local_demucs_available", lambda: True)
    monkeypatch.setattr(stems.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "Traceback...\nRuntimeError: modelo não encontrado"))
    with pytest.raises(stems.StemsError, match="modelo não encontrado"):
        stems.separate_local("vid123", audio)
    assert not stems.has_stems("vid123")


def test_separate_local_uses_cache(tmp_dirs, audio, monkeypatch):
    make_cached_stems(tmp_dirs["STEMS_DIR"] / "vid123")
    monkeypatch.setattr(stems, "local_demucs_available", lambda: False)
    assert stems.separate_local("vid123", audio).backend == "cache"


# ----------------------------------------------------------------------
# Rede (pago): só roda com token E com DJSK_PAID_TESTS=1
# ----------------------------------------------------------------------
@pytest.mark.network
def test_real_replicate_separation(tmp_dirs, monkeypatch):
    if not config.get_secret("REPLICATE_API_TOKEN"):
        pytest.skip("REPLICATE_API_TOKEN não definido")
    if os.getenv("DJSK_PAID_TESTS") != "1":
        pytest.skip("teste pago (~US$ 0,01): defina DJSK_PAID_TESTS=1 para rodar")
    from tests.conftest import synth_track, write_audio

    monkeypatch.setattr(stems.requests, "get", _REAL_GET)
    monkeypatch.setattr(util, "setup_ssl", _REAL_SETUP_SSL)
    audio = write_audio(tmp_dirs["CACHE_DIR"] / "sintetico.mp3", synth_track(duration_s=15.0))
    b = Budget(path=tmp_dirs["DATA_DIR"] / "orcamento.json")
    res = stems.separate_replicate("sintetico", audio, b)
    assert res.backend == "replicate" and res.cost_usd > 0
    assert all(Path(p).stat().st_size > 0 for p in res.stems.values())
    assert len(b.entries) == 1
