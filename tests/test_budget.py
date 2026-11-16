"""Testes do medidor e teto de gasto (seção 7)."""
from __future__ import annotations

import json
import threading

import pytest

import config
from modules.budget import Budget


def make_budget(tmp_path, **kw) -> Budget:
    params = {"path": tmp_path / "orcamento.json", "cap_brl": 10.0, "usd_per_second": 0.000225,
              "usd_brl": 5.40, "credit_usd": 2.00}
    params.update(kw)
    return Budget(**params)


def test_defaults_come_from_config_at_call_time(tmp_dirs, monkeypatch):
    monkeypatch.setattr(config, "BUDGET_CAP_BRL", 7.0)
    b = Budget()
    assert b.path == tmp_dirs["DATA_DIR"] / "orcamento.json"
    assert b.cap_brl == 7.0
    assert b.usd_per_second == config.REPLICATE_USD_PER_SECOND
    assert b.usd_brl == config.USD_BRL
    assert b.credit_usd == config.REPLICATE_CREDIT_USD
    assert b.entries == [] and b.spent_brl == 0.0 and b.fraction_used == 0.0


def test_cost_from_predict_time(tmp_path):
    b = make_budget(tmp_path)
    usd, brl = b.cost_from_predict_time(70.0)
    assert usd == pytest.approx(70.0 * 0.000225)
    assert brl == pytest.approx(70.0 * 0.000225 * 5.40)
    assert b.cost_from_predict_time(-3) == (0.0, 0.0)


def test_record_and_persistence_between_instances(tmp_path):
    b = make_budget(tmp_path)
    e = b.record("vid1", "pred1", 70.0)
    assert set(e) >= {"timestamp", "video_id", "prediction_id", "predict_time_s", "usd", "brl", "status"}
    assert e["status"] == "succeeded" and e["estimated"] is False
    assert e["usd"] == pytest.approx(0.01575)
    b.record("vid2", "pred2", 30.0, status="failed")

    b2 = make_budget(tmp_path)
    assert [x["video_id"] for x in b2.entries] == ["vid1", "vid2"]
    assert b2.entries[1]["status"] == "failed"
    assert b2.spent_usd == pytest.approx(100.0 * 0.000225)
    assert b2.spent_brl == pytest.approx(100.0 * 0.000225 * 5.40)
    assert b2.remaining_brl == pytest.approx(10.0 - b2.spent_brl)
    assert b2.remaining_credit_usd == pytest.approx(2.0 - b2.spent_usd)
    assert b2.fraction_used == pytest.approx(b2.spent_brl / 10.0)

    data = json.loads((tmp_path / "orcamento.json").read_text(encoding="utf-8"))
    assert len(data["entries"]) == 2


def test_instances_see_each_others_writes(tmp_path):
    a = make_budget(tmp_path)
    b = make_budget(tmp_path)
    a.record("v1", "p1", 10.0)
    assert len(b.entries) == 1
    b.record("v2", "p2", 10.0)
    assert len(a.entries) == 2


def test_record_without_predict_time_uses_estimate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "STEM_COST_ESTIMATE_USD", 0.02)
    b = make_budget(tmp_path)
    e = b.record("vid", None, None, status="failed")
    assert e["usd"] == pytest.approx(0.02)
    assert e["brl"] == pytest.approx(0.02 * 5.40)
    assert e["predict_time_s"] is None and e["estimated"] is True


def test_can_run_ok(tmp_path):
    ok, reason = make_budget(tmp_path).can_run()
    assert ok and reason == ""


def test_spending_exactly_up_to_cap_locks(tmp_path):
    # Teto R$ 1,00; estimativa US$ 0,10 = R$ 0,50 a câmbio 5.
    b = make_budget(tmp_path, cap_brl=1.0, usd_brl=5.0, usd_per_second=0.001, credit_usd=100.0)
    assert b.can_run(0.1) == (True, "")
    b.record("v1", "p1", 100.0)                 # US$ 0,10 = R$ 0,50 gastos
    ok, reason = b.can_run(0.1)                 # 0,50 + 0,50 == teto -> trava
    assert not ok
    assert "Teto de gasto" in reason and "R$" in reason
    assert b.can_run(0.09)[0]                   # 0,50 + 0,45 < teto -> libera


def test_spent_at_cap_locks_even_with_zero_estimate(tmp_path):
    b = make_budget(tmp_path, cap_brl=1.0, usd_brl=5.0, usd_per_second=0.001, credit_usd=100.0)
    b.record("v1", "p1", 200.0)                 # US$ 0,20 = R$ 1,00
    assert b.fraction_used == pytest.approx(1.0)
    assert b.remaining_brl == pytest.approx(0.0)
    assert not b.can_run(0.0)[0]


def test_replicate_credit_in_usd_locks(tmp_path):
    # Teto em reais folgado; o crédito de US$ 0,05 é o que segura.
    b = make_budget(tmp_path, cap_brl=1000.0, usd_brl=5.0, usd_per_second=0.001, credit_usd=0.05)
    b.record("v1", "p1", 30.0)                  # US$ 0,03
    assert b.can_run(0.02) == (True, "")        # 0,03 + 0,02 == crédito: ainda cabe
    ok, reason = b.can_run(0.03)                # 0,06 > 0,05
    assert not ok
    assert "crédito da Replicate" in reason and "US$" in reason


def test_insufficient_credit_flag_persists_and_resets(tmp_path):
    b = make_budget(tmp_path)
    assert not b.insufficient_credit
    b.mark_insufficient_credit("402 Payment Required")
    ok, reason = b.can_run()
    assert not ok and "saldo insuficiente" in reason and "402" in reason

    b2 = make_budget(tmp_path)
    assert b2.insufficient_credit
    assert b2.insufficient_credit_message == "402 Payment Required"
    assert not b2.can_run()[0]

    b2.reset_insufficient_credit()
    assert not make_budget(tmp_path).insufficient_credit
    assert b.can_run() == (True, "")


def test_corrupted_json_is_kept_aside_and_starts_empty(tmp_path):
    path = tmp_path / "orcamento.json"
    path.write_text('{"entries": [{"usd": 1.0, ', encoding="utf-8")
    b = make_budget(tmp_path)
    assert b.entries == [] and b.spent_brl == 0.0
    backup = tmp_path / "orcamento.json.corrompido"
    assert backup.read_text(encoding="utf-8") == '{"entries": [{"usd": 1.0, '
    assert not path.exists()

    b.record("v1", "p1", 10.0)
    assert len(make_budget(tmp_path).entries) == 1

    # Segunda corrupção não sobrescreve o primeiro backup.
    path.write_text("lixo", encoding="utf-8")
    assert make_budget(tmp_path).entries == []
    backups = sorted(p.name for p in tmp_path.glob("*.corrompido"))
    assert len(backups) == 2 and "orcamento.json.corrompido" in backups


def test_unexpected_structure_is_treated_as_corrupted(tmp_path):
    (tmp_path / "orcamento.json").write_text('"texto"', encoding="utf-8")
    assert make_budget(tmp_path).entries == []
    assert (tmp_path / "orcamento.json.corrompido").exists()


def test_legacy_list_format_and_bad_entries_are_tolerated(tmp_path):
    (tmp_path / "orcamento.json").write_text(
        json.dumps([{"video_id": "a", "usd": 0.01, "brl": 0.054}, "lixo", {"usd": "x"}]),
        encoding="utf-8")
    b = make_budget(tmp_path)
    assert len(b.entries) == 2
    assert b.spent_usd == pytest.approx(0.01)
    assert b.spent_brl == pytest.approx(0.054)


def test_concurrent_records_from_threads(tmp_path):
    b = make_budget(tmp_path)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for j in range(i, 50, 8):
                b.record(f"v{j}", f"p{j}", 1.0)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(b.entries) == 50
    fresh = make_budget(tmp_path)
    assert len(fresh.entries) == 50
    assert {e["video_id"] for e in fresh.entries} == {f"v{j}" for j in range(50)}
    assert fresh.spent_usd == pytest.approx(50 * 0.000225)


def test_concurrent_records_from_separate_instances(tmp_path):
    instances = [make_budget(tmp_path) for _ in range(4)]

    def worker(k: int) -> None:
        for j in range(10):
            instances[k].record(f"i{k}-{j}", None, 1.0)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(make_budget(tmp_path).entries) == 40
