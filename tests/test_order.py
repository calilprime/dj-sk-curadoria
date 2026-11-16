"""Testes da ordenação do set (4.6). Não dependem de outros módulos além de models/config."""
from __future__ import annotations

import itertools
import logging
import random
import time

import numpy as np
import pytest

import config
from modules import order
from modules.models import Analysis

W_DEFAULT = {"bpm": 1.0, "energia": 0.8, "tom": 0.2, "voz": 0.5}
CAMELOT_CODES = [f"{n}{letter}" for n in range(1, 13) for letter in "AB"]


def make(vid: str, bpm: float = 130.0, e_in: float = 0.5, e_out: float = 0.5,
         e_mean: float = 0.5, camelot: str | None = None, bpm_manual: float | None = None) -> Analysis:
    return Analysis(video_id=vid, duration_s=180.0, bpm=bpm, bpm_raw=bpm, bpm_manual=bpm_manual,
                    energy_in=e_in, energy_out=e_out, energy_mean=e_mean, camelot=camelot)


def random_analyses(n: int, rng: np.random.Generator, unknown_key_p: float = 0.2) -> list[Analysis]:
    out = []
    for i in range(n):
        cam = None if rng.random() < unknown_key_p else CAMELOT_CODES[int(rng.integers(24))]
        out.append(make(f"v{i:03d}", bpm=float(rng.uniform(125, 155)), e_in=float(rng.random()),
                        e_out=float(rng.random()), e_mean=float(rng.random()), camelot=cam))
    return out


def brute_force_best(analyses: list[Analysis], first: str, weights, voice_lookup=None) -> float:
    by_id = {a.video_id: a for a in analyses}
    others = [v for v in by_id if v != first]

    def cost(x, y):
        return order.transition(by_id[x], by_id[y], weights, voice_lookup).cost

    return min(order.path_cost([first, *p], cost) for p in itertools.permutations(others))


# ---------------------------------------------------------------------------
# Distância Camelot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a, b, expected", [
    ("8A", "8A", 0),
    ("8A", "9A", 1),
    ("8A", "7A", 1),
    ("8A", "8B", 1),
    ("12A", "1A", 1),
    ("1B", "12B", 1),
    ("8A", "10A", 2),
    ("8A", "2A", 6),
    ("8A", "3B", 6),       # circular 5 + 1 pela letra
    ("8A", "2B", 7),       # circular 6 + 1 pela letra (máximo)
    ("8A", "9B", 2),
    ("8a", " 9A ", 1),     # tolera minúscula e espaços
    ("08A", "8A", 0),
])
def test_camelot_distance_table(a, b, expected):
    assert order.camelot_distance(a, b) == expected


@pytest.mark.parametrize("a, b", [
    (None, "8A"), ("8A", None), (None, None), ("", "8A"), ("13A", "1A"), ("0A", "1A"),
    ("8C", "8A"), ("A", "8A"), ("xyz", "8A"), ("8", "8A"),
])
def test_camelot_distance_invalid_is_none(a, b):
    assert order.camelot_distance(a, b) is None


def test_camelot_distance_symmetric_and_bounded():
    for a in CAMELOT_CODES:
        for b in CAMELOT_CODES:
            d = order.camelot_distance(a, b)
            assert d == order.camelot_distance(b, a)
            assert 0 <= d <= 7
            assert (d == 0) == (a == b)


# ---------------------------------------------------------------------------
# Custo, classificação e path_cost
# ---------------------------------------------------------------------------

def test_transition_cost_formula():
    a = make("a", bpm=128, e_out=0.9, camelot="8A")
    b = make("b", bpm=132, e_in=0.6, camelot="10B")
    t = order.transition(a, b, W_DEFAULT,
                         voice_lookup=lambda x, y: ("voz de um lado", 0.1))
    expected = (1.0 * 4 / config.BPM_DIFF_NORM + 0.8 * 0.3 + 0.2 * 3 / config.CAMELOT_DIST_NORM
                + 0.5 * config.VOICE_COST["voz de um lado"])
    assert t.cost == pytest.approx(expected)
    assert (t.from_id, t.to_id) == ("a", "b")
    assert t.bpm_diff == pytest.approx(4)
    assert t.energy_diff == pytest.approx(0.3)
    assert t.camelot_dist == 3
    assert t.voice == "voz de um lado"
    assert t.voice_clash == pytest.approx(0.1)
    assert t.label == order.classify(expected, W_DEFAULT)


def test_transition_is_asymmetric_in_energy():
    a = make("a", e_in=0.1, e_out=0.9)
    b = make("b", e_in=0.9, e_out=0.1)
    w = {"energia": 1.0}
    assert order.transition(a, b, w).cost == pytest.approx(0.0)
    assert order.transition(b, a, w).cost == pytest.approx(0.0 + abs(0.1 - 0.1))
    c = make("c", e_in=0.2, e_out=0.2)
    assert order.transition(a, c, w).cost != pytest.approx(order.transition(c, a, w).cost)


def test_transition_uses_effective_bpm():
    a = make("a", bpm=130, bpm_manual=140)
    b = make("b", bpm=140)
    t = order.transition(a, b, {"bpm": 1.0})
    assert t.bpm_diff == pytest.approx(0.0)
    assert t.cost == pytest.approx(0.0)


def test_unknown_key_gives_zero_tone_term_and_none_distance():
    a = make("a", camelot=None)
    b = make("b", camelot="8A")
    t = order.transition(a, b, {"tom": 1.0})
    assert t.camelot_dist is None
    assert t.cost == 0.0
    assert t.label == "suave"


def test_missing_weights_count_as_zero():
    a = make("a", bpm=120, e_out=1.0, camelot="1A")
    b = make("b", bpm=160, e_in=0.0, camelot="7B")
    t = order.transition(a, b, {"bpm": 1.0})       # sem energia/tom/voz
    assert t.cost == pytest.approx(40 / config.BPM_DIFF_NORM)


def test_zero_weights_all_smooth_and_zero_cost():
    an = [make("a", bpm=120, e_out=1.0, camelot="1A"), make("b", bpm=160, e_in=0.0, camelot="7B"),
          make("c", bpm=140, camelot="4A")]
    zero = {"bpm": 0, "energia": 0, "tom": 0, "voz": 0}
    res = order.order_set(an, zero, voice_lookup=lambda x, y: ("choque de voz", 1.0))
    assert res.total_cost == 0.0
    assert [t.label for t in res.transitions] == ["suave", "suave"]
    assert all(t.cost == 0.0 for t in res.transitions)
    assert order.order_set(an, {}).total_cost == 0.0


def test_negative_or_invalid_weights_are_ignored():
    a, b = make("a", bpm=120), make("b", bpm=160)
    t = order.transition(a, b, {"bpm": -1.0, "energia": "x", "tom": float("nan")})
    assert t.cost == 0.0


def test_classify_thresholds():
    w = {"bpm": 1.0, "energia": 1.0}           # soma 2
    labels = config.TRANSITION_LABELS
    assert order.classify(0.0, w) == labels[0][1]
    assert order.classify(2 * labels[0][0], w) == labels[0][1]          # limite inclusivo
    assert order.classify(2 * labels[0][0] + 1e-6, w) == labels[1][1]
    assert order.classify(2 * (labels[1][0] + labels[2][0]) / 2, w) == labels[2][1]
    assert order.classify(100.0, w) == labels[-1][1]
    assert order.classify(float("inf"), w) == labels[-1][1]
    assert order.classify(5.0, {}) == "suave"
    assert order.classify(5.0, {"bpm": 0.0}) == "suave"


def test_label_ignores_weights_of_absent_terms():
    """Sem stems e sem tom, o rótulo é normalizado só por BPM e energia."""
    a = make("a", bpm=130, e_out=0.5)
    b = make("b", bpm=134, e_in=0.5)        # custo bpm = 0.5 -> normalizado 0.5/1.8 = 0.28
    w = {"bpm": 1.0, "energia": 0.8, "tom": 5.0, "voz": 5.0}
    t = order.transition(a, b, w)
    assert t.label == order.classify(t.cost, {"bpm": 1.0, "energia": 0.8})
    assert t.label == "boa"


def test_path_cost():
    costs = {("a", "b"): 1.0, ("b", "c"): 2.5, ("c", "a"): 10.0}
    assert order.path_cost(["a", "b", "c"], lambda x, y: costs[(x, y)]) == pytest.approx(3.5)
    assert order.path_cost(["a"], lambda x, y: 1 / 0) == 0.0
    assert order.path_cost([], lambda x, y: 1 / 0) == 0.0


def test_voice_lookup_errors_and_unknown_labels_become_no_stems(caplog):
    a, b = make("a"), make("b")

    def boom(x, y):
        raise RuntimeError("stem corrompido")

    with caplog.at_level(logging.WARNING, logger="modules.order"):
        t = order.transition(a, b, W_DEFAULT, voice_lookup=boom)
    assert t.voice == "sem stems" and t.voice_clash == 0.0
    t2 = order.transition(a, b, W_DEFAULT, voice_lookup=lambda x, y: ("???", 0.5))
    assert t2.voice == "sem stems"
    t3 = order.transition(a, b, W_DEFAULT, voice_lookup=lambda x, y: ("limpa", 7.0))
    assert t3.voice == "limpa" and t3.voice_clash == 1.0


# ---------------------------------------------------------------------------
# order_set / evaluate_order
# ---------------------------------------------------------------------------

def test_empty_and_single():
    assert order.order_set([]).order == []
    assert order.order_set([]).transitions == []
    res = order.order_set([make("x")])
    assert res.order == ["x"] and res.transitions == [] and res.total_cost == 0.0


def test_two_tracks_default_start_lowest_energy():
    res = order.order_set([make("a", e_mean=0.9), make("b", e_mean=0.2)], W_DEFAULT)
    assert res.order == ["b", "a"]
    assert len(res.transitions) == 1


def test_bpm_ascending_recovered_when_only_bpm_weight():
    rng = random.Random(3)
    bpms = [124, 126.5, 128, 130, 133, 135, 138, 141, 144, 150]
    an = [make(f"t{i}", bpm=b, e_mean=0.5, e_in=rng.random(), e_out=rng.random(),
               camelot=rng.choice(CAMELOT_CODES)) for i, b in enumerate(bpms)]
    rng.shuffle(an)
    res = order.order_set(an, {"bpm": 1.0}, start_id="t0")
    assert res.order == [f"t{i}" for i in range(len(bpms))]
    assert res.total_cost == pytest.approx((150 - 124) / config.BPM_DIFF_NORM)
    assert len(res.transitions) == len(bpms) - 1
    for t, (x, y) in zip(res.transitions, zip(res.order, res.order[1:])):
        assert (t.from_id, t.to_id) == (x, y)


def test_start_id_is_respected_and_invalid_start_ignored(caplog):
    an = [make("a", e_mean=0.3, bpm=130), make("b", e_mean=0.1, bpm=140), make("c", e_mean=0.9, bpm=135)]
    assert order.order_set(an, W_DEFAULT, start_id="c").order[0] == "c"
    with caplog.at_level(logging.WARNING, logger="modules.order"):
        res = order.order_set(an, W_DEFAULT, start_id="nao_existe")
    assert res.order[0] == "b"        # menor energy_mean
    assert any("nao_existe" in r.getMessage() for r in caplog.records)


def test_deterministic_regardless_of_input_order():
    rng = np.random.default_rng(11)
    an = random_analyses(25, rng)
    ref = order.order_set(an, W_DEFAULT)
    for seed in range(5):
        shuffled = an[:]
        random.Random(seed).shuffle(shuffled)
        res = order.order_set(shuffled, W_DEFAULT)
        assert res.order == ref.order
        assert res.total_cost == ref.total_cost


def test_ties_broken_by_video_id():
    an = [make("c"), make("a"), make("b"), make("d")]          # tudo igual
    res = order.order_set(an, W_DEFAULT)
    assert res.order == ["a", "b", "c", "d"]


def test_duplicates_are_dropped():
    an = [make("a", bpm=130), make("a", bpm=150), make("b", bpm=131)]
    res = order.order_set(an, {"bpm": 1.0})
    assert sorted(res.order) == ["a", "b"]
    assert res.total_cost == pytest.approx(1 / config.BPM_DIFF_NORM)


def test_matches_brute_force_on_small_sets():
    """n <= 7, início fixo: compara com todas as permutações e registra a diferença."""
    rng = np.random.default_rng(2026)
    gaps = []
    for trial in range(150):
        n = int(rng.integers(3, 8))
        an = random_analyses(n, rng)
        weights = W_DEFAULT if trial % 2 == 0 else {
            k: float(rng.uniform(0, config.WEIGHT_SLIDER_MAX)) for k in order.WEIGHT_KEYS}
        res = order.order_set(an, weights)
        best = brute_force_best(an, res.order[0], weights)
        assert res.total_cost >= best - 1e-9
        gaps.append(res.total_cost / best - 1.0 if best > 0 else 0.0)
    gaps = np.array(gaps)
    optimal_share = float(np.mean(gaps < 1e-9))
    print(f"\nforça bruta: {optimal_share:.1%} ótimas, diferença máx {gaps.max():.2%}, "
          f"média {gaps.mean():.3%}")
    assert optimal_share >= 0.95
    assert gaps.max() <= 0.05


def test_brute_force_with_asymmetric_voice_lookup():
    """Voz direcional (A→B difere de B→A): a heurística ainda acha o ótimo ou quase."""
    rng = np.random.default_rng(7)
    labels = list(config.VOICE_COST)
    for _ in range(40):
        an = random_analyses(6, rng)
        table = {(a.video_id, b.video_id): labels[int(rng.integers(len(labels)))]
                 for a in an for b in an if a is not b}

        def lookup(x, y, table=table):
            return table[(x, y)], 0.5

        w = {"bpm": 0.5, "energia": 0.5, "voz": 2.0}
        res = order.order_set(an, w, voice_lookup=lookup)
        best = brute_force_best(an, res.order[0], w, lookup)
        assert res.total_cost <= best * 1.05 + 1e-9


def _brute_force_matrix(cost: np.ndarray, start: int) -> float:
    others = [i for i in range(cost.shape[0]) if i != start]
    return min(order._path_cost_idx(cost, np.array([start, *p]))
               for p in itertools.permutations(others))


def test_held_karp_is_exact_on_random_asymmetric_matrices():
    rng = np.random.default_rng(21)
    for _ in range(60):
        n = int(rng.integers(2, 9))
        cost = rng.random((n, n)) * rng.choice([1.0, 10.0])
        np.fill_diagonal(cost, 0.0)
        start = int(rng.integers(n))
        path = order._held_karp(cost, start)
        assert path[0] == start and sorted(path) == list(range(n))
        assert order._path_cost_idx(cost, np.array(path)) == pytest.approx(
            _brute_force_matrix(cost, start), abs=1e-9)


def test_held_karp_ties_pick_smallest_ids():
    assert order._held_karp(np.zeros((6, 6)), 3) == [3, 0, 1, 2, 4, 5]
    assert order._held_karp(np.zeros((1, 1)), 0) == [0]
    assert order._held_karp(np.zeros((2, 2)), 1) == [1, 0]


def test_strategy_by_size(monkeypatch):
    """Exato até _EXACT_MAX_TRACKS; vários inícios até MAX_TWO_OPT_TRACKS; depois só guloso."""
    calls: list[str] = []
    real_hk, real_ms = order._held_karp, order._multi_start_search
    monkeypatch.setattr(order, "_held_karp", lambda c, s: calls.append("exato") or real_hk(c, s))
    monkeypatch.setattr(order, "_multi_start_search",
                        lambda c, s: calls.append("multi") or real_ms(c, s))
    rng = np.random.default_rng(8)
    order.order_set(random_analyses(order._EXACT_MAX_TRACKS, rng), W_DEFAULT)
    order.order_set(random_analyses(order._EXACT_MAX_TRACKS + 1, rng), W_DEFAULT)
    order.order_set(random_analyses(config.MAX_TWO_OPT_TRACKS + 1, rng), W_DEFAULT)
    assert calls == ["exato", "multi"]


def test_multi_start_never_worse_than_single_start():
    rng = np.random.default_rng(13)
    for _ in range(20):
        n = int(rng.integers(13, 35))
        cost = rng.random((n, n))
        np.fill_diagonal(cost, 0.0)
        single = order._local_search(cost, order._nearest_neighbor(cost, 0))
        multi = order._multi_start_search(cost, 0)
        assert multi[0] == 0 and sorted(multi) == list(range(n))
        assert (order._path_cost_idx(cost, np.array(multi))
                <= order._path_cost_idx(cost, np.array(single)) + 1e-12)


def test_multi_start_close_to_exact_on_realistic_sets():
    """Logo acima do limite do exato, a heurística fica no ótimo ou muito perto (registra)."""
    rng = np.random.default_rng(99)
    labels = list(config.VOICE_COST)
    gaps = []
    for trial in range(30):
        n = int(rng.integers(13, 15))
        an = random_analyses(n, rng)
        ids = sorted(a.video_id for a in an)
        by_id = {a.video_id: a for a in an}
        table = {(x, y): labels[int(rng.integers(len(labels)))] for x in ids for y in ids}
        lookup = (lambda x, y, t=table: (t[(x, y)], 0.5)) if trial % 2 else None
        cost = np.array([[0.0 if a == b else order.transition(by_id[a], by_id[b], W_DEFAULT,
                                                               lookup).cost
                          for b in ids] for a in ids])
        best = order._path_cost_idx(cost, np.array(order._held_karp(cost, 0)))
        got = order._path_cost_idx(cost, np.array(order._multi_start_search(cost, 0)))
        assert got >= best - 1e-9
        gaps.append(got / best - 1.0 if best > 0 else 0.0)
    gaps = np.array(gaps)
    print(f"\nvários inícios vs exato (13-14 faixas): {np.mean(gaps < 1e-9):.0%} ótimas, "
          f"diferença máx {gaps.max():.2%}, média {gaps.mean():.3%}")
    assert np.mean(gaps < 1e-9) >= 0.8
    assert gaps.max() <= 0.05


def test_local_search_never_worse_than_nearest_neighbor():
    rng = np.random.default_rng(5)
    for _ in range(20):
        n = int(rng.integers(8, 40))
        cost = rng.random((n, n))
        np.fill_diagonal(cost, 0.0)
        nn = order._nearest_neighbor(cost, 0)
        ls = order._local_search(cost, nn)
        assert ls[0] == 0 and sorted(ls) == list(range(n))
        assert order._path_cost_idx(cost, np.array(ls)) <= order._path_cost_idx(cost, np.array(nn)) + 1e-12


def test_move_deltas_match_recomputed_cost():
    """Os ganhos do 2-opt/or-opt são exatos para custo assimétrico."""
    rng = np.random.default_rng(0)
    for _ in range(40):
        n = int(rng.integers(3, 9))
        cost = rng.random((n, n))
        np.fill_diagonal(cost, 0.0)
        p = rng.permutation(n)
        base = order._path_cost_idx(cost, p)
        perm = cost[np.ix_(p, p)]
        ii, jj, d = order._two_opt_deltas(perm)
        for i, j, dd in zip(ii, jj, d):
            q = order._apply_two_opt(p, int(i), int(j))
            assert order._path_cost_idx(cost, q) - base == pytest.approx(dd, abs=1e-9)
        for seg_len in range(1, min(3, n - 2) + 1):
            for rev in (False, True):
                starts, dm = order._or_opt_deltas(perm, seg_len, rev)
                for r, i in enumerate(starts):
                    for k in range(n):
                        if np.isinf(dm[r, k]):
                            continue
                        q = order._apply_or_opt(p, int(i), seg_len, k, rev)
                        assert q[0] == p[0] and sorted(q) == list(range(n))
                        assert order._path_cost_idx(cost, q) - base == pytest.approx(dm[r, k], abs=1e-9)


def test_above_max_two_opt_uses_only_nearest_neighbor(monkeypatch):
    rng = np.random.default_rng(9)
    an = random_analyses(12, rng)
    full = order.order_set(an, W_DEFAULT)
    calls = []
    monkeypatch.setattr(order, "_local_search", lambda cost, path: calls.append(1) or path)
    monkeypatch.setattr(config, "MAX_TWO_OPT_TRACKS", 5)
    nn_only = order.order_set(an, W_DEFAULT)
    assert calls == []
    assert nn_only.total_cost >= full.total_cost - 1e-9


def test_voice_lookup_changes_order():
    # Tudo igual, só a voz decide: a->b e b->c têm choque; a->c e c->b estão limpas.
    an = [make("a", e_mean=0.1), make("b"), make("c")]
    clashes = {("a", "b"), ("b", "c"), ("b", "a"), ("c", "a")}

    def lookup(x, y):
        return ("choque de voz", 0.8) if (x, y) in clashes else ("limpa", 0.0)

    without = order.order_set(an, W_DEFAULT)
    assert without.order == ["a", "b", "c"]
    with_voice = order.order_set(an, W_DEFAULT, voice_lookup=lookup)
    assert with_voice.order == ["a", "c", "b"]
    assert [t.voice for t in with_voice.transitions] == ["limpa", "limpa"]
    assert with_voice.total_cost == 0.0
    assert [t.label for t in with_voice.transitions] == ["suave", "suave"]


def test_evaluate_order_matches_order_set_and_manual_order():
    rng = np.random.default_rng(4)
    an = random_analyses(10, rng)
    by_id = {a.video_id: a for a in an}
    res = order.order_set(an, W_DEFAULT)
    ev = order.evaluate_order(res.order, by_id, W_DEFAULT)
    assert ev.order == res.order
    assert ev.total_cost == pytest.approx(res.total_cost)
    assert [t.to_dict() for t in ev.transitions] == [t.to_dict() for t in res.transitions]

    manual = list(reversed(res.order))
    ev2 = order.evaluate_order(manual, by_id, W_DEFAULT)
    assert ev2.order == manual
    assert len(ev2.transitions) == len(manual) - 1
    # A ordem manual nunca é melhor que a ótima com a mesma 1ª faixa (10 faixas: busca exata).
    best_same_start = order.order_set(an, W_DEFAULT, start_id=manual[0])
    assert best_same_start.order[0] == manual[0]
    assert ev2.total_cost >= best_same_start.total_cost - 1e-9
    assert ev2.total_cost == pytest.approx(sum(t.cost for t in ev2.transitions))
    assert ev2.total_cost == pytest.approx(
        order.path_cost(manual, lambda x, y: order.transition(by_id[x], by_id[y], W_DEFAULT).cost))


def test_evaluate_order_defaults_and_missing_ids():
    by_id = {"a": make("a", bpm=128), "b": make("b", bpm=136)}
    ev = order.evaluate_order(["a", "zzz", "b"], by_id)        # pesos padrão do config
    assert ev.order == ["a", "b"]
    assert len(ev.transitions) == 1
    assert ev.total_cost == pytest.approx(config.DEFAULT_WEIGHTS["bpm"] * 8 / config.BPM_DIFF_NORM)
    assert order.evaluate_order([], by_id).order == []
    one = order.evaluate_order(["a"], by_id)
    assert one.order == ["a"] and one.transitions == [] and one.total_cost == 0.0


def test_weights_none_reads_config_at_call_time(monkeypatch):
    an = [make("a", bpm=128, e_mean=0.1), make("b", bpm=136)]
    monkeypatch.setattr(config, "DEFAULT_WEIGHTS", {"bpm": 2.0})
    res = order.order_set(an)
    assert res.total_cost == pytest.approx(2.0 * 8 / config.BPM_DIFF_NORM)


@pytest.mark.parametrize("n, limit_s", [(12, 1.0), (30, 1.0), (60, 3.0), (100, 5.0)])
def test_performance(n, limit_s):
    rng = np.random.default_rng(n)
    an = random_analyses(n, rng)
    labels = list(config.VOICE_COST)

    def lookup(x, y):
        return labels[(hash(x) + 3 * hash(y)) % len(labels)], 0.3

    t0 = time.perf_counter()
    res = order.order_set(an, W_DEFAULT, voice_lookup=lookup)
    elapsed = time.perf_counter() - t0
    print(f"\n{n} faixas: {elapsed:.3f} s")
    assert sorted(res.order) == sorted(a.video_id for a in an)
    assert len(res.transitions) == n - 1
    assert elapsed < limit_s


# ---------------------------------------------------------------------------
# Modo de energia ("suave" x "subindo")
# ---------------------------------------------------------------------------

def test_energy_mode_rising_term_formula():
    a = make("a", e_out=0.9, e_mean=0.7)
    b = make("b", e_in=0.5, e_mean=0.4)
    w = {"energia": 1.0}
    suave = order.transition(a, b, w)
    assert suave.cost == pytest.approx(0.4)
    assert order.transition(a, b, w, energy_mode="suave").cost == pytest.approx(0.4)
    subindo = order.transition(a, b, w, energy_mode="subindo")
    assert subindo.cost == pytest.approx((0.7 - 0.4) + 0.25 * 0.4)
    assert subindo.energy_diff == pytest.approx(0.4)        # campo continua |saída − entrada|
    # Subir de energia não custa nada além da emenda saída→entrada.
    up = order.transition(b, a, w, energy_mode="subindo")
    assert up.cost == pytest.approx(0.25 * abs(b.energy_out - a.energy_in))


def test_energy_mode_unknown_falls_back_to_smooth(caplog):
    a, b = make("a", e_out=0.9, e_mean=0.9), make("b", e_in=0.1, e_mean=0.1)
    with caplog.at_level(logging.WARNING, logger="modules.order"):
        t = order.transition(a, b, {"energia": 1.0}, energy_mode="xyz")
    assert t.cost == pytest.approx(0.8)
    assert any("xyz" in r.getMessage() for r in caplog.records)


def test_energy_mode_rising_orders_by_energy_mean():
    """Só com peso de energia, o modo "subindo" põe o set em energy_mean crescente."""
    rng = random.Random(11)
    means = [0.05 + 0.12 * i for i in range(8)]
    an = [make(f"t{i}", e_mean=m, e_in=rng.uniform(0.48, 0.52), e_out=rng.uniform(0.48, 0.52))
          for i, m in enumerate(means)]
    rng.shuffle(an)
    res = order.order_set(an, {"energia": 1.0}, energy_mode="subindo")
    by_id = {a.video_id: a for a in an}
    got = [by_id[v].energy_mean for v in res.order]
    assert got == sorted(got)
    assert res.order == [f"t{i}" for i in range(len(means))]
    ev = order.evaluate_order(res.order, by_id, {"energia": 1.0}, energy_mode="subindo")
    assert ev.total_cost == pytest.approx(res.total_cost)


def test_energy_mode_changes_order_versus_smooth():
    # "suave" emenda saída→entrada (a→c→b); "subindo" prefere a energia média crescente.
    an = [make("a", e_mean=0.1, e_in=0.1, e_out=0.9),
          make("b", e_mean=0.5, e_in=0.1, e_out=0.5),
          make("c", e_mean=0.9, e_in=0.9, e_out=0.1)]
    w = {"energia": 1.0}
    assert order.order_set(an, w).order == ["a", "c", "b"]
    assert order.order_set(an, w, energy_mode="suave").order == ["a", "c", "b"]
    assert order.order_set(an, w, energy_mode="subindo").order == ["a", "b", "c"]
