"""Ordenação do set (seção 4.6). Sem LLM: custo explícito + heurística.

CONTRATO (implementar mantendo as assinaturas):

custo(A→B) = w_bpm·|ΔBPM|/BPM_DIFF_NORM
           + w_energia·|energia_saída(A) − energia_entrada(B)|
           + w_tom·distância_Camelot/CAMELOT_DIST_NORM   (0 se algum tom desconhecido)
           + w_voz·VOICE_COST[rótulo de voz]            (0 sem stems)
ΔBPM usa `Analysis.effective_bpm`. O custo é ASSIMÉTRICO (saída de A vs entrada de B e
janelas de voz direcionais): o 2-opt deve recalcular o custo do caminho, não usar o atalho
simétrico. Caminho aberto (não ciclo), com a 1ª faixa fixa.
Algoritmo: vizinho mais próximo a partir de `start_id` (padrão: menor energy_mean)
seguido de 2-opt até não melhorar (até MAX_TWO_OPT_TRACKS faixas).
Classificação: custo / soma dos pesos usados, comparado com TRANSITION_LABELS.

Detalhes da implementação:
- A matriz de custos (n×n) é calculada uma vez; a busca trabalha só com ela.
- Até _EXACT_MAX_TRACKS faixas a ordem é EXATA (Held-Karp de caminho aberto com início
  fixo, O(n²·2ⁿ) vetorizado com numpy); empates vão para a ordem com menores video_ids.
- Acima disso (até MAX_TWO_OPT_TRACKS): vizinho mais próximo com vários inícios (a 1ª faixa
  fica fixa e a 2ª varia entre as _MULTI_START_MAX mais baratas), cada um seguido de busca
  local; fica o melhor caminho. Acima de MAX_TWO_OPT_TRACKS, só o vizinho mais próximo.
- Busca local = 2-opt de caminho aberto + or-opt (mover um bloco de 1 a 3 faixas, também
  invertido). Os ganhos de cada movimento são calculados de forma exata para custo
  assimétrico (somas acumuladas das arestas nos dois sentidos) e cada movimento aceito é
  conferido recalculando o custo do caminho inteiro.
- "Pesos usados" na classificação: o peso do tom só conta se os dois tons são conhecidos e
  o peso da voz só conta se há stems dos dois lados (assim um termo ausente não "dilui" o
  rótulo da transição).

Modo de energia (`energy_mode`, opcional, padrão "suave"):
- "suave": termo de energia = |energia_saída(A) − energia_entrada(B)| (comportamento original).
- "subindo": o set deve crescer. Termo de energia = max(0, energy_mean(A) − energy_mean(B))
  + 0.25·|energia_saída(A) − energia_entrada(B)|: cair de energia é caro, subir é de graça,
  e a emenda saída→entrada ainda conta um pouco.
`Transition.energy_diff` continua sendo |energia_saída(A) − energia_entrada(B)| nos dois modos.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

import config
from modules.models import Analysis, Transition

logger = logging.getLogger(__name__)

# voice_lookup(from_id, to_id) -> (rótulo, fração de choque 0..1). Ver modules/mixwindows.py.
VoiceLookup = Callable[[str, str], tuple[str, float]]

WEIGHT_KEYS = ("bpm", "energia", "tom", "voz")
NO_STEMS_LABEL = "sem stems"
ENERGY_MODES = ("suave", "subindo")   # ver docstring do módulo
_RISING_EDGE_FACTOR = 0.25            # peso da emenda saída→entrada no modo "subindo"
_OR_OPT_MAX_SEGMENT = 3     # maior bloco de faixas movido de uma vez no or-opt
_EPS = 1e-9                 # melhora mínima para aceitar um movimento
_EXACT_MAX_TRACKS = 12      # até aqui, solução exata (2^11 subconjuntos × 11 faixas)
_MULTI_START_MAX = 16       # quantas 2ªs faixas diferentes o vizinho mais próximo testa


@dataclass
class OrderResult:
    order: list[str]                    # video_ids na ordem do set
    transitions: list[Transition] = field(default_factory=list)  # len(order) - 1
    total_cost: float = 0.0


# ---------------------------------------------------------------------------
# Custo de uma transição
# ---------------------------------------------------------------------------

def _parse_camelot(code: str | None) -> tuple[int, str] | None:
    """"8A" -> (8, "A"). Aceita minúsculas e espaços; None se inválido."""
    if not isinstance(code, str):
        return None
    s = code.strip().upper()
    if len(s) < 2:
        return None
    number, letter = s[:-1], s[-1]
    if letter not in ("A", "B") or not number.isdigit():
        return None
    n = int(number)
    if not 1 <= n <= 12:
        return None
    return n, letter


def camelot_distance(a: str | None, b: str | None) -> int | None:
    """Distância na roda Camelot. Mesmo código = 0; ±1 no número (mesma letra) = 1;
    mesmo número com letra trocada = 1; caso geral = distância circular do número (0..6)
    + 1 se a letra difere. None se algum lado é None/ inválido.
    """
    pa, pb = _parse_camelot(a), _parse_camelot(b)
    if pa is None or pb is None:
        return None
    d = abs(pa[0] - pb[0]) % 12
    d = min(d, 12 - d)
    return int(d + (1 if pa[1] != pb[1] else 0))


def _clean_weights(weights: dict[str, float] | None) -> dict[str, float]:
    """Os 4 pesos conhecidos como float >= 0 (ausente, inválido ou negativo = 0)."""
    src = config.DEFAULT_WEIGHTS if weights is None else weights
    out: dict[str, float] = {}
    for key in WEIGHT_KEYS:
        try:
            w = float(src.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            w = 0.0
        out[key] = w if math.isfinite(w) and w > 0 else 0.0
    return out


def _finite(x: float) -> float:
    return x if math.isfinite(x) else 0.0


def _lookup_voice(voice_lookup: VoiceLookup | None, from_id: str, to_id: str) -> tuple[str, float]:
    if voice_lookup is None:
        return NO_STEMS_LABEL, 0.0
    try:
        label, clash = voice_lookup(from_id, to_id)
    except Exception:  # um stem ruim não pode derrubar a ordenação
        logger.warning("Falha ao consultar a janela de voz %s -> %s; tratando como sem stems.",
                       from_id, to_id, exc_info=True)
        return NO_STEMS_LABEL, 0.0
    if label not in config.VOICE_COST:
        logger.warning("Rótulo de voz desconhecido %r (%s -> %s); tratando como sem stems.",
                       label, from_id, to_id)
        return NO_STEMS_LABEL, 0.0
    try:
        clash = float(clash)
    except (TypeError, ValueError):
        clash = 0.0
    return str(label), min(max(_finite(clash), 0.0), 1.0)


def _clean_energy_mode(energy_mode: str | None) -> str:
    """Modo de energia válido; desconhecido vira "suave" (com aviso no log)."""
    if energy_mode is None:
        return "suave"
    mode = str(energy_mode).strip().lower()
    if mode not in ENERGY_MODES:
        logger.warning("Modo de energia desconhecido %r; usando \"suave\".", energy_mode)
        return "suave"
    return mode


def _energy_term(a: Analysis, b: Analysis, energy_diff: float, mode: str) -> float:
    if mode == "subindo":
        drop = _finite(float(a.energy_mean) - float(b.energy_mean))
        return max(0.0, drop) + _RISING_EDGE_FACTOR * energy_diff
    return energy_diff


def transition(a: Analysis, b: Analysis, weights: dict[str, float],
               voice_lookup: VoiceLookup | None = None, *,
               energy_mode: str = "suave") -> Transition:
    """Calcula o custo e a classificação da transição A→B."""
    w = _clean_weights(weights)
    mode = _clean_energy_mode(energy_mode)
    bpm_diff = _finite(abs(float(a.effective_bpm) - float(b.effective_bpm)))
    energy_diff = _finite(abs(float(a.energy_out) - float(b.energy_in)))
    cam = camelot_distance(a.camelot, b.camelot)
    voice, clash = _lookup_voice(voice_lookup, a.video_id, b.video_id)

    terms = {
        "bpm": bpm_diff / config.BPM_DIFF_NORM,
        "energia": _energy_term(a, b, energy_diff, mode),
        "tom": cam / config.CAMELOT_DIST_NORM if cam is not None else 0.0,
        "voz": float(config.VOICE_COST.get(voice, 0.0)),
    }
    cost = float(sum(w[k] * terms[k] for k in WEIGHT_KEYS))

    used = dict(w)
    if cam is None:
        used["tom"] = 0.0
    if voice == NO_STEMS_LABEL:
        used["voz"] = 0.0

    return Transition(
        from_id=a.video_id, to_id=b.video_id, cost=cost, bpm_diff=bpm_diff,
        energy_diff=energy_diff, camelot_dist=cam, voice=voice, voice_clash=clash,
        label=classify(cost, used),
    )


def classify(cost: float, weights: dict[str, float]) -> str:
    """Rótulo de TRANSITION_LABELS a partir do custo normalizado pela soma dos pesos (>0)."""
    labels = config.TRANSITION_LABELS
    total = sum(_clean_weights(weights).values())
    if total <= 0:
        return labels[0][1]
    normalized = cost / total
    for threshold, label in labels:
        if normalized <= threshold:
            return label
    return labels[-1][1]   # NaN cai aqui


def path_cost(order: list[str], cost_fn: Callable[[str, str], float]) -> float:
    """Soma do custo das transições consecutivas (caminho aberto)."""
    return float(sum(cost_fn(a, b) for a, b in zip(order, order[1:])))


# ---------------------------------------------------------------------------
# Heurística: vizinho mais próximo + busca local (2-opt e or-opt) na matriz de custos
# ---------------------------------------------------------------------------

def _path_cost_idx(cost: np.ndarray, path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    return float(cost[path[:-1], path[1:]].sum())


def _nearest_neighbor(cost: np.ndarray, start: int, second: int | None = None) -> list[int]:
    """Caminho guloso a partir de `start`; empate vai para o menor índice (= menor video_id).

    `second` força a 2ª faixa (usado pelos vários inícios).
    """
    n = cost.shape[0]
    visited = np.zeros(n, dtype=bool)
    path = [start]
    visited[start] = True
    current = start
    if second is not None and second != start and n > 1:
        path.append(second)
        visited[second] = True
        current = second
    for _ in range(n - len(path)):
        row = np.where(visited, np.inf, cost[current])
        nxt = int(np.argmin(row))
        path.append(nxt)
        visited[nxt] = True
        current = nxt
    return path


def _edge_prefix(perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Somas acumuladas das arestas do caminho no sentido direto e no inverso.

    fwd[k] = Σ_{t<k} P[t, t+1]   bwd[k] = Σ_{t<k} P[t+1, t]
    """
    fwd = np.concatenate(([0.0], np.cumsum(np.diagonal(perm, 1))))
    bwd = np.concatenate(([0.0], np.cumsum(np.diagonal(perm, -1))))
    return fwd, bwd


def _two_opt_deltas(perm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Variação exata do custo ao inverter o trecho de posições [i..j] (1 <= i < j <= n-1).

    `perm[a, b]` é o custo da faixa na posição a para a da posição b.
    """
    n = perm.shape[0]
    fwd, bwd = _edge_prefix(perm)
    ii, jj = np.triu_indices(n, 1)
    keep = ii >= 1                      # a posição 0 (início do set) é fixa
    ii, jj = ii[keep], jj[keep]
    delta = (perm[ii - 1, jj] - perm[ii - 1, ii]
             + (bwd[jj] - bwd[ii]) - (fwd[jj] - fwd[ii]))
    has_next = jj < n - 1
    jn = np.minimum(jj + 1, n - 1)
    delta = delta + np.where(has_next, perm[ii, jn] - perm[jj, jn], 0.0)
    return ii, jj, delta


def _or_opt_deltas(perm: np.ndarray, seg_len: int, reverse: bool) -> tuple[np.ndarray, np.ndarray]:
    """Variação exata ao mover o bloco [i..i+L-1] para logo depois da posição k.

    Devolve (starts, delta[len(starts), n]); movimentos inválidos valem +inf.
    """
    n = perm.shape[0]
    starts = np.arange(1, n - seg_len + 1)
    ends = starts + seg_len - 1
    # Retirar o bloco: some (i-1→i) e (e→e+1); entra (i-1→e+1).
    removal = -perm[starts - 1, starts]
    has_after = ends < n - 1
    en = np.minimum(ends + 1, n - 1)
    removal = removal + np.where(has_after, perm[starts - 1, en] - perm[ends, en], 0.0)
    # Inserir entre k e k+1: some (k→k+1); entram (k→cabeça) e (cauda→k+1).
    k = np.arange(n)
    kn = np.minimum(k + 1, n - 1)
    k_has_next = k < n - 1
    drop_k = np.where(k_has_next, -perm[k, kn], 0.0)
    head, tail = (ends, starts) if reverse else (starts, ends)
    insert = (perm[k[None, :], head[:, None]]
              + np.where(k_has_next[None, :], perm[tail[:, None], kn[None, :]], 0.0)
              + drop_k[None, :])
    if reverse:
        fwd, bwd = _edge_prefix(perm)
        insert = insert + ((bwd[ends] - bwd[starts]) - (fwd[ends] - fwd[starts]))[:, None]
    delta = removal[:, None] + insert
    valid = (k[None, :] < starts[:, None] - 1) | (k[None, :] > ends[:, None])
    return starts, np.where(valid, delta, np.inf)


def _apply_two_opt(path: np.ndarray, i: int, j: int) -> np.ndarray:
    new = path.copy()
    new[i:j + 1] = path[i:j + 1][::-1]
    return new


def _apply_or_opt(path: np.ndarray, i: int, seg_len: int, k: int, reverse: bool) -> np.ndarray:
    seg = path[i:i + seg_len]
    if reverse:
        seg = seg[::-1]
    rest = np.concatenate((path[:i], path[i + seg_len:]))
    pos = k + 1 if k < i else k - seg_len + 1
    return np.concatenate((rest[:pos], seg, rest[pos:]))


def _best_move(perm: np.ndarray, path: np.ndarray) -> tuple[float, np.ndarray | None]:
    """Melhor movimento (menor delta) entre 2-opt e or-opt. Ordem fixa = determinístico."""
    n = len(path)
    best_delta, best_path = 0.0, None

    ii, jj, delta = _two_opt_deltas(perm)
    if delta.size:
        idx = int(np.argmin(delta))
        if delta[idx] < best_delta:
            best_delta = float(delta[idx])
            best_path = _apply_two_opt(path, int(ii[idx]), int(jj[idx]))

    for seg_len in range(1, min(_OR_OPT_MAX_SEGMENT, n - 2) + 1):
        for reverse in ((False,) if seg_len == 1 else (False, True)):
            starts, d = _or_opt_deltas(perm, seg_len, reverse)
            if not d.size:
                continue
            r, k = np.unravel_index(int(np.argmin(d)), d.shape)
            if d[r, k] < best_delta:
                best_delta = float(d[r, k])
                best_path = _apply_or_opt(path, int(starts[r]), seg_len, int(k), reverse)
    return best_delta, best_path


def _local_search(cost: np.ndarray, path: list[int], max_iters: int | None = None) -> list[int]:
    """2-opt + or-opt até não melhorar. A 1ª posição nunca sai do lugar."""
    p = np.asarray(path, dtype=np.intp)
    n = len(p)
    if n < 3:
        return p.tolist()
    current = _path_cost_idx(cost, p)
    for _ in range(max_iters or 100 * n):
        perm = cost[np.ix_(p, p)]
        delta, candidate = _best_move(perm, p)
        if candidate is None or delta >= -_EPS:
            break
        new_cost = _path_cost_idx(cost, candidate)
        if new_cost >= current - _EPS:   # conferência com o custo real do caminho
            break
        p, current = candidate, new_cost
    return p.tolist()


def _held_karp(cost: np.ndarray, start: int) -> list[int]:
    """Caminho aberto ÓTIMO com a 1ª faixa fixa (programação dinâmica sobre subconjuntos).

    g[R, j] = menor custo para, saindo da faixa j, visitar todas as faixas do conjunto R
    (j fora de R). A reconstrução escolhe, a cada passo, o menor índice que mantém o ótimo,
    então empates ficam com a ordem de menores video_ids.
    """
    n = cost.shape[0]
    others = np.array([i for i in range(n) if i != start], dtype=np.intp)
    m = len(others)
    if m <= 1:
        return [start, *others.tolist()]
    sub = cost[np.ix_(others, others)]          # sub[j, k]: others[j] -> others[k]
    full = 1 << m
    masks = np.arange(full, dtype=np.int64)
    bits = (masks[:, None] >> np.arange(m)) & 1
    sizes = bits.sum(axis=1)
    g = np.full((full, m), np.inf)
    g[0] = 0.0
    for size in range(1, m):                    # |R| = m nunca é usado (j precisa ficar fora)
        layer = masks[sizes == size]
        best = np.full((len(layer), m), np.inf)
        for k in range(m):
            rows = np.nonzero((layer >> k) & 1)[0]
            cand = g[layer[rows] ^ (1 << k), k][:, None] + sub[:, k][None, :]
            best[rows] = np.minimum(best[rows], cand)
        best[bits[layer].astype(bool)] = np.inf   # j dentro de R: estado inválido
        g[layer] = best

    path = [start]
    remaining = full - 1
    from_row = cost[start, others]
    while remaining:
        ks = np.nonzero(bits[remaining])[0]
        vals = from_row[ks] + g[remaining ^ (1 << ks), ks]
        target = float(vals.min())
        k = int(ks[np.nonzero(vals <= target + _EPS * max(1.0, abs(target)))[0][0]])
        path.append(int(others[k]))
        remaining ^= 1 << k
        from_row = sub[k]
    return path


def _multi_start_search(cost: np.ndarray, start: int) -> list[int]:
    """Vizinho mais próximo + busca local com várias 2ªs faixas; fica o caminho mais barato.

    A 1ª faixa nunca muda. Empate de custo fica com a ordem de menores índices.
    """
    n = cost.shape[0]
    candidates = sorted((i for i in range(n) if i != start), key=lambda i: (cost[start, i], i))
    best_path: list[int] | None = None
    best_cost = math.inf
    # Menos reinícios quando o set é grande: o tempo da busca local cresce com n².
    n_starts = min(_MULTI_START_MAX, max(4, 240 // max(n, 1)))
    for second in candidates[:n_starts]:
        path = _local_search(cost, _nearest_neighbor(cost, start, second))
        c = _path_cost_idx(cost, np.asarray(path, dtype=np.intp))
        if (best_path is None or c < best_cost - _EPS
                or (c <= best_cost + _EPS and path < best_path)):
            best_path, best_cost = path, min(c, best_cost)
    return best_path if best_path is not None else [start]


def _pick_start(ids: list[str], by_id: dict[str, Analysis], start_id: str | None) -> int:
    if start_id is not None:
        if start_id in by_id:
            return ids.index(start_id)
        logger.warning("Faixa inicial %r não está entre as aprovadas; usando a de menor energia.",
                       start_id)

    def key(vid: str) -> tuple[float, str]:
        e = float(by_id[vid].energy_mean)
        return (e if math.isfinite(e) else math.inf, vid)

    return ids.index(min(ids, key=key))


def order_set(analyses: list[Analysis], weights: dict[str, float] | None = None,
              start_id: str | None = None,
              voice_lookup: VoiceLookup | None = None, *,
              energy_mode: str = "suave") -> OrderResult:
    """Ordena as faixas aprovadas. Lista vazia -> OrderResult([]); 1 faixa -> sem transições.

    weights=None usa config.DEFAULT_WEIGHTS; chaves ausentes valem 0.
    energy_mode: "suave" (padrão) ou "subindo" (ver docstring do módulo).
    Determinístico (desempate por video_id).
    """
    w = _clean_weights(weights)
    mode = _clean_energy_mode(energy_mode)
    by_id: dict[str, Analysis] = {}
    for an in analyses:
        if an.video_id in by_id:
            logger.warning("Faixa repetida na ordenação (%s); mantendo a primeira.", an.video_id)
            continue
        by_id[an.video_id] = an
    ids = sorted(by_id)
    n = len(ids)
    if n == 0:
        return OrderResult([])
    if n == 1:
        return OrderResult([ids[0]], [], 0.0)

    trans: list[list[Transition | None]] = [[None] * n for _ in range(n)]
    cost = np.zeros((n, n), dtype=np.float64)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i != j:
                t = transition(by_id[a], by_id[b], w, voice_lookup, energy_mode=mode)
                trans[i][j] = t
                cost[i, j] = t.cost

    start = _pick_start(ids, by_id, start_id)
    limit = config.MAX_TWO_OPT_TRACKS
    if n <= min(_EXACT_MAX_TRACKS, limit):
        path = _held_karp(cost, start)
    elif n <= limit:
        path = _multi_start_search(cost, start)
    else:
        path = _nearest_neighbor(cost, start)

    transitions = [trans[i][j] for i, j in zip(path, path[1:])]
    return OrderResult(order=[ids[i] for i in path], transitions=transitions,
                       total_cost=float(sum(t.cost for t in transitions)))


def evaluate_order(order: list[str], analyses: dict[str, Analysis], weights: dict[str, float] | None = None,
                   voice_lookup: VoiceLookup | None = None, *,
                   energy_mode: str = "suave") -> OrderResult:
    """Recalcula transições/custo para uma ordem dada (ex.: o DJ reordenou manualmente).

    Ids sem análise são retirados da ordem devolvida (com aviso no log), para manter
    len(transitions) == len(order) - 1. energy_mode: igual ao de `order_set`.
    """
    w = _clean_weights(weights)
    mode = _clean_energy_mode(energy_mode)
    kept = []
    for vid in order:
        if vid in analyses:
            kept.append(vid)
        else:
            logger.warning("Faixa %s sem análise foi ignorada na avaliação da ordem.", vid)
    transitions = [transition(analyses[a], analyses[b], w, voice_lookup, energy_mode=mode)
                   for a, b in zip(kept, kept[1:])]
    return OrderResult(order=kept, transitions=transitions,
                       total_cost=float(sum(t.cost for t in transitions)))
