"""Configuração central do painel de curadoria do DJ SK.

Valores ajustáveis (pesos, faixa de BPM, taxas, teto de orçamento) ficam aqui.
Segredos NUNCA ficam aqui: são lidos do arquivo `.env` (veja `.env.example`).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv é opcional em tempo de teste
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Pastas
# ---------------------------------------------------------------------------
CACHE_DIR = BASE_DIR / "cache"            # áudios baixados ({videoId}.mp3) e análises ({videoId}.json)
STEMS_DIR = BASE_DIR / "stems"            # stems/{videoId}/vocals.mp3 etc.
EXPORT_DIR = BASE_DIR / "Setlist_Atual"   # exportação padrão do módulo export
SETLISTS_DIR = BASE_DIR / "Setlists"     # o painel cria uma subpasta com nome próprio a cada exportação
DATA_DIR = BASE_DIR / "data"              # orçamento, métricas, sessão salva, token do Spotify
PLAYLIST_NAME = "Setlist_Atual"           # nome do .m3u8 e do .csv exportados

# ---------------------------------------------------------------------------
# Referências e sugestões (4.1 e 4.2)
# ---------------------------------------------------------------------------
MAX_REFERENCES = 3
N_SUGGESTIONS = 20                 # quantas sugestões aparecem por página na triagem
MIN_TRACK_DURATION_S = 60          # descarta vinhetas e trechos curtos
MAX_TRACK_DURATION_S = 480         # descarta mixes/sets longos
# Versões que costumam não servir para tocar no set. A interface permite desligar o filtro.
EXCLUDE_KEYWORDS = ["slowed", "reverb", "sped up", "speed up", "8d audio", "nightcore"]
YOUTUBE_STYLE_QUERY = "funk mandelão"  # termo de estilo usado no plano B (YouTube Data API)

# ---------------------------------------------------------------------------
# Download (4.3)
# ---------------------------------------------------------------------------
MAX_PARALLEL_DOWNLOADS = 4
AUDIO_CODEC = "mp3"
AUDIO_QUALITY = "320"              # kbps do mp3 gerado pelo ffmpeg
DOWNLOAD_RETRIES = 2
LOCAL_AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif")

# ---------------------------------------------------------------------------
# Análise de áudio (4.4)
# ---------------------------------------------------------------------------
ANALYSIS_VERSION = 1               # aumente para invalidar análises antigas em cache
ANALYSIS_SAMPLE_RATE = 22050
BPM_MIN = 120.0                    # [VALIDAR] com o DJ SK a faixa típica do mandelão
BPM_MAX = 160.0
ENERGY_WINDOW_S = 1.0              # RMS por janela de 1 s
INTRO_THRESHOLD = 0.5              # energia normalizada que marca o fim da intro / início da saída
TRANSITION_WINDOW_S = 16.0         # janela de entrada/saída usada nas transições (~8 compassos a 130 BPM)
WAVEFORM_POINTS = 400              # pontos da waveform reduzida para desenho
KEY_MIN_CONFIDENCE = 0.3           # abaixo disso o tom é tratado como desconhecido

# ---------------------------------------------------------------------------
# Ordenação do set (4.6)
# ---------------------------------------------------------------------------
# Pesos iniciais dos sliders: BPM e energia altos, tom baixo.
DEFAULT_WEIGHTS = {"bpm": 1.0, "energia": 0.8, "tom": 0.2, "voz": 0.5}
WEIGHT_SLIDER_MAX = 2.0
BPM_DIFF_NORM = 8.0                # |ΔBPM| de 8 vale 1 ponto de custo antes do peso
CAMELOT_DIST_NORM = 6.0            # distância Camelot máxima útil
MAX_TWO_OPT_TRACKS = 60            # acima disso, só vizinho mais próximo
# Classificação de cada transição pelo custo normalizado (custo / soma dos pesos).
TRANSITION_LABELS = [(0.15, "suave"), (0.35, "boa"), (0.6, "atenção"), (float("inf"), "difícil")]

# ---------------------------------------------------------------------------
# Janelas de mixagem (4.8)
# ---------------------------------------------------------------------------
VOICE_HOP_S = 0.1                  # resolução da detecção de voz no stem
VOICE_THRESHOLD_DB = -35.0         # RMS do stem de voz acima disso (dBFS) conta como voz
VOICE_MIN_SEGMENT_S = 0.5          # trechos de voz mais curtos que isso são ignorados
VOICE_MERGE_GAP_S = 0.6            # lacunas menores que isso unem dois trechos
VOICE_CLASH_MIN_OVERLAP = 0.2      # fração da janela com voz dos dois lados que já é choque
# Custo de voz que entra na ordenação (multiplicado por w_voz).
VOICE_COST = {"limpa": 0.0, "voz de um lado": 0.3, "choque de voz": 1.0, "sem stems": 0.0}

# ---------------------------------------------------------------------------
# Estúdio de Vocal / Replicate (4.7) e orçamento (7)
# ---------------------------------------------------------------------------
REPLICATE_MODEL = "cjwbw/demucs"
# Versão lida na aba API do modelo em 22/10/2026 (modelo comunitário exige a versão).
REPLICATE_VERSION = "25a173108cff36ef9f80f854c162d01df9e6528be175794b81158fa03836d953"
DEMUCS_MODEL_NAME = "htdemucs"     # separa em vocals, drums, bass, other
DEMUCS_OUTPUT_FORMAT = "mp3"
STEM_NAMES = ("vocals", "drums", "bass", "other")
REPLICATE_USD_PER_SECOND = 0.000225  # Nvidia T4 (hardware do cjwbw/demucs)
STEM_COST_ESTIMATE_USD = 0.016       # custo típico informado pela Replicate por execução
USD_BRL = 5.40                       # câmbio para converter o gasto (ajuste no dia)
BUDGET_CAP_BRL = 10.00               # capital total do projeto
REPLICATE_CREDIT_USD = 2.00          # crédito comprado na Replicate
REPLICATE_TIMEOUT_S = 600

# ---------------------------------------------------------------------------
# Medição de tempo (6)
# ---------------------------------------------------------------------------
BASELINE_SECONDS_PER_TRACK = (60.0, 90.0)  # processo manual medido com o DJ SK
TYPICAL_SET_SIZE = 100

# ---------------------------------------------------------------------------
# Exportação (4.9)
# ---------------------------------------------------------------------------
M3U_ABSOLUTE_PATHS = True          # o Rekordbox resolve melhor caminhos absolutos
CSV_DELIMITER = ";"                # Excel em português abre ';' direto
CSV_ENCODING = "utf-8-sig"         # BOM para o Excel reconhecer acentos
WRITE_ID3_TAGS = True              # grava título, artista, BPM e tom nos mp3 exportados

# ---------------------------------------------------------------------------
# Opcionais: Spotify (4.10) e Gemini (4.11)
# ---------------------------------------------------------------------------
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:8888/callback"  # o Spotify não aceita mais "localhost"
SPOTIFY_SCOPES = "playlist-read-private playlist-read-collaborative playlist-modify-private playlist-modify-public user-read-private"
SPOTIFY_DURATION_TOLERANCE_S = 4.0
GEMINI_MODEL = "gemini-flash-latest"  # o 2.5-flash não aceita mais chaves novas
# Tentados em ordem quando o principal está sobrecarregado (503), sem cota (429) ou sumiu (404).
GEMINI_FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-flash-lite-latest"]


def get_secret(name: str) -> str | None:
    """Lê um segredo do ambiente (.env). Devolve None se ausente ou vazio."""
    value = os.getenv(name, "").strip()
    return value or None


def ensure_dirs() -> None:
    for d in (CACHE_DIR, STEMS_DIR, EXPORT_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)
