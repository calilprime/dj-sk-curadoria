# Painel de Curadoria do DJ SK

Painel web **local** (Streamlit) que automatiza a curadoria e o download de músicas de funk mandelão para o DJ SK, que monta os sets no Rekordbox. A partir de 1 a 3 faixas de referência, o painel sugere músicas parecidas, baixa o áudio completo, mede BPM, energia e tom, deixa o DJ ouvir e aprovar cada uma, ordena o set e exporta uma pasta numerada com playlist `.m3u8` pronta para o Rekordbox.

> **Definição de valor:** transformar o trabalho repetitivo em eficiência, devolvendo às pessoas o seu recurso mais valioso: o tempo.

**Como medir:** hoje, aprovar e baixar uma música leva de 1 a 1,5 minuto (medido com o DJ SK), e uma playlist costuma ter mais de 100 músicas (100 a 150 minutos por set). O painel registra o tempo gasto por música e por set e mostra a comparação com essa linha de base.

## Sumário

- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração (.env)](#configuração-env)
- [Como rodar](#como-rodar)
- [Como usar](#como-usar)
- [Orçamento](#orçamento)
- [Medição de tempo](#medição-de-tempo)
- [Estrutura de pastas](#estrutura-de-pastas)
- [Status das validações](#status-das-validações)
- [Rodar os testes](#rodar-os-testes)
- [Solução de problemas](#solução-de-problemas)
- [Segurança](#segurança)

## Requisitos

| O quê | Detalhe |
|---|---|
| Windows 10 ou 11 | O painel roda no computador de quem usa; não é um site. |
| Python 3.11 ou mais novo | Testado com Python 3.14. Baixe em [python.org](https://www.python.org/downloads/) e marque **Add python.exe to PATH** na instalação. |
| ffmpeg no PATH | Converte o áudio baixado para mp3. Instale com `winget install Gyan.FFmpeg` e confira com `ffmpeg -version` num terminal novo. |
| Internet | Para as sugestões (YouTube Music) e os downloads. A análise e a exportação funcionam sem internet com arquivos de uma pasta local. |
| Rekordbox | Do lado do DJ, para importar o set exportado. |

## Instalação

### Jeito fácil: `instalar.bat`

Dê dois cliques em `instalar.bat` na pasta do projeto. Ele:

1. cria o ambiente virtual `.venv` com `py -3` (ou `python`) se ainda não existir, conferindo se o Python é 3.11 ou mais novo;
2. atualiza o `pip` e instala tudo do `requirements.txt`;
3. copia `.env.example` para `.env`, se o `.env` ainda não existir (um `.env` já preenchido nunca é sobrescrito);
4. confere se o `ffmpeg` está no PATH e avisa como instalar se não estiver.

Pode rodar de novo quando quiser: ele reaproveita o que já existe.

### Jeito manual (terminal `cmd`)

```bat
cd /d "C:\caminho\para\dj_sk_curadoria"
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
ffmpeg -version
```

O `requirements-lock.txt` guarda as versões exatas testadas em 22/10/2026. Use-o no lugar do `requirements.txt` se uma versão nova de alguma biblioteca quebrar algo. Exceção: o `yt-dlp` precisa estar sempre atualizado (veja [Solução de problemas](#solução-de-problemas)).

## Configuração (.env)

As chaves ficam **só** no arquivo `.env`, que nunca vai para o repositório. Copie o `.env.example` para `.env` (o `instalar.bat` faz isso) e preencha apenas o que for usar. **Nenhuma chave é obrigatória**: referências, sugestões, download, análise, triagem, ordenação e exportação funcionam sem nenhuma.

| Chave | Para quê | Sem a chave |
|---|---|---|
| `REPLICATE_API_TOKEN` | Estúdio de Vocal: separa a voz e os instrumentos (função paga, veja [Orçamento](#orçamento)). Crie em [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens). | O botão **🎙️ Extrair Acapella** fica desabilitado. |
| `YOUTUBE_API_KEY` | Plano B das sugestões (YouTube Data API v3), usado quando o rádio do YouTube Music traz poucas faixas. | O painel usa só o rádio do YouTube Music. |
| `SPOTIFY_CLIENT_ID` e `SPOTIFY_CLIENT_SECRET` | Importar playlists do Spotify do DJ SK como referências. No painel do app do Spotify, cadastre o redirect `http://127.0.0.1:8888/callback`. | O botão de importar do Spotify não aparece. |
| `GEMINI_API_KEY` | Uma frase curta explicando cada transição do set (Gemini, camada gratuita). | O recurso fica escondido. |

Os ajustes que não são segredo (faixa de BPM, pesos da ordenação, teto de orçamento, câmbio, pastas) ficam no `config.py`.

## Como rodar

- **Dois cliques em `iniciar.bat`**, ou, num terminal na pasta do projeto:

  ```bat
  .venv\Scripts\python.exe -m streamlit run app.py
  ```

- O navegador abre em `http://localhost:8501`. Na primeira vez o Streamlit pode pedir um e-mail no terminal: é só apertar **Enter**.
- Para encerrar, feche a janela do terminal ou aperte `Ctrl+C` nela.
- Opções extras do Streamlit passam pelo `iniciar.bat`, por exemplo `iniciar.bat --server.port 8502`.

## Como usar

O fluxo do DJ SK no painel:

1. **Referências.** Digite de 1 a 3 faixas no formato `Artista - Faixa` (também aceita link do YouTube ou YouTube Music) e clique em **Buscar sugestões**. Também dá para importar uma playlist do Spotify (opcional) ou usar uma pasta com músicas que já estão no computador.
2. **Triagem.** Aparecem 20 sugestões por página, ordenadas por quantas referências sugeriram a mesma faixa. O áudio completo de cada uma é baixado em segundo plano (até 4 ao mesmo tempo), com barra de progresso. Cada card mostra título, artista, duração, BPM, curva de energia, tamanho da intro e da saída, forma de onda e um player com os botões **Início**, **Meio** e **Final** (últimos 30 s). Se o BPM estiver errado, corrija no campo do card. Depois clique em **✅ Aprovar** ou **❌ Reprovar**.
3. **Setlist.** Ajuste os pesos (BPM, energia, tom, voz) e clique em **Gerar ordem**. O painel mostra a ordem, o custo e a classificação de cada transição (suave, boa, atenção, difícil) e um gráfico de BPM ao longo do set. A primeira faixa pode ser escolhida; por padrão é a de menor energia. A ordem é calculada por fórmula, sem IA.
4. **Estúdio de Vocal (opcional, pago).** Em cada faixa aprovada, **🎙️ Extrair Acapella** separa voz, bateria, baixo e outros em `stems/{videoId}/`. Com os stems, o painel marca onde o MC canta e classifica cada transição como **limpa**, **voz de um lado** ou **choque de voz**, e isso entra na ordenação.
5. **Exportar.** Cria a pasta `Setlist_Atual/` com cópias numeradas `NN - Artista - Título.mp3` (com título, artista, BPM e tom gravados nas tags), a playlist `Setlist_Atual.m3u8` e a planilha `Setlist_Atual.csv` (abre direto no Excel, com ordem, BPM, energia e classificação das transições).
6. **Rekordbox.** Importe o `Setlist_Atual.m3u8` (menu Arquivo > Importar > Importar playlist; em inglês, File > Import > Import Playlist) ou arraste a pasta `Setlist_Atual/` inteira (plano B: a numeração mantém a ordem).

No fim da sessão, as faixas reprovadas saem do cache. A curadoria em andamento fica salva em `data/sessao_atual.json`, então recarregar o navegador não perde as aprovações.

## Orçamento

- **Capital total do projeto: R$ 10,00.** Crédito comprado na Replicate: US$ 2,00.
- **O único gasto em dinheiro é o Estúdio de Vocal**: o modelo `cjwbw/demucs` na Replicate roda numa GPU Nvidia T4 a **US$ 0,000225 por segundo**, cerca de **US$ 0,016 por separação** (uns R$ 0,09 com câmbio de R$ 5,40). Dá para umas 115 separações antes de chegar ao teto.
- O custo real de cada separação é o tempo de GPU informado pela Replicate (`predict_time`) vezes a taxa por segundo, convertido para reais com o câmbio `USD_BRL` do `config.py` (ajuste no dia).
- Cada execução fica registrada em `data/orcamento.json`, e o topo do painel mostra `R$ gasto / R$ 10,00`.
- **Trava:** o botão do Estúdio de Vocal bloqueia quando o gasto mais a estimativa da próxima separação atingiria o teto, quando passaria do crédito da Replicate, ou quando a Replicate avisa que o saldo acabou.
- Plano B sem custo: Demucs local (`pip install demucs`, roda na CPU e é bem mais lento; não vem no `requirements.txt`).
- Todo o resto tem custo zero: YouTube Music, yt-dlp, librosa, Spotify e a camada gratuita do Gemini.

## Medição de tempo

Para provar o valor, o painel registra por sessão:

- **início**: o clique em **Buscar sugestões**;
- **fim**: a exportação do set;
- **faixas aprovadas** e **faixas revisadas**;
- **tempo médio por faixa aprovada** = (fim − início) ÷ aprovadas.

O resultado aparece ao lado da linha de base do processo manual (60 a 90 s por música) e é projetado para um set de 100 músicas. Exemplo: se 20 faixas aprovadas levaram 10 minutos, são 30 s por faixa, ou 50 minutos para 100 músicas contra 100 a 150 minutos à mão. O histórico fica em `data/metricas.json`.

## Estrutura de pastas

```
dj_sk_curadoria/
├── app.py                  # interface Streamlit
├── config.py               # pesos, faixa de BPM, taxas, teto de orçamento, pastas
├── requirements.txt        # dependências (requirements-lock.txt: versões exatas testadas)
├── .env.example            # modelo do .env (chaves opcionais, sem valores)
├── instalar.bat            # instalação no Windows
├── iniciar.bat             # abre o painel
├── modules/
│   ├── models.py           # estruturas compartilhadas (Track, Analysis, Transition...)
│   ├── util.py             # SSL, nomes de arquivo, JSON atômico, formatação
│   ├── suggest.py          # referências → sugestões (rádio do YouTube Music, plano B)
│   ├── download.py         # yt-dlp em cache e pasta local
│   ├── analyze.py          # librosa: BPM, energia, intro/saída, tom e Camelot
│   ├── jobs.py             # fila em segundo plano: download e análise
│   ├── order.py            # ordenação do set (vizinho mais próximo + 2-opt)
│   ├── stems.py            # Demucs via Replicate (e plano B local)
│   ├── mixwindows.py       # janelas de mixagem a partir do stem de voz
│   ├── budget.py           # medidor e teto de gasto
│   ├── metrics.py          # medição de tempo
│   ├── state.py            # sessão salva em disco
│   ├── viz.py              # gráficos (energia, forma de onda, BPM do set)
│   ├── export.py           # pasta numerada, .m3u8, CSV e tags
│   ├── spotify_import.py   # importar playlist do Spotify (opcional)
│   └── explain.py          # explicação das transições com Gemini (opcional)
├── scripts/
│   └── validar.py          # validações iniciais (seção 9 da especificação)
├── tests/                  # pytest
├── cache/                  # áudios baixados ({videoId}.mp3) e análises em JSON
├── stems/                  # saída da separação (stems/{videoId}/)
├── Setlist_Atual/          # exportação final
└── data/                   # orçamento, métricas, sessão salva, token do Spotify
```

As pastas `cache/`, `stems/`, `Setlist_Atual/` e `data/` são criadas pelo painel e ficam fora do repositório.

## Status das validações

Validações iniciais da especificação (seção 9), com os resultados de **22/10/2026**:

| # | Validação | Status | Resultado |
|---|---|---|---|
| 1 | `ytmusicapi`: rádio de uma faixa de mandelão, sem login | ✅ OK | O rádio sem login devolveu 50 faixas por referência, de funk, em cerca de 0,6 s. A 1ª faixa do rádio é a própria referência (o painel a remove). |
| 2 | `yt-dlp`: 5 e depois 20 faixas em paralelo | ✅ OK | 5 faixas em 24 s com 4 em paralelo (cerca de 6 MB cada, mp3 320). No painel, as 20 sugestões ficaram baixadas **e analisadas** em cerca de 30 s; num teste por script, 8 faixas levaram 26 s (download + análise). |
| 3 | `librosa` x Rekordbox: BPM e energia de 5 faixas | ⏳ Falta comparar com o Rekordbox | O `beat_track` puro deu 129,2 em 3 de 5 faixas, mas 80,7 e 107,7 nas outras duas. O estimador do painel escolhe o candidato mais forte dentro da faixa de BPM do `config.py` (120 a 160). Com ele, 4 das 5 faixas deram 130,0. A quinta (`bN8S3cRF9X0`) deu 160,0, porque a periodicidade dessa faixa está em 80/160 e não em 130: pode ser um funk mais acelerado, e se não for, o DJ corrige no card. Em 8 sugestões reais, os valores ficaram entre 123 e 133 (seis em 130,0). |
| 4 | Rekordbox: importar o `.m3u8` | ⏳ Pendente | Plano B: arrastar a pasta numerada. |
| 5 | Replicate: parâmetros do `cjwbw/demucs` | ✅ OK | Lidos na aba API: entrada `audio`, `model_name=htdemucs`, `output_format=mp3`; saída `vocals`, `drums`, `bass`, `other`. Roda numa Nvidia T4. |
| 6 | Conversa com o DJ SK (como ordena os sets, faixa de BPM típica) | ⏳ Pendente | Perguntas no fim de `scripts/validar.py tudo`. |

### Refazer as validações (`scripts/validar.py`)

Nenhum comando gasta dinheiro (o `replicate` só lê o esquema do modelo). Rode na pasta do projeto:

```bat
rem 1. Rádio sem login: 20 primeiras sugestões e % com cara de funk (mc, dj, funk, mandel, montagem, automotivo, beat)
.venv\Scripts\python.exe scripts\validar.py ytmusic "MC Fulano - Faixa" "MC Ciclano - Outra Faixa"

rem 2. Baixa 5 e depois 20 sugestões (4 em paralelo) e mede o tempo total e por faixa
.venv\Scripts\python.exe scripts\validar.py ytdlp --saida cache

rem 3. Analisa os áudios de uma pasta e salva data\validacao_librosa.csv
rem    (colunas bpm_rekordbox e tom_rekordbox vazias para preencher na comparação)
.venv\Scripts\python.exe scripts\validar.py librosa cache

rem 4. Gera data\Setlist_Teste\ (cópias numeradas + Setlist_Teste.m3u8) para importar no Rekordbox
.venv\Scripts\python.exe scripts\validar.py m3u8 "C:\pasta\com\mp3"

rem 5. Com REPLICATE_API_TOKEN no .env, lê o esquema do modelo pela API; sem token, explica como conferir na aba API
.venv\Scripts\python.exe scripts\validar.py replicate

rem Tudo em sequência (1 a 5) e o checklist da conversa com o DJ SK
.venv\Scripts\python.exe scripts\validar.py tudo
```

Use `--help` em qualquer comando para ver as opções (por exemplo `--lotes 5 20`, `--paralelo 4`, `--ids`, `--pular`). As faixas que já estão no cache não entram na medição do `ytdlp`.

## Rodar os testes

```bat
.venv\Scripts\python.exe -m pytest
```

- Por padrão os testes **não** acessam a internet. Os de rede ficam marcados e rodam com `.venv\Scripts\python.exe -m pytest -m network`.
- Os testes com áudio real só rodam se a variável `DJSK_AUDIO_FIXTURES` apontar para uma pasta com mp3 (fora do repositório, por direitos autorais):

  ```bat
  set DJSK_AUDIO_FIXTURES=C:\pasta\com\mp3
  .venv\Scripts\python.exe -m pytest
  ```

## Solução de problemas

| Sintoma | O que fazer |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` | Antivírus ou proxy interceptando o HTTPS com certificado próprio. O projeto já usa o `truststore` (certificados do Windows) em todas as chamadas; confira se ele está instalado (`.venv\Scripts\python.exe -m pip show truststore`). Se o erro aparecer no `pip`, atualize o `pip`, porque as versões recentes também usam os certificados do Windows. |
| Download falha (`HTTP Error 403`, "Sign in to confirm you're not a bot", formato indisponível) | O YouTube muda com frequência e o `yt-dlp` precisa acompanhar: `.venv\Scripts\python.exe -m pip install -U yt-dlp`. Se continuar, espere um pouco e tente de novo. |
| "ffmpeg não encontrado" | `winget install Gyan.FFmpeg`, feche e abra o terminal (ou reinicie o Windows) e confira com `ffmpeg -version`. |
| Porta ocupada / painel não abre | Outro painel pode estar aberto: feche a outra janela ou rode `iniciar.bat --server.port 8502` e abra `http://localhost:8502`. |
| Spotify: `INVALID_CLIENT: Invalid redirect URI` | No painel do app do Spotify, cadastre exatamente `http://127.0.0.1:8888/callback` (o Spotify não aceita mais `localhost`). A porta 8888 precisa estar livre durante o login. Em modo de desenvolvimento, o DJ SK precisa estar cadastrado em *User Management* (até 5 usuários), e só dá para ler playlists que ele criou ou colabora. |
| "Python não encontrado" ou abre a Microsoft Store | Instale o Python pelo [python.org](https://www.python.org/downloads/) com **Add python.exe to PATH**. Se ainda abrir a Store, desligue os atalhos `python.exe` em Configurações > Aplicativos > Configurações avançadas de aplicativos > Aliases de execução de aplicativo. |
| BPM pela metade ou em dobro | Corrija no campo do card (a correção fica salva). Se acontecer sempre, ajuste `BPM_MIN` e `BPM_MAX` no `config.py`. |
| Estúdio de Vocal travado | O teto de R$ 10,00 ou o crédito da Replicate acabou (veja o medidor no topo). Adicione crédito na Replicate ou use o plano B local. |

## Segurança

- **Nenhum segredo no código**, nos testes ou nos logs. As chaves ficam só no `.env`, que está no `.gitignore`; o `.env.example` não tem valores.
- O token do Spotify fica em `data/` e os áudios em `cache/`, `stems/` e `Setlist_Atual/`, todos fora do repositório.
- **Não hospede o painel publicamente** (nem no Streamlit Cloud): o download precisa rodar na máquina local, o painel não tem login e usa as suas chaves e o seu orçamento da Replicate. Por padrão o Streamlit também aceita conexões de outros aparelhos da mesma rede; para liberar só este computador, rode `iniciar.bat --server.address 127.0.0.1`.
- Os áudios baixados são para uso pessoal do DJ na curadoria; não redistribua.
