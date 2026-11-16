@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title Painel DJ SK

if not exist ".venv\Scripts\python.exe" goto :no_venv
if not exist "app.py" goto :no_app

echo Abrindo o painel do DJ SK no navegador (endereço http://localhost:8501)...
echo Para encerrar, feche esta janela ou aperte Ctrl+C.
echo Se o Streamlit pedir um e-mail na primeira vez, é só apertar Enter.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py %*
if errorlevel 1 goto :error
exit /b 0

:error
echo.
echo [ERRO] O painel parou com erro (veja a mensagem acima).
echo        Porta ocupada? Rode:  iniciar.bat --server.port 8502
pause
exit /b 1

:no_venv
echo [ERRO] Ambiente .venv não encontrado. Rode primeiro o instalar.bat.
pause
exit /b 1

:no_app
echo [ERRO] app.py não encontrado em "%~dp0".
pause
exit /b 1
