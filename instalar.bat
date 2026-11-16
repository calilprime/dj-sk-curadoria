@echo off
chcp 65001 >nul
setlocal EnableExtensions
cd /d "%~dp0"
title Painel DJ SK - Instalação

echo.
echo ================================================
echo   Painel de Curadoria do DJ SK - instalação
echo ================================================
echo.

set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" goto :venv_exists

rem Procura um Python 3.11 ou mais novo: primeiro o launcher "py -3", depois "python".
set "PYCMD="
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "PYCMD=py -3"
if defined PYCMD goto :create_venv
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "PYCMD=python"
if defined PYCMD goto :create_venv
echo [ERRO] Não encontrei o Python 3.11 ou mais novo.
echo        Instale pelo site https://www.python.org/downloads/ marcando a opção
echo        "Add python.exe to PATH" e rode este arquivo de novo.
goto :fail

:create_venv
echo Criando o ambiente virtual .venv com "%PYCMD%"...
%PYCMD% -m venv .venv
if errorlevel 1 goto :fail_venv
goto :check_version

:venv_exists
echo O ambiente virtual .venv já existe: não vou recriar.

:check_version
"%VPY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 goto :old_venv

echo.
echo Atualizando o pip...
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 echo [AVISO] Não consegui atualizar o pip, seguindo com a versão atual.

echo.
echo Instalando as dependências do requirements.txt (pode levar alguns minutos)...
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :fail_pip

echo.
if exist ".env" goto :env_exists
if not exist ".env.example" goto :env_done
copy ".env.example" ".env" >nul
echo Criei o arquivo .env a partir do .env.example.
echo Todas as chaves são opcionais: preencha só as que for usar (veja o README.md).
goto :env_done
:env_exists
echo O arquivo .env já existe: mantido como está.
:env_done

echo.
set "FFMPEG_MISSING="
where ffmpeg >nul 2>&1
if errorlevel 1 goto :no_ffmpeg
echo ffmpeg encontrado no PATH.
goto :done

:no_ffmpeg
set "FFMPEG_MISSING=1"
echo [AVISO] ffmpeg não encontrado no PATH. Sem ele o painel não consegue baixar e converter as músicas.
echo         Instale com:  winget install Gyan.FFmpeg
echo         Depois feche e abra o terminal (ou reinicie o Windows) e confira com:  ffmpeg -version

:done
echo.
if defined FFMPEG_MISSING echo Instalação concluída, mas falta o ffmpeg: veja o aviso acima.
if not defined FFMPEG_MISSING echo Instalação concluída.
echo Para abrir o painel, dê dois cliques em iniciar.bat.
echo.
pause
exit /b 0

:old_venv
echo [ERRO] A pasta .venv foi criada com um Python anterior ao 3.11.
echo        Apague a pasta .venv, instale o Python 3.11 ou mais novo e rode este arquivo de novo.
goto :fail

:fail_venv
echo [ERRO] Não consegui criar o ambiente virtual .venv (veja a mensagem acima).
goto :fail

:fail_pip
echo [ERRO] A instalação das dependências falhou (veja a mensagem acima).
echo        Se aparecer CERTIFICATE_VERIFY_FAILED, o antivírus ou um proxy está interceptando o HTTPS:
echo        versões recentes do pip usam os certificados do Windows; rode este arquivo de novo.
goto :fail

:fail
echo.
pause
exit /b 1
