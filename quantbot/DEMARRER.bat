@echo off
REM ======================================================================
REM  QBot — installation et premier lancement, sur Windows.
REM  Double-cliquez sur ce fichier. Rien d'autre a faire.
REM ======================================================================
setlocal
cd /d "%~dp0"
title QBot - installation et lancement

echo.
echo ======================================================================
echo   QBOT - INSTALLATION ET PREMIER LANCEMENT
echo ======================================================================
echo.

REM --- Python present ? -------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
  echo [X] Python n'est pas installe, ou pas dans le PATH.
  echo.
  echo     Installez-le depuis https://www.python.org/downloads/
  echo     IMPORTANT : cochez "Add Python to PATH" pendant l'installation.
  echo.
  pause
  exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [1/3] Python %PYVER% detecte.

REM --- Dependances ------------------------------------------------------
echo [2/3] Installation des bibliotheques (quelques minutes la premiere fois)...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
  echo.
  echo [X] L'installation des bibliotheques a echoue.
  echo     Reessayez, ou lancez a la main : python -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)
echo       Termine.

REM --- Lancement --------------------------------------------------------
echo [3/3] Lancement de l'analyse complete.
echo       Comptez 10 a 20 minutes. Le rapport s'ouvrira tout seul.
echo.
python scripts\start.py %*

echo.
echo ======================================================================
echo   Termine. Le rapport HTML se trouve dans runs\start\rapport.html
echo ======================================================================
echo.
pause
