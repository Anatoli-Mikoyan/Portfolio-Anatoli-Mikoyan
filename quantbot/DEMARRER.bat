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
set SILENCE=--quiet --no-warn-script-location
python -m pip install %SILENCE% --upgrade pip
REM Le noyau doit reussir ; hmmlearn est optionnel (extension C, paquets precompiles
REM tardifs apres chaque version de Python, utile au seul detecteur de regime par HMM).
python -m pip install %SILENCE% numpy pandas scipy torch scikit-learn PyYAML pytest
if errorlevel 1 (
  echo.
  echo [X] L'installation des bibliotheques essentielles a echoue.
  echo     Si torch refuse de s'installer, votre Python est peut-etre trop recent :
  echo     les paquets precompiles de PyTorch suivent avec quelques mois.
  echo.
  pause
  exit /b 1
)
python -m pip install %SILENCE% hmmlearn >nul 2>&1
if errorlevel 1 echo       [!] hmmlearn indisponible - non bloquant.
REM Verification deleguee a verifier.py : il isole chaque import et nomme le
REM paquet fautif, la ou un import groupe s'arrete au premier echec sans dire lequel.
python scripts\verifier.py
if errorlevel 1 (
  echo.
  echo [X] Installation incomplete - le detail est ci-dessus.
  pause
  exit /b 1
)
echo       Termine et verifie.

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
