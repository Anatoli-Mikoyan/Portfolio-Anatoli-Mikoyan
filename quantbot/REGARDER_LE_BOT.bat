@echo off
REM ======================================================================
REM  Regarder le bot trader, tout de suite.
REM  Double-cliquez sur ce fichier.
REM
REM  Pas besoin de MetaTrader, pas besoin que le marche soit ouvert.
REM  Onze mois de marche defilent en deux minutes, decision par decision,
REM  sur des prix reels que le modele n'avait jamais vus.
REM ======================================================================
setlocal
cd /d "%~dp0"
title QBot - regarder le bot trader

where python >nul 2>&1
if errorlevel 1 (
  echo [X] Python n'est pas installe, ou pas dans le PATH.
  pause
  exit /b 1
)

if not exist "runs\start\modele\agent.pt" (
  echo.
  echo [X] Aucun modele entraine.
  echo     Lancez DEMARRER.bat une fois, puis revenez ici.
  echo.
  pause
  exit /b 1
)

python scripts\regarder.py %*

echo.
pause
