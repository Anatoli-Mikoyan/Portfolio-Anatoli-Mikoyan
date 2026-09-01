@echo off
REM ======================================================================
REM  Rendez-vous hebdomadaire du bot.
REM
REM  A lancer une fois par semaine, PC allume dix minutes :
REM    1. ce fichier demarre le serveur
REM    2. vous ouvrez MetaTrader, l'EA se connecte et decide
REM    3. le serveur vous dit que c'est fait et s'arrete
REM    4. vous eteignez
REM
REM  Pourquoi une fois par semaine plutot qu'en continu : la mesure le dit.
REM  Sur la periode de test, une decision par semaine a donne -0,53 % contre
REM  -11,00 % pour une decision par heure. L'ecart, ce sont les frais.
REM ======================================================================
setlocal
cd /d "%~dp0"
title QBot - rendez-vous hebdomadaire

if not exist "runs\start\modele\agent.pt" (
  echo [X] Aucun modele entraine. Lancez DEMARRER.bat une fois.
  pause
  exit /b 1
)

echo.
echo ======================================================================
echo   RENDEZ-VOUS HEBDOMADAIRE
echo ======================================================================
echo.
echo   Le serveur va demarrer et attendre UNE decision, puis s'arreter.
echo   Ouvrez MetaTrader des que vous voyez "En attente de 1 decision".
echo.

python scripts\mt5.py demarrer --ordres --stop-apres 1

echo.
echo   Pour voir ou en est le compte :  python scripts\mt5.py bilan
echo.
pause
