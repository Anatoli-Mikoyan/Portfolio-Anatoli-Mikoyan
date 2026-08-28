@echo off
REM ======================================================================
REM  QBot <-> MetaTrader 5 : installation du pont et lancement du serveur.
REM  Double-cliquez sur ce fichier.
REM
REM  Ce que ca fait, dans l'ordre :
REM    1. copie l'Expert Advisor dans votre dossier MetaTrader
REM    2. verifie que la chaine complete repond (sans MetaTrader)
REM    3. lance le serveur auquel MetaTrader se connectera
REM
REM  Le trading REEL n'est PAS arme par ce fichier. Deux verrous restent
REM  fermes : InpDryRun=true dans l'EA, et pas de --reel ici.
REM ======================================================================
setlocal
cd /d "%~dp0"
title QBot - pont MetaTrader 5

echo.
echo ======================================================================
echo   QBOT ^<-^> METATRADER 5
echo ======================================================================

where python >nul 2>&1
if errorlevel 1 (
  echo [X] Python n'est pas installe, ou pas dans le PATH.
  echo     Lancez d'abord DEMARRER.bat.
  pause
  exit /b 1
)

if not exist "runs\start\modele\agent.pt" (
  echo.
  echo [X] Aucun modele entraine.
  echo.
  echo     Le serveur n'a rien a servir tant que le modele n'existe pas.
  echo     Lancez DEMARRER.bat une fois ^(10 a 20 minutes^), puis revenez ici.
  echo.
  pause
  exit /b 1
)

echo.
echo --- 1/3 : copie de l'Expert Advisor dans MetaTrader ------------------
python scripts\mt5.py installer
if errorlevel 1 (
  echo.
  echo     La copie automatique a echoue - lisez le message ci-dessus.
  echo     Vous pouvez copier mql5\QBotBridge.mq5 a la main dans
  echo     MQL5\Experts de votre dossier de donnees MetaTrader.
  echo.
  pause
)

echo.
echo --- 2/3 : verification de la chaine ^(sans MetaTrader^) ----------------
python scripts\mt5.py tester
if errorlevel 1 (
  echo.
  echo [X] La chaine ne repond pas. Inutile d'aller plus loin :
  echo     le probleme est cote Python, pas cote MetaTrader.
  echo.
  pause
  exit /b 1
)

echo.
echo --- 3/3 : demarrage du serveur ---------------------------------------
echo.
echo     LAISSEZ CETTE FENETRE OUVERTE.
echo     Allez dans MetaTrader et glissez QBotBridge sur un graphique.
echo     Chaque bougie, la decision s'affichera ici.
echo.
python scripts\mt5.py demarrer

echo.
echo ======================================================================
echo   Serveur arrete. MetaTrader ne recevra plus de reponse et restera
echo   a plat : l'EA ne prend aucune decision par lui-meme.
echo ======================================================================
pause
