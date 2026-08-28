# ======================================================================================
#  QBot — installation et premier lancement, en une seule commande.
#
#      irm https://raw.githubusercontent.com/Anatoli-Mikoyan/Portfolio-Anatoli-Mikoyan/refs/heads/claude/algorithmic-trading-bot-dql-hjyjok/quantbot/install.ps1 | iex
#
#  Ce script installe Python si besoin, télécharge le projet, installe ses
#  dépendances, lance l'analyse complète et ouvre le rapport dans le navigateur.
#  Il n'écrit que dans %USERPROFILE%\QBot et ne touche à rien d'autre.
# ======================================================================================
$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Repo    = "Anatoli-Mikoyan/Portfolio-Anatoli-Mikoyan"
$Branche = "claude/algorithmic-trading-bot-dql-hjyjok"
$Racine  = Join-Path $env:USERPROFILE "QBot"

function Titre($t) { Write-Host "`n$t" -ForegroundColor Cyan }
function Ok($t)    { Write-Host "  [ok] $t" -ForegroundColor Green }
function Info($t)  { Write-Host "  $t" -ForegroundColor Gray }
function Souci($t) { Write-Host "  [!] $t" -ForegroundColor Yellow }

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  QBOT - LABORATOIRE DE TRADING QUANTITATIF" -ForegroundColor Cyan
Write-Host "  Installation et premier lancement" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# ------------------------------------------------------------------ 1. Python
Titre "[1/5] Python"

function Trouver-Python {
    foreach ($c in @("python", "python3", "py")) {
        try {
            $v = & $c --version 2>&1
            if ($LASTEXITCODE -eq 0 -and "$v" -match "Python 3\.(\d+)") {
                if ([int]$Matches[1] -ge 9) { return $c }
            }
        } catch { }
    }
    return $null
}

$Py = Trouver-Python
if (-not $Py) {
    Souci "Python 3.9+ introuvable. Installation via winget..."
    try {
        winget install --id Python.Python.3.12 --source winget `
                       --accept-package-agreements --accept-source-agreements | Out-Null
        # winget ne rafraîchit pas le PATH de la session courante : on le recharge
        # à la main, sinon la commande python reste introuvable jusqu'au redémarrage.
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [Environment]::GetEnvironmentVariable("Path", "User")
        $Py = Trouver-Python
    } catch {
        Souci "L'installation automatique a echoue."
    }
}
if (-not $Py) {
    Write-Host ""
    Write-Host "  Installez Python manuellement : https://www.python.org/downloads/" -ForegroundColor Red
    Write-Host "  IMPORTANT : cochez 'Add Python to PATH' pendant l'installation," -ForegroundColor Red
    Write-Host "  puis relancez cette commande." -ForegroundColor Red
    Write-Host ""
    return
}
Ok ((& $Py --version 2>&1) -replace "`n", "")

# ------------------------------------------------------------------ 2. Téléchargement
Titre "[2/5] Telechargement du projet"

$Zip = Join-Path $env:TEMP "qbot-$(Get-Random).zip"
$Tmp = Join-Path $env:TEMP "qbot-extract-$(Get-Random)"
$Url = "https://codeload.github.com/$Repo/zip/refs/heads/$Branche"

Info "Depuis github.com/$Repo"
Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
Ok ("{0:N1} Mo telecharges" -f ((Get-Item $Zip).Length / 1MB))

Expand-Archive -Path $Zip -DestinationPath $Tmp -Force
# GitHub nomme le dossier d'apres la branche, en remplacant les '/' par des '-'.
# On le retrouve plutot que de le supposer : le nom changerait avec la branche.
$Source = Get-ChildItem -Path $Tmp -Directory | Select-Object -First 1
$Source = Join-Path $Source.FullName "quantbot"
if (-not (Test-Path $Source)) { throw "Dossier 'quantbot' introuvable dans l'archive." }

if (Test-Path $Racine) { Remove-Item $Racine -Recurse -Force }
Move-Item $Source $Racine
Remove-Item $Zip -Force
Remove-Item $Tmp -Recurse -Force
Ok "Installe dans $Racine"

# ------------------------------------------------------------------ 3. Dépendances
Titre "[3/5] Bibliotheques Python"
Info "Quelques minutes la premiere fois (PyTorch est volumineux)."
Push-Location $Racine

# --no-warn-script-location : pip signale sinon que pip.exe n'est pas dans le PATH.
# L'avertissement est sans objet ici, on appelle toujours "python -m pip".
$Silence = @("--quiet", "--no-warn-script-location")
& $Py -m pip install @Silence --upgrade pip

# Les dependances sont installees en DEUX temps. Le noyau doit reussir ; hmmlearn est
# une extension C dont les paquets precompiles arrivent tard apres chaque nouvelle
# version de Python, et il ne sert qu'au detecteur de regime par HMM. Le lier au reste
# ferait echouer toute l'installation pour une brique optionnelle.
$Noyau = @("numpy>=1.24", "pandas>=2.0", "scipy>=1.10", "torch>=2.0",
           "scikit-learn>=1.3", "PyYAML>=6.0", "pytest>=7.4")
& $Py -m pip install @Silence @Noyau
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Souci "L'installation des bibliotheques essentielles a echoue."
    Info "Reessayez a la main :  cd $Racine"
    Info "puis :  $Py -m pip install numpy pandas scipy torch scikit-learn PyYAML pytest"
    Info "Si torch refuse de s'installer, votre version de Python est peut-etre trop"
    Info "recente : les paquets precompiles de PyTorch suivent avec quelques mois."
    return
}

& $Py -m pip install @Silence "hmmlearn>=0.3" 2>$null | Out-Null
$Hmm = ($LASTEXITCODE -eq 0)
if (-not $Hmm) {
    Souci "hmmlearn indisponible pour cette version de Python - ce n'est pas bloquant."
    Info "Seul le detecteur de regime par HMM sera inactif ; les detecteurs par regles"
    Info "et par clustering fonctionnent, et l'analyse complete aussi."
}

# Verification par import : la seule preuve que l'installation a reellement abouti.
$Verif = & $Py -c "import numpy,pandas,scipy,sklearn,torch;print('OK',torch.__version__)" 2>&1
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Souci "Les bibliotheques ne s'importent pas :"
    Write-Host "  $Verif" -ForegroundColor Red
    return
}
Ok "Installees et verifiees ($Verif)"

# ------------------------------------------------------------------ 4. Raccourci
Titre "[4/5] Raccourci"
$Cmd = Join-Path $Racine "qbot.cmd"
@"
@echo off
REM Relance l'analyse QBot. Passez des options : qbot --rapide
cd /d "%~dp0"
$Py scripts\start.py %*
pause
"@ | Set-Content -Path $Cmd -Encoding ASCII
Ok "Pour relancer plus tard : double-cliquez sur $Cmd"

# ------------------------------------------------------------------ 5. Lancement
Titre "[5/5] Analyse complete"
Info "10 a 20 minutes. Chaque etape affiche son verdict au fur et a mesure."
Info "Le rapport s'ouvrira tout seul dans votre navigateur."
Write-Host ""

& $Py scripts\start.py
Pop-Location

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "  Termine." -ForegroundColor Cyan
Write-Host "  Rapport      : $Racine\runs\start\rapport.html" -ForegroundColor Cyan
Write-Host "  Relancer     : $Cmd" -ForegroundColor Cyan
Write-Host "  Vos donnees  : $Cmd --csv chemin\vers\fichier.csv" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
