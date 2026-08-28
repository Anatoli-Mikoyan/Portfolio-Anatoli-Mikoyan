//+------------------------------------------------------------------+
//|                                                 QBotBridge.mq5   |
//|   Pont MetaTrader 5 <-> moteur d'inférence Python (qbot)          |
//|                                                                  |
//|   L'EA ne contient AUCUNE logique de décision : il collecte les   |
//|   barres, les envoie au modèle, reçoit une exposition cible et    |
//|   l'exécute. Toute l'intelligence vit côté Python, où elle est    |
//|   backtestée et validée. Cette séparation garantit que ce qui     |
//|   trade en réel est exactement ce qui a été mesuré en backtest.   |
//|                                                                  |
//|   PRÉREQUIS : Outils > Options > Expert Advisors > cocher         |
//|   « Autoriser WebRequest / connexions » et AJOUTER l'adresse      |
//|   du serveur (ex. 127.0.0.1) à la liste autorisée, sinon          |
//|   SocketConnect() échouera avec l'erreur 5273.                    |
//|                                                                  |
//|   Aucune DLL n'est requise (sockets natifs, build >= 2085).       |
//+------------------------------------------------------------------+
#property copyright "qbot"
#property version   "1.00"
#property strict
#property description "Pont d'exécution pour agent RL distributionnel entraîné en Python"

#include <Trade/Trade.mqh>
#include <Trade/PositionInfo.mqh>

//====================================================================
//  PARAMÈTRES
//====================================================================
input group "=== Connexion ==="
input string  InpHost              = "127.0.0.1";  // Adresse du serveur d'inférence
input int     InpPort              = 8912;         // Port
input int     InpTimeoutMs         = 3000;         // Délai max d'une requête (ms)
input int     InpHistoryBars       = 1200;         // Barres envoyées (>= min_bars du serveur, cf. type "info")

input group "=== Exécution ==="
input long    InpMagic             = 770011;       // Magic number
input double  InpMaxExposure       = 1.0;          // Exposition maximale (fraction du capital)
input double  InpMinRebalance      = 0.05;         // Bande morte : rebalancement minimal
input int     InpSlippagePoints    = 20;           // Déviation max autorisée (points)
input bool    InpUseServerSLTP     = true;         // Utiliser le SL/TP calculé par le modèle
input bool    InpCloseOnDisconnect = false;        // Fermer les positions si le serveur tombe

input group "=== Garde-fous locaux (indépendants du serveur) ==="
input double  InpMaxSpreadPoints   = 30;           // Spread max toléré (points)
input double  InpMaxDailyLossPct   = 3.0;          // Perte journalière max (%)
input double  InpMaxDrawdownPct    = 20.0;         // Drawdown max sur le compte (%)
input double  InpMinEquity         = 0.0;          // Équité plancher (0 = désactivé)
input bool    InpTradeOnNewBarOnly = true;         // Décider uniquement à la clôture d'une barre

input group "=== Supervision (§17) ==="
input int     InpStatusEveryBars   = 24;           // Interroger la supervision toutes les N barres (0 = jamais)
input bool    InpShowPanel         = true;         // Afficher le panneau de supervision sur le graphique
input bool    InpBlockOnCritical   = false;        // Passer à plat si le serveur signale une alerte critique

input group "=== Journalisation ==="
input bool    InpVerbose           = true;         // Journal détaillé
input bool    InpDryRun            = true;         // true = aucun ordre réellement envoyé

//====================================================================
//  ÉTAT GLOBAL
//====================================================================
CTrade         trade;
CPositionInfo  posInfo;

int      g_socket        = INVALID_HANDLE;
datetime g_lastBarTime   = 0;
datetime g_lastDayStamp  = 0;
double   g_dayStartEquity= 0.0;
double   g_peakEquity    = 0.0;
double   g_entryPrice    = 0.0;
int      g_barsInPos     = 0;
int      g_reconnects    = 0;
int      g_failedReqs    = 0;
int      g_barsSinceStatus = 0;
string   g_lastStatus     = "";
bool     g_serverCritical = false;
bool     g_halted        = false;
string   g_haltReason    = "";

//====================================================================
//  INITIALISATION
//====================================================================
int OnInit()
{
   trade.SetExpertMagicNumber(InpMagic);
   trade.SetDeviationInPoints(InpSlippagePoints);
   trade.SetTypeFillingBySymbol(_Symbol);
   trade.LogLevel(LOG_LEVEL_ERRORS);

   g_peakEquity     = AccountInfoDouble(ACCOUNT_EQUITY);
   g_dayStartEquity = g_peakEquity;
   g_lastDayStamp   = TodayStamp();

   if(InpHistoryBars < 600)
   {
      Print("ERREUR : InpHistoryBars trop faible. Le serveur a besoin de plusieurs centaines "
            "de barres pour que toutes les features (variance ratio, rang de volatilité) soient définies.");
      return(INIT_PARAMETERS_INCORRECT);
   }
   if(InpMaxExposure <= 0.0 || InpMaxExposure > 5.0)
   {
      Print("ERREUR : InpMaxExposure doit être dans ]0, 5].");
      return(INIT_PARAMETERS_INCORRECT);
   }

   if(!ConnectToServer())
      Print("AVERTISSEMENT : connexion initiale impossible. Nouvelles tentatives à chaque barre.");

   PrintFormat("QBotBridge initialisé | %s %s | magic=%d | dry_run=%s",
               _Symbol, EnumToString((ENUM_TIMEFRAMES)_Period), (int)InpMagic,
               (InpDryRun ? "OUI" : "NON - TRADING RÉEL"));
   if(!InpDryRun)
      Print(">>> ATTENTION : InpDryRun=false, cet EA enverra de VRAIS ordres. <<<");

   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   CloseSocket();
   PrintFormat("QBotBridge arrêté (raison=%d) | reconnexions=%d | requêtes échouées=%d",
               reason, g_reconnects, g_failedReqs);
}

//====================================================================
//  SUPERVISION (§17)
//====================================================================
// Le serveur Python calcule dérive des features, écart attendu/réalisé, coûts réels et
// alertes. L'EA n'en refait rien : il interroge, affiche et — si on le lui demande —
// se met à plat. Dupliquer ces calculs côté MQL5 introduirait deux vérités, et c'est
// toujours la mauvaise qu'on croit.
void RequestStatus()
{
   if(InpStatusEveryBars <= 0)
      return;
   g_barsSinceStatus++;
   if(g_barsSinceStatus < InpStatusEveryBars)
      return;
   g_barsSinceStatus = 0;

   string response = "";
   if(!SendAndReceive("{\"type\":\"status\"}", response))
   {
      if(InpVerbose)
         Print("Supervision : pas de réponse du serveur.");
      return;
   }
   if(!JsonGetBool(response, "ok"))
   {
      if(InpVerbose)
         PrintFormat("Supervision indisponible : %s", JsonGetString(response, "error"));
      return;
   }

   string driftStatus  = JsonGetString(response, "drift_status");
   int    driftCrit    = (int)JsonGetNumber(response, "drift_critical");
   string driftWorst   = JsonGetString(response, "drift_worst");
   int    alertCount   = (int)JsonGetNumber(response, "alert_count");
   string alertWorst   = JsonGetString(response, "alert_worst");
   string reconVerdict = JsonGetString(response, "recon_verdict");
   string tcaVerdict   = JsonGetString(response, "tca_verdict");
   double tcaRatio     = JsonGetNumber(response, "tca_ratio");
   bool   journalOk    = JsonGetBool(response, "journal_ok");
   double sharpe       = JsonGetNumber(response, "sharpe_rolling");
   double dd           = JsonGetNumber(response, "drawdown");
   double p99          = JsonGetNumber(response, "p99_latency_ms");
   int    nBars        = (int)JsonGetNumber(response, "n_bars");

   g_serverCritical = (alertWorst == "critical");

   PrintFormat("SUPERVISION | barres=%d Sharpe=%.2f DD=%.2f%% | dérive=%s (%d crit., pire=%s) "
               "| coûts=%.2fx | alertes=%d (%s) | journal=%s",
               nBars, sharpe, dd * 100.0, driftStatus, driftCrit, driftWorst,
               tcaRatio, alertCount, alertWorst, (journalOk ? "intègre" : "COMPROMIS"));

   if(InpShowPanel)
   {
      g_lastStatus = StringFormat(
         "QBot — supervision\n"
         "Barres observées : %d\n"
         "Sharpe glissant  : %.2f\n"
         "Drawdown         : %.2f %%\n"
         "Latence p99      : %.0f ms\n"
         "Dérive features  : %s (%d critiques)\n"
         "Attendu/réalisé  : %s\n"
         "Coûts exécution  : %s (%.2fx)\n"
         "Alertes          : %d — pire : %s\n"
         "Journal d'audit  : %s",
         nBars, sharpe, dd * 100.0, p99, driftStatus, driftCrit,
         reconVerdict, tcaVerdict, tcaRatio, alertCount, alertWorst,
         (journalOk ? "intègre" : "COMPROMIS"));
      Comment(g_lastStatus);
   }

   if(!journalOk)
      Print("ALERTE : la chaîne du journal d'audit est rompue côté serveur. "
            "Une décision a été modifiée ou supprimée après écriture.");
}

//====================================================================
//  BOUCLE PRINCIPALE
//====================================================================
void OnTick()
{
   UpdateAccountState();

   if(g_halted)
      return;

   if(!CheckLocalGuards())
      return;

   datetime barTime = iTime(_Symbol, _Period, 0);
   if(InpTradeOnNewBarOnly && barTime == g_lastBarTime)
      return;
   g_lastBarTime = barTime;

   if(PositionSelectByMagic())
      g_barsInPos++;
   else
   {
      g_barsInPos = 0;
      g_entryPrice = 0.0;
   }

   if(!EnsureConnected())
   {
      g_failedReqs++;
      if(InpCloseOnDisconnect)
      {
         Print("Serveur injoignable -> fermeture de sécurité des positions.");
         ClosePosition();
      }
      return;
   }

   string request = BuildPredictRequest();
   if(request == "")
      return;

   string response = "";
   if(!SendAndReceive(request, response))
   {
      g_failedReqs++;
      CloseSocket();                       // force une reconnexion propre au prochain tour
      if(InpCloseOnDisconnect)
         ClosePosition();
      return;
   }

   ProcessResponse(response);
   RequestStatus();
}

//====================================================================
//  RÉSEAU
//====================================================================
bool ConnectToServer()
{
   CloseSocket();
   g_socket = SocketCreate();
   if(g_socket == INVALID_HANDLE)
   {
      PrintFormat("SocketCreate a échoué (erreur %d). Vérifier que les sockets sont autorisés.",
                  GetLastError());
      return(false);
   }
   if(!SocketConnect(g_socket, InpHost, InpPort, InpTimeoutMs))
   {
      int err = GetLastError();
      PrintFormat("SocketConnect %s:%d a échoué (erreur %d).%s",
                  InpHost, InpPort, err,
                  (err == 5273 ? " -> Ajouter l'adresse dans Outils > Options > Expert Advisors." : ""));
      CloseSocket();
      return(false);
   }
   g_reconnects++;
   if(InpVerbose)
      PrintFormat("Connecté au serveur d'inférence %s:%d", InpHost, InpPort);
   return(true);
}

bool EnsureConnected()
{
   if(g_socket != INVALID_HANDLE && SocketIsConnected(g_socket))
      return(true);
   return(ConnectToServer());
}

void CloseSocket()
{
   if(g_socket != INVALID_HANDLE)
   {
      SocketClose(g_socket);
      g_socket = INVALID_HANDLE;
   }
}

//+------------------------------------------------------------------+
//| Envoie une ligne JSON et lit la réponse jusqu'au '\n'.            |
//| TCP est un flux : une réponse peut arriver en plusieurs morceaux, |
//| d'où la boucle d'accumulation jusqu'au délimiteur.                |
//+------------------------------------------------------------------+
bool SendAndReceive(const string request, string &response)
{
   uchar req[];
   int len = StringToCharArray(request + "\n", req, 0, WHOLE_ARRAY, CP_UTF8) - 1;
   if(len <= 0)
      return(false);

   if(SocketSend(g_socket, req, len) != len)
   {
      PrintFormat("SocketSend incomplet (erreur %d)", GetLastError());
      return(false);
   }

   string acc = "";
   uint deadline = GetTickCount() + (uint)InpTimeoutMs;
   uchar buf[];

   while(GetTickCount() < deadline)
   {
      uint avail = SocketIsReadable(g_socket);
      if(avail > 0)
      {
         int n = SocketRead(g_socket, buf, avail, 200);
         if(n > 0)
         {
            acc += CharArrayToString(buf, 0, n, CP_UTF8);
            int nl = StringFind(acc, "\n");
            if(nl >= 0)
            {
               response = StringSubstr(acc, 0, nl);
               return(true);
            }
         }
         else if(n < 0)
         {
            PrintFormat("SocketRead a échoué (erreur %d)", GetLastError());
            return(false);
         }
      }
      else
      {
         Sleep(5);
      }
   }
   PrintFormat("Délai dépassé (%d ms) en attente de la réponse du serveur.", InpTimeoutMs);
   return(false);
}

//====================================================================
//  CONSTRUCTION DE LA REQUÊTE
//====================================================================
//+------------------------------------------------------------------+
//| Nature du compte connecté : demo, concours ou réel.              |
//|                                                                  |
//| Transmise au serveur pour qu'il puisse refuser d'armer les ordres |
//| sur un compte réel sans autorisation explicite. Le terminal est   |
//| la seule source fiable de cette information : ni le solde, ni le  |
//| nom du courtier ne permettent de distinguer une démo d'un réel.   |
//+------------------------------------------------------------------+
string AccountTypeName()
{
   long mode = AccountInfoInteger(ACCOUNT_TRADE_MODE);
   if(mode == ACCOUNT_TRADE_MODE_DEMO)    return("demo");
   if(mode == ACCOUNT_TRADE_MODE_CONTEST) return("contest");
   return("real");
}

string BuildPredictRequest()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   // start_pos = 1 : on ignore délibérément la barre EN COURS de formation.
   // Le modèle a été entraîné sur des barres CLÔTURÉES ; envoyer une barre incomplète
   // ferait varier les features à chaque tick et produirait des décisions instables.
   int copied = CopyRates(_Symbol, _Period, 1, InpHistoryBars, rates);
   if(copied < InpHistoryBars)
   {
      PrintFormat("Historique insuffisant : %d barres disponibles sur %d demandées.",
                  copied, InpHistoryBars);
      return("");
   }

   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

   string sb = "{\"type\":\"predict\",\"version\":\"1.0\",";
   sb += "\"symbol\":\"" + _Symbol + "\",";
   sb += "\"timeframe\":\"" + EnumToString((ENUM_TIMEFRAMES)_Period) + "\",";
   sb += "\"equity\":" + DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY), 2) + ",";
   sb += "\"balance\":" + DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2) + ",";
   sb += "\"peak_equity\":" + DoubleToString(g_peakEquity, 2) + ",";
   sb += "\"current_exposure\":" + DoubleToString(CurrentExposure(), 6) + ",";
   sb += "\"bars_in_position\":" + IntegerToString(g_barsInPos) + ",";
   sb += "\"entry_price\":" + DoubleToString(g_entryPrice, _Digits) + ",";
   sb += "\"magic\":" + IntegerToString((int)InpMagic) + ",";
   sb += "\"account_type\":\"" + AccountTypeName() + "\",";
   sb += "\"bars\":[";

   for(int i = 0; i < copied; i++)
   {
      if(i > 0) sb += ",";
      sb += "[" + IntegerToString((long)rates[i].time) + ","
                + DoubleToString(rates[i].open,  _Digits) + ","
                + DoubleToString(rates[i].high,  _Digits) + ","
                + DoubleToString(rates[i].low,   _Digits) + ","
                + DoubleToString(rates[i].close, _Digits) + ","
                + DoubleToString((double)rates[i].tick_volume, 0) + ","
                + DoubleToString(rates[i].spread * point, _Digits) + "]";
   }
   sb += "]}";
   return(sb);
}

//====================================================================
//  TRAITEMENT DE LA RÉPONSE
//====================================================================
void ProcessResponse(const string json)
{
   if(!JsonGetBool(json, "ok"))
   {
      string err = JsonGetString(json, "error");
      string st  = JsonGetString(json, "status");
      PrintFormat("Le serveur a refusé la requête [%s] : %s", st, err);
      // Réponse "liquidate" : le coupe-circuit Python a été déclenché.
      if(st == "liquidate")
      {
         ClosePosition();
         Halt("coupe-circuit serveur : " + err);
      }
      return;
   }

   double target  = JsonGetNumber(json, "target_exposure");
   double conf    = JsonGetNumber(json, "confidence");
   string status  = JsonGetString(json, "status");
   double slDist  = JsonGetNumber(json, "sl_distance");
   double tpDist  = JsonGetNumber(json, "tp_distance");
   double latency = JsonGetNumber(json, "latency_ms");

   if(status == "liquidate")
   {
      ClosePosition();
      Halt("coupe-circuit serveur");
      return;
   }

   target = MathMax(-InpMaxExposure, MathMin(InpMaxExposure, target));

   // Alerte critique côté serveur, et l'opérateur a demandé d'y réagir. On bloque le
   // RENFORCEMENT, jamais la réduction : un mode de sécurité qui empêcherait aussi de
   // fermer serait plus dangereux que le problème qu'il traite.
   // Désactivé par défaut : une couche d'observation qui pilote les positions devient
   // elle-même un risque opérationnel. On l'active après avoir observé ses alertes.
   double current = CurrentExposure();
   if(InpBlockOnCritical && g_serverCritical && MathAbs(target) > MathAbs(current))
   {
      Print("Alerte critique côté serveur : renforcement bloqué, réduction toujours permise.");
      target = current;
   }

   if(InpVerbose)
      PrintFormat("Décision : exposition=%.4f | actuelle=%.4f | confiance=%.2f | statut=%s | %.1f ms",
                  target, CurrentExposure(), conf, status, latency);

   ApplyTargetExposure(target, slDist, tpDist);
}

//====================================================================
//  EXÉCUTION
//====================================================================
double CurrentExposure()
{
   if(!PositionSelectByMagic())
      return(0.0);

   double volume   = PositionGetDouble(POSITION_VOLUME);
   long   type     = PositionGetInteger(POSITION_TYPE);
   double contract = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double price    = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity <= 0.0)
      return(0.0);

   double notional = volume * contract * price;
   double expo     = notional / equity;
   return(type == POSITION_TYPE_SELL ? -expo : expo);
}

//+------------------------------------------------------------------+
//| Convertit une exposition (fraction du capital) en volume, en      |
//| respectant pas de lot, minimum et maximum du symbole.             |
//| L'arrondi est INFÉRIEUR : ne jamais dépasser le risque demandé.   |
//+------------------------------------------------------------------+
double ExposureToLots(double exposure)
{
   double equity   = AccountInfoDouble(ACCOUNT_EQUITY);
   double contract = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_CONTRACT_SIZE);
   double price    = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   if(equity <= 0.0 || contract <= 0.0 || price <= 0.0)
      return(0.0);

   double step   = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(step <= 0.0) step = 0.01;

   double raw  = (MathAbs(exposure) * equity) / (contract * price);
   double lots = MathFloor(raw / step) * step;

   if(lots < minLot)
      return(0.0);                      // exposition trop petite pour être exprimable
   if(lots > maxLot)
      lots = maxLot;
   return(NormalizeDouble(lots, 2));
}

void ApplyTargetExposure(double target, double slDist, double tpDist)
{
   double current = CurrentExposure();
   double delta   = target - current;

   // Bande morte : un micro-ajustement coûte le spread pour un bénéfice nul.
   if(MathAbs(delta) < InpMinRebalance)
      return;

   bool hasPos = PositionSelectByMagic();

   // Inversion de sens : fermer d'abord, la couverture (hedging) n'est pas gérée ici.
   if(hasPos && target * current < 0.0)
   {
      ClosePosition();
      hasPos = false;
      current = 0.0;
      delta = target;
   }

   if(MathAbs(target) < InpMinRebalance)
   {
      if(hasPos) ClosePosition();
      return;
   }

   double targetLots = ExposureToLots(target);
   if(targetLots <= 0.0)
   {
      if(hasPos) ClosePosition();
      return;
   }

   if(!hasPos)
   {
      OpenPosition(target > 0.0, targetLots, slDist, tpDist);
      return;
   }

   double currentLots = PositionGetDouble(POSITION_VOLUME);
   double lotDelta    = targetLots - currentLots;
   double step        = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(step <= 0.0) step = 0.01;

   if(MathAbs(lotDelta) < step)
      return;

   if(lotDelta > 0.0)
      OpenPosition(target > 0.0, NormalizeDouble(lotDelta, 2), 0.0, 0.0);   // renforcement
   else
      PartialClose(NormalizeDouble(-lotDelta, 2));                          // allègement
}

void OpenPosition(bool isLong, double lots, double slDist, double tpDist)
{
   if(InpDryRun)
   {
      PrintFormat("[DRY-RUN] %s %.2f lot(s) sur %s", (isLong ? "ACHAT" : "VENTE"), lots, _Symbol);
      return;
   }

   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double entry = isLong ? ask : bid;

   double sl = 0.0, tp = 0.0;
   if(InpUseServerSLTP && slDist > 0.0)
   {
      sl = isLong ? entry - slDist : entry + slDist;
      tp = (tpDist > 0.0) ? (isLong ? entry + tpDist : entry - tpDist) : 0.0;
      sl = NormalizeDouble(sl, _Digits);
      tp = NormalizeDouble(tp, _Digits);
      if(!ValidateStops(isLong, entry, sl, tp))
      {
         sl = 0.0;
         tp = 0.0;
      }
   }

   bool ok = isLong ? trade.Buy(lots, _Symbol, 0.0, sl, tp, "qbot")
                    : trade.Sell(lots, _Symbol, 0.0, sl, tp, "qbot");
   if(ok)
   {
      g_entryPrice = entry;
      g_barsInPos  = 0;
      PrintFormat("%s %.2f lot(s) @ %.5f | SL=%.5f TP=%.5f",
                  (isLong ? "ACHAT" : "VENTE"), lots, entry, sl, tp);
   }
   else
   {
      PrintFormat("Ordre rejeté : retcode=%d (%s)", trade.ResultRetcode(),
                  trade.ResultRetcodeDescription());
   }
}

//+------------------------------------------------------------------+
//| Le courtier impose une distance minimale (SYMBOL_TRADE_STOPS_LEVEL)|
//| entre le prix et les stops. Un SL trop proche fait rejeter l'ordre |
//| ENTIER : mieux vaut ouvrir sans stop et le poser ensuite.         |
//+------------------------------------------------------------------+
bool ValidateStops(bool isLong, double entry, double sl, double tp)
{
   long   stopLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double point     = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double minDist   = stopLevel * point;
   if(minDist <= 0.0)
      return(true);

   if(sl > 0.0 && MathAbs(entry - sl) < minDist) return(false);
   if(tp > 0.0 && MathAbs(entry - tp) < minDist) return(false);
   return(true);
}

void PartialClose(double lots)
{
   if(InpDryRun)
   {
      PrintFormat("[DRY-RUN] Allègement de %.2f lot(s)", lots);
      return;
   }
   if(!trade.PositionClosePartial(_Symbol, lots))
      PrintFormat("Allègement refusé : retcode=%d", trade.ResultRetcode());
}

void ClosePosition()
{
   if(!PositionSelectByMagic())
      return;
   if(InpDryRun)
   {
      Print("[DRY-RUN] Fermeture de la position");
      return;
   }
   if(trade.PositionClose(_Symbol))
   {
      g_entryPrice = 0.0;
      g_barsInPos  = 0;
      Print("Position fermée.");
   }
   else
      PrintFormat("Fermeture refusée : retcode=%d", trade.ResultRetcode());
}

bool PositionSelectByMagic()
{
   if(!PositionSelect(_Symbol))
      return(false);
   return(PositionGetInteger(POSITION_MAGIC) == InpMagic);
}

//====================================================================
//  GARDE-FOUS LOCAUX
//====================================================================
//  Deuxième ligne de défense, volontairement redondante avec celle du
//  serveur : si le processus Python plante, se fige ou renvoie une
//  valeur aberrante, ces contrôles restent actifs dans le terminal.
//====================================================================
void UpdateAccountState()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   if(equity > g_peakEquity)
      g_peakEquity = equity;

   datetime today = TodayStamp();
   if(today != g_lastDayStamp)
   {
      g_lastDayStamp   = today;
      g_dayStartEquity = equity;
   }

   if(PositionSelectByMagic() && g_entryPrice == 0.0)
      g_entryPrice = PositionGetDouble(POSITION_PRICE_OPEN);
}

bool CheckLocalGuards()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   if(InpMinEquity > 0.0 && equity < InpMinEquity)
   {
      ClosePosition();
      Halt(StringFormat("équité %.2f sous le plancher %.2f", equity, InpMinEquity));
      return(false);
   }

   if(InpMaxDrawdownPct > 0.0 && g_peakEquity > 0.0)
   {
      double dd = (equity / g_peakEquity - 1.0) * 100.0;
      if(dd <= -InpMaxDrawdownPct)
      {
         ClosePosition();
         Halt(StringFormat("drawdown %.2f%% <= -%.2f%%", dd, InpMaxDrawdownPct));
         return(false);
      }
   }

   if(InpMaxDailyLossPct > 0.0 && g_dayStartEquity > 0.0)
   {
      double dayPnl = (equity / g_dayStartEquity - 1.0) * 100.0;
      if(dayPnl <= -InpMaxDailyLossPct)
      {
         ClosePosition();
         if(InpVerbose)
            PrintFormat("Perte journalière %.2f%% atteinte : arrêt jusqu'à demain.", dayPnl);
         return(false);
      }
   }

   if(InpMaxSpreadPoints > 0.0)
   {
      double spread = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      if(spread > InpMaxSpreadPoints)
      {
         if(InpVerbose)
            PrintFormat("Spread %.0f points > %.0f : aucune décision prise.",
                        spread, InpMaxSpreadPoints);
         return(false);
      }
   }
   return(true);
}

void Halt(const string reason)
{
   g_halted     = true;
   g_haltReason = reason;
   PrintFormat("*** ARRÊT DE L'EA : %s — intervention manuelle requise. ***", reason);
   Alert("QBotBridge arrêté : ", reason);
}

datetime TodayStamp()
{
   MqlDateTime t;
   TimeToStruct(TimeCurrent(), t);
   t.hour = 0; t.min = 0; t.sec = 0;
   return(StructToTime(t));
}

//====================================================================
//  ANALYSE JSON MINIMALE
//====================================================================
//  MQL5 n'a pas de parseur JSON natif. Les réponses du serveur ont un
//  schéma fixe et plat : une extraction par clé suffit et évite
//  d'embarquer une bibliothèque entière.
//====================================================================
int JsonValueStart(const string json, const string key)
{
   int p = StringFind(json, "\"" + key + "\":");
   if(p < 0)
      return(-1);
   return(p + StringLen(key) + 3);
}

double JsonGetNumber(const string json, const string key)
{
   int start = JsonValueStart(json, key);
   if(start < 0)
      return(0.0);

   int len = StringLen(json);
   int end = start;
   while(end < len)
   {
      ushort c = StringGetCharacter(json, end);
      // chiffres, signe, point décimal et notation exponentielle
      if((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
         end++;
      else
         break;
   }
   if(end == start)
      return(0.0);
   return(StringToDouble(StringSubstr(json, start, end - start)));
}

string JsonGetString(const string json, const string key)
{
   int start = JsonValueStart(json, key);
   if(start < 0)
      return("");
   if(StringGetCharacter(json, start) != '"')
      return("");
   start++;
   int end = StringFind(json, "\"", start);
   if(end < 0)
      return("");
   return(StringSubstr(json, start, end - start));
}

bool JsonGetBool(const string json, const string key)
{
   int start = JsonValueStart(json, key);
   if(start < 0)
      return(false);
   return(StringSubstr(json, start, 4) == "true");
}
//+------------------------------------------------------------------+
