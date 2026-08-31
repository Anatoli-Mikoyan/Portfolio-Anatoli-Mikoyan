"""Assemblage de la matrice de features.

Deux garanties non négociables :

1. **Causalité stricte** — toute normalisation est glissante ou étendue, jamais globale.
   Un simple `StandardScaler().fit(X)` sur tout l'historique fait fuir la moyenne et
   l'écart-type du futur dans le passé : c'est la fuite de données la plus fréquente en
   ML financier, et elle suffit à produire un Sharpe fantôme de 3+.
2. **Parité backtest / live** — le même objet, le même code, produisent les features en
   backtest et en production. Toute divergence ici est un « training-serving skew » qui
   ne se détecte qu'en perdant de l'argent réel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..utils.logging import get_logger
from .fracdiff import find_min_ffd, frac_diff_ffd
from .microstructure import build_microstructure_features
from .regime import build_calendar_features, build_regime_features
from .technical import build_technical_features

log = get_logger("features.pipeline")


@dataclass
class FeaturePipeline:
    """Construit et normalise la matrice de features de façon causale et reproductible."""

    cfg: FeatureConfig = field(default_factory=FeatureConfig)
    feature_names: List[str] = field(default_factory=list)
    fracdiff_d: Optional[float] = None
    _fitted: bool = False

    # --- longueur d'historique nécessaire pour reproduire EXACTEMENT les features -------
    @property
    def _feature_warmup(self) -> int:
        """Barres nécessaires pour que la feature la plus « longue » soit pleinement définie.

        Attention aux moyennes exponentielles (EMA, RSI, ATR, ADX, MACD) : leur mémoire est
        infinie. Tronquer l'historique ne les rend pas NaN, il les rend SILENCIEUSEMENT
        FAUSSES. On provisionne 5 fois leur span, seuil au-delà duquel le poids résiduel
        du passé tronqué est négligeable (0.99^500 ≈ 0.7 %).
        """
        ema_like = max(
            [*self.cfg.ema_windows, self.cfg.atr_window, self.cfg.adx_window,
             *self.cfg.rsi_windows, 26],
            default=30,
        )
        return int(max(
            max(self.cfg.returns_windows, default=1),
            max(self.cfg.vol_windows, default=1),
            self.cfg.bb_window, self.cfg.donchian_window,
            5 * ema_like,          # convergence des lissages exponentiels
            260,                   # z-score du spread (200) + fenêtres microstructure
            520,                   # percentile de volatilité : vol(20) + rang(500)
            275,                   # variance ratio : fenêtre 252 + décalage q
            265,                   # entropie plug-in : fenêtre 250 + longueur de mot
            160,                   # exposant de Hurst : fenêtre 128 + lags
            350,                   # poids de différenciation fractionnaire (seuil 1e-4)
        ))

    @property
    def min_history(self) -> int:
        """Historique minimal à fournir pour que la DERNIÈRE ligne soit identique à celle
        qu'aurait produite un calcul sur l'historique complet.

        La composition est ADDITIVE et non un maximum : la normalisation glissante
        s'applique APRÈS les features, et sa fenêtre doit elle aussi être entièrement
        remplie. Prendre le maximum des deux — l'erreur naturelle — laisse le z-score
        final se calculer sur une fenêtre partielle, donc avec une moyenne et un
        écart-type différents de ceux du backtest : un écart entraînement/service
        invisible mais bien réel.
        """
        scaler_need = self.cfg.scaler_window if self.cfg.scaler in ("rolling_zscore", "rank") else 0
        return int(self._feature_warmup + scaler_need + 50)

    # ------------------------------------------------------------------------------------
    def build_raw(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construit les features BRUTES (non normalisées)."""
        blocks = [build_technical_features(df, self.cfg)]

        if self.cfg.use_fracdiff:
            d = self.fracdiff_d if self.fracdiff_d is not None else self.cfg.fracdiff_d
            if d is None:
                d, stat, corr = find_min_ffd(df["close"], thresh=self.cfg.fracdiff_thresh)
                log.info("Fracdiff : d*=%.2f (ADF=%.2f, mémoire conservée=%.3f)", d, stat, corr)
            self.fracdiff_d = float(d)
            ffd = frac_diff_ffd(np.log(df["close"]), self.fracdiff_d, self.cfg.fracdiff_thresh)
            blocks.append(ffd.rename("ffd_logp").to_frame())

        if self.cfg.use_microstructure:
            blocks.append(build_microstructure_features(df))
        if self.cfg.use_regime:
            blocks.append(build_regime_features(df))
        if self.cfg.use_calendar:
            blocks.append(build_calendar_features(df.index))

        raw = pd.concat(blocks, axis=1)
        raw = raw.loc[:, ~raw.columns.duplicated()]
        return raw.replace([np.inf, -np.inf], np.nan)

    # ------------------------------------------------------------------------------------
    @staticmethod
    def _safe_zscore(raw: pd.DataFrame, mu: pd.DataFrame, sd: pd.DataFrame) -> pd.DataFrame:
        """z-score robuste aux fenêtres de variance nulle.

        Diviser par un écart-type nul produirait des NaN qui, propagés par un `dropna()`,
        peuvent vider silencieusement toute la matrice de features. Une feature constante
        sur sa fenêtre porte une information nulle : sa valeur normalisée est donc 0.
        """
        denom = sd.mask(sd <= 1e-12)
        out = (raw - mu) / denom
        degenerate = (sd <= 1e-12) & mu.notna() & raw.notna()
        return out.mask(degenerate, 0.0)

    def _scale(self, raw: pd.DataFrame) -> pd.DataFrame:
        mode, w = self.cfg.scaler, self.cfg.scaler_window
        if mode == "none":
            return raw
        if mode == "rolling_zscore":
            mu = raw.rolling(w, min_periods=max(w // 4, 20)).mean()
            sd = raw.rolling(w, min_periods=max(w // 4, 20)).std(ddof=0)
            out = self._safe_zscore(raw, mu, sd)
        elif mode == "expanding_zscore":
            mu = raw.expanding(min_periods=50).mean()
            sd = raw.expanding(min_periods=50).std(ddof=0)
            out = self._safe_zscore(raw, mu, sd)
        elif mode == "rank":
            # Rang glissant dans [-1, 1] : totalement insensible aux outliers et aux
            # changements d'échelle, au prix d'une perte d'information sur l'amplitude.
            out = raw.rolling(w, min_periods=max(w // 4, 20)).rank(pct=True) * 2.0 - 1.0
        else:
            raise ValueError(f"scaler inconnu : {mode}")

        # Les features déjà bornées (sin/cos, drapeaux de session) n'ont pas à être z-scorées :
        # les normaliser détruirait leur sémantique et amplifierait leur bruit.
        passthrough = [c for c in raw.columns
                       if c.startswith(("hour_", "dow_", "month_", "sess_", "dc_break"))]
        for c in passthrough:
            out[c] = raw[c]

        s = self.cfg.winsorize_sigma
        return out.clip(lower=-s, upper=s) if s and s > 0 else out

    # ------------------------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """À utiliser sur le segment d'entraînement : fige la liste des colonnes et d*."""
        raw = self.build_raw(df)
        scaled = self._scale(raw)
        if self.cfg.dropna:
            scaled = scaled.dropna()
        # Colonnes dégénérées : variance nulle => aucune information, uniquement du bruit
        # numérique et un paramètre de plus à surapprendre.
        keep = [c for c in scaled.columns if float(scaled[c].std(ddof=0)) > 1e-10]
        dropped = sorted(set(scaled.columns) - set(keep))
        if dropped:
            log.info("Colonnes constantes retirées : %s", dropped)
        scaled = scaled[keep]
        self.feature_names = list(scaled.columns)
        self._fitted = True
        log.info("Matrice de features : %d lignes x %d colonnes", *scaled.shape)
        return scaled

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """À utiliser sur les segments de test/live : impose exactement le même schéma."""
        if not self._fitted:
            raise RuntimeError("FeaturePipeline non ajusté : appeler fit_transform d'abord.")
        raw = self.build_raw(df)
        scaled = self._scale(raw)
        missing = [c for c in self.feature_names if c not in scaled.columns]
        for c in missing:
            scaled[c] = 0.0
        if missing:
            log.warning("Colonnes absentes en transform, remplies à 0 : %s", missing)
        scaled = scaled[self.feature_names]
        return scaled.dropna() if self.cfg.dropna else scaled

    def transform_latest(self, df_window: pd.DataFrame, n_rows: int = 1) -> np.ndarray:
        """Chemin d'inférence live : renvoie les `n_rows` dernières lignes de features.

        `df_window` doit contenir au moins `min_history + n_rows` barres, sinon les
        fenêtres longues (variance ratio, rank de vol) renverraient des NaN silencieux.
        """
        if len(df_window) < self.min_history:
            raise ValueError(
                f"Historique insuffisant pour l'inférence : {len(df_window)} barres "
                f"fournies, {self.min_history} requises."
            )
        out = self.transform(df_window)
        if len(out) < n_rows:
            raise ValueError(self._pourquoi_aucune_ligne(df_window, len(out), n_rows))
        return out.tail(n_rows).to_numpy(dtype=np.float32)

    # -----------------------------------------------------------------------------------
    def _pourquoi_aucune_ligne(self, df_window: pd.DataFrame, obtenues: int,
                               demandees: int) -> str:
        """Explique la disette de features, au lieu de la constater.

        « Seulement 0 lignes de features valides » est vrai et inutilisable : l'historique
        est suffisant, la cause est ailleurs et l'utilisateur n'a aucun moyen de la
        deviner. Or elle est presque toujours la même — une colonne d'entrée constante ou
        nulle rend une poignée de features indéfinies sur TOUTE la fenêtre, et le
        `dropna` supprime alors chaque ligne, y compris celles dont les soixante autres
        features étaient parfaitement calculées.

        Le cas concret : un flux MetaTrader qui ne renseigne pas le volume. `amihud`
        divise par le volume en notionnel, `kyle_lambda` régresse dessus, `vpin` le
        normalise : trois colonnes vides suffisent à tout emporter.

        La CAUSE passe en tête du message : MetaTrader tronque les longues lignes de son
        journal, et une explication placée après le constat n'atteindrait jamais l'écran
        de celui qui en a besoin. Le mode d'emploi complet part dans le journal du
        serveur, qui ne tronque pas.

        On ne comble pas ces colonnes d'office : le modèle a été entraîné avec de vraies
        valeurs, l'alimenter de zéros le ferait décider sur des entrées qu'il n'a jamais
        vues. On refuse, mais on dit quoi réparer.
        """
        fautives = []
        for colonne in ("volume", "spread"):
            if colonne not in df_window.columns:
                continue
            valeurs = pd.to_numeric(df_window[colonne], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(valeurs).any():
                fautives.append((colonne, "aucune valeur exploitable"))
            elif float(np.nanmax(np.abs(valeurs))) == 0.0:
                fautives.append((colonne, "zéro sur toute la fenêtre"))
            elif float(np.nanstd(valeurs)) == 0.0:
                fautives.append((colonne, "constante sur toute la fenêtre"))

        if not fautives:
            return (f"Historique trop court malgré {len(df_window)} barres : "
                    f"{obtenues} ligne(s) de features valides sur {demandees} demandée(s). "
                    "Fournir davantage de barres.")

        noms = ", ".join(f"{c} ({raison})" for c, raison in fautives)
        log.error(
            "Flux incomplet : %s. Les indicateurs de microstructure (Amihud, Kyle, VPIN, "
            "z-score du spread) divisent par cette grandeur : indéfinis sur toute la "
            "fenêtre, toutes les lignes sont écartées — y compris celles dont les autres "
            "features étaient correctes.\n"
            "  Côté MetaTrader, le flux du courtier ne fournit pas cette donnée.\n"
            "  - vérifier Observation du marché > clic droit sur le symbole > Spécification ;\n"
            "  - essayer un autre symbole, ou un autre courtier de démonstration.\n"
            "  Les valeurs ne sont pas comblées d'office : le modèle a été entraîné avec "
            "de vraies valeurs, lui en donner de fausses le ferait décider sur des "
            "entrées jamais rencontrées.", noms)

        return (f"Flux incomplet — {noms} : indicateurs de microstructure indéfinis, "
                f"{obtenues} ligne exploitable sur {demandees}. Détail dans le journal "
                "du serveur.")

    # --- persistance ---------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import json

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.cfg.__dict__.items()},
            "feature_names": self.feature_names,
            "fracdiff_d": self.fracdiff_d,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePipeline":
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg_raw = payload["cfg"]
        for k, v in list(cfg_raw.items()):
            if isinstance(v, list):
                cfg_raw[k] = tuple(v)
        pipe = cls(cfg=FeatureConfig(**cfg_raw))
        pipe.feature_names = payload["feature_names"]
        pipe.fracdiff_d = payload["fracdiff_d"]
        pipe._fitted = True
        return pipe


def align_features_prices(features: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Réaligne features et OHLCV sur l'intersection stricte de leurs index."""
    idx = features.index.intersection(df.index)
    return features.loc[idx], df.loc[idx]
