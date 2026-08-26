"""Réseaux de neurones du Rainbow.

Contient les trois extensions structurelles du Rainbow (Hessel et al., 2018) :
  * NoisyNet    — exploration paramétrique apprise, remplace l'ε-greedy
  * Dueling     — séparation V(s) / A(s,a)
  * Distributional — on modélise la DISTRIBUTION du retour, pas seulement son espérance

Le point 3 est particulièrement pertinent en finance : deux stratégies de même espérance
mais de distributions différentes (l'une régulière, l'autre à queue gauche épaisse) sont
strictement différentes pour un gérant. Un agent qui n'apprend que E[Q] est structurellement
aveugle au risque de queue. QR-DQN apprend les quantiles du retour, ce qui rend le
risque de queue directement observable — et permet d'agir dessus (voir `risk_measure`).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =======================================================================================
# NoisyNet
# =======================================================================================
class NoisyLinear(nn.Module):
    """Couche linéaire bruitée à bruit gaussien factorisé (Fortunato et al., 2017).

    L'exploration devient un paramètre APPRIS, propre à chaque état : l'agent explore
    encore là où il est incertain et cesse d'explorer là où il est confiant. Un ε-greedy
    global, lui, injecte du bruit uniforme y compris dans des états déjà maîtrisés — ce
    qui, en trading, coûte directement de l'argent en transactions inutiles.
    """

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features, self.out_features, self.sigma0 = in_features, out_features, sigma0

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        # `None` => le bruit suit le mode du module (train/eval), comportement par défaut.
        # `True`/`False` force explicitement l'état, ce qui permet d'explorer en mode eval
        # (bruit actif, dropout inactif) — voir `RainbowAgent.act`.
        self.noise_override: Optional[bool] = None

        self.reset_parameters()
        self.reset_noise()

    @property
    def noise_enabled(self) -> bool:
        return self.training if self.noise_override is None else self.noise_override

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma0 * bound)
        self.bias_sigma.data.fill_(self.sigma0 * bound)

    @staticmethod
    def _scale_noise(size: int, device) -> torch.Tensor:
        x = torch.randn(size, device=device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        """Bruit factorisé : p+q tirages au lieu de p*q — coût négligeable."""
        device = self.weight_mu.device
        eps_in = self._scale_noise(self.in_features, device)
        eps_out = self._scale_noise(self.out_features, device)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_enabled:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu   # politique déterministe
        return F.linear(x, w, b)

    def extra_repr(self) -> str:  # pragma: no cover - affichage
        return f"in={self.in_features}, out={self.out_features}, sigma0={self.sigma0}"


def linear_layer(in_f: int, out_f: int, noisy: bool, sigma0: float = 0.5) -> nn.Module:
    return NoisyLinear(in_f, out_f, sigma0) if noisy else nn.Linear(in_f, out_f)


# =======================================================================================
# Encodeurs
# =======================================================================================
class MLPEncoder(nn.Module):
    """Encodeur dense sur la fenêtre aplatie. Rapide, robuste, difficile à battre
    quand les features sont déjà riches en information temporelle."""

    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout: float, layer_norm: bool):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            if layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUEncoder(nn.Module):
    """Encodeur récurrent : traite la fenêtre comme une séquence.

    Utile quand l'ORDRE des événements compte au-delà de ce que capturent les indicateurs
    (par exemple la séquence d'un squeeze de volatilité suivi d'un breakout).
    """

    def __init__(self, n_features: int, window: int, hidden: int, extra_dim: int,
                 mlp_hidden: Tuple[int, ...], dropout: float, layer_norm: bool):
        super().__init__()
        self.n_features, self.window, self.extra_dim = n_features, window, extra_dim
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.norm = nn.LayerNorm(hidden) if layer_norm else nn.Identity()
        self.head = MLPEncoder(hidden + extra_dim, mlp_hidden, dropout, layer_norm)
        self.out_dim = self.head.out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = self.window * self.n_features
        seq = x[:, :seq_len].view(-1, self.window, self.n_features)
        extra = x[:, seq_len:]
        _, h = self.gru(seq)
        z = self.norm(h[-1])
        return self.head(torch.cat([z, extra], dim=1) if self.extra_dim else z)


class CausalConv1d(nn.Module):
    """Convolution 1D strictement causale : la sortie à t ne dépend que des entrées ≤ t.

    Une `nn.Conv1d` avec `padding=k//2` regarde le futur — erreur silencieuse et
    dévastatrice en série temporelle financière. On pad uniquement à gauche.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, 0)))


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network à dilatations croissantes.

    Champ réceptif exponentiel en profondeur, entraînement parallélisable (contrairement
    au GRU) et gradients stables sur longues fenêtres.
    """

    def __init__(self, n_features: int, window: int, channels: Tuple[int, ...], kernel: int,
                 extra_dim: int, mlp_hidden: Tuple[int, ...], dropout: float, layer_norm: bool):
        super().__init__()
        self.n_features, self.window, self.extra_dim = n_features, window, extra_dim
        blocks: list[nn.Module] = []
        in_ch = n_features
        for i, ch in enumerate(channels):
            blocks += [CausalConv1d(in_ch, ch, kernel, dilation=2 ** i), nn.SiLU()]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(in_ch) if layer_norm else nn.Identity()
        self.head = MLPEncoder(in_ch + extra_dim, mlp_hidden, dropout, layer_norm)
        self.out_dim = self.head.out_dim
        self.receptive_field = 1 + sum((kernel - 1) * (2 ** i) for i in range(len(channels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = self.window * self.n_features
        seq = x[:, :seq_len].view(-1, self.window, self.n_features).transpose(1, 2)
        extra = x[:, seq_len:]
        z = self.norm(self.tcn(seq)[:, :, -1])   # dernier pas de temps uniquement
        return self.head(torch.cat([z, extra], dim=1) if self.extra_dim else z)


# =======================================================================================
# Réseau Q complet
# =======================================================================================
class QNetwork(nn.Module):
    """Réseau Q Rainbow : encodeur + tête duelling + sortie distributionnelle.

    Sortie selon `distributional` :
      * "qr"   -> (B, A, n_quantiles) : quantiles du retour, régression quantile
      * "c51"  -> (B, A, n_atoms)     : probabilités sur un support fixe
      * "none" -> (B, A)              : Q scalaire classique
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        n_features: int,
        window: int,
        extra_dim: int,
        encoder: str = "mlp",
        hidden_sizes: Tuple[int, ...] = (256, 256),
        encoder_hidden: int = 128,
        tcn_channels: Tuple[int, ...] = (64, 64),
        tcn_kernel: int = 3,
        dropout: float = 0.1,
        layer_norm: bool = True,
        noisy: bool = True,
        sigma0: float = 0.5,
        dueling: bool = True,
        distributional: str = "qr",
        n_quantiles: int = 51,
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        super().__init__()
        self.n_actions = n_actions
        self.dueling = dueling
        self.noisy = noisy
        self.distributional = distributional

        if distributional == "qr":
            self.n_outputs = n_quantiles
        elif distributional == "c51":
            self.n_outputs = n_atoms
            self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))
            self.delta_z = (v_max - v_min) / (n_atoms - 1)
        elif distributional == "none":
            self.n_outputs = 1
        else:
            raise ValueError(f"distributional inconnu : {distributional}")

        if encoder == "mlp":
            self.encoder: nn.Module = MLPEncoder(obs_dim, hidden_sizes, dropout, layer_norm)
        elif encoder == "gru":
            self.encoder = GRUEncoder(n_features, window, encoder_hidden, extra_dim,
                                      hidden_sizes, dropout, layer_norm)
        elif encoder == "tcn":
            self.encoder = TCNEncoder(n_features, window, tcn_channels, tcn_kernel, extra_dim,
                                      hidden_sizes, dropout, layer_norm)
        else:
            raise ValueError(f"encoder inconnu : {encoder}")

        d = self.encoder.out_dim
        head_hidden = max(d // 2, 64)
        self.adv = nn.Sequential(
            linear_layer(d, head_hidden, noisy, sigma0), nn.SiLU(),
            linear_layer(head_hidden, n_actions * self.n_outputs, noisy, sigma0),
        )
        if dueling:
            self.val = nn.Sequential(
                linear_layer(d, head_hidden, noisy, sigma0), nn.SiLU(),
                linear_layer(head_hidden, self.n_outputs, noisy, sigma0),
            )

        if distributional == "qr":
            # Points médians des quantiles : τ_i = (2i+1) / 2N
            taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
            self.register_buffer("taus", taus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        a = self.adv(z).view(-1, self.n_actions, self.n_outputs)
        if self.dueling:
            v = self.val(z).view(-1, 1, self.n_outputs)
            # Identifiabilité : on centre l'avantage, sinon V et A ne sont définis
            # qu'à une constante près et l'apprentissage dérive.
            out = v + a - a.mean(dim=1, keepdim=True)
        else:
            out = a

        if self.distributional == "c51":
            return F.softmax(out, dim=2)
        if self.distributional == "none":
            return out.squeeze(2)
        return out

    # ---------------------------------------------------------------------------------
    def q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Espérance du retour par action — (B, A)."""
        out = self.forward(x)
        if self.distributional == "qr":
            return out.mean(dim=2)
        if self.distributional == "c51":
            return (out * self.support.view(1, 1, -1)).sum(dim=2)
        return out

    def risk_measure(self, x: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
        """CVaR_α du retour par action — (B, A).

        Disponible uniquement avec une tête distributionnelle. Permet une politique
        AVERSE AU RISQUE : choisir l'action maximisant le CVaR revient à optimiser le
        pire décile des scénarios plutôt que la moyenne. C'est exactement le critère
        qu'applique une table de trading professionnelle sous contrainte de risque.
        """
        out = self.forward(x)
        if self.distributional == "qr":
            k = max(int(alpha * out.shape[2]), 1)
            return out.sort(dim=2).values[:, :, :k].mean(dim=2)
        if self.distributional == "c51":
            cdf = out.cumsum(dim=2)
            mask = (cdf <= alpha).float()
            weights = out * mask
            denom = weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
            return ((weights / denom) * self.support.view(1, 1, -1)).sum(dim=2)
        return out   # tête non distributionnelle : pas d'information de queue

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def set_noise(self, active: Optional[bool]) -> None:
        """Force (ou libère) l'activation du bruit, indépendamment du mode train/eval.

        Permet de séparer deux stochasticités que l'on confond souvent : le bruit de
        NoisyNet, qui est la politique d'exploration APPRISE, et le dropout, qui n'est
        qu'un régularisateur de la passe d'apprentissage. Laisser le dropout actif au
        moment de choisir une action ajoute un aléa non maîtrisé au comportement, et fait
        diverger la politique exécutée de celle qui a été évaluée.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.noise_override = active
