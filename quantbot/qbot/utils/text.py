"""Rendu de tableaux texte à largeur garantie.

Les encadrés alignés à la main finissent toujours par se désaligner : il suffit qu'une
valeur dépasse la largeur prévue, ou qu'un accent compte pour deux octets, pour que la
bordure droite parte en biais. Comme ces encadrés sont ce que l'opérateur lit en
production, on les construit ici par programme — la largeur est une donnée, pas une
espérance.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

DEFAULT_WIDTH = 74


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(width - 1, 0)] + "…"


def _rule(left: str, right: str, title: str, width: int) -> str:
    """Ligne de bordure de largeur exactement `width`, titre optionnel incrusté."""
    if not title:
        return left + "─" * (width - 2) + right
    label = _clip(f" {title} ", width - 4)
    return left + "─" + label + "─" * (width - 3 - len(label)) + right


def box_top(title: str = "", width: int = DEFAULT_WIDTH) -> str:
    return _rule("┌", "┐", title, width)


def box_sep(title: str = "", width: int = DEFAULT_WIDTH) -> str:
    return _rule("├", "┤", title, width)


def box_bottom(width: int = DEFAULT_WIDTH) -> str:
    return "└" + "─" * (width - 2) + "┘"


def box_row(label: str, value: str = "", width: int = DEFAULT_WIDTH,
            value_width: int = 34) -> str:
    """Une ligne « libellé ......... valeur » de largeur exactement `width`."""
    inner = width - 4                       # deux bordures + une espace de chaque côté
    value = _clip(str(value), value_width)
    label_room = inner - len(value) - 1
    lab = _clip(str(label), max(label_room, 0))
    pad = inner - len(lab) - len(value)
    return "│ " + lab + " " * max(pad, 1) + value + " │"


def box_text(text: str, width: int = DEFAULT_WIDTH) -> str:
    inner = width - 4
    return "│ " + _clip(str(text), inner).ljust(inner) + " │"


def render_box(title: str, sections: Sequence[Tuple[Optional[str], Sequence[Tuple[str, str]]]],
               width: int = DEFAULT_WIDTH) -> str:
    """Encadré complet : un titre, puis des sections de paires (libellé, valeur)."""
    lines = [box_top(title, width)]
    for i, (section_title, rows) in enumerate(sections):
        if section_title is not None:
            lines.append(box_sep(section_title, width))
        elif i > 0:
            lines.append(box_sep("", width))
        for label, value in rows:
            lines.append(box_row(label, value, width))
    lines.append(box_bottom(width))
    return "\n".join(lines)


def table(headers: Sequence[str], rows: Iterable[Sequence[object]],
          aligns: Optional[Sequence[str]] = None) -> str:
    """Tableau texte simple, colonnes dimensionnées sur le contenu."""
    rows = [[str(c) for c in r] for r in rows]
    heads = [str(h) for h in headers]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(heads)]
    aligns = list(aligns or ["<"] + [">"] * (len(heads) - 1))
    out = ["  ".join(h.ljust(w) for h, w in zip(heads, widths)),
           "  ".join("─" * w for w in widths)]
    for r in rows:
        out.append("  ".join(f"{c:{a}{w}}" for c, a, w in zip(r, aligns, widths)))
    return "\n".join(out)
