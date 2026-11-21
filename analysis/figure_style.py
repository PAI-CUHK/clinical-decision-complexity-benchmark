"""Shared print style for benchmark figures.

Figures are drawn at their intended two-column width so labels are checked at
the same scale at which a reader will see them.
"""

from contextlib import contextmanager
from pathlib import Path
import re

import matplotlib as mpl
from matplotlib import font_manager


TWO_COLUMN = 7.16
ONE_COLUMN = 3.5
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#777777"

# This host has metrically compatible Arial/Times faces (Arimo/Tinos) rather
# than Microsoft Arial/Times New Roman. Register them explicitly so Matplotlib
# embeds the intended compatible faces instead of silently falling back to
# DejaVu Sans.
_COMPATIBLE_FONTS = (
    "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-Italic.ttf",
    "/usr/share/fonts/truetype/croscore/Arimo-BoldItalic.ttf",
    "/usr/share/fonts/truetype/croscore/Tinos-Regular.ttf",
    "/usr/share/fonts/truetype/croscore/Tinos-Bold.ttf",
    "/usr/share/fonts/truetype/croscore/Tinos-Italic.ttf",
    "/usr/share/fonts/truetype/croscore/Tinos-BoldItalic.ttf",
)
for _font_path in _COMPATIBLE_FONTS:
    if Path(_font_path).is_file():
        font_manager.fontManager.addfont(_font_path)

# Fixed by model identity, so sorting/filtering never changes a model's color.
MODEL_STYLE = {
    "gpt_claude-sonnet-5": (BLUE, "o"),
    "gpt_gpt-5.6-sol-ca": (ORANGE, "s"),
    "gpt_qwen3.5-397b-a17b": (GREEN, "^"),
    "vllm_Mistral_4": ("#CC79A7", "D"),
    "vllm_Qwen3_6_35B_A3B": ("#9467BD", "v"),
    "vllm_Qwen3_8_27B": ("#8C6D31", "P"),
    "vllm_deepseek_v4_flash": ("#4D6475", "X"),
    "vllm_gpt_oss_120b": ("#222222", "h"),
}


def apply_borderless_style():
    """Disable canvas, axes, colorbar, and legend borders for figure exports."""
    mpl.rcParams.update({
        "figure.edgecolor": "none",
        "axes.edgecolor": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.bottom": False,
        "axes.spines.left": False,
        "legend.frameon": False,
        "savefig.edgecolor": "none",
    })


def apply_style():
    apply_borderless_style()
    mpl.rcParams.update({
        "font.family": "Arimo",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "grid.color": "#D9D9D9",
        "grid.linewidth": 0.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def add_subfigure_label(ax, label, y=-0.34):
    """Place a subfigure label below an axes, centered at 8 pt."""
    # Keep the label in axes coordinates so later subplots_adjust/tight layout
    # calls move it with its panel. A figure-level artist avoids occlusion by
    # neighboring axes while the axes transform keeps placement dynamic.
    text = ax.figure.text(0.5, y, f"({label})", transform=ax.transAxes,
                          ha="center", va="top", fontsize=8, family="Tinos",
                          clip_on=False, in_layout=True, zorder=100)
    text._subfigure_axes = ax
    text._subfigure_y = y
    return text


def move_title_tag_below(ax, y=-0.34):
    """Move a leading ``(a)`` title tag below the axes and retain its title."""
    title = ax.get_title()
    match = re.match(r"^\(([a-z])\)\s*(.*)$", title, flags=re.DOTALL)
    if not match:
        return False
    ax.set_title(match.group(2))
    add_subfigure_label(ax, match.group(1), y=y)
    return True


def standardize_subfigure_labels(fig, y=-0.34):
    """Move existing panel tags from titles/text to centered 8 pt labels below."""
    for ax in fig.axes:
        move_title_tag_below(ax, y=y)
        for text in list(ax.texts):
            match = re.fullmatch(r"\(([a-z])\)", text.get_text().strip())
            if match and 0 <= text.get_position()[1] <= 1:
                text.remove()
                add_subfigure_label(ax, match.group(1), y=y)


def save_figure(fig, path, formats=("png", "pdf", "svg")):
    standardize_subfigure_labels(fig)
    base = Path(path).with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(Path(str(base) + "." + fmt), dpi=600 if fmt == "png" else None)


@contextmanager
def isolate_axes(fig, axes):
    """Exclude other panels from exports instead of relying on viewport clipping."""
    axes = set(axes)
    hidden = []
    for artist in fig.get_children():
        if (artist is fig.patch or artist in axes
                or getattr(artist, "_subfigure_axes", None) in axes):
            continue
        hidden.append((artist, artist.get_visible()))
        artist.set_visible(False)
    try:
        yield
    finally:
        for artist, visible in hidden:
            artist.set_visible(visible)


