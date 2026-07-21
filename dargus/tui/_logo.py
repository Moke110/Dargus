"""Dargus - Clinical Efficacy Prediction System ASCII logo.

15-line hand-built block logo rendered with rich.
Layout:
  lines 0-5 :  D (top 6)    + gap + "RUG"   (thin outlined, grey70 dim)
  lines 6-14:  D (bottom 9) + gap + indent + "ARGUS" (thick filled, white bold)
"""

from rich.style import Style
from rich.text import Text

# ---------------------------------------------------------------- styles
STYLE_D = Style(color="white", bold=True)
STYLE_ARGUS = Style(color="white", bold=True)
STYLE_RUG = Style(color="grey70", dim=True)
STYLE_TAGLINE = Style(color="grey70", italic=True)

GAP = "   "  # gap between D and the word blocks
INDENT = ""  # no indent: D->A spacing equals D->R spacing
SPACING = "  "  # letter spacing inside "ARGUS"

# ------------------------------------------------------- D: 15 lines x 16 cols, thick filled
# straight left edge, large-radius (4-row) rounded right corners
D = [
    "█████████╗       ",
    "████████████╗    ",
    "██████████████╗  ",
    "███╔════════███╗",
    "███║        ███║",
    "███║        ███║",
    "███║        ███║",
    "███║        ███║",
    "███║        ███║",
    "███║        ███║",
    "███║        ███║",
    "███╚════════███╝",
    "██████████████╝  ",
    "████████████╝    ",
    "█████████╝       ",
]

# ------------------------------------------------- small letters ("RUG"): 6 lines x 7 cols, thin
R_SMALL = [
    "╔═══╗  ",
    "║   ║  ",
    "╠═══╝  ",
    "║ ╚╗   ",
    "║  ╚╗  ",
    "╚   ╚═ ",
]

U_SMALL = [
    "║   ║  ",
    "║   ║  ",
    "║   ║  ",
    "║   ║  ",
    "║   ║  ",
    "╚═══╝  ",
]

G_SMALL = [
    "╔═══╗  ",
    "║   ╝  ",
    "║      ",
    "║  ═╗  ",
    "║   ║  ",
    "╚═══╝  ",
]

# ------------------------------------------------- tall letters ("ARGUS"): 9 lines x 9 cols, thick
A_TALL = [
    "  █████  ",
    " ███████ ",
    "███   ███",
    "██║ ● ██║",  # the pupil - Argus, the hundred-eyed giant
    "██║   ██║",
    "█████████",
    "██║   ██║",
    "██║   ██║",
    "███   ███",
]

R_TALL = [
    " ███████╗",
    "███╔═══██",
    "██║   ██║",
    "██║   ██║",
    "██╚═══██║",
    "██████╔╝ ",
    "██║  ██║ ",
    "██║   ██║",
    "███   ███",
]

G_TALL = [
    " ███████╗",
    "███╔═══██",
    "██║   ╚═╝",
    "██║      ",
    "██║  ████",
    "██║   ██║",
    "██║   ██║",
    "██╚═══██║",
    " ╚═════╝ ",
]

U_TALL = [
    "██║   ██║",
    "██║   ██║",
    "██║   ██║",
    "██║   ██║",
    "██║   ██║",
    "██║   ██║",
    "██║   ██║",
    "██╚═══╝██",
    " ╚═════╝ ",
]

S_TALL = [
    " ███████╗",
    "███╔═══██",
    "██║   ╚═╝",
    "██╚═══╗  ",
    " ╚██████╗",
    "  ╚═══██║",
    "╔═╗   ██║",
    "██╔═══██║",
    " ╚═════╝ ",
]

TAGLINE = " Clinical Efficacy Prediction System"


# ---------------------------------------------------------------- assembly
def build_logo() -> list[Text]:
    rug = [r + u + g for r, u, g in zip(R_SMALL, U_SMALL, G_SMALL)]
    argus = [SPACING.join(chars) for chars in zip(A_TALL, R_TALL, G_TALL, U_TALL, S_TALL)]

    lines: list[Text] = []
    for i in range(15):
        line = Text()
        line.append(D[i].ljust(17), style=STYLE_D)
        if i < 6:  # "RUG", top-aligned with D
            line.append(GAP + rug[i], style=STYLE_RUG)
        else:  # "ARGUS", bottom-aligned with D
            line.append(GAP + INDENT + argus[i - 6], style=STYLE_ARGUS)
        lines.append(line)
    # Pad all lines to uniform width
    max_w = max(len(line.plain) for line in lines)
    for line in lines:
        pad = max_w - len(line.plain)
        if pad:
            line.append(" " * pad)
    return lines
