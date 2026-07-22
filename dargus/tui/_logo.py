"""Dargus - Clinical Efficacy Prediction System ASCII logo.

Single-row "DARGUS" wordmark rendered with rich.
  D:     8 lines x 10 cols, thick filled, white bold
  ARGUS: 6 lines x 7 cols each, thick filled, white bold, baseline-aligned with D
Every letter carries a hollow ring pupil (○, the eye of Argus),
each at a different relative position inside its letter.
"""

from rich.style import Style
from rich.text import Text

# ---------------------------------------------------------------- styles
STYLE_D = Style(color="white", bold=True)
STYLE_ARGUS = Style(color="white", bold=True)
STYLE_TAGLINE = Style(color="grey70", italic=True)

GAP = "  "  # gap between D and A (same as letter spacing)
SPACING = "  "  # letter spacing inside "ARGUS"

# ------------------------------------------------------- D: 8 lines x 10 cols, thick filled
# straight left edge, rounded right corners; pupil: upper-left of the counter
D = [
    "█████╗    ",
    "████████╗ ",
    "██╔════██╗",
    "██║○   ██║",
    "██║    ██║",
    "██╚════██║",
    "████████╝ ",
    "█████╝    ",
]

# ------------------------------------------------- "ARGUS": 6 lines x 7 cols, thick filled
# pupil positions:
#   A: bottom-right of the enclosed counter (above the crossbar)
#   R: upper-right of the bowl counter
#   G: interior whitespace, upper-left
#   U: interior whitespace, left side, vertically centered
#   S: interior whitespace of the lower half, upper-right
A = [
    " █████ ",
    "██   ██",
    "██  ○██",
    "███████",
    "██   ██",
    "██   ██",
]

R = [
    "█████╗ ",
    "██  ○██",
    "██   ██",
    "██████╝",
    "██  ██ ",
    "██   ██",
]

G = [
    " █████╗",
    "██   ═╝",
    "██○    ",
    "██  ███",
    "██   ██",
    " █████╝",
]

U = [
    "██   ██",
    "██   ██",
    "██   ██",
    "██   ██",
    "██○  ██",
    "╚█████╝",
]

S = [
    " █████╗",
    "██     ",
    "╚█████╗",
    "    ○██",
    "██   ██",
    "╚█████╝",
]

TAGLINE = " Full-stack Biomedicine Database and Clinical Efficacy Prediction System"


# ---------------------------------------------------------------- assembly
def build_logo() -> list[Text]:
    argus = [SPACING.join(chars) for chars in zip(A, R, G, U, S)]
    offset = len(D) - len(A)  # baseline-align ARGUS with D

    lines: list[Text] = []
    for i in range(len(D)):
        line = Text()
        line.append(D[i], style=STYLE_D)
        if i >= offset:
            line.append(GAP + argus[i - offset], style=STYLE_ARGUS)
        lines.append(line)
    return lines
