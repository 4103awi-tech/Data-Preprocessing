"""
Data Preprocessing Studio
-------------------------
A desktop app (PySide6 + pandas + scikit-learn + matplotlib) for exploring and
cleaning tabular data: missing values, duplicates, encoding, scaling, outliers,
train/test split and export.

Run:  python preprocessing_gui.py
"""
import base64
import io
import math
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import (QColor, QIcon, QLinearGradient, QPainter, QPen,
                           QPixmap, QPolygonF, QRadialGradient, QKeySequence, QShortcut)
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox,
    QDoubleSpinBox, QFileDialog, QFrame, QGraphicsDropShadowEffect, QHBoxLayout,
    QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder, MaxAbsScaler, MinMaxScaler, OrdinalEncoder, RobustScaler,
    StandardScaler,
)

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
BG = "#020814"
CARD = "#071224"
CARD2 = "#0b1830"
BORDER = "#142a4a"
ACCENT = "#1e6bff"
TEXT = "#e8eefc"
MUTED = "#7a8db0"
PALETTE = ["#1e6bff", "#19c15a", "#8a4dff", "#ff8a00", "#12b5c9",
           "#5ad12a", "#7a3dff", "#00c2ff", "#ffb300"]

PLOT_BG = "#061022"

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SCALE = 1.0


def S(v):
    """Scale a pixel size by the current UI scale."""
    return int(round(v * SCALE))


def build_qss(scale=1.0):
    S = lambda v: int(round(v * scale))
    return f"""
QMainWindow {{ background:#01050e; }}
QWidget#root {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:1,
                stop:0 #01040c, stop:0.55 #030a1a, stop:1 #061228); }}
QWidget {{ color:{TEXT}; font-family:'Segoe UI','Inter',sans-serif; font-size:{S(13)}px; }}
QLabel {{ background:transparent; }}
QFrame#card {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #08162e, stop:1 #050e1d);
               border:1px solid #142a4a; border-radius:16px; }}
QFrame#sidebar {{ background:transparent; }}
QLabel#title {{ font-size:{S(30)}px; font-weight:700; }}
QLabel#accent {{ font-size:{S(30)}px; font-weight:700; color:#27a3ff; }}
QLabel#subtitle {{ color:#a0b2d4; font-size:{S(14)}px; }}
QLabel#cardtitle {{ font-size:{S(19)}px; font-weight:700; }}
QLabel#muted {{ color:{MUTED}; }}
QLabel#ok {{ color:#3ddc84; font-weight:600; }}
QLabel#statvalue {{ font-size:{S(19)}px; font-weight:700; }}
QPushButton {{ background:{CARD2}; border:1px solid {BORDER}; border-radius:10px;
               padding:{S(9)}px {S(16)}px; font-weight:600; }}
QPushButton:hover {{ border-color:#3d8bff; }}
QPushButton:disabled {{ color:#3a4a6a; }}
QPushButton#nav {{ text-align:left; border:none; background:transparent;
                   padding:{S(11)}px {S(14)}px; font-size:{S(14)}px; color:#b0c0de; border-radius:10px; }}
QPushButton#nav:checked {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1e6bff, stop:1 #2a86ff);
                           color:white; }}
QPushButton#nav:hover:!checked {{ background:#0a1a38; }}
QPushButton#tab {{ padding:{S(12)}px {S(8)}px; font-size:{S(14)}px; border-radius:12px;
                   background:#08162e; border:1px solid #142a4a; }}
QPushButton#tab:checked {{ background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1e6bff, stop:1 #2a86ff);
                           border-color:#4a8bff; color:white; }}
QPushButton#excel {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #17944a, stop:1 #0e6b33);
                     border:1px solid #25d366; font-size:{S(15)}px; padding:{S(13)}px {S(22)}px; }}
QPushButton#csv {{ background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #2a7bff, stop:1 #1555d6);
                   border:1px solid #5a9aff; font-size:{S(15)}px; padding:{S(13)}px {S(22)}px; }}
QPushButton#blue {{ background:#1e6bff; border-color:#4a8bff; }}
QPushButton#teal {{ background:#0e8fa8; border-color:#12b5c9; }}
QPushButton#purple {{ background:#6a35d8; border-color:#8a4dff; }}
QPushButton#green {{ background:#19a64a; border-color:#25d366; }}
QPushButton#toggle {{ background:#08162e; }}
QPushButton#toggle:checked {{ background:#1e6bff; border-color:#4a8bff; }}
QTableWidget {{ background:#040e1f; alternate-background-color:#07152e; gridline-color:#102040;
               border:1px solid {BORDER}; border-radius:8px; font-size:{S(13)}px; }}
QHeaderView::section {{ background:#0c1d3d; color:{TEXT}; padding:{S(5)}px; border:none;
                       border-right:1px solid {BORDER}; font-weight:600; }}
QListWidget, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background:{CARD2}; border:1px solid {BORDER}; border-radius:8px; padding:{S(6)}px; }}
QListWidget::item:selected {{ background:{ACCENT}; color:white; }}
QComboBox QAbstractItemView {{ background:{CARD2}; selection-background-color:{ACCENT}; }}
QScrollArea {{ border:none; background:transparent; }}
QScrollArea > QWidget > QWidget {{ background:transparent; }}
QScrollBar:vertical {{ background:#040e1f; width:{S(10)}px; border-radius:5px; }}
QScrollBar::handle:vertical {{ background:#1a3660; border-radius:5px; min-height:24px; }}
QScrollBar:horizontal {{ background:#040e1f; height:{S(10)}px; border-radius:5px; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background:none; }}
QScrollBar::handle:horizontal {{ background:#1a3660; border-radius:5px; min-width:24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
QSplitter::handle {{ background:transparent; }}
QStatusBar {{ color:{MUTED}; }}
QCheckBox::indicator {{ width:16px; height:16px; }}
"""


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fill_table(table: QTableWidget, df: pd.DataFrame, index=False,
               index_label="index", max_rows=1000):
    d = df.head(max_rows)
    if index:
        d = d.rename_axis(index_label).reset_index()
    table.clear()
    table.setRowCount(len(d))
    table.setColumnCount(len(d.columns))
    table.setHorizontalHeaderLabels([str(c) for c in d.columns])
    for r, row in enumerate(d.itertuples(index=False)):
        for c, v in enumerate(row):
            if isinstance(v, (float, np.floating)):
                txt = "NaN" if np.isnan(v) else f"{v:.4g}"
            else:
                txt = str(v)
            item = QTableWidgetItem(txt)
            item.setTextAlignment(Qt.AlignCenter)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()


def fit_table(t, rows):
    rh = int(30 * SCALE)
    t.verticalHeader().setDefaultSectionSize(rh)
    t.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    t.setFixedHeight(t.horizontalHeader().sizeHint().height() + rh * rows
                     + t.horizontalScrollBar().sizeHint().height() + 8)


def make_table(fixed_rows=None):
    t = QTableWidget()
    t.setAlternatingRowColors(True)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setStretchLastSection(True)
    t.verticalHeader().setDefaultSectionSize(int(30 * SCALE))
    if fixed_rows:
        fit_table(t, fixed_rows)
    return t


def add_shadow(widget, blur=24, offset=(0, 6), color="#000000", alpha=140):
    """Apply a soft drop shadow to a widget."""
    eff = QGraphicsDropShadowEffect(widget)
    eff.setBlurRadius(blur)
    eff.setOffset(*offset)
    c = QColor(color)
    c.setAlpha(alpha)
    eff.setColor(c)
    widget.setGraphicsEffect(eff)


# ---- vector icons (drawn with QPainter, no image files needed) ----
def _draw_icon(p, n, col):
    P = lambda *pts: QPolygonF([QPointF(x, y) for x, y in pts])
    L = lambda a, b, c, d: p.drawLine(QPointF(a, b), QPointF(c, d))
    E = lambda x, y, rx, ry=None: p.drawEllipse(QPointF(x, y), rx, rx if ry is None else ry)
    if n == "chart":
        p.setBrush(col)
        p.setPen(Qt.NoPen)
        for x, y, h in [(3, 13, 8), (10, 5, 16), (17, 9, 12)]:
            p.drawRoundedRect(QRectF(x, y, 4, h), 1, 1)
    elif n == "warning":
        p.drawPolygon(P((12, 3), (22, 20), (2, 20)))
        L(12, 9, 12, 14)
        E(12, 17, 0.4)
    elif n == "copy":
        p.drawRoundedRect(QRectF(8, 8, 13, 13), 2, 2)
        p.drawPolyline(P((5, 16), (3, 16), (3, 3), (16, 3), (16, 5)))
    elif n == "code":
        p.drawPolyline(P((8, 7), (3, 12), (8, 17)))
        p.drawPolyline(P((16, 7), (21, 12), (16, 17)))
        L(14, 5, 10, 19)
    elif n == "scale":
        L(12, 4, 12, 20)
        L(6, 20, 18, 20)
        L(4, 8, 20, 8)
        p.drawPolygon(P((4, 8), (1, 15), (7, 15)))
        p.drawPolygon(P((20, 8), (17, 15), (23, 15)))
    elif n == "target":
        E(12, 12, 9)
        E(12, 12, 5)
        E(12, 12, 1.2)
    elif n == "scissors":
        E(7, 18, 3)
        E(17, 18, 3)
        L(9, 16, 18, 4)
        L(15, 16, 6, 4)
    elif n == "download":
        L(12, 4, 12, 15)
        p.drawPolyline(P((7, 11), (12, 16), (17, 11)))
        L(5, 20, 19, 20)
    elif n == "gear":
        E(12, 12, 6)
        E(12, 12, 2.2)
        for k in range(8):
            a = k * math.pi / 4
            L(12 + 6 * math.cos(a), 12 + 6 * math.sin(a), 12 + 9 * math.cos(a), 12 + 9 * math.sin(a))
    elif n == "db":
        E(12, 6, 8, 3)
        L(4, 6, 4, 18)
        L(20, 6, 20, 18)
        p.drawArc(QRectF(4, 9, 16, 6), 180 * 16, 180 * 16)
        p.drawArc(QRectF(4, 15, 16, 6), 180 * 16, 180 * 16)
    elif n == "table":
        p.drawRoundedRect(QRectF(3, 4, 18, 16), 2, 2)
        L(3, 10, 21, 10)
        L(9, 10, 9, 20)
        L(15, 10, 15, 20)
    elif n == "file":
        p.drawPolygon(P((6, 3), (14, 3), (19, 8), (19, 21), (6, 21)))
        p.drawPolyline(P((14, 3), (14, 8), (19, 8)))
        L(9, 13, 16, 13)
        L(9, 17, 16, 17)
    elif n == "check":
        p.drawPolyline(P((5, 12), (10, 17), (19, 7)))
    elif n == "info":
        E(12, 12, 9)
        L(12, 11, 12, 17)
        E(12, 7.5, 0.5)
    elif n == "trend":
        p.drawPolyline(P((3, 18), (9, 11), (13, 14), (21, 5)))
        p.drawPolyline(P((15, 5), (21, 5), (21, 11)))
    elif n == "eye":
        p.drawEllipse(QRectF(2.5, 6.5, 19, 11))
        E(12, 12, 3)
    elif n == "cube":
        p.drawPolygon(P((12, 3), (21, 8), (21, 16), (12, 21), (3, 16), (3, 8)))
        p.drawPolyline(P((3, 8), (12, 13), (21, 8)))
        L(12, 13, 12, 21)


def make_pixmap(name, color=TEXT, size=22):
    S = size * 2
    pm = QPixmap(S, S)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.scale(S / 24, S / 24)
    pen = QPen(QColor(color), 1.9)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    p.setBrush(Qt.NoBrush)
    _draw_icon(p, name, QColor(color))
    p.end()
    pm.setDevicePixelRatio(2.0)
    return pm


def make_icon(name, color=TEXT, size=22):
    return QIcon(make_pixmap(name, color, size))


def badge(icon, c1, c2, size=46):
    """Rounded gradient square with a white icon, used on stat cards."""
    S = size * 2
    pm = QPixmap(S, S)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    g = QLinearGradient(0, 0, S, S)
    g.setColorAt(0, QColor(c1))
    g.setColorAt(1, QColor(c2))
    p.setBrush(g)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(QRectF(0, 0, S, S), 22, 22)
    ip = make_pixmap(icon, "#ffffff", int(size * 0.55))
    ip.setDevicePixelRatio(1.0)
    off = (S - ip.width()) // 2
    p.drawPixmap(off, off, ip)
    p.end()
    pm.setDevicePixelRatio(2.0)
    lbl = QLabel()
    lbl.setPixmap(pm)
    lbl.setFixedSize(size, size)
    return lbl


class RobotWidget(QWidget):
    """Glowing robot mascot. full=False -> head only (top-left logo),
    full=True -> head + body with ground glow (bottom-left mascot)."""

    def __init__(self, full=False, size=110):
        super().__init__()
        self.full = full
        self.pm = QPixmap(os.path.join(ASSET_DIR, "robot_full.png" if full else "robot_head.png"))
        if self.pm.isNull():
            self.pm = QPixmap()
            self.pm.loadFromData(base64.b64decode(ROBOT_FULL_B64 if full else ROBOT_HEAD_B64))
        if self.pm.isNull():
            self.pm = None
            self.H = 150 if full else 100
        else:
            self.H = 100 * self.pm.height() / self.pm.width()
        self.setFixedSize(size, int(size * self.H / 100))
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        if self.pm is not None:
            p.drawPixmap(self.rect(), self.pm)
            p.end()
            return
        p.scale(self.width() / 100, self.width() / 100)
        p.setPen(Qt.NoPen)

        def glow(cx, cy, r, color, a=110):
            g = QRadialGradient(cx, cy, r)
            c = QColor(color)
            c.setAlpha(a)
            g.setColorAt(0, c)
            c2 = QColor(color)
            c2.setAlpha(0)
            g.setColorAt(1, c2)
            p.setBrush(g)
            p.drawEllipse(QPointF(cx, cy), r, r)

        # big soft aura behind the robot
        glow(50, 48 if not self.full else 62, 58 if not self.full else 66, "#2b8cff", 120)

        # ground glow + light line (full only)
        if self.full:
            g = QRadialGradient(50, 138, 40)
            g.setColorAt(0, QColor(60, 190, 255, 170))
            g.setColorAt(1, QColor(60, 190, 255, 0))
            p.setBrush(g)
            p.drawEllipse(QPointF(50, 138), 40, 7)

        # headphone band (behind head)
        pen = QPen(QColor("#1e6bff"), 3.2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(16, 16, 68, 60), 0, 180 * 16)
        p.setPen(Qt.NoPen)

        # body (full only)
        if self.full:
            g = QLinearGradient(0, 78, 0, 120)
            g.setColorAt(0, QColor("#ffffff"))
            g.setColorAt(1, QColor("#9fb4e0"))
            p.setBrush(g)
            p.drawRoundedRect(QRectF(31, 76, 38, 42), 14, 14)
            p.setBrush(QColor("#dfe8fa"))
            p.drawRoundedRect(QRectF(19, 82, 10, 28), 5, 5)
            p.drawRoundedRect(QRectF(71, 82, 10, 28), 5, 5)
            p.setBrush(QColor("#1e6bff"))
            p.drawEllipse(QPointF(24, 112), 5, 5)
            p.drawEllipse(QPointF(76, 112), 5, 5)
            glow(50, 96, 11, "#35e0ff", 200)
            p.setBrush(QColor("#7ff0ff"))
            p.drawEllipse(QPointF(50, 96), 4, 4)

        # antenna
        pen = QPen(QColor("#9fb4e0"), 2)
        p.setPen(pen)
        p.drawLine(QPointF(50, 28), QPointF(50, 19))
        p.setPen(Qt.NoPen)
        glow(50, 16, 8, "#35e0ff", 200)
        p.setBrush(QColor("#7ff0ff"))
        p.drawEllipse(QPointF(50, 16), 3.2, 3.2)

        # head
        g = QLinearGradient(0, 28, 0, 72)
        g.setColorAt(0, QColor("#ffffff"))
        g.setColorAt(1, QColor("#b3c4e8"))
        p.setBrush(g)
        p.drawRoundedRect(QRectF(22, 28, 56, 44), 16, 16)

        # ear cups
        p.setBrush(QColor("#1e6bff"))
        p.drawRoundedRect(QRectF(12, 40, 11, 24), 5, 5)
        p.drawRoundedRect(QRectF(77, 40, 11, 24), 5, 5)

        # visor + glowing eyes
        p.setBrush(QColor("#0a1530"))
        p.drawRoundedRect(QRectF(28, 36, 44, 27), 11, 11)
        for ex in (41, 59):
            glow(ex, 49, 9, "#35e0ff", 170)
            p.setBrush(QColor("#6ff0ff"))
            p.drawEllipse(QPointF(ex, 49), 4.6, 4.6)
        p.end()


def card(title=None, subtitle=None, icon=None, icon_color="#27a3ff"):
    frame = QFrame()
    frame.setObjectName("card")
    add_shadow(frame, blur=32, offset=(0, 8), alpha=120)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(18, 16, 18, 16)
    lay.setSpacing(10)
    frame.head = None
    if title:
        head = QHBoxLayout()
        head.setSpacing(12)
        if icon:
            il = QLabel()
            il.setPixmap(make_pixmap(icon, icon_color, S(40)))
            il.setFixedSize(S(44), S(44))
            head.addWidget(il)
        tv = QVBoxLayout()
        tv.setSpacing(2)
        t = QLabel(title)
        t.setObjectName("cardtitle")
        tv.addWidget(t)
        if subtitle:
            s_ = QLabel(subtitle)
            s_.setObjectName("muted")
            s_.setWordWrap(True)
            tv.addWidget(s_)
        head.addLayout(tv, 1)
        lay.addLayout(head)
        frame.head = head
    return frame, lay


class CircuitWidget(QWidget):
    """Faint circuit-board line that decorates the header."""

    def __init__(self):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setMinimumHeight(60)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(QPen(QColor(39, 163, 255, 80), 1.4))
        pts = [(0, .62), (.18, .62), (.26, .30), (.62, .30), (.70, .62), (1, .62)]
        p.drawPolyline(QPolygonF([QPointF(x * w, y * h) for x, y in pts]))
        p.setPen(QPen(QColor(39, 163, 255, 45), 1))
        p.drawLine(QPointF(w * .30, h * .74), QPointF(w * .82, h * .74))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(90, 205, 255, 170))
        for x, y in [(.26, .30), (.62, .30), (.45, .74)]:
            p.drawEllipse(QPointF(x * w, y * h), 3, 3)
        p.end()


def styled_axes(ax):
    ax.set_facecolor(PLOT_BG)
    ax.tick_params(colors=TEXT, labelsize=9 * SCALE, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.get_offset_text().set_color(MUTED)
    ax.yaxis.get_offset_text().set_color(MUTED)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(alpha=0.1, color="#1a3660", linestyle="--", linewidth=0.6)


def short(name, n=22):
    """Shorten long column names and wrap at the underscore nearest the middle."""
    name = str(name)
    if len(name) > 14 and "_" in name:
        idx = [i for i, ch in enumerate(name) if ch == "_"]
        k = min(idx, key=lambda i: abs(i - len(name) / 2))
        name = name[:k + 1] + "\n" + name[k + 1:]
    return name if len(name) <= n + 1 else name[:n] + "…"


def draw_distributions(fig: Figure, df: pd.DataFrame, mode="hist", cols=None, limit=2, start=0):
    """Draws 2 distributions per page, vertically stacked."""
    fig.clear()
    fig.set_facecolor(CARD)
    cols = cols if cols is not None else df.select_dtypes(include=np.number).columns.tolist()
    cols = cols[start:start + limit]
    if not cols:
        fig.text(0.5, 0.5, "No numeric columns to plot", ha="center", color=MUTED)
        return 0

    nrows = len(cols)
    for i, c in enumerate(cols):
        ax = fig.add_subplot(nrows, 1, i + 1)
        styled_axes(ax)
        data = df[c].dropna()
        color = PALETTE[(start + i) % len(PALETTE)]

        if mode == "hist":
            counts, bins, patches = ax.hist(data, bins=30, color=color, edgecolor=PLOT_BG,
                                            linewidth=0.4, alpha=0.95)
            ax.hist(data, bins=30, color=color, edgecolor="none", alpha=0.15,
                    histtype="stepfilled")
        else:
            ax.boxplot(data, vert=True, patch_artist=True, widths=0.45,
                       boxprops=dict(facecolor=color, color=color, linewidth=1.2, alpha=0.9),
                       medianprops=dict(color="#ffffff", linewidth=1.8),
                       whiskerprops=dict(color=TEXT, linewidth=1.0),
                       capprops=dict(color=TEXT, linewidth=1.2),
                       flierprops=dict(marker="o", markerfacecolor=color,
                                       markeredgecolor=color, markersize=3, alpha=0.35))
            ax.set_xticks([])
            ax.axhline(data.median(), color="#ffffff", linewidth=0.6, linestyle=":", alpha=0.5)

        ax.set_title(short(c), color=TEXT, fontsize=10.5 * SCALE, fontweight="bold", pad=8)

    fig.tight_layout(pad=2.5, h_pad=3.5, w_pad=2.6)

    try:
        r = fig.canvas.get_renderer()
        inv = fig.transFigure.inverted()
        for ax in fig.axes:
            bb = ax.get_tightbbox(r).transformed(inv)
            pad = 0.015
            fig.add_artist(FancyBboxPatch(
                (bb.x0 - pad * 1.6, bb.y0 - pad * 1.6),
                bb.width + 3.2 * pad, bb.height + 3.2 * pad,
                boxstyle="round,pad=0,rounding_size=0.025", transform=fig.transFigure,
                fc=PLOT_BG, ec="#1e6bff", lw=0, alpha=0.18, zorder=-2))
            fig.add_artist(FancyBboxPatch(
                (bb.x0 - pad, bb.y0 - pad), bb.width + 2 * pad, bb.height + 2 * pad,
                boxstyle="round,pad=0,rounding_size=0.02", transform=fig.transFigure,
                fc=PLOT_BG, ec="#1a3660", lw=1.2, zorder=-1))
    except Exception:
        pass
    return len(cols)


class PickerBox(QWidget):
    """Multi-select list of columns with Select all / Clear buttons."""

    def __init__(self, label="Columns"):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(label))
        self.list = QListWidget()
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        a = QPushButton("Select all")
        n = QPushButton("Clear")
        a.clicked.connect(self.list.selectAll)
        n.clicked.connect(self.list.clearSelection)
        row.addWidget(a)
        row.addWidget(n)
        lay.addLayout(row)

    def set_columns(self, cols):
        keep = set(self.selected())
        self.list.clear()
        self.list.addItems([str(c) for c in cols])
        for i in range(self.list.count()):
            if self.list.item(i).text() in keep:
                self.list.item(i).setSelected(True)

    def selected(self):
        return [i.text() for i in self.list.selectedItems()]


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
class ExplorePage(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.mode = "hist"
        self.page = 0
        self._rt = QTimer(self)
        self._rt.setSingleShot(True)
        self._rt.timeout.connect(self.refresh_plot)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(16)

        left, ll = card("Data Overview", "Preview structure, statistics and column distributions",
                        icon="db")
        btns = QHBoxLayout()
        btns.setSpacing(8)
        for icon, text, name, slot in [
            ("eye", "Preview", "blue", lambda: self.view.setCurrentIndex(0)),
            ("info", "Info", "teal", self.show_info),
            ("chart", "Statistics", "purple", self.show_stats),
            ("cube", "Data Types", "purple", self.show_dtypes),
            ("chart", "Histograms", "green", lambda: self.set_mode("hist")),
        ]:
            b = QPushButton(text)
            b.setObjectName(name)
            b.setIcon(make_icon(icon, "#ffffff", S(16)))
            b.setIconSize(QSize(S(16), S(16)))
            b.setStyleSheet(f"padding:{S(7)}px {S(8)}px;")
            b.clicked.connect(slot)
            btns.addWidget(b)
        ll.addLayout(btns)

        self.view = QStackedWidget()
        pv = QScrollArea()
        pv.setWidgetResizable(True)
        pv.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        pl = QVBoxLayout(inner)
        pl.setContentsMargins(0, 0, 6, 0)
        pl.setSpacing(6)
        pl.addWidget(self._h("First 5 Rows", "table"))
        self.head_t = make_table(5)
        pl.addWidget(self.head_t)
        pl.addSpacing(8)
        pl.addWidget(self._h("Last 5 Rows", "info"))
        self.tail_t = make_table(5)
        pl.addWidget(self.tail_t)
        pl.addStretch()
        pv.setWidget(inner)
        self.view.addWidget(pv)
        self.info_txt = QPlainTextEdit()
        self.info_txt.setReadOnly(True)
        self.info_txt.setStyleSheet("font-family:Consolas,monospace;")
        self.view.addWidget(self.info_txt)
        self.gen_t = make_table()
        self.view.addWidget(self.gen_t)
        ll.addWidget(self.view, 1)
        left.setMinimumWidth(S(380))

        right, rl = card("Numeric Feature Distributions", "Visual insights into your data",
                         icon="chart")
        self.b_hist = QPushButton("Histograms")
        self.b_box = QPushButton("Box Plots")
        for b, ic in ((self.b_hist, "chart"), (self.b_box, "table")):
            b.setObjectName("toggle")
            b.setCheckable(True)
            b.setIcon(make_icon(ic, "#ffffff", S(16)))
            b.setIconSize(QSize(S(16), S(16)))
            right.head.addWidget(b)
        self.b_hist.setChecked(True)
        self.b_hist.clicked.connect(lambda: self.set_mode("hist"))
        self.b_box.clicked.connect(lambda: self.set_mode("box"))

        self.fig = Figure(figsize=(6, 6), facecolor=CARD)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background:transparent;")
        self.canvas.setMinimumHeight(S(300))
        rl.addWidget(self.canvas, 1)

        bottom = QHBoxLayout()
        self.plot_lbl = QLabel("")
        self.plot_lbl.setObjectName("ok")
        bottom.addWidget(self.plot_lbl, 1)
        self.prev = QPushButton("◀")
        self.next = QPushButton("▶")
        self.page_lbl = QLabel("")
        self.page_lbl.setObjectName("muted")
        for b in (self.prev, self.next):
            b.setFixedWidth(S(40))
        self.prev.clicked.connect(lambda: self.change_page(-1))
        self.next.clicked.connect(lambda: self.change_page(1))
        bottom.addWidget(self.prev)
        bottom.addWidget(self.page_lbl)
        bottom.addWidget(self.next)
        rl.addLayout(bottom)
        right.setMinimumWidth(S(420))
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(12)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 6)
        split.splitterMoved.connect(lambda *_: self._rt.start(150))
        root.addWidget(split)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rt.start(150)

    @staticmethod
    def _h(text, icon):
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        i = QLabel()
        i.setPixmap(make_pixmap(icon, "#27a3ff", S(20)))
        i.setFixedSize(S(22), S(22))
        h.addWidget(i)
        l = QLabel(text)
        l.setStyleSheet("font-size:15px;font-weight:700;")
        h.addWidget(l)
        h.addStretch()
        return w

    def set_mode(self, mode):
        self.mode = mode
        self.b_hist.setChecked(mode == "hist")
        self.b_box.setChecked(mode == "box")
        self.refresh_plot()

    def show_info(self):
        buf = io.StringIO()
        self.app.df.info(buf=buf)
        self.info_txt.setPlainText(buf.getvalue())
        self.view.setCurrentIndex(1)

    def show_stats(self):
        df = self.app.df
        if df.empty:
            return
        fill_table(self.gen_t, df.describe(include="all").T, index=True, index_label="column")
        self.view.setCurrentIndex(2)

    def show_dtypes(self):
        df = self.app.df
        d = pd.DataFrame({"dtype": df.dtypes.astype(str), "non-null": df.notna().sum(),
                          "unique": df.nunique()})
        fill_table(self.gen_t, d, index=True, index_label="column")
        self.view.setCurrentIndex(2)

    def change_page(self, d):
        self.page += d
        self.refresh_plot()

    def refresh_plot(self):
        per = 2
        total = len(self.app.df.select_dtypes(include=np.number).columns)
        maxpage = max((total - 1) // per, 0)
        self.page = max(0, min(self.page, maxpage))
        n = draw_distributions(self.fig, self.app.df, self.mode,
                               start=self.page * per, limit=per)
        self.canvas.draw_idle()
        self.page_lbl.setText(f"{self.page * per + 1}–{self.page * per + n} of {total}" if n else "")
        self.prev.setEnabled(self.page > 0)
        self.next.setEnabled(self.page < maxpage)
        self.plot_lbl.setText(f"✔ Plotted {n} numeric distributions" if n else "")

    def refresh(self):
        df = self.app.df
        fill_table(self.head_t, df.head(5), index=True)
        fill_table(self.tail_t, df.tail(5), index=True)
        if self.view.currentIndex() == 2:
            self.show_dtypes()
        self.refresh_plot()


class MissingPage(QWidget):
    STRATS = ["Drop rows with missing", "Drop columns with missing", "Fill with mean",
              "Fill with median", "Fill with mode", "Fill with constant",
              "Forward fill", "Backward fill"]

    def __init__(self, app):
        super().__init__()
        self.app = app
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        left, ll = card("Missing Values", "Missing count and percentage per column")
        self.table = make_table()
        ll.addWidget(self.table)
        root.addWidget(left, 6)

        right, rl = card("Handle Missing Values", "Select columns (none selected = all columns with missing values)")
        self.picker = PickerBox("Columns")
        rl.addWidget(self.picker, 1)
        rl.addWidget(QLabel("Strategy"))
        self.strat = QComboBox()
        self.strat.addItems(self.STRATS)
        rl.addWidget(self.strat)
        self.const = QLineEdit()
        self.const.setPlaceholderText("Constant value (for 'Fill with constant')")
        rl.addWidget(self.const)
        b = QPushButton("Apply")
        b.setObjectName("blue")
        b.clicked.connect(self.apply)
        rl.addWidget(b)
        root.addWidget(right, 4)

    def refresh(self):
        df = self.app.df
        d = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                          "missing %": (df.isna().mean() * 100).round(2)})
        fill_table(self.table, d, index=True, index_label="column")
        self.picker.set_columns(df.columns)

    def apply(self):
        df = self.app.df.copy()
        sel = self.picker.selected() or [c for c in df.columns if df[c].isna().any()]
        sel = [c for c in sel if df[c].isna().any()]
        if not sel:
            self.app.toast("No missing values in the selected columns.")
            return
        s = self.strat.currentText()
        skipped = []
        if s.startswith("Drop rows"):
            df = df.dropna(subset=sel)
        elif s.startswith("Drop columns"):
            df = df.drop(columns=sel)
        else:
            for c in sel:
                if s == "Fill with mean" or s == "Fill with median":
                    if not pd.api.types.is_numeric_dtype(df[c]):
                        skipped.append(c)
                        continue
                    df[c] = df[c].fillna(df[c].mean() if "mean" in s else df[c].median())
                elif s == "Fill with mode":
                    m = df[c].mode()
                    if not m.empty:
                        df[c] = df[c].fillna(m.iloc[0])
                elif s == "Fill with constant":
                    v = self.const.text()
                    if pd.api.types.is_numeric_dtype(df[c]):
                        try:
                            v = float(v)
                        except ValueError:
                            skipped.append(c)
                            continue
                    df[c] = df[c].fillna(v)
                elif s == "Forward fill":
                    df[c] = df[c].ffill()
                elif s == "Backward fill":
                    df[c] = df[c].bfill()
        msg = f"Missing values handled: {s}"
        if skipped:
            msg += f" (skipped non-applicable: {', '.join(skipped)})"
        self.app.commit(df, msg)


class DuplicatesPage(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        left, ll = card("Duplicate Rows", "Rows that repeat across the selected columns")
        self.count = QLabel("")
        self.count.setObjectName("ok")
        ll.addWidget(self.count)
        self.table = make_table()
        ll.addWidget(self.table, 1)
        root.addWidget(left, 6)

        right, rl = card("Remove Duplicates", "Select columns to compare (none = all columns)")
        self.picker = PickerBox("Subset columns")
        rl.addWidget(self.picker, 1)
        rl.addWidget(QLabel("Keep"))
        self.keep = QComboBox()
        self.keep.addItems(["first", "last", "none (drop all duplicates)"])
        rl.addWidget(self.keep)
        f = QPushButton("Find duplicates")
        f.setObjectName("teal")
        f.clicked.connect(self.find)
        r = QPushButton("Remove duplicates")
        r.setObjectName("blue")
        r.clicked.connect(self.remove)
        rl.addWidget(f)
        rl.addWidget(r)
        root.addWidget(right, 4)

    def _kw(self):
        k = self.keep.currentText().split()[0]
        return dict(subset=self.picker.selected() or None, keep=False if k == "none" else k)

    def find(self):
        df = self.app.df
        mask = df.duplicated(subset=self._kw()["subset"], keep=False)
        self.count.setText(f"{int(df.duplicated(**self._kw()).sum())} duplicate rows "
                           f"({int(mask.sum())} rows involved)")
        fill_table(self.table, df[mask].sort_values(by=self._kw()["subset"] or list(df.columns)),
                   index=True)

    def remove(self):
        df = self.app.df
        n = len(df)
        out = df.drop_duplicates(**self._kw())
        self.app.commit(out, f"Removed {n - len(out)} duplicate rows")

    def refresh(self):
        self.picker.set_columns(self.app.df.columns)
        self.count.setText(f"{int(self.app.df.duplicated().sum())} fully duplicated rows")
        self.table.setRowCount(0)


class EncodingPage(QWidget):
    METHODS = ["Label Encoding", "Ordinal Encoding", "One-Hot Encoding", "Frequency Encoding"]

    def __init__(self, app):
        super().__init__()
        self.app = app
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        # ---- LEFT: stacked summary + live preview
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(14)

        sum_card, sl = card("Categorical Columns", "Cardinality of non-numeric columns")
        self.table = make_table()
        sl.addWidget(self.table)
        lv.addWidget(sum_card)

        prev_card, pl = card("Data Preview", "Live preview of the current dataset (top 50 rows)")
        self.preview = make_table()
        pl.addWidget(self.preview, 1)
        lv.addWidget(prev_card, 1)

        root.addWidget(left, 6)

        # ---- RIGHT: options
        right, rl = card("Encode Categorical Data", "Select the columns to encode")
        self.picker = PickerBox("Categorical columns")
        rl.addWidget(self.picker, 1)
        rl.addWidget(QLabel("Method"))
        self.method = QComboBox()
        self.method.addItems(self.METHODS)
        rl.addWidget(self.method)
        self.drop_first = QCheckBox("One-Hot: drop first category")
        rl.addWidget(self.drop_first)
        b = QPushButton("Apply")
        b.setObjectName("blue")
        b.clicked.connect(self.apply)
        rl.addWidget(b)
        root.addWidget(right, 4)

    def cat_cols(self):
        return self.app.df.select_dtypes(exclude=np.number).columns.tolist()

    def refresh(self):
        df = self.app.df
        cc = self.cat_cols()
        d = pd.DataFrame({"dtype": df[cc].dtypes.astype(str), "unique": df[cc].nunique()}) \
            if cc else pd.DataFrame(columns=["dtype", "unique"])
        fill_table(self.table, d, index=True, index_label="column")
        self.picker.set_columns(cc)
        # live preview of the current dataset
        fill_table(self.preview, df.head(50), index=True)

    def apply(self):
        cols = self.picker.selected()
        if not cols:
            self.app.toast("Select at least one column.")
            return
        df = self.app.df.copy()
        m = self.method.currentText()
        if m == "Label Encoding":
            for c in cols:
                df[c] = LabelEncoder().fit_transform(df[c].astype(str))
        elif m == "Ordinal Encoding":
            df[cols] = OrdinalEncoder().fit_transform(df[cols].astype(str))
        elif m == "One-Hot Encoding":
            df = pd.get_dummies(df, columns=cols, drop_first=self.drop_first.isChecked(), dtype=int)
        else:
            for c in cols:
                df[c] = df[c].map(df[c].value_counts(normalize=True))
        self.app.commit(df, f"{m} applied to {len(cols)} column(s)")

        # jump the preview to the first affected column
        first = str(cols[0])
        for c in range(self.preview.columnCount()):
            it = self.preview.horizontalHeaderItem(c)
            if it and it.text() == first:
                self.preview.scrollToItem(self.preview.item(0, c),
                                          QAbstractItemView.PositionAtCenter)
                break


class ScalingPage(QWidget):
    METHODS = {"Standard (Z-score)": StandardScaler, "Min-Max (0-1)": MinMaxScaler,
               "Robust (median/IQR)": RobustScaler, "Max-Abs": MaxAbsScaler}

    def __init__(self, app):
        super().__init__()
        self.app = app
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        # ---- LEFT: statistics + live preview
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(14)

        stat_card, sl = card("Numeric Statistics", "Mean, spread and range before/after scaling")
        self.table = make_table()
        sl.addWidget(self.table)
        lv.addWidget(stat_card)

        prev_card, pl = card("Data Preview", "Live preview of the current dataset (top 50 rows)")
        self.preview = make_table()
        pl.addWidget(self.preview, 1)
        lv.addWidget(prev_card, 1)

        root.addWidget(left, 6)

        right, rl = card("Feature Scaling", "Select the numeric columns to scale")
        self.picker = PickerBox("Numeric columns")
        rl.addWidget(self.picker, 1)
        rl.addWidget(QLabel("Method"))
        self.method = QComboBox()
        self.method.addItems(list(self.METHODS))
        rl.addWidget(self.method)
        b = QPushButton("Apply")
        b.setObjectName("blue")
        b.clicked.connect(self.apply)
        rl.addWidget(b)
        root.addWidget(right, 4)

    def refresh(self):
        num = self.app.df.select_dtypes(include=np.number)
        fill_table(self.table, num.describe().T[["mean", "std", "min", "max"]] if not num.empty
                   else pd.DataFrame(), index=True, index_label="column")
        self.picker.set_columns(num.columns)
        fill_table(self.preview, self.app.df.head(50), index=True)

    def apply(self):
        cols = self.picker.selected()
        if not cols:
            self.app.toast("Select at least one numeric column.")
            return
        df = self.app.df.copy()
        name = self.method.currentText()
        df[cols] = self.METHODS[name]().fit_transform(df[cols].astype(float))
        self.app.commit(df, f"{name} scaling applied to {len(cols)} column(s)")


class OutliersPage(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        # ---- LEFT: detection table + box-plot chart + preview
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(14)

        det_card, dl = card("Outlier Detection", "Detected outliers per selected column")
        self.table = make_table()
        self.table.setMaximumHeight(200)
        dl.addWidget(self.table)
        self.fig = Figure(figsize=(5, 3), facecolor=CARD)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(S(220))
        dl.addWidget(self.canvas, 1)
        lv.addWidget(det_card)

        prev_card, pl = card("Data Preview", "Live preview of the current dataset (top 50 rows)")
        self.preview = make_table()
        pl.addWidget(self.preview, 1)
        lv.addWidget(prev_card, 1)

        root.addWidget(left, 6)

        right, rl = card("Handle Outliers", "Select numeric columns")
        self.picker = PickerBox("Numeric columns")
        rl.addWidget(self.picker, 1)
        rl.addWidget(QLabel("Detection method"))
        self.method = QComboBox()
        self.method.addItems(["IQR", "Z-score"])
        self.method.currentTextChanged.connect(
            lambda m: self.k.setValue(1.5 if m == "IQR" else 3.0))
        rl.addWidget(self.method)
        rl.addWidget(QLabel("Threshold (IQR multiplier / Z)"))
        self.k = QDoubleSpinBox()
        self.k.setRange(0.5, 10)
        self.k.setSingleStep(0.1)
        self.k.setValue(1.5)
        rl.addWidget(self.k)
        rl.addWidget(QLabel("Action"))
        self.action = QComboBox()
        self.action.addItems(["Remove rows", "Cap to bounds", "Replace with median"])
        rl.addWidget(self.action)
        d = QPushButton("Detect")
        d.setObjectName("teal")
        d.clicked.connect(self.detect)
        a = QPushButton("Apply")
        a.setObjectName("blue")
        a.clicked.connect(self.apply)
        rl.addWidget(d)
        rl.addWidget(a)
        root.addWidget(right, 4)

    def bounds(self, s):
        k = self.k.value()
        if self.method.currentText() == "IQR":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            return q1 - k * iqr, q3 + k * iqr
        return s.mean() - k * s.std(), s.mean() + k * s.std()

    def detect(self):
        cols = self.picker.selected()
        if not cols:
            self.app.toast("Select at least one numeric column.")
            return
        df = self.app.df
        rows = {}
        for c in cols:
            lo, hi = self.bounds(df[c].dropna())
            n = int(((df[c] < lo) | (df[c] > hi)).sum())
            rows[c] = {"lower": lo, "upper": hi, "outliers": n,
                       "outliers %": round(n / max(len(df), 1) * 100, 2)}
        fill_table(self.table, pd.DataFrame(rows).T, index=True, index_label="column")
        draw_distributions(self.fig, df, "box", cols=cols, limit=2)
        self.canvas.draw_idle()

    def apply(self):
        cols = self.picker.selected()
        if not cols:
            self.app.toast("Select at least one numeric column.")
            return
        df = self.app.df.copy()
        act = self.action.currentText()
        mask_any = pd.Series(False, index=df.index)
        for c in cols:
            lo, hi = self.bounds(df[c].dropna())
            m = (df[c] < lo) | (df[c] > hi)
            if act == "Cap to bounds":
                df[c] = df[c].clip(lo, hi)
            elif act == "Replace with median":
                df.loc[m, c] = df[c].median()
            mask_any |= m
        if act == "Remove rows":
            n = int(mask_any.sum())
            df = df[~mask_any]
            self.app.commit(df, f"Removed {n} outlier rows")
        else:
            self.app.commit(df, f"Outliers handled ({act.lower()})")

    def refresh(self):
        self.picker.set_columns(self.app.df.select_dtypes(include=np.number).columns)
        fill_table(self.preview, self.app.df.head(50), index=True)


class SplitPage(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)
        left, ll = card("Train / Test Split", "Result of the latest split")
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        ll.addWidget(self.result, 1)
        root.addWidget(left, 6)

        right, rl = card("Split Settings")
        rl.addWidget(QLabel("Target column (optional)"))
        self.target = QComboBox()
        rl.addWidget(self.target)
        rl.addWidget(QLabel("Test size"))
        self.test = QDoubleSpinBox()
        self.test.setRange(0.05, 0.95)
        self.test.setSingleStep(0.05)
        self.test.setValue(0.2)
        rl.addWidget(self.test)
        rl.addWidget(QLabel("Random state"))
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.seed.setValue(42)
        rl.addWidget(self.seed)
        self.shuffle = QCheckBox("Shuffle")
        self.shuffle.setChecked(True)
        self.stratify = QCheckBox("Stratify by target")
        rl.addWidget(self.shuffle)
        rl.addWidget(self.stratify)
        rl.addStretch()
        b = QPushButton("Split data")
        b.setObjectName("blue")
        b.clicked.connect(self.split)
        rl.addWidget(b)
        root.addWidget(right, 4)

    def refresh(self):
        cur = self.target.currentText()
        self.target.clear()
        self.target.addItems(["(none)"] + [str(c) for c in self.app.df.columns])
        i = self.target.findText(cur)
        self.target.setCurrentIndex(max(i, 0))

    def split(self):
        df = self.app.df
        t = self.target.currentText()
        strat = df[t] if (self.stratify.isChecked() and t != "(none)") else None
        try:
            if t == "(none)":
                tr, te = train_test_split(df, test_size=self.test.value(),
                                          random_state=self.seed.value(),
                                          shuffle=self.shuffle.isChecked())
                self.app.split = {"train": tr, "test": te}
                txt = f"Train: {tr.shape}\nTest:  {te.shape}"
            else:
                X, y = df.drop(columns=[t]), df[t]
                Xtr, Xte, ytr, yte = train_test_split(
                    X, y, test_size=self.test.value(), random_state=self.seed.value(),
                    shuffle=self.shuffle.isChecked(), stratify=strat)
                self.app.split = {"train": pd.concat([Xtr, ytr], axis=1),
                                  "test": pd.concat([Xte, yte], axis=1)}
                txt = (f"X_train: {Xtr.shape}\nX_test:  {Xte.shape}\n"
                       f"y_train: {ytr.shape}\ny_test:  {yte.shape}\nTarget: {t}")
        except ValueError as e:
            QMessageBox.warning(self, "Split failed", str(e))
            return
        self.result.setPlainText(txt + "\n\nGo to the Export tab to save the splits.")
        self.app.toast("Data split successfully.")


class ExportPage(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        c, cl = card("Export Processed Data", "Save the cleaned dataset or the train/test splits")
        self.summary = QLabel("")
        self.summary.setObjectName("ok")
        cl.addWidget(self.summary)
        row = QHBoxLayout()
        for text, name, slot in [("Export CSV", "blue", self.to_csv),
                                 ("Export Excel", "green", self.to_excel),
                                 ("Export Train/Test CSVs", "purple", self.to_split)]:
            b = QPushButton(text)
            b.setObjectName(name)
            b.clicked.connect(slot)
            row.addWidget(b)
        cl.addLayout(row)
        self.table = make_table()
        cl.addWidget(self.table, 1)
        root.addWidget(c)

    def refresh(self):
        df = self.app.df
        self.summary.setText(f"Dataset ready: {df.shape[0]:,} rows × {df.shape[1]} columns")
        fill_table(self.table, df.head(50), index=True)

    def to_csv(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save CSV", "processed.csv", "CSV (*.csv)")
        if p:
            self.app.df.to_csv(p, index=False)
            self.app.toast(f"Saved {p}")

    def to_excel(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save Excel", "processed.xlsx", "Excel (*.xlsx)")
        if p:
            self.app.df.to_excel(p, index=False)
            self.app.toast(f"Saved {p}")

    def to_split(self):
        if not self.app.split:
            QMessageBox.information(self, "No split", "Create a split in the Split tab first.")
            return
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.app.split["train"].to_csv(os.path.join(d, "train.csv"), index=False)
            self.app.split["test"].to_csv(os.path.join(d, "test.csv"), index=False)
            self.app.toast(f"Saved train.csv and test.csv to {d}")


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------
class MainWindow(QMainWindow):
    NAMES = ["Explore", "Missing", "Duplicates", "Encoding", "Scaling", "Outliers", "Split", "Export"]
    ICON_NAMES = ["chart", "warning", "copy", "code", "scale", "target", "scissors", "download"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Preprocessing Studio")
        self.resize(1500, 880)
        self.setMinimumSize(900, 560)
        self.df = pd.DataFrame()
        self.original = pd.DataFrame()
        self.history = []
        self.split = None

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 12, 16, 10)
        outer.setSpacing(18)

        # ---- sidebar
        side = QFrame()
        side.setObjectName("sidebar")
        side.setFixedWidth(S(170))
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(4)
        sl.addWidget(RobotWidget(False, S(112)), 0, Qt.AlignHCenter)
        sl.addSpacing(12)
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_btns = []
        for i, (n, ic) in enumerate(zip(self.NAMES, self.ICON_NAMES)):
            b = QPushButton(n)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setIcon(make_icon(ic, "#dbe6ff", S(22)))
            b.setIconSize(QSize(S(22), S(22)))
            b.clicked.connect(lambda _=False, i=i: self.goto(i))
            self.nav_group.addButton(b)
            self.nav_btns.append(b)
            sl.addWidget(b)
        st = QPushButton("Settings")
        st.setObjectName("nav")
        st.setIcon(make_icon("gear", "#dbe6ff", S(22)))
        st.setIconSize(QSize(S(22), S(22)))
        st.clicked.connect(self.about)
        sl.addWidget(st)
        sl.addStretch()
        sl.addWidget(RobotWidget(True, S(120)), 0, Qt.AlignHCenter)
        tag = QLabel("Data Insight\nMade Easy")
        tag.setAlignment(Qt.AlignCenter)
        tag.setStyleSheet("font-size:14px;font-weight:600;color:#dbe6ff;")
        sl.addWidget(tag)
        line = QFrame()
        line.setFixedSize(S(110), S(3))
        line.setStyleSheet("background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
                           "stop:0 transparent, stop:0.5 #27a3ff, stop:1 transparent);"
                           "border-radius:1px;")
        sl.addSpacing(4)
        sl.addWidget(line, 0, Qt.AlignHCenter)
        outer.addWidget(side)

        # ---- main column
        col = QVBoxLayout()
        col.setSpacing(12)
        outer.addLayout(col, 1)

        head = QHBoxLayout()
        tl = QVBoxLayout()
        tl.setSpacing(0)
        trow = QHBoxLayout()
        trow.setSpacing(10)
        t1 = QLabel("Data Preprocessing")
        t1.setObjectName("title")
        t2 = QLabel("Studio")
        t2.setObjectName("accent")
        trow.addWidget(t1)
        trow.addWidget(t2)
        trow.addStretch()
        tl.addLayout(trow)
        sub = QLabel("Professional preprocessing & visualization for your data")
        sub.setObjectName("subtitle")
        tl.addWidget(sub)
        head.addLayout(tl, 0)
        head.addWidget(CircuitWidget(), 1)
        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.clicked.connect(self.undo)
        self.reset_btn = QPushButton("⟲ Reset")
        self.reset_btn.clicked.connect(self.reset)
        be = QPushButton(" Load Excel")
        be.setObjectName("excel")
        be.setIcon(make_icon("table", "#ffffff", S(24)))
        be.setIconSize(QSize(S(24), S(24)))
        be.clicked.connect(self.load_excel)
        bc = QPushButton(" Load CSV")
        bc.setObjectName("csv")
        bc.setIcon(make_icon("file", "#ffffff", S(24)))
        bc.setIconSize(QSize(S(24), S(24)))
        bc.clicked.connect(self.load_csv)
        for w in (self.undo_btn, self.reset_btn, be, bc):
            head.addWidget(w)
        col.addLayout(head)

        # ---- top tabs
        tabs = QHBoxLayout()
        tabs.setSpacing(10)
        self.tab_group = QButtonGroup(self)
        self.tab_group.setExclusive(True)
        self.tab_btns = []
        for i, (n, ic) in enumerate(zip(self.NAMES, self.ICON_NAMES)):
            b = QPushButton(n)
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setIcon(make_icon(ic, "#ffffff", S(22)))
            b.setIconSize(QSize(S(22), S(22)))
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, i=i: self.goto(i))
            self.tab_group.addButton(b)
            self.tab_btns.append(b)
            tabs.addWidget(b)
        col.addLayout(tabs)

        # ---- pages
        self.stack = QStackedWidget()
        self.pages = [ExplorePage(self), MissingPage(self), DuplicatesPage(self),
                      EncodingPage(self), ScalingPage(self), OutliersPage(self),
                      SplitPage(self), ExportPage(self)]
        for p in self.pages:
            sa = QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QFrame.NoFrame)
            sa.setWidget(p)
            self.stack.addWidget(sa)
        col.addWidget(self.stack, 1)

        # ---- stat cards
        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.stat_labels = {}
        for key, title, icon, c1, c2 in [
            ("rows", "Total Rows", "db", "#19c15a", "#0d8a3e"),
            ("cols", "Total Columns", "table", "#8a4dff", "#5a2bd0"),
            ("num", "Numeric Features", "check", "#19c15a", "#0d8a3e"),
            ("miss", "Missing Values", "info", "#2a86ff", "#1555d6"),
        ]:
            f, fl = card()
            add_shadow(f, blur=28, offset=(0, 6), alpha=100)
            fl.setContentsMargins(14, 10, 14, 10)
            r = QHBoxLayout()
            r.setSpacing(12)
            r.addWidget(badge(icon, c1, c2, S(40)))
            v = QVBoxLayout()
            v.setSpacing(0)
            tl_ = QLabel(title)
            tl_.setObjectName("muted")
            val = QLabel("–")
            val.setObjectName("statvalue")
            v.addWidget(tl_)
            v.addWidget(val)
            r.addLayout(v)
            r.addStretch()
            fl.addLayout(r)
            self.stat_labels[key] = val
            stats.addWidget(f, 1)
        self.ready_frame = QFrame()
        self.ready_frame.setObjectName("readycard")
        add_shadow(self.ready_frame, blur=24, offset=(0, 6), alpha=110)
        rl_ = QHBoxLayout(self.ready_frame)
        rl_.setContentsMargins(18, 10, 18, 10)
        tr = QLabel()
        tr.setPixmap(make_pixmap("trend", "#27a3ff", S(46)))
        tr.setFixedSize(S(52), S(46))
        rl_.addWidget(tr)
        self.ready = QLabel("Load a dataset to begin")
        self.ready.setAlignment(Qt.AlignCenter)
        self.ready.setWordWrap(True)
        rl_.addWidget(self.ready, 1)
        self.ready_icon = QLabel()
        self.ready_icon.setFixedSize(S(30), S(30))
        rl_.addWidget(self.ready_icon)
        stats.addWidget(self.ready_frame, 2)
        col.addLayout(stats)

        self.goto(0)
        self.update_stats()
        self.statusBar().showMessage("Ready. Load a CSV or Excel file.  (Ctrl +/- to zoom the interface)")
        for keys, fn in [("Ctrl+=", lambda: self.set_zoom(SCALE + 0.1)),
                         ("Ctrl++", lambda: self.set_zoom(SCALE + 0.1)),
                         ("Ctrl+-", lambda: self.set_zoom(SCALE - 0.1)),
                         ("Ctrl+0", lambda: self.set_zoom(1.0))]:
            QShortcut(QKeySequence(keys), self, activated=fn)

    def goto(self, i):
        self.stack.setCurrentIndex(i)
        self.nav_btns[i].setChecked(True)
        self.tab_btns[i].setChecked(True)

    def about(self):
        v, ok = QInputDialog.getInt(self, "Settings", "Interface zoom (%)  -  Ctrl +/- also works",
                                    int(round(SCALE * 100)), 80, 160, 5)
        if ok:
            self.set_zoom(v / 100)

    def set_zoom(self, z):
        global SCALE
        SCALE = max(0.8, min(1.6, round(z, 2)))
        QApplication.instance().setStyleSheet(build_qss(SCALE))
        ex = self.pages[0]
        for t in (ex.head_t, ex.tail_t):
            fit_table(t, 5)
        ex.refresh_plot()
        self.toast(f"Interface zoom {int(round(SCALE * 100))}%")

    def toast(self, msg):
        self.statusBar().showMessage(msg, 8000)

    def load_csv(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv);;All files (*)")
        if p:
            self._load(lambda: pd.read_csv(p), p)

    def load_excel(self):
        p, _ = QFileDialog.getOpenFileName(self, "Open Excel", "", "Excel files (*.xlsx *.xls)")
        if not p:
            return
        try:
            sheets = pd.ExcelFile(p).sheet_names
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return
        sheet = sheets[0]
        if len(sheets) > 1:
            sheet, ok = QInputDialog.getItem(self, "Sheet", "Choose sheet:", sheets, 0, False)
            if not ok:
                return
        self._load(lambda: pd.read_excel(p, sheet_name=sheet), p)

    def _load(self, reader, path):
        try:
            df = reader()
        except Exception as e:
            QMessageBox.critical(self, "Could not load file", str(e))
            return
        self.original = df.copy()
        self.df = df
        self.history = []
        self.split = None
        self.refresh_all()
        self.toast(f"Loaded {os.path.basename(path)}: {df.shape[0]:,} rows × {df.shape[1]} columns")

    def commit(self, new_df, message):
        self.history.append(self.df.copy())
        self.history = self.history[-20:]
        self.df = new_df
        self.split = None
        self.refresh_all()
        self.toast(message)

    def undo(self):
        if not self.history:
            self.toast("Nothing to undo.")
            return
        self.df = self.history.pop()
        self.refresh_all()
        self.toast("Undid last step.")

    def reset(self):
        if self.original.empty:
            return
        self.df = self.original.copy()
        self.history = []
        self.split = None
        self.refresh_all()
        self.toast("Reset to the original dataset.")

    def update_stats(self):
        df = self.df
        total = df.size
        miss = int(df.isna().sum().sum())
        pct = (miss / total * 100) if total else 0
        self.stat_labels["rows"].setText(f"{len(df):,}")
        self.stat_labels["cols"].setText(str(df.shape[1]))
        self.stat_labels["num"].setText(str(len(df.select_dtypes(include=np.number).columns)))
        self.stat_labels["miss"].setText(f"{pct:.1f}%")
        if df.empty:
            text, col, border, bg1, bg2, icon = ("Load a dataset to begin", "#7a8db0", "#142a4a",
                                                 "#071224", "#050e1d", "info")
        elif miss == 0:
            text, col, border, bg1, bg2, icon = ("Data Ready for Analysis", "#e8fff3", "#19d3ff",
                                                 "#0b3a6e", "#0a2a58", "check")
        else:
            text, col, border, bg1, bg2, icon = (f"{miss:,} missing values · needs cleaning",
                                                 "#ffd166", "#ffb300", "#3a2a0c", "#2a1d08", "warning")
        self.ready.setText(text)
        self.ready.setStyleSheet(f"color:{col};font-size:{S(14)}px;font-weight:700;")
        self.ready_frame.setStyleSheet(
            "QFrame#readycard { background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            f"stop:0 {bg1}, stop:1 {bg2}); border:1.5px solid {border}; border-radius:16px; }}")
        self.ready_icon.setPixmap(make_pixmap(icon, border if icon != "check" else "#3ddc84", S(28)))

    def refresh_all(self):
        self.update_stats()
        for p in self.pages:
            p.refresh()


# ---- embedded robot artwork (used when the assets/ folder is missing) ----
ROBOT_HEAD_B64 = (
    "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"
)

ROBOT_FULL_B64 = (
    "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"
)


def main():
    global SCALE
    app = QApplication(sys.argv)
    g = app.primaryScreen().availableGeometry()
    SCALE = max(0.7, min(1.0, g.height() / 900))
    app.setStyleSheet(build_qss(SCALE))
    w = MainWindow()
    w.resize(min(1500, int(g.width() * 0.94)), min(880, int(g.height() * 0.92)))
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()