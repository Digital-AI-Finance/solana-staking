"""
PIPE (Private Investment in Public Equity) Structure Diagram

Shows capital flow and stakeholder positions in a SOL-focused PIPE deal.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'models'))
from utils import CHART_RCPARAMS, COLORS

plt.rcParams.update(CHART_RCPARAMS)

fig, ax = plt.subplots(figsize=(10, 7))

# Turn off axis
ax.set_xlim(0, 10)
ax.set_ylim(0, 8)
ax.axis('off')

# Define box style
def draw_box(ax, x, y, w, h, text, color, text_color='white'):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.2",
                         facecolor=color, edgecolor='white', linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=11, fontweight='bold', color=text_color, wrap=True)

# Define arrow style
def draw_arrow(ax, start, end, text='', color='gray'):
    arrow = FancyArrowPatch(start, end, arrowstyle='->', mutation_scale=15,
                           color=color, linewidth=2)
    ax.add_patch(arrow)
    if text:
        mid = ((start[0] + end[0])/2, (start[1] + end[1])/2 + 0.2)
        ax.text(mid[0], mid[1], text, ha='center', va='bottom', fontsize=9,
               color=color, fontweight='bold')

# Institutional Investors (top left)
draw_box(ax, 0.5, 6, 2.5, 1.2, 'Institutional\nInvestors', COLORS['ACCENT_BLUE'])

# PIPE Vehicle (center)
draw_box(ax, 3.5, 5.5, 3, 1.2, 'PIPE Vehicle\n($500M)', COLORS['SOL_PURPLE'])

# SOL Treasury Company (center bottom)
draw_box(ax, 3.5, 3, 3, 1.5, 'SOL Treasury Corp\n(Public Company)', COLORS['SOL_PURPLE'])

# SOL Assets (bottom center)
draw_box(ax, 3.5, 0.5, 3, 1.2, 'Staked SOL\n3.3M SOL @ 7.5% APY', COLORS['SOL_GREEN'], 'white')

# Convertible Bond Holders (right)
draw_box(ax, 7.5, 5.5, 2, 1.2, 'Convertible\nBond Holders', COLORS['WARNING_ORANGE'])

# Equity Shareholders (bottom right)
draw_box(ax, 7.5, 3, 2, 1.2, 'Public\nShareholders', COLORS['ACCENT_BLUE'])

# Arrows with labels
# Investors -> PIPE Vehicle
draw_arrow(ax, (3, 6.6), (3.5, 6.1), '$500M Cash', COLORS['SOL_GREEN'])

# PIPE Vehicle -> Treasury Company
draw_arrow(ax, (5, 5.5), (5, 4.5), 'Capital\nInjection', COLORS['SOL_GREEN'])

# Treasury Company -> SOL Assets
draw_arrow(ax, (5, 3), (5, 1.7), 'SOL\nPurchase', COLORS['SOL_PURPLE'])

# SOL Assets -> Treasury Company (staking rewards)
draw_arrow(ax, (6.7, 1.1), (6.7, 3), '+7.5% APY', COLORS['SOL_GREEN'])

# Convertible holders -> Treasury
draw_arrow(ax, (7.5, 5.8), (6.5, 4.2), 'Debt/Equity\nConversion', COLORS['WARNING_ORANGE'])

# Treasury -> Shareholders
draw_arrow(ax, (6.5, 3.3), (7.5, 3.3), 'Value\nAccrual', COLORS['SOL_PURPLE'])

# Add term sheet summary box
terms = (
    "PIPE Term Sheet\n"
    "-------------------\n"
    "Size: $500M\n"
    "Discount: 5-10% to market\n"
    "Lock-up: 6-12 months\n"
    "Use: 100% SOL acquisition\n"
    "Structure: Common + Warrants"
)
ax.text(0.5, 2.8, terms, fontsize=9, fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor=COLORS['TEXT_SECONDARY'], alpha=0.95))

# Title
ax.text(5, 7.6, 'PIPE Structure for SOL Treasury Accumulation',
        ha='center', fontsize=16, fontweight='bold', color=COLORS['SOL_PURPLE'])

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'chart.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("Chart saved: 05_pipe_structure/chart.pdf")
