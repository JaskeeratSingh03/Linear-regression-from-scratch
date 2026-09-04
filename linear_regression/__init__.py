"""Linear regression from scratch — stdlib only."""

from .model import LinearRegression
from .dataset import make_synthetic, load_csv
from .visualize import ascii_line_plot, ascii_scatter_with_line

__all__ = [
    "LinearRegression",
    "make_synthetic",
    "load_csv",
    "ascii_line_plot",
    "ascii_scatter_with_line",
]
