"""Reopen and display a saved GPU_HYPE figure pickle (a ``.pplot`` file).

Training runs (and the results notebook) can persist Matplotlib figures with
``Display.save_pickle``. This small utility reopens one for inspection.

Usage:
    python DisplayPKL.py path/to/figure.pplot
"""

import sys

from gpu import Display

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python DisplayPKL.py <figure.pplot>")
        sys.exit(1)
    Display.show_pickle(sys.argv[1])
