"""Shared path setup: run any experiment from anywhere; data lands in results/data, figures in results/figures."""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src"); DATA = os.path.join(ROOT, "results", "data"); FIGDIR = os.path.join(ROOT, "results", "figures") + os.sep
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGDIR, exist_ok=True)
if SRC not in sys.path: sys.path.insert(0, SRC)
if os.path.dirname(os.path.abspath(__file__)) not in sys.path: sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(DATA)
__all__ = ["ROOT", "SRC", "DATA", "FIGDIR"]
