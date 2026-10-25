#!/usr/bin/env python3
"""Run food clustering on the supplied nutrition table."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vyndra.recommend import run_clustering
if __name__ == "__main__": run_clustering()
