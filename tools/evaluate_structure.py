"""Proposal C evaluation: C0/B0/B3 with the shared seven-condition protocol."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.evaluate_pfa import main

if __name__ == '__main__':
    main(proposal='C')
