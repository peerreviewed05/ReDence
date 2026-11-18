"""Convenient entry point for the non-trainable ReDence controller."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from redence.controller import main

if __name__ == "__main__":
    raise SystemExit(main())
