"""Train the ReDence Decider adapter."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.sft import main

if __name__ == "__main__":
    main("decider")
