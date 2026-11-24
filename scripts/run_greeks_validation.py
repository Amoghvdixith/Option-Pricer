"""Run the 'greeks iv' stage(s) of the experiment suite.

Usage: python scripts/run_greeks_validation.py [--mode fast|research]

Thin wrapper: all logic lives in the optpricer package (optpricer.experiments).
"""

import sys

from optpricer.experiments.run_all import main

if __name__ == "__main__":
    args = sys.argv[1:]
    sys.exit(main(args + ["--stages", *"greeks iv".split()]))
