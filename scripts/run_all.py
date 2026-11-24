"""Run every experiment and regenerate results/. Usage: python scripts/run_all.py [--mode fast|research]"""

import sys

from optpricer.experiments.run_all import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
