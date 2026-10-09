#!/usr/bin/env python3
"""Run the Claude LogP benchmark with a fixed ICL configuration."""

import sys
from submit_batch import main

STRATEGY = "random"
SHOTS = 10

if __name__ == "__main__":
    if any(arg in ("--strategy", "--shots") for arg in sys.argv[1:]):
        raise SystemExit(
            "This experiment has a fixed strategy and shot count. "
            "Use its dedicated filename instead."
        )

    sys.argv[1:1] = [
        "--strategy", STRATEGY,
        "--shots", str(SHOTS),
    ]
    main()
