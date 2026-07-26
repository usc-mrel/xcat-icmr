"""Allow invocation with ``python -m xcat_icmr``."""

from xcat_icmr.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
