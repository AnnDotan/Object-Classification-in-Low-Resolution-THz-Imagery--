# main.py
import argparse
import sys


def main() -> None:
    """Dispatch to the Lightning entry by default; --engine legacy keeps the
    hand-rolled trainer available for parity validation."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--engine", choices=["lightning", "legacy"], default="lightning")
    args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    if args.engine == "legacy":
        from src.runner import main as legacy_main
        legacy_main()
    else:
        from src.lightning.train import main as lightning_main
        lightning_main()


if __name__ == "__main__":
    main()
