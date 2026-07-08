from __future__ import annotations

import argparse

from .io import compare_puppi_outputs, run_puppi_on_npz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone PUPPI on PileFlow generator .npz output."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input generator .npz file containing full_* arrays.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output .npz file for standalone PUPPI arrays.",
    )

    parser.add_argument(
        "--n-pu",
        type=int,
        default=None,
        help="Optional override for n_pu. If omitted, use n_pu from input .npz.",
    )

    parser.add_argument("--R0", type=float, default=0.3)
    parser.add_argument("--Rmin", type=float, default=0.02)
    parser.add_argument("--w-cut", type=float, default=0.1)
    parser.add_argument("--eta-tracker", type=float, default=2.5)
    parser.add_argument("--max-const", type=int, default=500)
    parser.add_argument("--jet-R", type=float, default=0.4)

    parser.add_argument(
        "--compare-to-generator",
        action="store_true",
        help="Compare standalone output to puppi_* arrays already in the input file.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    run_puppi_on_npz(
        npz_path=args.input,
        output_path=args.output,
        n_pu_override=args.n_pu,
        R0=args.R0,
        Rmin=args.Rmin,
        w_cut=args.w_cut,
        eta_tracker=args.eta_tracker,
        max_const=args.max_const,
        jet_R=args.jet_R,
    )

    if args.compare_to_generator:
        diffs = compare_puppi_outputs(
            generator_npz_path=args.input,
            standalone_npz_path=args.output,
        )

        print()
        print("[PUPPI] Comparison to generator-side puppi_* arrays:")
        for key, value in diffs.items():
            print(f"  {key}: {value:.6e}")


if __name__ == "__main__":
    main()