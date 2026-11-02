"""No-immediate-inverse random policy baseline."""
from zeromethod.experiments.random_baseline import (
    _run_cli,
    evaluate_no_inverse_baseline,
)


def main(argv=None):
    _run_cli(
        evaluate_no_inverse_baseline,
        "Uniform-random cube policy that excludes the previous move's inverse.",
        argv,
    )


if __name__ == "__main__":
    main()
