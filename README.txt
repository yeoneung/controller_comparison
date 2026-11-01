Finite-memory controller-comparison tools

Requirements: Python 3.9 or later, NumPy 1.26.4, Matplotlib 3.9.2, and a
C99 compiler. Simulation libraries build on macOS or Linux. Run commands
from this extracted directory. No account, network download, or parent
project is required after dependencies have been installed.

Recorded observations and verification
  python code/sensitivity_examples.py
  python code/thermal_application.py --verify
  python code/thermal_application.py --check-kernel
  python structural_extension/code/structural_checks.py
  python structural_extension/code/gain_sweep.py --verify
  python structural_extension/code/gain_sweep.py --check-kernel
  python code/plot_results.py --out figures

The deterministic programs check six bounded-feedback cases and 720
sector-feedback sensitivities, finite gain changes, weighted coordinates,
and independent complex-step derivatives. The matrix asymptotic check is
a finite-search illustration of its analytic argument.

The thermal program checks all 24 configurations of the 30-day grid,
including larger controller changes with no memory improvement. Its
independent NumPy energy-balance check also replays selected full-horizon
contrasts. All recorded contrasts, seeds, bound curves, and paired timing
observations are supplied. The structural_extension directory contains
the annual 256-pair workload, shared input generation across all queries,
and horizon-independent memory preparation. Its README describes the
separate kernel check and cost accounting. plot_results.py reproduces
the two figures directly from the supplied arrays and tables. The third
panel of the sensitivity figure uses all six recorded matrix gain values
in structural_extension/evidence/structural_checks.json. Each displayed
maximum is over a finite length search, not a certified infinite supremum.

Regenerating stochastic observations and timings
  python code/thermal_application.py --run
  python structural_extension/code/gain_sweep.py --run
These commands overwrite reference observations and timing outputs.
Copy the directory first if the distributed observations must be kept.
Timings depend on hardware; imports, serialization, and separate research
validation are excluded from measured workload costs. Selection and
one-time compilation are included in the complete-cost comparisons.

Physical coefficients are design values, without building calibration.
The 0.001 K budget concerns truncation bias in the discrete simulation
model. The synthetic paths do not certify physical prediction accuracy
or Monte Carlo sampling error. Ordinary floating-point arithmetic is used.
SHA256.json records every distributed source and observation file.
