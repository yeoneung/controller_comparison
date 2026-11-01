Sector feedback and repeated controller-comparison checks

This self-contained directory contains additional deterministic checks and
an annual thermal-model controller sweep. It uses Python 3, NumPy 1.26.4,
and a C99 compiler on macOS or Linux.

Run from this directory:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/structural_checks.py
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/gain_sweep.py --verify
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/gain_sweep.py --check-kernel
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/gain_sweep.py --run

The first command checks unbounded sector functions, weighted coordinates,
log-gain sensitivities, finite increments, complex-step derivatives, and
the primitive-matrix scaling example. Its CSV and JSON outputs are supplied.
The matrix supremum uses a finite search; it is not a rounding certificate.

--verify recomputes the sufficient-memory curves and checks all retained
contrasts and timing summaries. --check-kernel compiles the C kernel and
compares a separate NumPy heat balance, also checking matched RNG suffixes.
--run regenerates all controller outputs, timings, and validation data.
It overwrites the recorded reference outputs in this directory.

The fixed protocol uses a one-year horizon of 105120 five-minute steps,
256 nominal gains and their 0.01 increments, 64 independent paths per
query, and three paired timing repetitions. Inputs and heat calculations
are shared across all queries. Both algorithms already use finite memory.
Memory selection processes one 288-step input period and a fixed search
cap, independently of the horizon. Reported complete costs include input
generation, allocation, nonlinear state updates, selection, and one-time
compilation. Imports, serialization, and separate full-horizon validation
are excluded. Three descriptive timing repetitions do not establish a
hardware-independent runtime or a high-confidence superiority guarantee.

All physical coefficients are illustrative, without building calibration.
The 0.001 K budget concerns discrete-model truncation bias, rather than
real-building accuracy or Monte Carlo sampling error. Full-horizon checks
use five prespecified gains and eight separate paths. Input tapes have
matching terminal suffixes at every tested memory length.
