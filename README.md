# Controller comparison experiments

Simulation tools and recorded synthetic results for comparing nearby feedback
controllers with finite retained histories.

| Experiment | Program | Recorded results |
| --- | --- | --- |
| Bounded-feedback sensitivity and finite gain changes | `code/sensitivity_examples.py` | `recorded_data/sensitivity_examples.csv`, `evidence/sensitivity_examples.json` |
| Unbounded sector feedback, weighted coordinates, and matrix gain scaling | `structural_extension/code/structural_checks.py` | `structural_extension/recorded_data/structural_checks.csv`, `structural_extension/evidence/structural_checks.json` |
| A 24-condition thermal-network comparison over 30 days | `code/thermal_application.py` | `recorded_data/thermal_observations.npz`, `evidence/thermal_application.json` |
| A year-long workload comparing 256 controller pairs | `structural_extension/code/gain_sweep.py` | `structural_extension/recorded_data/gain_sweep.npz`, `structural_extension/evidence/gain_sweep.json` |

## Setup

Use Python 3.9 or later, NumPy 1.26.4, Matplotlib 3.9.2, and a C99 compiler
(`cc`, Clang, or GCC) on macOS or Linux. From the repository root:

```sh
python -m pip install -r requirements.txt
```

## Check the recorded results

```sh
python code/sensitivity_examples.py
python structural_extension/code/structural_checks.py
python code/thermal_application.py --verify
python code/thermal_application.py --check-kernel
python structural_extension/code/gain_sweep.py --verify
python structural_extension/code/gain_sweep.py --check-kernel
```

These commands check the deterministic examples, memory bounds, recorded
contrasts, timing summaries, and independent C/NumPy implementations. They
do not rerun the full stochastic workloads. The matrix gain-scaling check
uses a finite length search.

## Recreate the figures

```sh
python code/plot_results.py --out figures
```

This creates two vector PDF figures and PNG previews from the supplied
observations: sensitivity bounds and matrix gain scaling, followed by
thermal-network memory choices and cost comparisons.

## Rerun the stochastic workloads

```sh
python code/thermal_application.py --run
python structural_extension/code/gain_sweep.py --run
```

Run these in a separate copy: they overwrite the recorded observations and
timings. Timing results depend on the hardware and compiler. The thermal
coefficients are illustrative design values. The temperature-error budget
controls truncation bias within the discrete model.

See [the detailed usage notes](README.txt) and the
[annual-workload protocol](structural_extension/README.txt) for cost accounting,
validation settings, and reproducibility details.
