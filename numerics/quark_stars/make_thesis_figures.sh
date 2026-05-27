#!/usr/bin/env bash
# Regenerate all Part 2 quark-star thesis figures.
#
# Usage (from repo root):
#   bash numerics/quark_stars/make_thesis_figures.sh
#
# Each script writes its plots directly to thesis/figures/quark_stars/.
# Expensive scans (sec4, sec5, sec6) are run in --plot-only mode and reuse
# cached data from numerics/quark_stars/output/.  Sec1 and sec2 scripts
# recompute from scratch (fast — pure QM, no QMD stellar pipeline).

set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

echo "=== Sec 1: QM EoS ==="
python -m numerics.quark_stars.sec1_qm_eos.run_qm_eos
python -m numerics.quark_stars.sec1_qm_eos.run_qm_stellar_eos

echo "=== Sec 2: QM Stars ==="
python -m numerics.quark_stars.sec2_qm_stars.plot_maxwell_schematic
python -m numerics.quark_stars.sec2_qm_stars.run_vacuum_scan
python -m numerics.quark_stars.sec2_qm_stars.run_qm_stars
python -m numerics.quark_stars.sec2_qm_stars.run_grav_bound

echo "=== Sec 4: QMD Stars (plot-only) ==="
python -m numerics.quark_stars.sec4_qmd_stars.run_benchmark --plot-only
python -m numerics.quark_stars.sec4_qmd_stars.run_stellar   --plot-only
python -m numerics.quark_stars.sec4_qmd_stars.run_truncation_variations --plot-only

echo "=== Sec 5: Parameter sensitivity (plot-only) ==="
python -m numerics.quark_stars.sec5_parameter_sensitivity.run_sweep --plot-only

echo "=== Sec 6: Observational constraints ==="
python -m numerics.quark_stars.sec6_observational.plot_observational_mr
python -m numerics.quark_stars.sec6_observational.run_observational_combos_finalize

echo "=== Done. All figures deployed to thesis/figures/quark_stars/ ==="
