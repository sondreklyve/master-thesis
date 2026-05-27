#!/usr/bin/env bash
# Regenerate all Part 2 quark-star thesis figures.
#
# Usage (from repo root):
#   bash numerics/quark_stars/make_thesis_figures.sh
#
# Expensive scans (sec4, sec5, sec6) are run in --plot-only mode and reuse
# cached data from numerics/quark_stars/output/.  Sec1 and sec2 scripts
# recompute from scratch (fast — pure QM, no QMD stellar pipeline).
#
# Outputs land in the correct thesis/figures/quark_stars/ subfolders.

set -euo pipefail
REPO=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$REPO"

FIGURES="thesis/figures/quark_stars"
OUTPUT="numerics/quark_stars/output"

echo "=== Sec 1: QM EoS ==="
python -m numerics.quark_stars.sec1_qm_eos.run_qm_eos
python -m numerics.quark_stars.sec1_qm_eos.run_qm_stellar_eos

cp "$OUTPUT/simple/plots/sigma_vs_mu_multi.pdf"              "$FIGURES/qm_eos/"
cp "$OUTPUT/simple/plots/number_density_vs_mu_multi.pdf"     "$FIGURES/qm_eos/"
cp "$OUTPUT/simple/plots/pressure_vs_mu_multi.pdf"           "$FIGURES/qm_eos/"
cp "$OUTPUT/simple/plots/pressure_vs_energy_density_multi.pdf" "$FIGURES/qm_eos/"
cp "$OUTPUT/simple/plots/speed_of_sound_vs_mu_multi.pdf"     "$FIGURES/qm_eos/"
cp "$OUTPUT/eos/pressure_vs_energy_density.pdf"              "$FIGURES/qm_stars/"

echo "=== Sec 2: QM Stars ==="
python -m numerics.quark_stars.sec2_qm_stars.plot_maxwell_schematic
python -m numerics.quark_stars.sec2_qm_stars.run_vacuum_scan
python -m numerics.quark_stars.sec2_qm_stars.run_qm_stars
python -m numerics.quark_stars.sec2_qm_stars.run_grav_bound

cp "$OUTPUT/vacuum/vacuum_potential_sigma_scan.pdf"          "$FIGURES/qm_stars/"
cp "$OUTPUT/stellar/mass_radius_combined.pdf"                "$FIGURES/qm_stars/"
cp "$OUTPUT/stellar/qm_grav_bound_mass_radius.pdf"           "$FIGURES/qm_stars/"
# maxwell_construction_schematic.pdf saved directly by the script above

echo "=== Sec 4: QMD Stars (plot-only) ==="
python -m numerics.quark_stars.sec4_qmd_stars.run_benchmark --plot-only
python -m numerics.quark_stars.sec4_qmd_stars.run_stellar   --plot-only
python -m numerics.quark_stars.sec4_qmd_stars.run_truncation_variations --plot-only

cp "$OUTPUT/qmd_benchmark/plots/qmd_benchmark_condensates.pdf"          "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_benchmark/plots/qmd_benchmark_condensate_comparison.pdf" "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_benchmark/plots/qmd_benchmark_pressure.pdf"             "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_benchmark/plots/qmd_benchmark_eos.pdf"                  "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_benchmark/plots/qmd_benchmark_cs2.pdf"                  "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_stellar/plots/qmd_stellar_condensates.pdf"              "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_stellar/plots/qmd_stellar_neutrality.pdf"               "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_stellar/plots/qmd_stellar_eos.pdf"                      "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_stellar/plots/qmd_stellar_cs2.pdf"                      "$FIGURES/qmd_stars/"
cp "$OUTPUT/qmd_stellar/plots/qmd_vs_qm_mass_radius.pdf"                "$FIGURES/qmd_stars/"
# qmd_truncation_parameter_variations.pdf saved directly by the script above

echo "=== Sec 5: Parameter sensitivity (plot-only) ==="
python -m numerics.quark_stars.sec5_parameter_sensitivity.run_sweep --plot-only

cp "$OUTPUT/section2/plots/section2_MR_gdelta.pdf"           "$FIGURES/parameter_sensitivity/"
cp "$OUTPUT/section2/plots/section2_MR_mdelta.pdf"           "$FIGURES/parameter_sensitivity/"
cp "$OUTPUT/section2/plots/section2_MR_lamdelta.pdf"         "$FIGURES/parameter_sensitivity/"
cp "$OUTPUT/section2/plots/section2_MR_lam3.pdf"             "$FIGURES/parameter_sensitivity/"
cp "$OUTPUT/section2/plots/section2_condensates_gdelta.pdf"  "$FIGURES/parameter_sensitivity/"
cp "$OUTPUT/section2/plots/section2_condensates_mdelta.pdf"  "$FIGURES/parameter_sensitivity/"
cp "$OUTPUT/section2/plots/section2_cs2_gdelta.pdf"          "$FIGURES/parameter_sensitivity/"

echo "=== Sec 6: Observational constraints ==="
python -m numerics.quark_stars.sec6_observational.plot_observational_mr
python -m numerics.quark_stars.sec6_observational.run_observational_combos_finalize
# Both scripts save directly to thesis/figures/quark_stars/observational/

echo "=== Done. All figures deployed to $FIGURES/ ==="
