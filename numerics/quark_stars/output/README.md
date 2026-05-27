# Output directory structure

## Authoritative outputs (used in the thesis)

### `qmd_benchmark/`
Common-chemical-potential benchmark results for the QMD model (Section 1 of the QMD chapter).
- `data/` — raw benchmark EoS tables
- `plots/` — benchmark figures (condensates, pressure, EoS, cs², condensate comparison)
  - All `.pdf` files here are referenced in `thesis/part2/quark_stars.tex`

### `qmd_stellar/`
Neutral stellar EoS and TOV results for the QMD baseline (Section 1 of the QMD chapter).
- `data/` — stellar EoS tables, TOV output, QM reference stars
- `plots/` — stellar figures (condensates, EoS, cs², mass-radius, neutrality, QMD vs QM)
  - All `.pdf` files here are referenced in `thesis/part2/quark_stars.tex`

### `section2/`
Parameter sensitivity study (Section 2 of the QMD chapter). One-at-a-time variations
of the four diquark-sector parameters: g_Δ (Runs A, B), m_Δ (Runs C, D), λ_Δ (Runs E, F),
λ_3 (Runs G, H).
- `data/` — per-run benchmark and stellar EoS files; naming pattern `section2_{stellar|benchmark}_{param}_{value}[_eos].txt`
- `plots/` — thesis figures (MR curves and condensate/cs² detail for g_Δ variation)
- `plots/diagnostic/` — diagnostic plots not referenced in the thesis
- `section2_summary.csv` — master table: M_max, R(M_max), onset, cs² peak, asymptotic gap for all 9 runs

### `simple/`
Simple Quark-Meson model results (Chapter overview, Section 1 of the QMD chapter).
- `data/` — simple QM EoS tables
- `plots/` — sigma/density/pressure/EoS/cs² figures; referenced in `thesis/part2/quark_stars.tex`

### `stellar/`
Quark stars from the simple QM model (Section 2 of the QMD chapter, before the QMD extension).
- `mass_radius_combined.pdf` and associated data; referenced in `thesis/part2/quark_stars.tex`

### `eos/`
Neutral QM EoS with Maxwell construction (Section 2 of the QMD chapter, before QMD).
- `pressure_vs_energy_density.pdf` and associated data; referenced in `thesis/part2/quark_stars.tex`

## Notes

- All `.pdf` files in `plots/` subdirectories that are referenced in the thesis have been
  mirrored to `thesis/figures/quark_stars/`. If a plot is regenerated, copy it over to keep
  the thesis figure directory current.
- The `plots/diagnostic/` subdirectories contain figures generated for debugging or inspection
  that are not part of the thesis.
- Do NOT delete the `data/` files — they are the authoritative numerical results.
