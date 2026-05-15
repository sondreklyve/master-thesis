"""Parameter definitions for the two-flavor Quark-Meson-Diquark (QMD) model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QMDParameters:
    """Input and derived parameters for the QMD model with 2SC diquark condensate.

    The chiral sector uses the same vacuum conventions as the QM model:
      g        = m_q / f_pi
      lambda_0 = 3 * (m_sigma^2 - m_pi^2) / f_pi^2   (identical to QMFittedParameters.lambda_parameter)

    The diquark sector adds:
      g_delta    = g_delta_factor * g
      lambda_3   = lambda_3_factor   * lambda_0   (sigma-diquark quartic coupling)
      lambda_delta = lambda_delta_factor * lambda_0   (diquark quartic self-coupling)

    Convention note -- t_loop4_factor:
      The phi^2 * delta^2 log-correction term in Omega_trunc is:

          t_loop4 = t_loop4_factor * g^2 * g_delta^2 / (4pi)^2
                    * [log(mq^2/C) - F(mpi^2) - mpi^2*F'(mpi^2)]
                    * phi^2 * delta^2

      where C = g^2*phi^2 + g_delta^2*delta^2 and phi, delta are the bare
      (unrescaled) condensate fields.

      Both papers use the same field conventions, but disagree on the coefficient:
        Andersen2024 (arXiv:2408.12361), Eq. (36): coefficient = 4  (default here)
        Andersen2025 (arXiv:2502.10229), Eq. (29): coefficient = 8

      Our thesis appendix cites Andersen2024, so the default is 4.
      Use t_loop4_factor=8.0 (QMD_SET_A_REFERENCE) to reproduce
      GeneralFunctions_2SC.py / Condensates_and_cs_final.ipynb (Andersen2025).

      Secondary note: Andersen2024 uses chiral-limit parameters mpi=0, msigma=500 MeV
      for its numerics; our code uses physical masses mpi=140, msigma=600 MeV,
      consistent with Andersen2025.  The analytic formula is valid for both choices.
    """

    # --- chiral / vacuum sector ---
    m_pi_mev: float = 140.0
    f_pi_mev: float = 93.0
    m_q_mev: float = 300.0
    m_sigma_mev: float = 600.0

    # --- diquark sector ---
    m_delta_mev: float = 500.0
    g_delta_factor: float = 2.0
    lambda_3_factor: float = 1.0
    lambda_delta_factor: float = 0.25

    # Coefficient for the phi^2*delta^2 log-correction term (see class docstring).
    # 4.0  = Andersen2024 (arXiv:2408.12361) Eq. (36)  = thesis appendix  [DEFAULT]
    # 8.0  = Andersen2025 (arXiv:2502.10229) Eq. (29)  = reference notebook
    t_loop4_factor: float = 4.0

    # --- options ---
    # Set False to inspect the analytic vacuum potential in isolation.  This
    # omits the T=0 quasiparticle medium term and the residual BCS integral.
    include_medium_term: bool = True

    # Include the finite residual integral Omega_{1,num} from Eq. A.23.
    # This is part of the full one-loop QMD potential.  Setting it False keeps
    # the older diagnostic truncation but is not the physical 2SC potential.
    include_omega_1_num: bool = True

    # Numerical cutoff used for the finite Omega_{1,num} and gapless-mode
    # quadratures.  The reference notebook Condensates_and_cs_final.ipynb uses
    # Lambda_cutoff = 3000 MeV.
    residual_cutoff_mev: float = 3000.0

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def g(self) -> float:
        """Yukawa coupling: g = m_q / f_pi."""
        return self.m_q_mev / self.f_pi_mev

    @property
    def g_delta(self) -> float:
        """Diquark Yukawa coupling: g_delta = g_delta_factor * g."""
        return self.g_delta_factor * self.g

    @property
    def lambda_0(self) -> float:
        """Meson quartic coupling, same formula as QM fit_qm_parameters:
        lambda_0 = 3 * (m_sigma^2 - m_pi^2) / f_pi^2."""
        return 3.0 * (self.m_sigma_mev**2 - self.m_pi_mev**2) / self.f_pi_mev**2

    @property
    def lambda_3(self) -> float:
        """Sigma-diquark quartic coupling: lambda_3 = lambda_3_factor * lambda_0."""
        return self.lambda_3_factor * self.lambda_0

    @property
    def lambda_delta(self) -> float:
        """Diquark quartic self-coupling: lambda_delta = lambda_delta_factor * lambda_0."""
        return self.lambda_delta_factor * self.lambda_0

    def describe(self) -> str:
        """Return a human-readable summary of all input and derived parameter values."""
        lines = [
            f"  m_pi       = {self.m_pi_mev:.1f} MeV",
            f"  f_pi       = {self.f_pi_mev:.1f} MeV",
            f"  m_q        = {self.m_q_mev:.1f} MeV",
            f"  m_sigma    = {self.m_sigma_mev:.1f} MeV",
            f"  m_delta    = {self.m_delta_mev:.1f} MeV",
            f"  g          = {self.g:.4f}  (= m_q/f_pi)",
            f"  g_delta    = {self.g_delta:.4f}  (= {self.g_delta_factor:.2f} * g)",
            f"  lambda_0   = {self.lambda_0:.4f}",
            f"  lambda_3   = {self.lambda_3:.4f}  (= {self.lambda_3_factor:.3f} * lambda_0)",
            f"  lambda_d   = {self.lambda_delta:.4f}  (= {self.lambda_delta_factor:.3f} * lambda_0)",
            f"  t_loop4_factor = {self.t_loop4_factor:.1f}  "
            f"({'thesis appendix' if self.t_loop4_factor == 4.0 else 'reference code (GeneralFunctions_2SC)' if self.t_loop4_factor == 8.0 else 'custom'})",
            f"  include_medium_term = {self.include_medium_term}",
            f"  include_omega_1_num = {self.include_omega_1_num}",
            f"  residual_cutoff = {self.residual_cutoff_mev:.1f} MeV",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Named parameter sets
# ---------------------------------------------------------------------------

# SET_A matches the single parameter set used in the reference notebook
# (Condensates_and_cs_final.ipynb, arXiv:2502.10229):
#   gΔ = 2*mq/fπ  →  g_delta_factor=2.0
#   mΔ = 500 MeV
#   λ3 = λ0       →  lambda_3_factor=1.0
#   λΔ = λ0/4     →  lambda_delta_factor=0.25
# Uses thesis appendix convention: t_loop4_factor=4.0 (default).
QMD_SET_A = QMDParameters(
    m_delta_mev=500.0,
    g_delta_factor=2.0,
    lambda_3_factor=1.0,
    lambda_delta_factor=0.25,
)

# SET_A with the reference-code t_loop4 coefficient (8.0 instead of 4.0).
# Use this set to reproduce the Condensates_and_cs_final.ipynb results exactly.
# All physical parameters are identical to SET_A; only the phi^2*Delta^2
# logarithmic coefficient differs.
QMD_SET_A_REFERENCE = QMDParameters(
    m_delta_mev=500.0,
    g_delta_factor=2.0,
    lambda_3_factor=1.0,
    lambda_delta_factor=0.25,
    t_loop4_factor=8.0,
)

# SET_B is an independent exploration set (not from any reference paper).
# Physical motivation: heavier diquark (mΔ=900 MeV), weaker diquark Yukawa,
# no sigma-diquark tree interaction (lambda_3=0).
# lambda_delta_factor=0.2 is our own parametric choice (reference does not
# define a second parameter set).
QMD_SET_B = QMDParameters(
    m_delta_mev=900.0,
    g_delta_factor=1.5,
    lambda_3_factor=0.0,
    lambda_delta_factor=0.2,
)


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

ALL_SETS: dict[str, QMDParameters] = {
    "SET_A": QMD_SET_A,
    "SET_A_REFERENCE": QMD_SET_A_REFERENCE,
    "SET_B": QMD_SET_B,
}


def print_convention_summary() -> None:
    """Print a table comparing all named QMD parameter sets and their conventions."""
    print("QMD parameter sets and conventions")
    print("=" * 60)
    print(
        "  t_loop4_factor=4.0  →  Andersen2024 (arXiv:2408.12361), Eq. (36)\n"
        "                          = thesis appendix Eq. A.22\n"
        "  t_loop4_factor=8.0  →  Andersen2025 (arXiv:2502.10229), Eq. (29)\n"
        "                          = GeneralFunctions_2SC.py / reference notebook\n"
        "\n"
        "  Discrepancy is between the two papers (same bare fields, different coeff).\n"
        "  Thesis cites Andersen2024  →  default 4.0 is correct for thesis.\n"
    )
    for name, params in ALL_SETS.items():
        print(f"--- {name} ---")
        print(params.describe())
        print()
