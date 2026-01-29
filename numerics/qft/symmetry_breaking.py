import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

# --- phi^4 potential ---
# V(phi) = 1/2 m^2 phi^2 + (lambda/4!) phi^4
def V(phi, m2, lam):
    return 0.5 * m2 * phi**2 + (lam / 24.0) * phi**4

# choose parameters (just for illustration)
lam = 1.0
m2_pos = 1.0     # unbroken case: single minimum
m2_neg = -1.0    # broken case: double well

# phi range
phi = np.linspace(-3.0, 3.0, 800)

V_pos = V(phi, m2_pos, lam)
V_neg = V(phi, m2_neg, lam)

# locations of minima for m^2 < 0: phi = ±v, v = sqrt(6|m^2|/lambda)
v = np.sqrt(6.0 * abs(m2_neg) / lam)

# --- colors (viridis-like) ---
cmap = plt.cm.viridis
color_pos = cmap(0.25)
color_neg = cmap(0.65)

plt.figure(figsize=(8, 5.5))

# plot potentials
plt.plot(phi, V_pos, color=color_pos, linewidth=2.2, label=r"$m^2>0$ (single minimum)")
plt.plot(phi, V_neg, color=color_neg, linewidth=2.2, label=r"$m^2<0$ (double well)")

# mark minima in the broken case
Vmin = V(v, m2_neg, lam)
plt.scatter([ -v, v ], [ Vmin, Vmin ], color="black", s=35, zorder=5, label=r"Minima at $\phi=\pm v$")

# labels and title
plt.xlabel(r"Field value $\phi$", fontsize=18)
plt.ylabel(r"Potential $V(\phi)$", fontsize=18)
plt.title(r"Classical $\phi^4$ Potential and Vacuum Structure", fontsize=20)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=13, frameon=True)
plt.grid(True, linestyle=":", alpha=0.5)

plt.tight_layout()
# plt.savefig("phi4_potential.pdf", dpi=300)
plt.show()
