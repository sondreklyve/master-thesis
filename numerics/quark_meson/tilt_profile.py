import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

cmap = plt.cm.viridis
color_pos = cmap(0.65)

# --- Potential ---
def V(phi, m2, lam, h):
    return 0.5*m2*phi**2 + 0.25*lam*phi**4 - h*phi

# parameters
lam = 1.0
m2  = -4.0
h   = 1.2   # explicit symmetry breaking

# sigma range
sigma = np.linspace(-3.2, 3.2, 2000)
Vvals = V(sigma, m2, lam, h)

# find minimum
imin = np.argmin(Vvals)
sigma_min = sigma[imin]
Vmin = Vvals[imin]

# shift so minimum is at zero
Vvals = Vvals - Vmin

# --- plot ---
plt.figure(figsize=(8,5.5))
plt.plot(sigma, Vvals, color=color_pos, linewidth=2.2)
plt.scatter(sigma_min, 0.0, color="black", s=40, zorder=5)

plt.xlabel(r"$\sigma \ (\pi=0)$", fontsize=18)
plt.ylabel(r"$V(\sigma,\pi=0)$", fontsize=18)
plt.title(r"Tilted Quark--Meson Potential ($h \neq 0$)", fontsize=20)

plt.grid(True, linestyle=":", alpha=0.6)

ax = plt.gca()
ax.tick_params(axis='both', which='both',
               labelbottom=False, labelleft=False)

plt.tight_layout()
plt.show()
