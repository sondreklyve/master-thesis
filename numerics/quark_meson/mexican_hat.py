import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 14

# --- Potential ---
def V_r(r, m2, lam):
    return 0.5 * m2 * r**2 + 0.25 * lam * r**4

# --- parameters ---
lam = 1.0
m2  = -4.0
v0  = np.sqrt(abs(m2)/lam)

# --- disk domain ---
Rmax = 2.5
nr, nt = 260, 360
r = np.linspace(0.0, Rmax, nr)
t = np.linspace(0.0, 2*np.pi, nt)
R, T = np.meshgrid(r, t)

sigma = R * np.cos(T)
pi    = R * np.sin(T)

V = V_r(R, m2, lam)

# shift constant so minimum is 0
Vmin = V_r(v0, m2, lam)
Z = V - Vmin

fig = plt.figure(figsize=(9.0, 6.5))
ax = fig.add_subplot(111, projection="3d")

# orthographic view like textbook
ax.set_proj_type("ortho")

# smooth surface with subtle mesh lines
surf = ax.plot_surface(
    sigma, pi, Z,
    cmap="viridis_r",
    linewidth=0.3,
    edgecolor=(0.2,0.2,0.2,0.25),
    antialiased=True
)

# remove default box
ax.set_axis_off()

# ---- Custom thin axes ----
axis_len = 3.0
z_len = 5.0

# thin axis lines
ax.plot([0, axis_len], [0, 0], [0, 0], color="black", linewidth=1.2)
ax.plot([0, 0], [0, axis_len], [0, 0], color="black", linewidth=1.2)
ax.plot([0, 0], [0, 0], [0, z_len], color="black", linewidth=1.2)

# small arrow tips
arrow = 0.25
ax.plot([axis_len, axis_len-arrow], [0,  arrow/2], [0, 0], color="black", linewidth=1.2)
ax.plot([axis_len, axis_len-arrow], [0, -arrow/2], [0, 0], color="black", linewidth=1.2)

ax.plot([0,  arrow/2], [axis_len, axis_len-arrow], [0, 0], color="black", linewidth=1.2)
ax.plot([0, -arrow/2], [axis_len, axis_len-arrow], [0, 0], color="black", linewidth=1.2)

ax.plot([0,  arrow/2], [0, 0], [z_len, z_len-arrow], color="black", linewidth=1.2)
ax.plot([0, -arrow/2], [0, 0], [z_len, z_len-arrow], color="black", linewidth=1.2)

# labels
ax.text(axis_len, -0.15, 0.3, r"$\sigma_0$", fontsize=15)
ax.text(0, axis_len*0.95, 0.25, r"$\pi_0$", fontsize=15)
ax.text(0, 0, z_len*1.08, r"$V(\sigma_0,\pi_0)$", fontsize=15)

ax.view_init(elev=25, azim=40)

ax.set_box_aspect((1,1,0.45))


plt.tight_layout()
plt.show()
