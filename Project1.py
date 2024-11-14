import Project1_Functions as fn
import numpy as np
import matplotlib.pyplot as plt
import time
start_time = time.time()

#A=Hydrogen-> RED DOTS
#B=Nitrogen-> BLUE DOTS

################################################################################################################################################
################################################################################################################################################
############################################    Ideal Mixture  #################################################################################
################################################################################################################################################
################################################################################################################################################

Title="Ideal Mixture"
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019,7)

params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': 0,
            'epsilon_BB': 0,
            'epsilon_AB': 0,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k))
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_H = np.zeros((len(mus_A), len(Ts)))
mean_coverage_N = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    lattice, coverage_H, coverage_N = fn.run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_H[i // len(Ts), i % len(Ts)] = np.mean(coverage_H[-1000:])
    mean_coverage_N[i // len(Ts), i % len(Ts)] = np.mean(coverage_N[-1000:])

Ts=Ts/(8.617*10**-5) #Convert to Kelvin

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5),dpi=150)
fontisize=8
# Mean coverage of H
axs[0].pcolormesh(mus_A, Ts, mean_coverage_H.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$ (eV)')
axs[0].set_ylabel(r'$T(K)$')

# Mean coverage of N
axs[1].pcolormesh(mus_A, Ts, mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$ (eV)')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_H.T + mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_{tot}\rangle$')
axs[2].set_xlabel(r'$\mu_H$ (eV)')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)
fig.suptitle(Title, fontsize=15, y=0.95)

fig.text(0.05, 0.85, 'a)', fontsize=12, fontweight='bold', ha='center', va='center')

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = fn.plot_lattice(final_lattice[0, 3], axs[3], "")
axs[3].set_title(rf'$\mu_H = {mus_A[0]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = fn.plot_lattice(final_lattice[3, 3], axs[4],"")
axs[4].set_title(rf'$\mu_H = {mus_A[3]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = 0 eV and T = 0.01 / k
axs[5] = fn.plot_lattice(final_lattice[6, 3], axs[5], "")
axs[5].set_title(rf'$\mu_H = {mus_A[6]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

fig.text(0.05, 0.45, 'b)', fontsize=12, fontweight='bold', ha='center', va='center')

plt.tight_layout()
plt.close()

fn.save_Phase_Lattice(fig,Title)
optimal_conditions= fn.find_optimal_conditions(mus_A,Ts,final_lattice,2,4)
fn.create_table(optimal_conditions, Title)
fn.save_plots(optimal_conditions,final_lattice,Title)
print(f"Process finished {Title} --- {time.time() - start_time} seconds ---")

################################################################################################################################################
################################################################################################################################################
############################################    Repulsion Interactions  ########################################################################
################################################################################################################################################
################################################################################################################################################

# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)

params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': 0.05,
            'epsilon_BB': 0.05,
            'epsilon_AB': 0.05,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k))
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_H = np.zeros((len(mus_A), len(Ts)))
mean_coverage_N = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    lattice, coverage_H, coverage_N = fn.run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_H[i // len(Ts), i % len(Ts)] = np.mean(coverage_H[-1000:])
    mean_coverage_N[i // len(Ts), i % len(Ts)] = np.mean(coverage_N[-1000:])

Ts=Ts/(8.617*10**-5) #Convert to Kelvin

Title="Repulsion Interactions"
# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5),dpi=150)
fontisize=8
# Mean coverage of H
axs[0].pcolormesh(mus_A, Ts, mean_coverage_H.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$ (eV)')
axs[0].set_ylabel(r'$T(K)$')

# Mean coverage of N
axs[1].pcolormesh(mus_A, Ts, mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$ (eV)')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_H.T + mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_{tot}\rangle$')
axs[2].set_xlabel(r'$\mu_H$ (eV)')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)
fig.suptitle(Title, fontsize=15, y=0.95)

fig.text(0.05, 0.85, 'a)', fontsize=12, fontweight='bold', ha='center', va='center')

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = fn.plot_lattice(final_lattice[0, 3], axs[3], "")
axs[3].set_title(rf'$\mu_H = {mus_A[0]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = fn.plot_lattice(final_lattice[3, 3], axs[4],"")
axs[4].set_title(rf'$\mu_H = {mus_A[3]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = 0 eV and T = 0.01 / k
axs[5] = fn.plot_lattice(final_lattice[6, 3], axs[5], "")
axs[5].set_title(rf'$\mu_H = {mus_A[6]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

fig.text(0.05, 0.45, 'b)', fontsize=12, fontweight='bold', ha='center', va='center')

plt.tight_layout()
plt.close()

fn.save_Phase_Lattice(fig,Title)
optimal_conditions= fn.find_optimal_conditions(mus_A,Ts,final_lattice,2,4)
fn.create_table(optimal_conditions, Title)
fn.save_plots(optimal_conditions,final_lattice,Title)
print(f"Process finished {Title} --- {time.time() - start_time} seconds ---")

################################################################################################################################################
################################################################################################################################################
############################################    Attractive Interactions  #######################################################################
################################################################################################################################################
################################################################################################################################################

# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)

params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': -0.05,
            'epsilon_BB': -0.05,
            'epsilon_AB': -0.05,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k))
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_H = np.zeros((len(mus_A), len(Ts)))
mean_coverage_N = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    lattice, coverage_H, coverage_N = fn.run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_H[i // len(Ts), i % len(Ts)] = np.mean(coverage_H[-1000:])
    mean_coverage_N[i // len(Ts), i % len(Ts)] = np.mean(coverage_N[-1000:])

Ts=Ts/(8.617*10**-5) #Convert to Kelvin

Title="Attractive Interactions"
# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5),dpi=150)
fontisize=8
# Mean coverage of H
axs[0].pcolormesh(mus_A, Ts, mean_coverage_H.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$ (eV)')
axs[0].set_ylabel(r'$T(K)$')

# Mean coverage of N
axs[1].pcolormesh(mus_A, Ts, mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$ (eV)')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_H.T + mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_{tot}\rangle$')
axs[2].set_xlabel(r'$\mu_H$ (eV)')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)
fig.suptitle(Title, fontsize=15, y=0.95)

fig.text(0.05, 0.85, 'a)', fontsize=12, fontweight='bold', ha='center', va='center')

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = fn.plot_lattice(final_lattice[0, 3], axs[3], "")
axs[3].set_title(rf'$\mu_H = {mus_A[0]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = fn.plot_lattice(final_lattice[3, 3], axs[4],"")
axs[4].set_title(rf'$\mu_H = {mus_A[3]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = 0 eV and T = 0.01 / k
axs[5] = fn.plot_lattice(final_lattice[6, 3], axs[5], "")
axs[5].set_title(rf'$\mu_H = {mus_A[6]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

fig.text(0.05, 0.45, 'b)', fontsize=12, fontweight='bold', ha='center', va='center')

plt.tight_layout()
plt.close()

fn.save_Phase_Lattice(fig,Title)
optimal_conditions= fn.find_optimal_conditions(mus_A,Ts,final_lattice,2,4)
fn.create_table(optimal_conditions, Title)
fn.save_plots(optimal_conditions,final_lattice,Title)
print(f"Process finished {Title} --- {time.time() - start_time} seconds ---")

################################################################################################################################################
################################################################################################################################################
############################################   Immiscible Nitrogen and Hydrogen  ###############################################################
################################################################################################################################################
################################################################################################################################################

# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)

params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': -0.05,
            'epsilon_BB': -0.05,
            'epsilon_AB': 0.05,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k))
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_H = np.zeros((len(mus_A), len(Ts)))
mean_coverage_N = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    lattice, coverage_H, coverage_N = fn.run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_H[i // len(Ts), i % len(Ts)] = np.mean(coverage_H[-1000:])
    mean_coverage_N[i // len(Ts), i % len(Ts)] = np.mean(coverage_N[-1000:])

Ts=Ts/(8.617*10**-5) #Convert to Kelvin

Title="Immiscible Interactions"

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5),dpi=150)
fontisize=8
# Mean coverage of H
axs[0].pcolormesh(mus_A, Ts, mean_coverage_H.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$ (eV)')
axs[0].set_ylabel(r'$T(K)$')

# Mean coverage of N
axs[1].pcolormesh(mus_A, Ts, mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$ (eV)')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_H.T + mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_{tot}\rangle$')
axs[2].set_xlabel(r'$\mu_H$ (eV)')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)
fig.suptitle(Title, fontsize=15, y=0.95)

fig.text(0.05, 0.85, 'a)', fontsize=12, fontweight='bold', ha='center', va='center')

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = fn.plot_lattice(final_lattice[0, 3], axs[3], "")
axs[3].set_title(rf'$\mu_H = {mus_A[0]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = fn.plot_lattice(final_lattice[3, 3], axs[4],"")
axs[4].set_title(rf'$\mu_H = {mus_A[3]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = 0 eV and T = 0.01 / k
axs[5] = fn.plot_lattice(final_lattice[6, 3], axs[5], "")
axs[5].set_title(rf'$\mu_H = {mus_A[6]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

fig.text(0.05, 0.45, 'b)', fontsize=12, fontweight='bold', ha='center', va='center')

plt.tight_layout()
plt.close()

fn.save_Phase_Lattice(fig,Title)
optimal_conditions= fn.find_optimal_conditions(mus_A,Ts,final_lattice,2,4)
fn.create_table(optimal_conditions, Title)
fn.save_plots(optimal_conditions,final_lattice,Title)
print(f"Process finished {Title} --- {time.time() - start_time} seconds ---")

################################################################################################################################################
################################################################################################################################################
############################################   Like Dissolves Unlike  ##########################################################################
################################################################################################################################################
################################################################################################################################################

# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)

params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': 0.05,
            'epsilon_BB': 0.05,
            'epsilon_AB': -0.05,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k))
        })

# Run the simulation
np.random.seed(42)
final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_H = np.zeros((len(mus_A), len(Ts)))
mean_coverage_N = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    lattice, coverage_H, coverage_N = fn.run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_H[i // len(Ts), i % len(Ts)] = np.mean(coverage_H[-1000:])
    mean_coverage_N[i // len(Ts), i % len(Ts)] = np.mean(coverage_N[-1000:])

Ts=Ts/(8.617*10**-5) #Convert to Kelvin

Title="Like Dissolves Unlike"

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(6.5, 4.5),dpi=150)
fontisize=8
# Mean coverage of H
axs[0].pcolormesh(mus_A, Ts, mean_coverage_H.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_H \rangle$')
axs[0].set_xlabel(r'$\mu_H$ (eV)')
axs[0].set_ylabel(r'$T(K)$')

# Mean coverage of N
axs[1].pcolormesh(mus_A, Ts, mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_N \rangle$')
axs[1].set_xlabel(r'$\mu_H$ (eV)')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts, mean_coverage_H.T + mean_coverage_N.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_{tot}\rangle$')
axs[2].set_xlabel(r'$\mu_H$ (eV)')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)
fig.suptitle(Title, fontsize=15, y=0.95)

fig.text(0.05, 0.85, 'a)', fontsize=12, fontweight='bold', ha='center', va='center')

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = fn.plot_lattice(final_lattice[0, 3], axs[3], "")
axs[3].set_title(rf'$\mu_H = {mus_A[0]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = fn.plot_lattice(final_lattice[3, 3], axs[4],"")
axs[4].set_title(rf'$\mu_H = {mus_A[3]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

# mu_A = 0 eV and T = 0.01 / k
axs[5] = fn.plot_lattice(final_lattice[6, 3], axs[5], "")
axs[5].set_title(rf'$\mu_H = {mus_A[6]}$ eV, $T = {Ts[3]:.0f}$ K', fontsize=fontisize)

fig.text(0.05, 0.45, 'b)', fontsize=12, fontweight='bold', ha='center', va='center')

plt.tight_layout()
plt.close()

fn.save_Phase_Lattice(fig,Title)
optimal_conditions= fn.find_optimal_conditions(mus_A,Ts,final_lattice,2,4)
fn.create_table(optimal_conditions, Title)
fn.save_plots(optimal_conditions,final_lattice,Title)
print(f"Process finished {Title} --- {time.time() - start_time} seconds ---")
