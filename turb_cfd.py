import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PHYSICAL AND FLOW PARAMETERS (Air at ~20°C)
# =============================================================================
rho = 1.225         # Density (kg/m³)
nu = 1.5e-5         # Kinematic viscosity (m²/s)

# Channel geometry and acoustic oscillation
H = 0.2             # Total channel height (m)
R = H / 2.0         # Channel half-height (m)
f = 10.73           # Oscillation frequency (Hz)
omega = 2 * np.pi * f

# Standing wave parameters in the resonator
P_amp = 11800.0     # Pressure at antinode (Pa)
L_tubo = 16.0       # Tube length (m)
lambda_onda = 2 * L_tubo
k = (2 * np.pi) / lambda_onda
dpdx_amp = k * P_amp # Maximum pressure gradient (Pa/m)

# =============================================================================
# 2. SPATIAL AND TEMPORAL DISCRETIZATION
# =============================================================================
Ny = 300            # Number of grid points
y = np.linspace(0, H, Ny)
dy = y[1] - y[0]

# Target position for time series (e.g., y = 100 mm)
y_alvo_m = 0.1    
idx_alvo = np.argmin(np.abs(y - y_alvo_m))
y_real_mm = y[idx_alvo] * 1000.0

# Time step and cycle control
dt = 3e-7           # Time step (s)
ciclos = 5
T = 1.0 / f
t_final = ciclos * T
Nt = int(t_final / dt)
t = np.linspace(0, t_final, Nt)

# =============================================================================
# 3. ARRAY INITIALIZATION AND STORAGE
# =============================================================================
u = np.zeros(Ny)
u_new = np.zeros(Ny)

# Phase control for the last cycle
inicio_ultimo_ciclo = t_final - T
# Targets: 0, 0.2*(T/2), 0.4*(T/2), 0.6*(T/2), 0.8*(T/2), 1.0*(T/2)
fases_alvo = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
perfis_ultimo_ciclo = {}
nu_t_ultimo_ciclo = {}  # Storage for turbulent viscosity profiles at the same phases

# Time history at target position
t_ciclo = []
u_alvo_ciclo = []
p_ciclo = []
dpdx_ciclo = []

# =============================================================================
# 4. CFD SOLVER (1D NAVIER-STOKES + PRANDTL / VAN DRIEST MIXING MODEL)
# =============================================================================
print("Starting CFD simulation...")

for n in range(Nt):
    tempo_atual = t[n]
    
    # Acoustic forcing
    dpdx = dpdx_amp * np.cos(omega * tempo_atual)
    p_atual = 0.0  # Pressure node at the analysis position
    
    # --- Velocity gradient (dudy) ---
    dudy = np.zeros(Ny)
    dudy[1:-1] = (u[2:] - u[:-2]) / (2 * dy)     # Internal nodes (central)
    dudy[0] = (u[1] - u[0]) / dy                 # Bottom wall (forward)
    dudy[-1] = (u[-1] - u[-2]) / dy               # Top wall (backward)
    
    # --- Turbulence Model (Symmetric Van Driest) ---
    u_tau_bottom = np.sqrt(nu * np.abs(dudy[0]))
    u_tau_top = np.sqrt(nu * np.abs(dudy[-1]))
    
    Re_tau_bottom = (u_tau_bottom * R) / nu
    Re_tau_top = (u_tau_top * R) / nu
    Re_tau = np.where(y <= R, Re_tau_bottom, Re_tau_top)
    
    # Dimensionless distance to channel center
    y_tilde = np.abs(y - R) / R
    poly = 0.14 - 0.08 * (y_tilde**2) - 0.06 * (y_tilde**4)
    damping = 1.0 - np.exp((y_tilde - 1.0) * Re_tau / 26.0)
    l_m = R * poly * damping
    
    # Turbulent and effective viscosity
    nu_t = (l_m**2) * np.abs(dudy)
    nu_eff = nu + nu_t
    
    # --- Effective Diffusive Term ---
    nu_eff_plus = 0.5 * (nu_eff[2:] + nu_eff[1:-1])
    nu_eff_minus = 0.5 * (nu_eff[1:-1] + nu_eff[:-2])
    diff_term = (nu_eff_plus * (u[2:] - u[1:-1]) - nu_eff_minus * (u[1:-1] - u[:-2])) / (dy**2)
    
    # --- Time Advance (Explicit Euler) ---
    u_new[1:-1] = u[1:-1] + dt * (- (1/rho) * dpdx + diff_term)
    
    # --- BOUNDARY CONDITIONS (No-slip walls) ---
    u_new[0] = 0.0   
    u_new[-1] = 0.0  
    
    u = u_new.copy()
    
    # --- Last Cycle Data Logging ---
    if tempo_atual >= inicio_ultimo_ciclo:
        fase_atual = (tempo_atual - inicio_ultimo_ciclo) / T
        
        # Save profiles at specific phases
        for alvo in fases_alvo:
            if alvo not in perfis_ultimo_ciclo and fase_atual >= (alvo - 1e-5):
                perfis_ultimo_ciclo[alvo] = u.copy()
                nu_t_ultimo_ciclo[alvo] = nu_t.copy()
                
        t_ciclo.append(tempo_atual - inicio_ultimo_ciclo)
        u_alvo_ciclo.append(u[idx_alvo])
        p_ciclo.append(p_atual)
        dpdx_ciclo.append(dpdx)

print("Simulation completed!")

# =============================================================================
# 5. POST-PROCESSING
# =============================================================================
t_ciclo = np.array(t_ciclo)
u_alvo_ciclo = np.array(u_alvo_ciclo)
p_ciclo = np.array(p_ciclo)
dpdx_ciclo = np.array(dpdx_ciclo)

dudt_alvo_ciclo = np.gradient(u_alvo_ciclo, dt)
fase_normalizada = t_ciclo / T
y_mm = y * 1000.0

# =============================================================================
# 6. VISUALIZATION
# =============================================================================

# Formatting settings
cores = ['black', 'blue', 'green', 'orange', 'red', 'purple']
labels_fase = ['t = 0', 't = 0.2 T/2', 't = 0.4 T/2', 't = 0.6 T/2', 't = 0.8 T/2', 't = T/2']
fases_ordenadas = sorted(perfis_ultimo_ciclo.keys())

# --- Figure 1: Velocity Profiles in the Last Cycle ---
plt.figure(figsize=(9, 6))
for fase, cor, label in zip(fases_ordenadas, cores, labels_fase):
    plt.plot(perfis_ultimo_ciclo[fase], y_mm, label=label, color=cor, linewidth=2)

plt.axhline(y_real_mm, color='gray', linestyle=':', label=f'Target y = {y_real_mm:.1f} mm')
plt.axvline(0, color='gray', linestyle='--', linewidth=1)
plt.title('Turbulent Velocity Profiles (Last Cycle)', fontsize=13, fontweight='bold')
plt.xlabel('Velocity u (m/s)', fontsize=11)
plt.ylabel('Vertical Position y (mm)', fontsize=11)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# --- Figure 2: Time Series at Target Position ---
fig, axs = plt.subplots(4, 1, figsize=(11, 9), sharex=True)

axs[0].plot(fase_normalizada, u_alvo_ciclo, color='#1f77b4', linewidth=2)
axs[0].set_ylabel('u [m/s]', fontsize=11)
axs[0].set_title(f'Flow Dynamics at y = {y_real_mm:.1f} mm (Last Cycle)', fontsize=13, pad=10, fontweight='bold')
axs[0].grid(True, alpha=0.3)

axs[1].plot(fase_normalizada, dudt_alvo_ciclo, color='#e377c2', linewidth=2)
axs[1].set_ylabel('du/dt [m/s²]', fontsize=11)
axs[1].grid(True, alpha=0.3)

axs[2].plot(fase_normalizada, p_ciclo, color='#2ca02c', linewidth=2)
axs[2].set_ylabel('p [Pa]', fontsize=11)
axs[2].grid(True, alpha=0.3)

axs[3].plot(fase_normalizada, dpdx_ciclo, color='#d62728', linewidth=2)
axs[3].set_ylabel('dp/dx [Pa/m]', fontsize=11)
axs[3].set_xlabel('Normalized Phase (t/T)', fontsize=11)
axs[3].grid(True, alpha=0.3)

plt.xlim(0, 1)
plt.tight_layout()

# --- Figure 3: Turbulent Viscosity Ratio Profiles in the Last Cycle ---
plt.figure(figsize=(9, 6))
for fase, cor, label in zip(fases_ordenadas, cores, labels_fase):
    plt.plot(nu_t_ultimo_ciclo[fase] / nu, y_mm, label=label, color=cor, linewidth=2)

plt.title('Turbulent Viscosity Ratio $\\nu_t/\\nu$ (Last Cycle)', fontsize=13, fontweight='bold')
plt.xlabel('$\\nu_t/\\nu$', fontsize=11)
plt.ylabel('Vertical Position y (mm)', fontsize=11)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
