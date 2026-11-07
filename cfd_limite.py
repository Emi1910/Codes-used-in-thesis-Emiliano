import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. Physical and Fluid Parameters (Air at ~20°C)
# =============================================================================
rho = 1.225         # Density (kg/m^3)
nu = 1.5e-5         # Kinematic viscosity (m^2/s)

# =============================================================================
# 2. Geometry and Flow Parameters
# =============================================================================
H = 0.1            # Channel height: 30 mm (in meters)
f = 10.73            # Oscillation frequency (Hz)
omega = 2 * np.pi * f
u_top_amp = 30.0    # Top wall velocity amplitude (m/s)

# Standing Wave Parameters
P_amp = 2*5900.0     # Max pressure amplitude at the ends (Pa)
L_tubo = 16.0        # Total length representing 1/2 wave (m)

# Pressure gradient amplitude (dp/dx) at the middle of the tube (x=L/2)
lambda_onda = 2 * L_tubo                # Total wavelength (16 m)
k = (2 * np.pi) / lambda_onda           # Wavenumber (rad/m)
dpdx_amp = k * P_amp                    # dp/dx amplitude at the pressure node (approx 5497.8 Pa/m)

# =============================================================================
# 3. Numerical Mesh Parameters (FTCS)
# =============================================================================
Ny = 1000            # Number of grid points (y)
y = np.linspace(0, H, Ny)
dy = y[1] - y[0]

# Time step (von Neumann stability criterion)
dt_max = (dy**2) / (2 * nu)
dt = min(1e-4, dt_max * 0.9)

ciclos = 5
T = 1.0 / f
t_final = ciclos * T
Nt = int(t_final / dt)
t = np.linspace(0, t_final, Nt)

# =============================================================================
# 4. Initialization and Simulation
# =============================================================================
u = np.zeros(Ny)
u_new = np.zeros(Ny)

perfis_ultimo_ciclo = {}
inicio_ultimo_ciclo = t_final - T

# Target phases: 0°, 45°, 90°, 135°, 180° (as fractions of period T)
fases_alvo = [0.0, 0.125, 0.25, 0.375, 0.5]

for n in range(Nt):
    tempo_atual = t[n]
    
    # Both in phase over time for point x = L/2
    dpdx = dpdx_amp * np.cos(omega * tempo_atual)
    u_top = u_top_amp * np.cos(omega * tempo_atual)
    
    # 1D Navier-Stokes (Explicit Scheme)
    u_new[1:-1] = u[1:-1] + dt * (
        - (1/rho) * dpdx 
        + nu * (u[2:] - 2*u[1:-1] + u[:-2]) / (dy**2)
    )
    
    # Boundary Conditions
    u_new[0] = 0.0      # Bottom wall (No-slip)
    #u_new[-1] = u_top   # Top moving wall (in phase with dp/dx)
    u_new[-1] = 0
    
    u = u_new.copy()
    
    # Collect data during the last cycle for plotting
    if tempo_atual >= inicio_ultimo_ciclo:
        fase_atual = (tempo_atual - inicio_ultimo_ciclo) / T
        for alvo in fases_alvo:
            if alvo not in perfis_ultimo_ciclo and fase_atual >= alvo:
                perfis_ultimo_ciclo[alvo] = u.copy()

# =============================================================================
# 5. Plotting Results
# =============================================================================
plt.figure(figsize=(10, 7))

# Colors and labels for the 5 selected phases
cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
labels_fase = [
    'Phase 0° (t=0)', 
    'Phase 45° (t=T/8)', 
    'Phase 90° (t=T/4)', 
    'Phase 135° (t=3T/8)', 
    'Phase 180° (t=T/2)'
]

y_mm = y * 1000  

# Ensure keys are sorted to correctly match the colors and labels
fases_ordenadas = sorted(perfis_ultimo_ciclo.keys())

for fase, cor, label in zip(fases_ordenadas, cores, labels_fase):
    plt.plot(perfis_ultimo_ciclo[fase], y_mm, label=label, color=cor, linewidth=2)

plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title(f'Velocity Profile - Standing Wave Midpoint (x=4m)\ndp/dx Amplitude: {dpdx_amp:.0f} Pa/m', fontsize=14)
plt.xlabel('Velocity u (m/s)', fontsize=12)
plt.ylabel('Vertical position y (mm)', fontsize=12)
plt.legend(loc='best', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
