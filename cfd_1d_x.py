import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1. PHYSICAL AND ACOUSTIC PARAMETERS (Air at 20°C)
# =============================================================================
rho0 = 1.225         # Air density (kg/m³)
c = 343.0            # Speed of sound (m/s)
L_tubo = 15.983      # Total duct length (m)

f = 10.73            # Operating frequency (Hz)
omega = 2 * np.pi * f
k = omega / c        # Wave number (rad/m)

# Boundary pressure amplitude at ends (11800 Pa)
P_borda = 2 * 5900.0 
dpdx_amp = k * P_borda # Maximum pressure gradient amplitude (Pa/m)

# =============================================================================
# 2. SPATIAL MESH AND TIME DISCRETIZATION (1D FDTD)
# =============================================================================
Nx = 800
x = np.linspace(0, L_tubo, Nx)
dx = x[1] - x[0]

dt = 0.1 * dx / c    # CFL stability condition
ciclos = 200
T = 1.0 / f
t_final = ciclos * T
Nt = int(t_final / dt)
t = np.linspace(0, t_final, Nt)

# =============================================================================
# 3. ARRAYS INITIALIZATION AND ACOUSTIC FIELD
# =============================================================================
p = P_borda * np.sin(k * (x - L_tubo / 2.0))
u = np.zeros(Nx)

p_new = np.zeros(Nx)
u_new = np.zeros(Nx)

inicio_ultimo_ciclo = t_final - T
perfis = {}

# =============================================================================
# 4. SIMULATION LOOP (1D FDTD)
# =============================================================================
for n in range(Nt):
    tempo_atual = t[n]

    # 1. Update Velocity u
    u_new[1:-1] = u[1:-1] - (dt / (2 * rho0 * dx)) * (p[2:] - p[:-2])
    u_new[0] = u[0] - (dt / (rho0 * dx)) * (p[1] - p[0])
    u_new[-1] = u[-1] - (dt / (rho0 * dx)) * (p[-1] - p[-2])

    # 2. Update Pressure p in internal nodes
    p_new[1:-1] = p[1:-1] - (rho0 * (c**2) * dt / (2 * dx)) * (u_new[2:] - u_new[:-2])
    
    # 3. Harmonic Boundary Conditions at Endpoints
    p_new[0] = P_borda * np.sin(k * (0 - L_tubo / 2.0)) * np.cos(omega * tempo_atual)
    p_new[-1] = P_borda * np.sin(k * (L_tubo - L_tubo / 2.0)) * np.cos(omega * tempo_atual)

    u = u_new.copy()
    p = p_new.copy()

    # Save profile at phase t/T = 1/4 (90°)
    if tempo_atual >= inicio_ultimo_ciclo:
        fase_atual = (tempo_atual - inicio_ultimo_ciclo) / T
        if 't4' not in perfis and fase_atual >= 0.25:
            dpdx_t4_num = np.gradient(p, dx)
            dpdx_t4_an = dpdx_amp * np.cos(k * (x - L_tubo / 2.0)) * np.cos(omega * tempo_atual)
            perfis['t4'] = {
                'p': p.copy(), 
                'u': u.copy(), 
                'dpdx_num': dpdx_t4_num, 
                'dpdx_an': dpdx_t4_an
            }

# Save profile at end of cycle (phase t/T = 0)
dpdx_fim_num = np.gradient(p, dx)
dpdx_fim_an = dpdx_amp * np.cos(k * (x - L_tubo / 2.0)) * np.cos(omega * t[-1])
perfis['fim'] = {
    'p': p.copy(), 
    'u': u.copy(), 
    'dpdx_num': dpdx_fim_num, 
    'dpdx_an': dpdx_fim_an
}

# =============================================================================
# 5. ORGANIZED VISUALIZATION (ENGLISH)
# =============================================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

c_fim, c_t4 = '#1f77b4', '#ff7f0e'
lbl_fim, lbl_t4 = 'Phase t/T = 0', 'Phase t/T = 1/4'
x_no = L_tubo / 2.0

# Subplot 1: Acoustic Pressure
axs[0].plot(x, perfis['fim']['p'], color=c_fim, linewidth=2, label=lbl_fim)
axs[0].plot(x, perfis['t4']['p'], color=c_t4, linewidth=2, linestyle='--', label=lbl_t4)
axs[0].axvline(x_no, color='gray', linestyle=':', linewidth=1.5, label=f'Pressure Node (x = {x_no:.2f} m)')
axs[0].set_ylabel("p' [Pa]", fontsize=11)
axs[0].set_title("Acoustic Pressure Spatial Profile $p'(x)$", fontsize=12, pad=8)
axs[0].legend(loc='upper right', fontsize=9)
axs[0].grid(True, alpha=0.3)

# Subplot 2: Acoustic Velocity
axs[1].plot(x, perfis['fim']['u'], color=c_fim, linewidth=2, label=lbl_fim)
axs[1].plot(x, perfis['t4']['u'], color=c_t4, linewidth=2, linestyle='--', label=lbl_t4)
axs[1].axvline(x_no, color='gray', linestyle=':', linewidth=1.5)
axs[1].set_ylabel('u [m/s]', fontsize=11)
axs[1].set_title('Acoustic Velocity Spatial Profile $u(x)$', fontsize=12, pad=8)
axs[1].grid(True, alpha=0.3)

# Subplot 3: Spatial Pressure Gradient
axs[2].plot(x, perfis['fim']['dpdx_num'], color=c_fim, linewidth=2, label=f'{lbl_fim} (Num)')
axs[2].plot(x, perfis['t4']['dpdx_num'], color=c_t4, linewidth=2, linestyle='--', label=f'{lbl_t4} (Num)')
axs[2].plot(x, perfis['fim']['dpdx_an'], color='black', linewidth=1.2, linestyle=':', label='Analytical')
axs[2].axvline(x_no, color='gray', linestyle=':', linewidth=1.5)
axs[2].set_ylabel('dp/dx [Pa/m]', fontsize=11)
axs[2].set_xlabel('Position along duct x [m]', fontsize=11)
axs[2].set_title('Spatial Pressure Gradient $\partial p / \partial x$', fontsize=12, pad=8)
axs[2].legend(loc='upper right', fontsize=9)
axs[2].grid(True, alpha=0.3)

plt.xlim(0, L_tubo)
plt.tight_layout()
plt.show()
