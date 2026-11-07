import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

# =============================================================================
# 1. PARAMETERS & INPUT CONTROLS
# =============================================================================
caminho_pasta = r'/home/emiliano/Documentos/IC/Pulsator-selected/260608/air'
descartar_iniciais = 40       

# Filter Settings
filtro_parede = True          # Enable/Disable wall detection filter
pct_corte_freestream = 0.8    # Cutoff percentage for freestream region (0.0 to hide line)
vel_alvo = 20               # Target freestream velocity (m/s)
tolerancia = 0.5            # Velocity tolerance (m/s)

# Analisys X-Cuts (mm) - Used ONLY for calculating the mean freestream velocity
x_esquerda = 5.0              # Left cutoff boundary for velocity analysis
x_direita = 45.0              # Right cutoff boundary for velocity analysis

# Quiver Settings
escala_seta = 1000             # Arrow scale factor (MAIOR = SETAS MENORES)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================
def detectar_altura_parede(df):
    """
    Detects the absolute vertical wall height (Y) for each individual frame.
    """
    linhas = df.groupby('y (mm)[mm]')['U[m/s]'].agg(['std', 'mean', 'count']).dropna().sort_index()
    max_vetores = linhas['count'].max()
    linhas_validas = linhas[linhas['count'] > 0.5 * max_vetores]
    
    if linhas_validas.empty:
        return df['y (mm)[mm]'].min()
        
    limite_std = linhas_validas['std'].quantile(0.75) * 1.5
    
    for y, row in linhas_validas.iterrows():
        if row['std'] <= limite_std:
            return y
            
    return linhas_validas.index.min()

def carregar_campos_filtrados(pasta, vel_alvo, tol, descartar, pct_corte, x_esq, x_dir, max_imagens=9):
    padrao = os.path.join(pasta, '*.csv')
    arquivos = sorted(glob.glob(padrao))
    
    if len(arquivos) > descartar:
        arquivos = arquivos[descartar:]
    else:
        print("Error: Insufficient number of files in the directory.")
        return []

    v_min = vel_alvo - tol
    v_max = vel_alvo + tol
    frames_selecionados = []
    
    print(f"Filtering fields with Freestream = {v_min:.1f} to {v_max:.1f} m/s...")

    for arquivo in arquivos:
        if len(frames_selecionados) >= max_imagens:
            break
            
        try:
            df = pd.read_csv(arquivo, skiprows=8, sep=';', decimal=',')
            df_valido = df[df['Status'] == 0].copy()
            
            if not df_valido.empty:
                
                # Detecta a parede absoluta e filtra os dados abaixo dela
                if filtro_parede:
                    y_parede_abs = detectar_altura_parede(df_valido)
                    df_valido = df_valido[df_valido['y (mm)[mm]'] >= y_parede_abs]
                else:
                    y_parede_abs = df_valido['y (mm)[mm]'].min()
                
                if df_valido.empty: continue

                # Calcula o limite do freestream com base na porcentagem da altura restante
                y_max_frame = df_valido['y (mm)[mm]'].max()
                y_min_frame = df_valido['y (mm)[mm]'].min()
                y_limite_freestream = y_min_frame + (pct_corte) * (y_max_frame - y_min_frame)
                
                df_topo = df_valido[df_valido['y (mm)[mm]'] >= y_limite_freestream]
                
                if not df_topo.empty:
                    # Aplica o recorte lateral de X apenas para calcular a média
                    df_topo_analise = df_topo[(df_topo['x (mm)[mm]'] >= x_esq) & (df_topo['x (mm)[mm]'] <= x_dir)]
                    
                    if not df_topo_analise.empty:
                        media_freestream = df_topo_analise['U[m/s]'].mean()
                        
                        if v_min <= media_freestream <= v_max:
                            # Extrai as coordenadas ABSOLUTAS (sem corte em X) para plotagem
                            X = df_valido['x (mm)[mm]'].values
                            Y = df_valido['y (mm)[mm]'].values
                            U = df_valido['U[m/s]'].values
                            V = df_valido['V[m/s]'].values
                            Mag = np.hypot(U, V)
                            
                            frames_selecionados.append({
                                'nome': os.path.basename(arquivo),
                                'u_fs': media_freestream,
                                'y_parede': y_parede_abs,
                                'y_fs': y_limite_freestream,
                                'X': X, 'Y': Y, 'U': U, 'V': V, 'Mag': Mag
                            })
                        
        except Exception as e:
            pass 

    return frames_selecionados

# =============================================================================
# 3. 3x3 GRID PLOTTING
# =============================================================================
dados_piv = carregar_campos_filtrados(caminho_pasta, vel_alvo, tolerancia, descartar_iniciais, pct_corte_freestream, x_esquerda, x_direita, 9)

if not dados_piv:
    print(f"No files found with Freestream between {vel_alvo-tolerancia} and {vel_alvo+tolerancia} m/s.")
    exit()

print(f"Found {len(dados_piv)} fields. Generating grid (Close the plot window to select full view)...")

fig_grid, axs = plt.subplots(3, 3, figsize=(15, 10))
titulo_str = f"Velocity Field | Target Freestream: {vel_alvo} ± {tolerancia} m/s"
fig_grid.suptitle(titulo_str, fontsize=16, fontweight='bold')

cmap = 'jet'

for idx, ax in enumerate(axs.flat):
    if idx >= len(dados_piv):
        ax.axis('off')
        continue
        
    dado = dados_piv[idx]
    u_fs = dado['u_fs']
    
    # Escala de +- 10 m/s
    vmin_din = u_fs - 10.0
    vmax_din = u_fs + 5.0
    
    q = ax.quiver(dado['X'], dado['Y'], 
                  dado['U'], dado['V'], 
                  dado['Mag'], 
                  cmap=cmap, clim=(vmin_din, vmax_din), scale=escala_seta, width=0.002)
    
    # Proporção 1:1 física (não estica os eixos)
    ax.set_aspect('equal')
    
    # Linha do Freestream Absoluto (somente se pct_corte_freestream for maior que 0)
    if pct_corte_freestream > 0.0:
        ax.axhline(dado['y_fs'], color='black', linestyle='--', alpha=0.8, linewidth=1.5)
    
    # Linha da Parede Absoluta
    if filtro_parede:
        ax.axhline(dado['y_parede'], color='red', linestyle=':', alpha=0.8, linewidth=1.5)
        
    ax.set_title(f"Frame {idx+1}: {dado['nome']}\n$U_{{fs}}$ = {u_fs:.2f} m/s", fontsize=9)
    
    # Limites baseados EXATAMENTE no min e max absoluto dos vetores válidos
    ax.set_xlim(dado['X'].min(), dado['X'].max())
    ax.set_ylim(dado['Y'].min(), dado['Y'].max())
    
    if idx % 3 != 0: ax.set_ylabel('')
    else: ax.set_ylabel('y (mm)', fontsize=10)
    
    if idx < 6: ax.set_xlabel('')
    else: ax.set_xlabel('x (mm)', fontsize=10)

cbar = fig_grid.colorbar(q, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Velocity Magnitude (m/s)', fontsize=11)

plt.show()

# =============================================================================
# 4. SINGLE FULL-SIZE PLOT SELECTION
# =============================================================================
try:
    print("-" * 50)
    escolha = int(input("Select Frame number to expand (1 to 9) or 0 to exit: "))
    
    if 1 <= escolha <= len(dados_piv):
        idx_escolha = escolha - 1
        d = dados_piv[idx_escolha]
        u_fs = d['u_fs']
        
        # Escala de +- 10 m/s
        vmin_din = u_fs - 15
        vmax_din = u_fs + 8
        
        # Janela quadrada (10x10)
        fig_single, ax_single = plt.subplots(figsize=(10, 10))
        
        q_single = ax_single.quiver(d['X'], d['Y'], 
                                    d['U'], d['V'], 
                                    d['Mag'], 
                                    cmap=cmap, clim=(vmin_din, vmax_din), scale=escala_seta*0.8, width=0.0015)
        
        # Proporção 1:1 física na visualização expandida
        ax_single.set_aspect('equal')
        
        # Linha do Freestream Absoluto (somente se pct_corte_freestream for maior que 0)
        if pct_corte_freestream > 0.0:
            ax_single.axhline(d['y_fs'], color='black', linestyle='--', linewidth=2, label=f"Freestream Limit (y = {d['y_fs']:.1f} mm)")
        
        if filtro_parede:
            ax_single.axhline(d['y_parede'], color='red', linestyle=':', linewidth=2, label=f"Wall Location (y = {d['y_parede']:.1f} mm)")
            
        ax_single.set_title(f"Velocity Field | $U_{{fs}}$ = {u_fs:.2f} m/s", fontsize=16, pad=15, fontweight='bold')
        ax_single.set_xlabel("x (mm)", fontsize=12)
        ax_single.set_ylabel("y (mm)", fontsize=12)
        
        ax_single.set_xlim(d['X'].min(), d['X'].max())
        ax_single.set_ylim(d['Y'].min(), d['Y'].max())
        
        # Só exibe a legenda se houver algo para mostrar
        if pct_corte_freestream > 0.0 or filtro_parede:
            ax_single.legend(loc='upper right', fontsize=10)
        
        cbar_single = fig_single.colorbar(q_single, ax=ax_single, fraction=0.046, pad=0.04)
        cbar_single.set_label('Velocity Magnitude (m/s)', fontsize=12)
        
        plt.tight_layout()
        plt.show()
    else:
        print("Exiting.")
except ValueError:
    print("Exiting.")
