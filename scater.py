import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def plot_freestream_sequence(caminho_pasta, pct_corte=0.0, descartar_iniciais=40, x_esquerda=5.0, x_direita=45.0):
    padrao = os.path.join(caminho_pasta, '*.csv')
    arquivos = sorted(glob.glob(padrao))
    
    if len(arquivos) > descartar_iniciais:
        arquivos = arquivos[descartar_iniciais:]
    else:
        print("Error: Insufficient number of files in the folder.")
        return

    numeros_frame = []
    velocidades_freestream = []

    print(f"Processing {len(arquivos)} files to extract freestream velocities...")
    print(f"Filters applied: y >= {pct_corte}% height | x between {x_esquerda} and {x_direita} mm")

    for idx, arquivo in enumerate(arquivos):
        try:
            df = pd.read_csv(arquivo, skiprows=8, sep=';', decimal=',')
            df_valido = df[df['Status'] == 0]
            
            if not df_valido.empty:
                y_min = df_valido['y (mm)[mm]'].min()
                y_max = df_valido['y (mm)[mm]'].max()
                y_corte_mm = y_min + (pct_corte / 100.0) * (y_max - y_min)
                
                # Aplica todos os filtros de região (Y e X)
                df_analise = df_valido[
                    (df_valido['y (mm)[mm]'] >= y_corte_mm) &
                    (df_valido['x (mm)[mm]'] >= x_esquerda) &
                    (df_valido['x (mm)[mm]'] <= x_direita)
                ]
                
                if not df_analise.empty:
                    v_avg = df_analise['U[m/s]'].mean()
                    
                    numeros_frame.append(idx + 1)
                    velocidades_freestream.append(v_avg)
                    
        except Exception as e:
            pass # Ignora erros de leitura e pula o arquivo

    if not velocidades_freestream:
        print("No valid data could be extracted.")
        return

    # --- PLOTAGEM DO GRÁFICO (EM INGLÊS) ---
    print("Plotting sequential data...")
    
    fig, ax = plt.subplots(figsize=(11, 5))
    
    # Plotamos as linhas finas e os pontos em destaque
    ax.plot(numeros_frame, velocidades_freestream, 
            marker='o', linestyle='-', color='dodgerblue', 
            markersize=4, linewidth=1, alpha=0.8, 
            label='v_avg per frame')
    
    # Adiciona uma linha vermelha tracejada mostrando a média global
    media_global = np.mean(velocidades_freestream)
    ax.axhline(media_global, color='red', linestyle='--', linewidth=1.5, 
               label=f'Global Mean ({media_global:.2f} m/s)')

    ax.set_title('Sequential Variation of Average Freestream Velocity', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Frame Sequence Number', fontsize=12)
    ax.set_ylabel('Average Freestream Velocity U (m/s)', fontsize=12)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

# --- EXECUÇÃO ---
pasta_dados = r'/home/emiliano/Documentos/IC/Pulsator-selected/260225_first_measurements/high res'

# Mantenha os mesmos parâmetros usados no Analise_2.py
plot_freestream_sequence(
    caminho_pasta=pasta_dados, 
    pct_corte=0.0, 
    descartar_iniciais=40, 
    x_esquerda=5.0, 
    x_direita=45.0
)
