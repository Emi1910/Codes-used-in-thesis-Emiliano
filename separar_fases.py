import pandas as pd
import glob
import os
import math
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def detectar_altura_parede(df):
    """Identifies the Y wall height for EACH individual frame."""
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

def calcular_perfil_medio(lista_perfis, lista_ids):
    """Calcula a média de uma lista de perfis filtrados pelos IDs exatos."""
    dfs_selecionados = [p['raw_df'] for p in lista_perfis if p['id_imagem'] in lista_ids]
    
    if not dfs_selecionados:
        return None, None, 0
        
    df_combinado = pd.concat(dfs_selecionados)
    df_medio = df_combinado.groupby('y_coord')['U[m/s]'].mean().reset_index()
    return df_medio['U[m/s]'].values, df_medio['y_coord'].values, len(dfs_selecionados)

def analisar_e_separar_fases_sequencial(caminho_da_pasta, vel_alvo, tolerancia=1.0, descartar_iniciais=40, filtrar_parede=True, pct_corte=0.0, x_esquerda=5.0, x_direita=45.0):
    padrao_busca = os.path.join(caminho_da_pasta, '*.csv')
    arquivos_csv = sorted(glob.glob(padrao_busca))
    
    if len(arquivos_csv) <= descartar_iniciais:
        print("Warning: Insufficient number of files.")
        return
        
    arquivos_csv = arquivos_csv[descartar_iniciais:]
    v_min_faixa = vel_alvo - tolerancia
    v_max_faixa = vel_alvo + tolerancia
    
    perfis_encontrados = []
    print(f"Searching for images with freestream between {v_min_faixa:.1f} and {v_max_faixa:.1f} m/s...")
    print(f"Filters applied: y >= {pct_corte*100}% height | x between {x_esquerda} and {x_direita} mm")
    
    for idx_arquivo, arquivo in enumerate(arquivos_csv):
        try:
            df = pd.read_csv(arquivo, skiprows=8, sep=';', decimal=',')
            df_valido = df[df['Status'] == 0].copy()
            
            if not df_valido.empty:
                # Aplica o limite X para a extração do perfil e cálculo do freestream
                df_valido = df_valido[(df_valido['x (mm)[mm]'] >= x_esquerda) & (df_valido['x (mm)[mm]'] <= x_direita)]
                if df_valido.empty: continue

                if filtrar_parede:
                    y_parede_frame = detectar_altura_parede(df_valido)
                    df_valido = df_valido[df_valido['y (mm)[mm]'] >= y_parede_frame]
                    if df_valido.empty: continue
                    df_valido['y_relativa'] = df_valido['y (mm)[mm]'] - y_parede_frame
                    # Filtra até 30 mm acima da parede para o perfil da camada limite
                    df_valido = df_valido[(df_valido['y_relativa'] >= 0.0) & (df_valido['y_relativa'] <= 30.0)]
                else:
                    df_valido['y_relativa'] = df_valido['y (mm)[mm]']
                
                if df_valido.empty: continue

                y_max = df_valido['y (mm)[mm]'].max()
                y_min = df_valido['y (mm)[mm]'].min()
                
                # Aplica o percentual de corte para definir a região de freestream
                y_limite_freestream = y_min + (y_max - y_min) * pct_corte
                
                df_topo = df_valido[df_valido['y (mm)[mm]'] >= y_limite_freestream]
                
                if not df_topo.empty:
                    media_freestream = df_topo['U[m/s]'].mean()
                    
                    if v_min_faixa <= media_freestream <= v_max_faixa:
                        df_valido['y_coord'] = df_valido['y_relativa'].round(3)
                        perfil_y = df_valido.groupby('y_coord')['U[m/s]'].mean().reset_index()
                        
                        perfis_encontrados.append({
                            'id_imagem': idx_arquivo + descartar_iniciais + 1, # Adiciona 1 para bater com a numeração humana de frames
                            'u_freestream': media_freestream,
                            'y_vals': perfil_y['y_coord'].values,
                            'u_vals': perfil_y['U[m/s]'].values,
                            'raw_df': df_valido[['y_coord', 'U[m/s]']]
                        })
        except Exception:
            pass 

    total_imagens = len(perfis_encontrados)
    if total_imagens == 0:
        print(f"No images found in the range {v_min_faixa:.1f} to {v_max_faixa:.1f} m/s.")
        return

    # Lista os IDs válidos encontrados para orientar o usuário
    ids_disponiveis = [p['id_imagem'] for p in perfis_encontrados]
    print(f"\nSuccess! Found {total_imagens} images.")
    print(f"Available Image IDs in this range: {ids_disponiveis}\n")

    # --- 1. EXIBIÇÃO EM GRADES 3x3 ---
    imagens_por_figura = 9
    num_figuras = math.ceil(total_imagens / imagens_por_figura)

    for fig_idx in range(num_figuras):
        inicio = fig_idx * imagens_por_figura
        fim = inicio + imagens_por_figura
        lote_atual = perfis_encontrados[inicio:fim]

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(14, 10), sharex=True, sharey=True)
        texto_titulo = (f"Individual Boundary Layer Profiles | Range: {vel_alvo} ± {tolerancia} m/s\n"
                        #f"Page {fig_idx + 1} of {num_figuras} (Images {inicio + 1} to {min(fim, total_imagens)})"
                        )
        fig.suptitle(texto_titulo, fontsize=16, fontweight='bold', y=0.97)

        axes_flat = axes.flatten()
        for i, ax in enumerate(axes_flat):
            if i < len(lote_atual):
                dados = lote_atual[i]
                ax.plot(dados['u_vals'], dados['y_vals'], color='darkblue', marker='o', linestyle='-', linewidth=1.5, markersize=5, zorder=3)
                ax.axvline(0, color='gray', linewidth=1.5, zorder=1)
                ax.set_title(f"Img: {dados['id_imagem']} | U: {dados['u_freestream']:.2f}", fontsize=12, pad=6)
                ax.grid(True, linestyle=':', alpha=0.6)
                
                if i % 3 == 0: ax.set_ylabel('Y (mm)', fontsize=12)
                if i >= 6: ax.set_xlabel('U (m/s)', fontsize=12)
            else:
                ax.axis('off')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90) 
        print(f"Displaying page {fig_idx + 1} of {num_figuras}... (Close the window to proceed)")
        plt.show(block=True)

    print("\n" + "="*50)
    print(" SEQUENTIAL PHASE CLASSIFICATION")
    print("="*50)

    # --- 2. FASE GROWING ---
    growing_input = input("\nEnter IDs for the GROWING phase (e.g. 45, 48, 52): ")
    growing_ids = [int(x.strip()) for x in growing_input.split(',') if x.strip().isdigit()]
    u_grow, y_grow, n_grow = calcular_perfil_medio(perfis_encontrados, growing_ids)

    if u_grow is not None:
        print(f"Matched {n_grow} images. Plotting GROWING phase profile...")
        plt.figure(figsize=(6, 8))
        plt.plot(u_grow, y_grow, label=f'Growing Phase (n={n_grow})', color='firebrick', marker='^', linestyle='-', linewidth=2, markersize=6)
        plt.axvline(0, color='gray', linewidth=1.5, zorder=1)
        plt.title(f'Mean Profile: GROWING Phase\nFreestream Target: {vel_alvo} ± {tolerancia} m/s', fontsize=14, pad=15)
        plt.xlabel('Mean Velocity $\overline{U}$ [m/s]', fontsize=12)
        plt.ylabel('Y Position from Wall (mm)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(f"WARNING: None of the entered IDs {growing_ids} matched the available images: {ids_disponiveis}")

    # --- 3. FASE SHRINKING ---
    shrinking_input = input("\nEnter IDs for the SHRINKING phase (e.g. 60, 65, 71): ")
    shrinking_ids = [int(x.strip()) for x in shrinking_input.split(',') if x.strip().isdigit()]
    u_shrink, y_shrink, n_shrink = calcular_perfil_medio(perfis_encontrados, shrinking_ids)

    if u_shrink is not None:
        print(f"Matched {n_shrink} images. Plotting SHRINKING phase profile...")
        plt.figure(figsize=(6, 8))
        plt.plot(u_shrink, y_shrink, label=f'Shrinking Phase (n={n_shrink})', color='dodgerblue', marker='v', linestyle='-', linewidth=2, markersize=6)
        plt.axvline(0, color='gray', linewidth=1.5, zorder=1)
        plt.title(f'Mean Profile: SHRINKING Phase\nFreestream Target: {vel_alvo} ± {tolerancia} m/s', fontsize=14, pad=15)
        plt.xlabel('Mean Velocity $\overline{U}$ [m/s]', fontsize=12)
        plt.ylabel('Y Position from Wall (mm)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(f"WARNING: None of the entered IDs {shrinking_ids} matched the available images: {ids_disponiveis}")

    # --- 4. GRÁFICO COMBINADO ---
    if u_grow is not None or u_shrink is not None:
        print("\nGenerating final comparative plot (Combined)...")
        plt.figure(figsize=(8, 10))
        
        if u_grow is not None:
            plt.plot(u_grow, y_grow, label=f'Growing Phase (n={n_grow})', color='firebrick', marker='^', linestyle='-', linewidth=2, markersize=6)
        if u_shrink is not None:
            plt.plot(u_shrink, y_shrink, label=f'Shrinking Phase (n={n_shrink})', color='dodgerblue', marker='v', linestyle='-', linewidth=2, markersize=6)
                     
        plt.axvline(0, color='gray', linewidth=1.5, zorder=1)
        plt.title(f'Mean Boundary Layer Profiles by Phase\nFreestream Target: {vel_alvo} ± {tolerancia} m/s', fontsize=14, pad=15)
        plt.xlabel('Mean Velocity $\overline{U}$ [m/s]', fontsize=12)
        plt.ylabel('Y Position from Wall (mm)', fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.show(block=True)

# --- EXECUÇÃO ---
pasta_dados = r'file_path'

analisar_e_separar_fases_sequencial(
    caminho_da_pasta=pasta_dados, 
    vel_alvo=26.0, 
    tolerancia=0.5,
    filtrar_parede=True,
    pct_corte=0.5,
    x_esquerda=5.0,
    x_direita=45.0
)
