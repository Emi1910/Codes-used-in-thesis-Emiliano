import pandas as pd
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

def analisar_histograma_piv(caminho_pasta, pct_corte=0, descartar_iniciais=40, n_amostras=10, x_esquerda=5.0, x_direita=45.0):
    padrao = os.path.join(caminho_pasta, '*.csv')
    arquivos = sorted(glob.glob(padrao))
    
    if len(arquivos) > descartar_iniciais:
        arquivos = arquivos[descartar_iniciais:]
    else:
        print("Error: Insufficient number of files in the folder.")
        return

    dados_instantes = []
    y_corte_mm = None

    print(f"Step 1: Processing {len(arquivos)} files for the region ABOVE {pct_corte}% of the height...")
    print(f"Applying X-axis cuts: x >= {x_esquerda} mm and x <= {x_direita} mm")
    
    for arquivo in arquivos:
        try:
            df = pd.read_csv(arquivo, skiprows=8, sep=';', decimal=',')
            df_valido = df[df['Status'] == 0]
            
            if not df_valido.empty:
                y_min = df_valido['y (mm)[mm]'].min()
                y_max = df_valido['y (mm)[mm]'].max()
                y_corte_mm = y_min + (pct_corte / 100.0) * (y_max - y_min)
                
                # Filtra a região Y superior
                df_regiao_superior = df_valido[df_valido['y (mm)[mm]'] >= y_corte_mm]
                
                if not df_regiao_superior.empty:
                    # Aplica o recorte lateral no eixo X apenas para a análise estatística
                    df_regiao_analise = df_regiao_superior[
                        (df_regiao_superior['x (mm)[mm]'] >= x_esquerda) & 
                        (df_regiao_superior['x (mm)[mm]'] <= x_direita)
                    ]
                    
                    if not df_regiao_analise.empty:
                        dados_instantes.append({
                            'media': df_regiao_analise['U[m/s]'].mean(),
                            'std': df_regiao_analise['U[m/s]'].std()
                        })
        except Exception as e:
            print(f"Error reading {arquivo}: {e}")

    if len(dados_instantes) < n_amostras * 2:
        print("No valid data could be extracted.")
        return

    velocidades_freestream = [d['media'] for d in dados_instantes]

    # --- STEP 2: ANÁLISE ESTATÍSTICA ROBUSTA (Top/Bottom Extremes) ---
    dados_ordenados = sorted(dados_instantes, key=lambda x: x['media'])
    
    menores = dados_ordenados[:n_amostras]
    maiores = dados_ordenados[-n_amostras:]
    
    u_min_avg = np.mean([d['media'] for d in menores])
    u_max_avg = np.mean([d['media'] for d in maiores])
    
    std_min_prop = np.sqrt(np.sum([d['std']**2 for d in menores])) / n_amostras
    std_max_prop = np.sqrt(np.sum([d['std']**2 for d in maiores])) / n_amostras
    
    A_fit_abs = (u_max_avg - u_min_avg) / 2.0
    desvio_propagado = np.sqrt(std_max_prop**2 + std_min_prop**2) / 2.0
    C_fit = np.mean(velocidades_freestream)

    # --- STEP 3: COMPARAÇÃO ESTATÍSTICA (CDF e R²) ---
    # Ordena apenas as velocidades para construir a CDF empírica
    vel_array_ord = np.sort(np.array(velocidades_freestream))
    n_pontos = len(vel_array_ord)
    
    # CDF Empírica (distribuição real dos dados)
    cdf_empirica = np.arange(1, n_pontos + 1) / n_pontos
    
    # CDF Teórica (distribuição arco-seno ideal)
    # np.clip confina os valores no domínio [-1, 1] para evitar erro numérico no arcsin em dados com ruído além da amplitude
    termos_seno = np.clip((vel_array_ord - C_fit) / A_fit_abs, -1.0, 1.0)
    cdf_teorica = 0.5 + (1.0 / np.pi) * np.arcsin(termos_seno)
    
    # Métricas de similaridade
    erro_maximo_ks = np.max(np.abs(cdf_empirica - cdf_teorica)) # Distância máxima (KS statistic)
    
    ss_res = np.sum((cdf_empirica - cdf_teorica)**2) # Soma dos resíduos quadrados
    ss_tot = np.sum((cdf_empirica - np.mean(cdf_empirica))**2) # Variância total
    r_quadrado = 1 - (ss_res / ss_tot) # R²

    print(f"\n=== STATISTICAL PARAMETERS ===")
    print(f"Evaluated Region: y >= {y_corte_mm:.3f} mm")
    print(f"Calculated Amplitude (A): {A_fit_abs:.4f} m/s")
    print(f"Propagated Std Dev of Amplitude: {desvio_propagado:.4f} m/s")
    print(f"Mean Velocity Offset (C): {C_fit:.4f} m/s")
    print(f"\n=== GOODNESS OF FIT (CDF) ===")
    print(f"R-squared (R²): {r_quadrado:.4f} (1.0 = Perfect fit to Sine)")
    print(f"Max CDF Divergence (K-S stat): {erro_maximo_ks:.4f} (Closer to 0 is better)")

    # --- STEP 4: PLOTAGEM DO HISTOGRAMA EM INGLÊS ---
    fig, ax = plt.subplots(figsize=(9, 6.5))

    contagens, bins, _ = ax.hist(velocidades_freestream, bins=25, density=True, alpha=0.6, 
                                 color='blue', edgecolor='black', 
                                 label='Experimental PIV Data')
    
    u_teorico = np.linspace(C_fit - A_fit_abs + 0.005, C_fit + A_fit_abs - 0.005, 1000)
    pdf_teorica = 1.0 / (np.pi * np.sqrt(A_fit_abs**2 - (u_teorico - C_fit)**2))
    
    ax.plot(u_teorico, pdf_teorica, color='red', linewidth=2.5, 
             label=f'Theoretical Sine PDF (A = {A_fit_abs:.2f} m/s)')
    
    # Adiciona a anotação do R² no gráfico
    texto_stats = f"$R^2$ = {r_quadrado:.3f}\nMax Div = {erro_maximo_ks:.3f}"
    ax.text(0.05, 0.95, texto_stats, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'Velocity Distribution vs. Sine PDF (y ≥ {y_corte_mm:.2f} mm)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Freestream Velocity U [m/s]', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    
    ax.set_ylim(0, min(np.max(contagens) * 1.6, 2.0)) 
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(fontsize=11, loc='upper right') # Movido para a direita para não sobrepor o texto de stats

    plt.tight_layout()
    plt.show() 
    
    print("\nAnalysis completed!")

# --- EXECUÇÃO ---
pasta_dados = r'/home/emiliano/Documentos/IC/Pulsator-selected/260608/He plus air'
analisar_histograma_piv(pasta_dados, pct_corte=80, n_amostras=10, x_esquerda=5.0, x_direita=45.0)
