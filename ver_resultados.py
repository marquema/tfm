import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.environment_trading import PortfolioEnv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.environment_trading import PortfolioEnv

def extraer_datos(log_dir):
    # Buscamos el archivo de eventos más reciente
    event_file = glob(f"{log_dir}/**/*.tfevents*", recursive=True)[-1]
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Extraemos la recompensa media (ep_rew_mean)
    if 'rollout/ep_rew_mean' in ea.Tags()['scalars']:
        df = pd.DataFrame(ea.Scalars('rollout/ep_rew_mean'))
        return df
    return None

def plot_training_progress(log_dir):
    # 1. Buscar el archivo de logs (tfevents)
    files = glob.glob(os.path.join(log_dir, "**/*tfevents*"), recursive=True)
    if not files:
        print("No se han encontrado archivos de log en ./logs/")
        return

    # 2. Cargar el último archivo generado
    latest_file = max(files, key=os.path.getctime)
    event_acc = EventAccumulator(latest_file)
    event_acc.Reload()

    # 3. Extraer la métrica de recompensa media
    # Stable Baselines 3 usa 'rollout/ep_rew_mean'
    if 'rollout/ep_rew_mean' in event_acc.Tags()['scalars']:
        # EXTRAER LOS DATOS CORRECTAMENTE
        events = event_acc.Scalars('rollout/ep_rew_mean')
        
        # Extraemos cada atributo de los objetos ScalarEvent
        step_nums = [e.step for e in events]
        vals = [e.value for e in events]
        
        plt.figure(figsize=(10, 5))
        plt.plot(step_nums, vals, label='Recompensa Media (Train)', color='#2ca02c', linewidth=2)
        
        # Añadimos una media móvil para suavizar la gráfica (opcional pero recomendado)
        if len(vals) > 10:
            suavizado = pd.Series(vals).rolling(window=10).mean()
            plt.plot(step_nums, suavizado, label='Tendencia (Media Móvil)', linestyle='--', color='red')

        plt.title('Progreso del Entrenamiento de la IA (Fase 3)')
        plt.xlabel('Pasos (Steps)')
        plt.ylabel('Log-Return Medio')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
    else:
        print("Aún no hay suficientes datos de 'rollout/ep_rew_mean'. ¡Espera un poco más!")

def ejecutar_backtest():
    print("--- Iniciando Backtest (Datos no vistos) ---")
    
    # 1. Cargar el mejor modelo guardado
    model_path = "models/best_model/best_model.zip"
    model = PPO.load(model_path)
    
    # 2. Crear el entorno de TEST (el mismo 20% que usamos en validación)
    df_f = pd.read_csv('data/features_normalizadas.csv')
    split_idx = int(len(df_f) * 0.8)
    env = PortfolioEnv('data/features_normalizadas.csv', 'data/precios_originales.csv', start_idx=split_idx)
    
    # 3. Ejecutar el episodio de test
    obs, _ = env.reset()
    done = False
    history_ia = []
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        history_ia.append(info['value'])
    
    # 4. Calcular el Benchmark: "Buy & Hold" (Invertir todo a partes iguales y no tocar)
    # Usamos los precios originales para el realismo
    precios_test = env.df_precios
    retornos_activos = precios_test.pct_change().dropna()
    # Estrategia: 1/N (reparto equitativo entre todos los activos)
    retorno_bh = retornos_activos.mean(axis=1) 
    cum_bh = 10000 * (1 + retorno_bh).cumprod() # Empezando con los mismos 10k
    
    # 5. Visualización de Resultados
    plt.figure(figsize=(12, 6))
    plt.plot(history_ia, label='IA Portfolio Manager (PPO)', color='blue', linewidth=2)
    plt.plot(cum_bh.values, label='Estrategia Buy & Hold (1/N)', color='gray', linestyle='--')
    
    plt.title('Comparativa Final: IA vs Mercado (Datos de Test 2025-2026)')
    plt.xlabel('Días de Negociación')
    plt.ylabel('Valor de la Cartera ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Métricas finales
    print(f"Resultado Final IA: ${history_ia[-1]:.2f}")
    print(f"Resultado Final B&H: ${cum_bh.values[-1]:.2f}")

def calcular_sharpe(retornos, rf=0.0):
    """Calcula el Sharpe Ratio anualizado (asumiendo 252 días de trading)"""
    if len(retornos) < 2: return 0
    mean_ret = retornos.mean()
    std_ret = retornos.std()
    if std_ret == 0: return 0
    return (mean_ret - rf) / std_ret * np.sqrt(252)

def ejecutar_backtest_pro():
    print("--- Iniciando Backtest de Alta Precisión ---")
    
    # 1. Cargar modelo y entorno de Test (20% final)
    model = PPO.load("models/best_model/best_model.zip")
    df_f = pd.read_csv('data/features_normalizadas.csv')
    split_idx = int(len(df_f) * 0.8)
    env = PortfolioEnv('data/features_normalizadas.csv', 'data/precios_originales.csv', start_idx=split_idx)
    
    # 2. Simulación con la IA
    obs, _ = env.reset()
    done = False
    valores_ia = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        valores_ia.append(info['value'])
    
    # 3. Datos para el Benchmark (Buy & Hold 1/N)
    precios_test = env.df_precios
    retornos_activos = precios_test.pct_change().dropna()
    retorno_bh = retornos_activos.mean(axis=1)
    valores_bh = 10000 * (1 + retorno_bh).cumprod().values

    # 4. CÁLCULO DE MÉTRICAS
    df_res = pd.DataFrame({
        'IA': valores_ia,
        'BH': np.insert(valores_bh, 0, 10000)[:len(valores_ia)] # Sincronizar longitudes
    })
    
    rets = df_res.pct_change().dropna()
    
    sharpe_ia = calcular_sharpe(rets['IA'])
    sharpe_bh = calcular_sharpe(rets['BH'])
    
    # Max Drawdown: La mayor caída desde un máximo histórico
    dd_ia = (df_res['IA'] / df_res['IA'].cummax() - 1).min()
    dd_bh = (df_res['BH'] / df_res['BH'].cummax() - 1).min()

    # 5. RESULTADOS POR PANTALLA (Para tu tabla del TFM)
    print("\n" + "="*30)
    print(f"{'MÉTRICA':<15} | {'IA (PPO)':<10} | {'B&H (1/N)':<10}")
    print("-" * 30)
    print(f"{'Final Value':<15} | ${df_res['IA'].iloc[-1]:>8.2f} | ${df_res['BH'].iloc[-1]:>8.2f}")
    print(f"{'Sharpe Ratio':<15} | {sharpe_ia:>10.2f} | {sharpe_bh:>10.2f}")
    print(f"{'Max Drawdown':<15} | {dd_ia*100:>9.2f}% | {dd_bh*100:>9.2f}%")
    print("="*30)

    # 6. GRÁFICA PARA LA MEMORIA
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['IA'], label=f'IA PPO (Sharpe: {sharpe_ia:.2f})', color='#1f77b4', lw=2)
    plt.plot(df_res['BH'], label=f'Benchmark B&H (Sharpe: {sharpe_bh:.2f})', color='#ff7f0e', linestyle='--')
    plt.fill_between(range(len(df_res)), df_res['IA'], df_res['IA'].cummax(), color='red', alpha=0.1, label='IA Drawdown Area')
    
    plt.title('Validación Ex-Post: IA Portfolio Manager vs Benchmark')
    plt.legend()
    plt.show()    


ejecutar_backtest()
plot_training_progress("./logs/")
ejecutar_backtest_pro()

df = extraer_datos("./logs/")
if df is not None:
    plt.plot(df['step'], df['value'])
    plt.title("Progreso de la IA (Reward)")
    plt.xlabel("Pasos")
    plt.ylabel("Recompensa Media")
    plt.show()
else:
    print("Aún no hay datos suficientes o el tag es distinto.")

