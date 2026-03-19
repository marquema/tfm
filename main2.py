# main.py
#from src.dataset import generar_dataset_hibrido
from environment_trading import PortfolioEnv
import pandas as pd

#universo = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']
#features, precios = generar_dataset_hibrido(universo, "2024-02-01", "2026-03-01")

# Guarda los datos para no saturar a Yahoo Finance
#features.to_csv("data/features.csv")
#precios.to_csv("data/precios.csv")
#print("¡Dataset listo y guardado!")

# Test rápido en tu main
if __name__ == "__main__":
    df_f = pd.read_csv("data/features_normalizadas.csv", index_col=0)
    df_p = pd.read_csv("data/precios_originales.csv", index_col=0)
    
    env = PortfolioEnv(df_f, df_p)
    obs, _ = env.reset()
    
    for _ in range(5):
        accion_aleatoria = env.action_space.sample() # La IA "tonta"
        obs, reward, done, _, info = env.step(accion_aleatoria)
        print(f"Valor cartera: {info['value']:.2f}$ | Reward: {reward:.4f}")