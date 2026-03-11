# main.py
from src.dataset import generar_dataset_hibrido

universo = ['IVV', 'BND', 'IBIT', 'MO', 'JNJ', 'SCU', 'AWK', 'CB']
features, precios = generar_dataset_hibrido(universo, "2024-02-01", "2026-03-01")

# Guarda los datos para no saturar a Yahoo Finance
features.to_csv("data/features.csv")
precios.to_csv("data/precios.csv")
print("¡Dataset listo y guardado!")