import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from src.environment_trading import PortfolioEnv

st.set_page_config(page_title="IA Portfolio Manager - TFM", layout="wide")

st.title("🚀 AI-Driven Portfolio Management Dashboard")
st.markdown("### Microservicio de Inferencia y Análisis de Rendimiento")

# --- BARRA LATERAL ---
st.sidebar.header("Configuración")
model_path = st.sidebar.text_input("Ruta del Modelo", "models/best_model/best_model.zip")
commission = st.sidebar.slider("Comisión por Operación (%)", 0.0, 0.5, 0.1) / 100

# --- CARGAR DATOS ---
@st.cache_data
def load_data():
    df_f = pd.read_csv('data/features_normalizadas.csv')
    df_p = pd.read_csv('data/precios_originales.csv')
    return df_f, df_p

df_f, df_p = load_data()

# --- EJECUTAR BACKTEST ---
if st.button('Lanzar Simulación en Datos de Test'):
    with st.spinner('La IA está operando en el mercado...'):
        # Inicializar entorno de test (20% final)
        split_idx = int(len(df_f) * 0.8)
        env = PortfolioEnv('data/features_normalizadas.csv', 'data/precios_originales.csv', 
                           start_idx=split_idx, commission=commission)
        model = PPO.load(model_path)
        
        obs, _ = env.reset()
        done = False
        history = []
        weights_history = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            history.append(info['value'])
            weights_history.append(action / (np.sum(action) + 1e-6))


        col1, col2, col3 = st.columns(3)
        final_val = history[-1]
        ret_total = ((final_val / 10000) - 1) * 100
        
        col1.metric("Valor Final Cartera", f"${final_val:,.2f}", f"{ret_total:.2f}%")
        col2.metric("Nº de Operaciones", len(history))
        col3.metric("Estado del Modelo", "Ready / Inferred")

        st.subheader("Evolución del Capital (Equity Curve)")
        st.line_chart(history)


        st.subheader("Asset Allocation Actual (Decisión de la IA)")
        

        last_weights = np.array(weights_history[-1]).flatten()
        
        # 2. DEBUG (Opcional: puedes borrar esto después de que funcione)
        st.write(f"Dimensiones de pesos: {last_weights.shape}") 
        tickers = df_p.columns[-7:]
        
        #print(last_weights)
        #df_aux = df_p[[:,-7]]
        #print(df_aux)
        #print(df_p.columns[:,-7])
        # 3. Solo dibujamos si coinciden las longitudes
        if len(last_weights) == len(tickers):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(
                last_weights, 
                labels=tickers, 
                autopct='%1.1f%%', 
                startangle=90, 
                colors=plt.cm.Paired.colors,
                wedgeprops={'edgecolor': 'white'}
            )
            ax.axis('equal') 
            st.pyplot(fig)
        else:
            st.error(f"Error de dimensiones: La IA devolvió {len(last_weights)} pesos pero tenemos {len(df_p.columns)} activos.")