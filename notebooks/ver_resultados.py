import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob
import os

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
        w_times, step_nums, vals = zip(*event_acc.Scalars('rollout/ep_rew_mean'))
        
        plt.figure(figsize=(10, 5))
        plt.plot(step_nums, vals, label='Recompensa Media (Train)')
        plt.title('Progreso del Entrenamiento de la IA')
        plt.xlabel('Pasos (Steps)')
        plt.ylabel('Log-Return Medio')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print("Aún no hay suficientes datos de 'rollout/ep_rew_mean'. ¡Espera un poco más!")

if __name__ == "__main__":
    plot_training_progress("../logs/")