"""
Gestión de la configuración del universo de activos.

Garantiza la coherencia entre los datos descargados, el modelo entrenado
y la simulación del inversor. Cada vez que se preparan datos con un conjunto
de activos, se guarda un fichero JSON que actúa como "contrato":

  data/universe_config.json

Cualquier componente posterior (entrenamiento, simulación, especulativo)
valida que su universo de activos coincide con el del dataset actual.
Si no coincide, lanza un error claro en lugar de crashear con KeyErrors
crípticos por columnas faltantes.

Esto resuelve el problema de entrenar con 8 activos y simular con 15,
o de usar un modelo especulativo ajustado con tickers del screener sobre
datos del universo core.
"""

import os
import json
from datetime import datetime
from typing import Optional


_CONFIG_PATH = os.path.join('data', 'universe_config.json')


def save_config(tickers: list, start_date: str, end_date: str,
                n_features: int, n_days: int) -> dict:
    """
    Guarda la configuración del universo actual tras preparar datos.

    Se llama desde el endpoint /admin/fase1/preparar-datos tras generar
    los CSVs exitosamente.

    Parameters
    ----------
    tickers    : lista de tickers usados (ej. ['IVV', 'BND', ...])
    start_date : fecha inicio del dataset
    end_date   : fecha fin del dataset
    n_features : número de columnas en normalized_features.csv
    n_days     : número de filas (días de trading)

    Returns
    -------
    dict con la configuración guardada.
    """
    config = {
        "tickers": sorted(tickers),
        "start_date": start_date,
        "end_date": end_date,
        "n_features": n_features,
        "n_days": n_days,
        "n_assets": len(tickers),
        "created_at": datetime.utcnow().isoformat(),
    }

    os.makedirs(os.path.dirname(_CONFIG_PATH), exist_ok=True)
    with open(_CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Configuración del universo guardada: {_CONFIG_PATH}")
    print(f"    {len(tickers)} activos: {tickers}")
    return config


def load_config() -> Optional[dict]:
    """
    Carga la configuración del universo actual.

    Returns
    -------
    dict con la configuración, o None si no existe el fichero
    (datos no preparados aún).
    """
    if not os.path.exists(_CONFIG_PATH):
        return None
    with open(_CONFIG_PATH, 'r') as f:
        return json.load(f)


def get_tickers_from_config() -> Optional[list]:
    """
    Retorna los tickers del universo actual (datos preparados).

    Returns
    -------
    list[str] o None si no hay configuración.
    """
    config = load_config()
    return config["tickers"] if config else None


def validate_compatibility(model_tickers: list, context: str = "modelo") -> bool:
    """
    Valida que los tickers de un modelo son compatibles con los datos actuales.

    Compara los tickers del modelo con los del dataset actual
    (universe_config.json). Si no coinciden, lanza un ValueError
    con un mensaje claro indicando qué sobra y qué falta.

    Parameters
    ----------
    model_tickers : lista de tickers con los que se entrenó el modelo
    context       : nombre del componente para el mensaje de error
                    (ej. "PPO", "Especulativo")

    Returns
    -------
    True si son compatibles.

    Raises
    ------
    ValueError si los tickers no coinciden.
    FileNotFoundError si no hay configuración (datos no preparados).
    """
    config = load_config()
    if config is None:
        raise FileNotFoundError(
            "No hay datos preparados. Ejecuta /admin/fase1/preparar-datos primero."
        )

    data_tickers = set(config["tickers"])
    model_set    = set(model_tickers)

    if data_tickers != model_set:
        missing = data_tickers - model_set
        extra   = model_set - data_tickers
        msg = (
            f"El {context} no es compatible con los datos actuales.\n"
            f"  Datos actuales: {sorted(data_tickers)}\n"
            f"  {context}: {sorted(model_set)}\n"
        )
        if missing:
            msg += f"  Faltan en el {context}: {sorted(missing)}\n"
        if extra:
            msg += f"  Sobran en el {context}: {sorted(extra)}\n"
        msg += f"  Solución: re-ejecuta el {context} con los datos actuales."
        raise ValueError(msg)

    return True


def save_model_metadata(model_dir: str, tickers: list, extra: dict = None):
    """
    Guarda metadatos del universo junto con un modelo entrenado.

    Crea un fichero universe_metadata.json en el directorio del modelo
    para poder validar la compatibilidad en el futuro.

    Parameters
    ----------
    model_dir : directorio donde se guarda el modelo (ej. 'models/best_model_academic/')
    tickers   : tickers con los que se entrenó
    extra     : datos adicionales (ej. n_features, fecha de entrenamiento)
    """
    metadata = {
        "tickers": sorted(tickers),
        "n_assets": len(tickers),
        "trained_at": datetime.utcnow().isoformat(),
    }
    if extra:
        metadata.update(extra)

    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'universe_metadata.json')
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def load_model_metadata(model_dir: str) -> Optional[dict]:
    """
    Carga los metadatos del universo de un modelo entrenado.

    Returns
    -------
    dict o None si no existe el fichero de metadatos.
    """
    path = os.path.join(model_dir, 'universe_metadata.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)
