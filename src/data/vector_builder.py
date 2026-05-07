from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_ORDER = ["carbon", "nitrogen", "oxygen", "sulfur", "phosphorus", "silicon", "iron", "magnesium", "sodium", "potassium"]


def _clip(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def derive_env_vector_from_exoplanet(row: pd.Series) -> list[float]:
    """Derive a deterministic 10D chemistry vector from NASA exoplanet columns.

    Uses physically motivated proxies from available columns:
    - st_teff, st_met, st_mass, pl_orbper, pl_rade
    Missing values fall back to neutral priors without randomization.
    """
    teff = float(row.get("st_teff", 5500.0)) if pd.notna(row.get("st_teff")) else 5500.0
    met = float(row.get("st_met", 0.0)) if pd.notna(row.get("st_met")) else 0.0
    smass = float(row.get("st_mass", 1.0)) if pd.notna(row.get("st_mass")) else 1.0
    orbper = float(row.get("pl_orbper", 365.0)) if pd.notna(row.get("pl_orbper")) else 365.0
    rade = float(row.get("pl_rade", 1.0)) if pd.notna(row.get("pl_rade")) else 1.0

    # normalized proxies
    temp_norm = _clip((teff - 2600.0) / 8000.0)
    metal_norm = _clip((met + 1.0) / 2.0)
    mass_norm = _clip((smass - 0.08) / 2.5)
    orbper_norm = _clip(np.log10(max(orbper, 0.1)) / 3.5)
    rade_norm = _clip((rade - 0.3) / 14.0)

    carbon = _clip(0.45 + 0.35 * metal_norm)
    nitrogen = _clip(0.40 + 0.25 * mass_norm)
    oxygen = _clip(0.35 + 0.40 * temp_norm)
    sulfur = _clip(0.25 + 0.30 * metal_norm + 0.10 * (1 - orbper_norm))
    phosphorus = _clip(0.30 + 0.20 * mass_norm)
    silicon = _clip(0.28 + 0.35 * rade_norm)
    iron = _clip(0.30 + 0.55 * metal_norm)
    magnesium = _clip(0.30 + 0.30 * metal_norm + 0.10 * mass_norm)
    sodium = _clip(0.25 + 0.25 * (1 - orbper_norm) + 0.10 * rade_norm)
    potassium = _clip(0.22 + 0.20 * (1 - orbper_norm) + 0.10 * mass_norm)

    return [carbon, nitrogen, oxygen, sulfur, phosphorus, silicon, iron, magnesium, sodium, potassium]


def derive_env_vector_from_extremophile_meta(meta: dict) -> list[float]:
    """Derive 10D vector from extremophile metadata (temperature/pH/salinity/metal proxies)."""
    temp = float(meta.get("temperature_c", 37.0))
    ph = float(meta.get("ph", 7.0))
    salinity = float(meta.get("salinity_psu", 35.0))
    sulfur_rich = float(meta.get("sulfur_rich", 0.3))
    iron_rich = float(meta.get("iron_rich", 0.3))
    radiation = float(meta.get("radiation_index", 0.2))

    temp_norm = _clip((temp - 0.0) / 120.0)
    acidity = _clip((7.0 - ph) / 7.0)
    alkalinity = _clip((ph - 7.0) / 7.0)
    salinity_norm = _clip(salinity / 300.0)

    carbon = _clip(0.35 + 0.15 * temp_norm + 0.10 * (1 - salinity_norm))
    nitrogen = _clip(0.30 + 0.20 * temp_norm + 0.10 * radiation)
    oxygen = _clip(0.45 - 0.20 * temp_norm + 0.10 * alkalinity)
    sulfur = _clip(0.20 + 0.65 * sulfur_rich + 0.10 * acidity)
    phosphorus = _clip(0.30 + 0.10 * temp_norm + 0.05 * salinity_norm)
    silicon = _clip(0.25 + 0.20 * temp_norm + 0.05 * alkalinity)
    iron = _clip(0.20 + 0.70 * iron_rich + 0.10 * acidity)
    magnesium = _clip(0.25 + 0.35 * salinity_norm)
    sodium = _clip(0.20 + 0.75 * salinity_norm)
    potassium = _clip(0.20 + 0.50 * salinity_norm)

    return [carbon, nitrogen, oxygen, sulfur, phosphorus, silicon, iron, magnesium, sodium, potassium]
