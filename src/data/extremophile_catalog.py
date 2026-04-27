from __future__ import annotations

# Curated Earth extremophiles + approximate normalized environment vectors
# [C, N, O, S, P, Si, Fe, Mg, Na, K]
EXTREMOPHILE_CATALOG = [
    {
        "organism": "Pyrococcus furiosus",
        "habitat": "hydrothermal_vent",
        "env_vector": [0.45, 0.35, 0.15, 0.85, 0.30, 0.20, 0.80, 0.60, 0.50, 0.40],
        "function_label": 6,
    },
    {
        "organism": "Thermus aquaticus",
        "habitat": "hot_spring",
        "env_vector": [0.50, 0.40, 0.30, 0.55, 0.35, 0.25, 0.50, 0.45, 0.30, 0.35],
        "function_label": 5,
    },
    {
        "organism": "Halobacterium salinarum",
        "habitat": "hypersaline_lake",
        "env_vector": [0.55, 0.40, 0.60, 0.45, 0.30, 0.20, 0.35, 0.40, 0.95, 0.70],
        "function_label": 4,
    },
    {
        "organism": "Acidithiobacillus ferrooxidans",
        "habitat": "acid_mine_drainage",
        "env_vector": [0.40, 0.35, 0.40, 0.80, 0.30, 0.20, 0.90, 0.45, 0.25, 0.30],
        "function_label": 7,
    },
    {
        "organism": "Deinococcus radiodurans",
        "habitat": "radiation_exposed_soil",
        "env_vector": [0.50, 0.45, 0.55, 0.40, 0.35, 0.20, 0.35, 0.35, 0.30, 0.35],
        "function_label": 3,
    },
]
