from pathlib import Path
import io

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CLEANED_DATA_PATH = BASE_DIR / "data" / "cleaned_environmental_data_10K.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "figures" / "exoplanet_vs_environmental_comparison.png"

def fetch_exoplanet_data():
    """
    Fetches confirmed exoplanets and their environmental parameters 
    from the NASA Exoplanet Archive.
    """
    url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
    
    # We write an ADQL (Astronomical Data Query Language) query to get specific columns
    # pl_name = Planet Name
    # pl_eqt = Equilibrium Temperature (K)
    # pl_insol = Insolation Flux (Earth Flux)
    # pl_rade = Planet Radius (Earth Radii)
    # st_spectype = Star Spectral Type (e.g., G2V for our Sun)
    
    query = """
        SELECT pl_name, pl_eqt, pl_insol, pl_rade, st_spectype, pl_orbper
        FROM pscomppars
        WHERE pl_eqt > 200 AND pl_eqt < 400
        AND pl_insol IS NOT NULL
    """
    
    params = {
        "request": "doQuery",
        "lang": "ADQL",
        "format": "csv",
        "query": query
    }
    
    print("Querying NASA Exoplanet Archive...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    
    # Load into Pandas
    df = pd.read_csv(io.StringIO(response.text))
    
    # Convert Kelvin to Celsius so it matches your UniProt/GOLD data
    df['pl_temp_C'] = df['pl_eqt'] - 273.15

    # Convert Insolation Flux to kJ/m^2/day (1 Earth Flux = 1361 W/m^2)
    df['pl_insol_kJ_m2_day'] = df['pl_insol'] * 117590
    
    return df


def load_cleaned_environmental_data(csv_path=DEFAULT_CLEANED_DATA_PATH):
    """Load the cleaned environmental dataset used for comparison."""
    df = pd.read_csv(csv_path)
    df["NASA_Temp_C"] = pd.to_numeric(df["NASA_Temp_C"], errors="coerce")
    
    # Calculate average solar radiation from srad columns
    srad_cols = [f'srad_{str(i).zfill(2)}' for i in range(1, 13)]
    srad_data = df[srad_cols].apply(pd.to_numeric, errors="coerce")
    df["Mean_Annual_Solar_Rad"] = srad_data.mean(axis=1)
    
    return df.dropna(subset=["NASA_Temp_C", "Mean_Annual_Solar_Rad"])


def print_dataset_summary(label, df, temp_col, rad_col):
    """Print a compact comparison summary for a dataset."""
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Rows used: {len(df):,}")
    print(f"Temperature: mean={df[temp_col].mean():.2f}, median={df[temp_col].median():.2f}, min={df[temp_col].min():.2f}, max={df[temp_col].max():.2f}")
    print(f"Solar radiation: mean={df[rad_col].mean():.2f}, median={df[rad_col].median():.2f}, min={df[rad_col].min():.2f}, max={df[rad_col].max():.2f}")


def create_comparison_plot(exoplanet_df, environmental_df, output_path=DEFAULT_OUTPUT_PATH):
    """Create a layered comparison plot for temperature and solar radiation."""
    sns.set_theme(style="whitegrid", context="talk")

    exo_plot = exoplanet_df[["pl_temp_C", "pl_insol_kJ_m2_day"]].copy()
    env_plot = environmental_df[["NASA_Temp_C", "Mean_Annual_Solar_Rad"]].copy()

    exo_plot = exo_plot.dropna()
    env_plot = env_plot.dropna()

    exo_plot = exo_plot[exo_plot["pl_insol_kJ_m2_day"] > 0]
    env_plot = env_plot[env_plot["Mean_Annual_Solar_Rad"] > 0]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(
        env_plot["NASA_Temp_C"],
        env_plot["Mean_Annual_Solar_Rad"],
        s=18,
        alpha=0.18,
        color="#6c757d",
        edgecolors="none",
        label=f"Earth data ({len(env_plot):,})",
    )

    ax.scatter(
        exo_plot["pl_temp_C"],
        exo_plot["pl_insol_kJ_m2_day"],
        s=50,
        alpha=0.9,
        color="#d62828",
        marker="^",
        edgecolors="black",
        linewidths=0.25,
        label=f"Exoplanets ({len(exo_plot):,})",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Solar Radiation (kJ/m²/day, log scale)")
    ax.set_title("Exoplanets vs Earth Data")
    ax.legend(frameon=True)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison plot saved to {output_path}")

if __name__ == "__main__":
    exo_df = fetch_exoplanet_data()
    env_df = load_cleaned_environmental_data()

    print(f"Fetched {len(exo_df)} exoplanets with temperature data.")
    print(f"Loaded {len(env_df)} cleaned environmental rows.")

    print_dataset_summary("Exoplanet dataset", exo_df, "pl_temp_C", "pl_insol_kJ_m2_day")
    print_dataset_summary("Earth dataset", env_df, "NASA_Temp_C", "Mean_Annual_Solar_Rad")

    create_comparison_plot(exo_df, env_df)